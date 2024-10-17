import argparse
from collections import Counter, defaultdict, namedtuple
import logging
import os, json, random
from pathlib import Path
import sys
import time
import PIL.Image as Image
import copy
import math
from collections import deque
import torch
# from thop import profile
from fvcore.nn import FlopCountAnalysis
# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from fvcore.nn import FlopCountAnalysis

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env
from robot_flamingo.data.data import preprocess_image, preprocess_text_calvin
from robot_flamingo.utils import world_to_tcp_frame, tcp_to_world_frame
import functools
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = -1

def merge_multi_list(res):
    tmp = []
    for l in res:
        tmp.extend(l)
    return tmp

def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success

def count_exit_ratio(exit_results, n_layers):
    count = Counter(exit_results)
    layer_ratio = []
    for i in range(0, n_layers):
        sr = count[i] / len(exit_results)
        layer_ratio.append(sr)
    return layer_ratio
    

def print_and_save(results, success_exit_results, fail_exit_results, step_results, success_llm_time_list, fail_llm_time_list, sequences, log_dir, n_layer, epoch=None):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    step_results = np.array(step_results)
    print(step_results.shape)
    print(step_results)
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")
    

    success_layer_ratio = count_exit_ratio(success_exit_results, n_layer)
    fail_layer_ratio = count_exit_ratio(fail_exit_results, n_layer)
    print(f"Early Exit (success tasks) | Total steps : {len(success_exit_results)} | VLM n_layer: {n_layer} | Average : {np.mean(success_exit_results)+1:.1f} | Min : {np.min(success_exit_results)+1} | Max : {np.max(success_exit_results)+1} | AVG LLM time: {np.mean(success_llm_time_list)*1000:.1f}ms")
    print(f"Early Exit (fail tasks) | Total steps : {len(fail_exit_results)} | VLM n_layer: {n_layer} | Average : {np.mean(fail_exit_results)+1:.1f} | Min : {np.min(fail_exit_results)+1} | Max : {np.max(fail_exit_results)+1} | AVG LLM time: {np.mean(fail_llm_time_list)*1000:.1f}ms")
    
    print(f"Total Successful steps: {np.sum(step_results)} | Avg steps per successful subtask: {np.mean(step_results):.1f} | Min: {np.min(step_results)} | Max: {np.max(step_results)}")
    
    print("Early exit rates for layer i in successful tasks:")
    for i, exit_ratio in enumerate(success_layer_ratio):
        print(f'{i+1}: {exit_ratio * 100:.1f}%')
    print("Early exit rates for layer i in failed tasks:")
    for i, exit_ratio in enumerate(fail_layer_ratio):
        print(f'{i+1}: {exit_ratio * 100:.1f}%')

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    sorted_tasks = sorted(total.keys())
    task_info = {}
    for task in sorted_tasks:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    current_data[epoch] = data

    print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )
    
    return avg_seq_len, np.mean(success_exit_results)+1


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype

def check_loaded_parameters(model, checkpoint):
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint.keys())

    # Check if there are keys in the checkpoint that are not in the model
    extra_keys = checkpoint_keys - model_keys
    if extra_keys and torch.distributed.get_rank() == 0:
        raise KeyError(f'{len(extra_keys)} keys in the checkpoint were not found in the model: {extra_keys}')

    # Check if there are keys in the model that are not in the checkpoint
    # missing_keys = model_keys - checkpoint_keys
    # if missing_keys and torch.distributed.get_rank() == 0:
    #     print(f'Warning: {len(missing_keys)} keys in the model were not found in the checkpoint: {missing_keys}')



def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env

class DebugEnv():
    
    def __init__(self) -> None:
        pass
    
    def get_random_obs(self):
        obs = {}
        obs['rgb_obs'] = {}
        obs['rgb_obs']['rgb_static'] = np.ones((200, 200, 3), dtype=np.uint8)
        obs['rgb_obs']['rgb_gripper'] = np.ones((84, 84, 3), dtype=np.uint8)
        obs['robot_obs'] = np.ones(15, dtype=np.float32)
        return obs
    
    def get_obs(self):
        return self.get_random_obs()
    
    def step(self, action):
        return self.get_random_obs()

    def reset(self, **kwargs):
        return

    def get_info(self):
        return


def make_env_debug(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class ModelWrapper(CalvinBaseModel):
    def __init__(self, model, tokenizer, image_processor, cast_dtype, use_diff, history_len=None, future_act_len=-1,
                 amp=False, exit_id=None, early_exit=False, exit_controller=None, multi_execution=1, use_action_ensemble=False):
        super().__init__()
        self.model = model
        self.replan = model.module.replan
        self.decoder_type = model.module.decoder_type
        self.cast_type = cast_dtype
        self.use_diff = use_diff
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.feature_cache = None
        self.dt_feat_cache = []
        self.fusion_mode = self.model.module.fusion_mode
        self.amp = amp
        self.head_type = model.module.head_type
        if self.amp and torch.distributed.get_rank()==0:
            print("Enable AMP during inference!")
        self.exit_id = exit_id
        self.dynamic_early_exit = early_exit
        self.exit_controller = exit_controller
        self.multi_execution = multi_execution
        self.use_action_ensemble = use_action_ensemble
        self.current_exit_layer = exit_id if isinstance(exit_id, int) else -1
        
        if self.model.module.act_step==1:
            if self.multi_execution > 1:
                if torch.distributed.get_rank()==0:
                    print(f'Repeatedly execute one predicted action {self.multi_execution} times!')
        else:
            assert self.multi_execution <= self.model.module.act_step
            if torch.distributed.get_rank()==0:
                print(f'Predict {self.model.module.act_step} actions and executed {self.multi_execution} actions!')
        
        if use_diff:
            self.diffusion_model = None
            self.normalizer = None
            if isinstance(self.model, DistributedDataParallel):
                self.diffusion_model = self.model.module.diffusion_model
            else:
                self.diffusion_model = self.model.diffusion_model
            action_dim = self.diffusion_model.data_dim
            horizon = self.diffusion_model.horizon
            self.normalizer = self.diffusion_model.normalizer
            self.action_hist_queue = deque(maxlen=history_len-1)
            self.action_hist_queue.extend([np.zeros(action_dim) for _ in range(history_len-1)])

            if horizon-history_len+1:
                self.supp = None
            self.hist_len = history_len-1
            self.action_dim = action_dim
            self.horizon = horizon
            self.future_act_len = future_act_len
        
        # if self.model.module.pad_length != -1:
        if self.model.module.pad_length == -1:
            history_len = self.model.module.window_size
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        

    def reset(self):
        """
        This is called
        """
        if self.use_diff:
            self.action_hist_queue = deque(maxlen=self.hist_len)
            self.action_hist_queue.extend([np.zeros(self.action_dim) for _ in range(self.hist_len)])
        if self.model.module.pad_length != -1:
            history_len = self.model.module.pad_length
        else:
            history_len = self.model.module.window_size
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.feature_cache = None
        self.dt_feat_cache = []
        
        # clear LSTM hidden states
        self.model.module.clear_all_exit_memory()
        
        # LSTM value net
        if self.exit_controller is not None:
            self.exit_controller.module.value_net.hidden_state = None
            self.exit_controller.module.value_net.history_memory = []

    def step(self, obs, goal, get_action=True,):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        
        if not get_action:
            return None

        # preprocess image
        image = obs["rgb_obs"]['rgb_static']
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image])
        # expand image dimension
        image_x = image_x.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
        # expand text dimension
        text_x, mask = self.text_process_fn([goal])

        assert self.model.module.pad_length == -1, "Padding for multi-exit net is not implemented. It requires store feature_cache for each exit separately."
        # fix window_size : ddp_model -> ... -> window_size
        # Inference: set lstm window_size = 1 and use hidden_state_memory
        # training: set window_size = 12 and set hidden_state = None and calculate it from scratch
        if self.model.module.pad_length != -1 and self.feature_cache is None:
            window_size = self.model.module.set_all_exit_window_size(self.model.module.pad_length)
            if self.exit_controller is not None:
                self.exit_controller.module.value_net.window_size = self.model.module.pad_length
        else:
            window_size = self.model.module.set_all_exit_window_size(1)
            if self.exit_controller is not None:
                self.exit_controller.module.value_net.window_size = 1

        gripper = None
        state = None

        if self.model.module.use_gripper:
            gripper = obs["rgb_obs"]['rgb_gripper']
            gripper = Image.fromarray(gripper)
            gripper = self.image_process_fn([gripper])
            # expand image dimension
            gripper = gripper.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
        
        # if self.model.module.use_state or self.model.module.sep_lm_head:
        if self.model.module.use_state or self.model.module.sep_lm_head:
            state = obs['robot_obs']
            state = torch.from_numpy(np.stack([state]))
            # if self.model.module.sep_lm_head:
            #     state = torch.cat([state[...,:6], state[...,[-1]]], dim=-1)
            if self.fusion_mode == 'two_way':
                state = state.repeat(2, 1)
            state = state.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
            state = state.to(torch.float32)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp):
            device = 'cuda'
            image_x = image_x.to(device)
            text_x = text_x.to(device)
            mask = mask.to(device)
            if gripper is not None:
                gripper = gripper.to(device)
            if state is not None:
                state = state.to(device)
            
            # if self.model.module.pad_length != -1:
            if len(self.img_queue) == 0:
                self.img_queue.append(image_x)
                for _ in range(self.model.module.pad_length - 1):
                    self.img_queue.append(image_x)
            else:
                self.img_queue.append(image_x)
            if len(self.gripper_queue) == 0 and gripper is not None:
                self.gripper_queue.append(gripper)
                for _ in range(self.model.module.pad_length - 1):
                    self.gripper_queue.append(gripper)
            else:
                self.gripper_queue.append(gripper)
            if len(self.state_queue) == 0 and state is not None:
                self.state_queue.append(state)
                for _ in range(self.model.module.pad_length - 1):
                    self.state_queue.append(state)
            else:
                self.state_queue.append(state)
            if len(self.mask_queue) == 0 and mask is not None:
                self.mask_queue.append(mask)
                for _ in range(self.model.module.pad_length - 1):
                    self.mask_queue.append(mask)
            if len(self.text_queue) == 0 and text_x is not None:
                self.text_queue.append(text_x)
                for _ in range(self.model.module.pad_length - 1):
                    self.text_queue.append(text_x)
            
            if self.model.module.pad_length != -1 and self.feature_cache is None:
                image_x = torch.cat(list(self.img_queue), dim=0)
                if gripper is not None:
                    gripper = torch.cat(list(self.gripper_queue), dim=0)
                if state is not None:
                    state = torch.cat(list(self.state_queue), dim=0)
                mask = torch.cat(list(self.mask_queue), dim=0)
                text_x = torch.cat(list(self.text_queue), dim=0)
            
            if self.fusion_mode == 'vit_concat':
                image_x = torch.cat(list(self.img_queue), dim=0)
                if gripper is not None:
                    gripper = torch.cat(list(self.gripper_queue), dim=0)
                if state is not None:
                    state = torch.cat(list(self.state_queue), dim=0)
                pass

            if self.use_diff:
                if self.fusion_mode == 'two_way':
                    vision_x = torch.cat([image_x, gripper], dim=0)
                    text_x = text_x.repeat(2, 1)
                    mask = mask.repeat(2, 1)
                    model_out = self.model(vision_x=vision_x, lang_x=text_x, attention_mask=mask, state_tensor = state, return_feature=True)
                else:
                    model_out = self.model(vision_x=image_x, lang_x=text_x, attention_mask=mask, vision_gripper = gripper, state_tensor = state, return_feature=True)

                if not get_action:
                    return None
                model_out = model_out.logits
                action_history = torch.tensor(np.stack(self.action_hist_queue, axis=0), dtype=torch.float, device=device).unsqueeze(0)
                action_history = self.normalizer.normalize(action_history)
                if self.supp is None:
                    self.supp = torch.zeros(
                        action_history.shape[0], self.horizon-self.hist_len, action_history.shape[-1], 
                        dtype=action_history.dtype,
                        device=action_history.device,
                    )
                action_history = torch.concat([action_history, self.supp], dim=1)
                act_mask = torch.zeros_like(action_history, device=action_history.device, dtype=torch.bool)
                act_mask[:,:self.hist_len,...] = 1.
                pred_action_seq = self.diffusion_model.conditional_sample(cond_data=action_history, cond_mask=act_mask, global_cond=model_out)
                pred_action_seq = self.normalizer.unnormalize(pred_action_seq)
                action = pred_action_seq[:,self.hist_len:,:]
                if self.future_act_len > 0:
                    action = action[:,:self.future_act_len,:]
                action = action[0]
                action = action.cpu().detach().to(dtype=torch.float16).numpy()
                action[...,-1] = action[...,-1] > 0.5
                action[...,-1] = (action[...,-1] - 0.5) * 2  # scale to -1 or 1
            else:
                if self.fusion_mode == 'two_way':
                    vision_x = torch.cat([image_x, gripper], dim=0)
                    text_x = text_x.repeat(2, 1)
                    mask = mask.repeat(2, 1)
                    assert False, "please update the code for dynamic exit"
                    action = self.model(vision_x=vision_x, lang_x=text_x, attention_mask=mask, state_tensor = state, return_feature=True)
                else:
                    eval_time=False
                    if eval_time:
                        torch.cuda.synchronize()
                        cur_time = time.time()                    
                    
                    action = self.model(vision_x=image_x, lang_x=text_x, attention_mask=mask, vision_gripper = gripper, state_tensor = state, return_feature=True, 
                                        deterministic=True, exit_id=self.exit_id, dynamic_early_exit=self.dynamic_early_exit, exit_controller=self.exit_controller)
                    
                    if eval_time:
                        torch.cuda.synchronize()
                        print(f"total time: {cur_time-time.time():.4f} seconds")
                        cur_time = time.time()
                # if self.model.module.pad_length != -1:
                #     if self.feature_cache is None:
                #         self.feature_cache = action.logits[-1]
                #     else:
                #         new_feat = torch.cat([self.feature_cache[1:], action.logits[-1]], dim=0)
                #         self.feature_cache = new_feat
                #         if not self.model.module.sep_lm_head:
                #             self.model.module.lang_encoder.lm_head.window_size = window_size
                #             lm_out = self.model.module.lang_encoder.lm_head(new_feat)
                #         else:
                #             self.model.module.lm_head.window_size = window_size
                #             lm_out = self.model.module.lm_head(new_feat)
                #         Output = namedtuple('Output', ['logits'])
                #         action = Output(lm_out)
                if hasattr(action, 'exit_layer'):
                    self.current_exit_layer = action.exit_layer
                if self.model.module.act_step == 1:
                    if not self.use_action_ensemble:
                        action = torch.concat((action.logits[0], action.logits[1] > 0.5), dim=2).squeeze(0)[-1] # support multi step history
                    else:
                        action = self.exit_controller.module.value_net.get_ensemble_action()
                        self.exit_controller.module.value_net.reset_actions()
                        action = torch.concat((action[0], action[1] > 0.5), dim=2).squeeze(0)[-1]
                        
                    action[-1] = (action[-1] - 0.5) * 2  # scale to -1 or 1
                    action = torch.stack([action]*self.multi_execution, dim=0)
                    
                else:
                    pose = action.logits[0]
                    gripper = action.logits[1] > 0.5
                    pose = pose.squeeze(0)[-1].view(self.model.module.act_step, -1)
                    gripper = gripper.squeeze(0)[-1].view(self.model.module.act_step, -1)
                    action = torch.cat([pose, gripper], dim=-1)
                    # action = action[0] # select first step action
                    action[:, -1] = (action[:, -1] - 0.5) * 2  # scale to -1 or 1
                    action = action[:self.multi_execution]
                
                action = action.cpu().detach().to(dtype=torch.float16).numpy()
                
        
        self.model.module.set_all_exit_window_size(window_size)
        
        if self.model.module.tcp_rel:
            raise NotImplementedError
            state = obs['robot_obs']
            state = torch.from_numpy(np.stack([state])).unsqueeze(0).float().cpu().detach()
            action = torch.from_numpy(np.stack([action])).unsqueeze(0).float().cpu().detach()
            action = tcp_to_world_frame(action, state)
            action=action.squeeze().to(dtype=torch.float16).numpy()
            
        return action


def evaluate_policy(model, env, epoch, calvin_conf_path, eval_log_dir=None, debug=False, create_plan_tsne=False, diverse_inst=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    if diverse_inst:
        with open('./enrich_lang_annotations.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_policy_ddp(model, env, epoch, calvin_conf_path, eval_log_dir=None, debug=False, create_plan_tsne=False, reset=False, diverse_inst=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open('./lang_annotation_cache.json', 'r') as f:
            val_annotations = json.load(f)
            print('Enable enriched annotation eval!')
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    with open('./eval_sequences.json', 'r') as f:
        eval_sequences = json.load(f)
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    assert NUM_SEQUENCES % device_num == 0
    interval_len = int(NUM_SEQUENCES // device_num)
    eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
    results = []
    success_exit_layers_list = [] # seq, subtask, timestep
    fail_exit_layers_list = [] # seq, subtask, timestep
    success_task_num_list = [] # seq, subtask
    fail_llm_time_list = []
    success_llm_time_list = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len
    n_layer = model.model.module.lang_encoder.config.n_layers

    eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result, success_seq_exit_layers, fail_seq_exit_layers, seq_success_task_steps, \
            success_llm_time, fail_llm_time = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir, base_sequence_i+local_sequence_i, reset=reset, diverse_inst=diverse_inst)
        success_exit_layers_list.append(success_seq_exit_layers)
        fail_exit_layers_list.append(fail_seq_exit_layers)
        success_task_num_list.append(seq_success_task_steps)
        success_llm_time_list.append(success_llm_time)
        fail_llm_time_list.append(fail_llm_time)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
            print("")
        local_sequence_i += 1

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]
    
    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    eval_sequences = extract_iter_from_tqdm(eval_sequences)

    res_tup = list(zip(results, success_exit_layers_list, fail_exit_layers_list, success_task_num_list, 
                       success_llm_time_list, fail_llm_time_list, eval_sequences))
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        
        res_list, success_exit_list, fail_exit_list, step_list, \
            success_llm_time_list, fail_llm_time_list, eval_seq_list = map(list, zip(*res_tup_list))

        ret = print_and_save(res_list, merge_multi_list(success_exit_list),merge_multi_list(fail_exit_list), 
                       merge_multi_list(step_list), merge_multi_list(success_llm_time_list), merge_multi_list(fail_llm_time_list),
                       eval_seq_list, eval_log_dir, n_layer, epoch)

        return ret


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir='', sequence_i=-1, reset=False, diverse_inst=False):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    success_task_step_list = []
    fail_exit_layers_list = []
    success_exit_layers_list = []
    fail_llm_time_list = []
    success_llm_time_list = []
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            success, exit_layers, num_steps, llm_time = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i, robot_obs=robot_obs, scene_obs=scene_obs, diverse_inst=diverse_inst)
        else:
            success, exit_layers, num_steps, llm_time = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i,diverse_inst=diverse_inst)
        
        n_layer = model.model.module.lang_encoder.config.n_layers
        layer_ratio = count_exit_ratio(exit_layers, n_layer)
        if subtask_i == 0: print('')
        print(("(success) " if success else "(fail) ") + f'{num_steps} steps: ' +  " ".join([f"{i + 1}/{n_layer} : {v * 100:.1f}% |" for i, v in enumerate(layer_ratio)]) + "|")
        
        if success:
            success_counter += 1
            success_task_step_list.append(num_steps)
            success_exit_layers_list.append(exit_layers)
            success_llm_time_list.append(llm_time)
        else:
            fail_exit_layers_list.append(exit_layers)
            fail_llm_time_list.append(llm_time)
            return success_counter, merge_multi_list(success_exit_layers_list), merge_multi_list(fail_exit_layers_list), success_task_step_list, merge_multi_list(success_llm_time_list), merge_multi_list(fail_llm_time_list)
    return success_counter, merge_multi_list(success_exit_layers_list), merge_multi_list(fail_exit_layers_list), success_task_step_list, merge_multi_list(success_llm_time_list), merge_multi_list(fail_llm_time_list)


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, eval_log_dir='', subtask_i=-1, sequence_i=-1, robot_obs=None, scene_obs=None, diverse_inst=False):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    planned_actions = []
    exit_layers = []
    llm_time_list = []
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split('\n')[0]
    if '\u2019' in lang_annotation:
        lang_annotation.replace('\u2019', '\'')
    model.reset()
    start_info = env.get_info()

    if debug:
        img_queue = []

    for step in range(EP_LEN):
        if model.replan != -1 and step % model.replan == 0:
            if model.model.module.refresh != -1:
                model.model.module.lang_encoder.lm_head.hidden_state = None
                model.model.module.lang_encoder.lm_head.history_memory = model.model.module.lang_encoder.lm_head.history_memory[-model.refresh:]
                # refresh hidden state for LSTM value net
                model.exit_controller.module.value_net.hidden_state = None
                model.exit_controller.module.value_net.history_memory = model.exit_controller.module.value_net.history_memory[-model.refresh:]
            else:
                model.reset()
        if hasattr(model, 'exit_controller') and model.exit_controller is not None:
            model.exit_controller.module.set_timestep(step)
        action = model.step(obs, lang_annotation, (len(planned_actions) == 0))
        exit_layers.append(model.current_exit_layer)
        llm_time_list.append(model.model.module.llm_inference_time)
        if len(planned_actions) == 0:
            if action.shape == (7,):
                planned_actions.append(action)
            else:
                planned_actions.extend([action[i] for i in range(action.shape[0])])
        action = planned_actions.pop(0)
        if model.use_diff:
            model.action_hist_queue.append(action)
        obs, _, _, current_info = env.step(action)
        if debug:
            img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
            img_queue.append(img_copy)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                # print(colored("success", "green"), end=" ")
                # img_clip = ImageSequenceClip(img_queue, fps=30)
                # img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
                save_screenshot_with_exit_info(img_queue, exit_layers, sequence_i, subtask_i, subtask, 'success', eval_log_dir)
            return True, exit_layers, step+1, llm_time_list
    if debug:
        # print(colored("fail", "red"), end=" ")
        # img_clip = ImageSequenceClip(img_queue, fps=30)
        # img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
         save_screenshot_with_exit_info(img_queue, exit_layers, sequence_i, subtask_i, subtask, 'fail', eval_log_dir)
    return False, exit_layers, step+1, llm_time_list

def save_screenshot_with_exit_info(images, exit_layers, seq_id, subtask_id, subtask, success, eval_log_dir, freq=5):
    assert len(images) == len(exit_layers)
    save_dir = os.path.join(eval_log_dir, f'{seq_id}-{subtask_id}-{subtask}-{success}')
    os.makedirs(save_dir, exist_ok=True)
    for t, (img, exit_id) in enumerate(zip(images, exit_layers)):
        if len(images) < 15 or t % freq == 0:
            img_path = os.path.join(save_dir, f'{t}_{exit_id}.jpg')
            img2 = Image.fromarray(img, 'RGB')
            # Save the image to a file
            img2.save(img_path)

def eval_one_epoch_calvin(args, model, dataset_path, image_processor, tokenizer, future_act_len=-1):

    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    wrapped_model = ModelWrapper(model, tokenizer, image_processor, cast_dtype, args.head_type=="diffusion", history_len=args.n_obs_steps, future_act_len=future_act_len, amp=args.amp)
    evaluate_policy(wrapped_model, env, 0, args.calvin_conf_path)


def eval_one_epoch_calvin_ddp(args, model, dataset_path, image_processor, tokenizer, eval_log_dir=None, debug=False, future_act_len=-1, reset=False, diverse_inst=False, exit_controller=None):

    global NUM_SEQUENCES
    NUM_SEQUENCES = args.num_seq
    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = None
    if args.head_type=="diffusion":
        hist_len = args.n_obs_steps
    elif args.pad_length != -1:
        hist_len = args.pad_length
    
    if args.eval_exit_mode == 'last':   
        if args.rank == 0: 
            if args.use_extra_exit:
                print("\nEvaluate the extra exit with features from the last layer !\n")
        wrapped_model = ModelWrapper(model, tokenizer, image_processor, cast_dtype, args.head_type=="diffusion", history_len=hist_len, future_act_len=future_act_len, amp=args.amp, exit_id=-1, multi_execution=args.multi_execution, use_action_ensemble=args.use_action_ensemble)
        results = evaluate_policy_ddp(wrapped_model, env, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir, debug=debug, reset=reset, diverse_inst=diverse_inst)
        
    elif args.eval_exit_mode == 'all':
        torch.distributed.barrier()
        if args.rank == 0: 
            if args.use_extra_exit:
                print("\nEvaluate the extra exit with features from a fixed layer (i=0,1,2,..) !\n")
            else:
                print("\nEvaluate all exits by numerical order!\n")
        for exit_id in reversed(model.module.get_all_exit_idx()):
            # if exit_id == 11: continue
            if args.rank == 0: print('#'*40 + '\n' + f'Evaluate the exit with exit_id={exit_id}!\n' + '#'*40 + '\n')
            wrapped_model = ModelWrapper(model, tokenizer, image_processor, cast_dtype, args.head_type=="diffusion", history_len=hist_len, future_act_len=future_act_len, amp=args.amp, exit_id=exit_id, multi_execution=args.multi_execution, use_action_ensemble=args.use_action_ensemble)
            results = evaluate_policy_ddp(wrapped_model, env, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir, debug=debug, reset=reset, diverse_inst=diverse_inst)
            torch.distributed.barrier() # don't conduct next eval until all threads reach
    
    elif args.eval_exit_mode == 'dynamic':
        if args.rank == 0: print("\nEvaluate with dynamic exit!\n")
        wrapped_model = ModelWrapper(model, tokenizer, image_processor, cast_dtype, args.head_type=="diffusion", history_len=hist_len, future_act_len=future_act_len, amp=args.amp, early_exit=True, exit_controller=exit_controller, multi_execution=args.multi_execution, use_action_ensemble=args.use_action_ensemble)
        results = evaluate_policy_ddp(wrapped_model, env, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir, debug=debug, reset=reset, diverse_inst=diverse_inst)

    else:
        raise NotImplementedError
    
    return  results

def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # evaluate a custom model
    if args.custom_model:
        model = CustomModel()
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, debug=args.debug)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = checkpoint.stem.split("=")[1]
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)


def generate_zero_shot_instr():
    random.seed(123)
    with open('./enrich_lang_annotations.json', 'r') as f:
            val_annotations = json.load(f)
    eval_sequences = get_sequences(NUM_SEQUENCES)
    
    all_res = []
    for initial_state, eval_sequence in eval_sequences:
        res = []
        for subtask_i, subtask in enumerate(eval_sequence):
            res.append(random.choice(val_annotations[subtask]))
        all_res.append(res)
    with open('./lang_annotation_cache.json', 'w') as f:
        json.dump(all_res, f, indent=1)


def save_sequences():
    random.seed(123)
    eval_sequences = get_sequences(NUM_SEQUENCES)
    with open('./eval_sequences.json', 'w') as f:
        json.dump(eval_sequences, f)
