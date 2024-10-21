import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, Tuple
from tqdm.auto import tqdm
import math
from abc import abstractmethod
import numpy as np
import scipy.stats

def get_similarity(f_x1s, f_x2s, detach_f1=False):
    if detach_f1:
        f_x1s = f_x1s.detach()
    f_x1 = F.normalize(f_x1s, p=2., dim=-1, eps=1e-5)
    f_x2 = F.normalize(f_x2s, p=2., dim=-1, eps=1e-5)
    # loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
    sim = (f_x1 * f_x2).sum(-1)
    return sim


def logits_to_expected_value(logits, bin_boundaries):
    # Assuming logits is a tensor with shape (batch, num_bin)
    ndim = logits.ndim
    if ndim == 3:
        d1, d2 = logits.shape[:2]
        logits = logits.flatten(0, 1)
    batch, num_bin = logits.shape
    # Create a tensor representing the midpoints of each bin
    bin_midpoints = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    # Expand bin_midpoints to match the batch size
    bin_midpoints = bin_midpoints.unsqueeze(0).expand(batch, num_bin)
    # Convert logits to probabilities (softmax)
    probabilities = torch.softmax(logits, dim=1)
    # Calculate the expected value as the weighted sum of bin midpoints
    expected_values = torch.sum(probabilities * bin_midpoints, dim=1)
    if ndim == 3:
        expected_values = expected_values.reshape((d1, d2))
    return expected_values

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


class BaseValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # dummy for DDP wrapper
        self.head = torch.nn.Linear(10, 10)
      
    @abstractmethod  
    def clear_hidden_state(self) -> None:
        pass
    
    @abstractmethod
    def update_memory(self):
        pass
    
    @abstractmethod
    def set_bin_boundaries(self, bin_boundaries):
        pass
           
   
 
class ActionValueNet(BaseValueNet):
    def __init__(self,  exit_list, exit_head, interval, window_size, threshold_type) -> None:
        super().__init__()
        self.exit_list = exit_list
        self.exit_head = exit_head
        self.interval = interval # exit interval
        self.action_list = []
        self.window_size = window_size
        self.threshold_type = threshold_type
        
    def reset_actions(self):
        self.action_list = []
        
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def update_exit_hidden_state(self):
        """called after early exiting and executing an action in the environment"""
        self.exit_head.module.update_hidden_state()
        
    def get_ensemble_action(self):
        assert len(self.action_list) > 0
        actions, grippers = zip(*self.action_list[-2:])
        return torch.stack(actions, dim=0).mean(0), torch.stack(grippers, dim=0).mean(0)
        
    def forward(  # type: ignore
        self,
        feats: torch.Tensor,
        i=None,
        mode='infer',
        rand_layer_feat=None,
        ):
        
        def get_delta(action1, action2):
            delta = torch.abs(action1 - action2)
            if self.threshold_type == 'mean':
                delta = delta.mean(-1)
            elif self.threshold_type == 'L2':
                delta = delta.pow(2).mean(-1).pow(0.5)
            elif self.threshold_type == 'max':
                delta = delta.max(-1)[0]
            elif self.threshold_type == 'cosine':
                delta = 1 - get_similarity(action1, action2)
            else:
                raise NotImplementedError
            return delta
            
        
        if mode == 'infer':
            assert i > 0, 'the first layer similarity is not implemented yet'
            if i > 0 and i - self.interval < 0:
                # since we don't have an action before the first action, we can
                # 1. produce pseudo action using previous layer's feature
                # 2. using gripper confidence
                prev_action = self.exit_head(feats[i-1], update_hidden_state=False)
            else:
                prev_action = self.action_list[-1]
            
            action = self.exit_head(feats[i], update_hidden_state=False)
            self.action_list.append(action)   
            delta = get_delta(action[0], prev_action[0])

            return delta
        else:
            assert 0 not in self.exit_list
            exits_feat = [feats[i] for i in [0]+self.exit_list] # (n_exit+1, bs * action_seq_len, lang_len, d)
            
            exits_action_list = []
            lang_len, d = feats[0].shape[1:]
            for seq_id in range(self.window_size//2-1, self.window_size-1):
                # (bs * action_seq_len, lang_len, d) -> (bs, seq_id, lang_len, d)
                prev_time_feat = rand_layer_feat.reshape(-1, self.window_size, lang_len, d)[:, :seq_id, :, :]
                
                exit_action = [] # (exit+1, bs, dim)
                for i in [0]+self.exit_list:
                    # (bs * action_seq_len, lang_len, d) -> (bs, 1, lang_len, d)
                    last_time_feat = feats[i].reshape(-1, self.window_size, lang_len, d)[:, seq_id:seq_id+1, :, :]
                    combined_feat = torch.concat([prev_time_feat, last_time_feat], dim=1) # (bs, seq_id+1, lang_len, d)
            
                    self.exit_head.last_action = True
                    action = self.exit_head(combined_feat) # (bs, dim)
                    self.exit_head.last_action = False
                    exit_action.append(action[0].squeeze(1))   # (exit+1, bs, dim)
                exits_action_list.append(torch.stack(exit_action)) # (seq_action/2, exit+1, bs, dim)
            exits_action_list = torch.stack(exits_action_list).permute(1, 2, 0, 3)  # (exit+1, bs, seq_action/2, dim)
                    
            prev_actions = exits_action_list[:-1] # (n_exit, bs, action_seq_len, dim)
            last_actions = exits_action_list[1:] # (n_exit, bs, action_seq_len, dim)
            delta = get_delta(prev_actions, last_actions).flatten(1,2) # (n_exit, bs*action_seq_len)
            return delta
                    
        
class ExitController(torch.nn.Module):
    def __init__(self, value_net, exit_id_list, steps_per_stage, exit_dist='exp', leq=True, max_layer=12):
        super().__init__()
        self.value_net = value_net
        self.thresholds = None
        self.leq = leq
        self.exit_id_list = exit_id_list
        self.num_exit = len(self.exit_id_list)
        self.steps_per_stage = steps_per_stage
        self.exit_dist = exit_dist
        self.max_layer = min(max_layer - 1, self.exit_id_list[-1])
        # for debug
        # self.history_values = [[] for i in range(num_exit)]
        
    def _set_threshold_value(self, thresholds):
        real_num_exit = len([x for x in self.exit_id_list if x <= self.max_layer])
        assert len(thresholds) == real_num_exit
        self.thresholds = {self.exit_id_list[i] : thresholds[i]  for i in range(real_num_exit)}
        if torch.distributed.get_rank() == 0:
            print('setting thresholds, ', thresholds)
        
        
    def set_threshold(self, args, model, dataloader, exit_ratio, model_name, values=None):  
        if values is None:  
            device_id = torch.distributed.get_rank()
            num_devices = torch.distributed.get_world_size()
            if isinstance(self.value_net, ActionValueNet):
                pred_value_list, target_value_list = generate_action_values(args, model, self.value_net, dataloader, device_id=device_id)
            else:
                raise NotImplementedError

            # Initialize tensors to hold the gathered data
            pred_value_gathered = [torch.zeros_like(pred_value_list) for _ in range(num_devices)]
            # target_value_gathered = [torch.zeros_like(target_value_list) for _ in range(num_devices)]

            # Gather data
            torch.distributed.all_gather(pred_value_gathered, pred_value_list)
            # torch.distributed.all_gather(target_value_gathered, target_value_list)
            pred_value_gathered = torch.cat(pred_value_gathered, dim=1)
            # target_value_gathered = torch.cat(target_value_gathered, dim=1)
        else:
            pred_value_gathered = values
            
        n_stage, n_sample = pred_value_gathered.size() # (exit, bs * seq_length)

        _, sorted_idx = pred_value_gathered.sort(dim=1, descending=not self.leq)

        filtered = torch.zeros(n_sample)
        real_num_exit = len([x for x in self.exit_id_list if x <= self.max_layer])
        
        T = torch.Tensor(real_num_exit).fill_(-1e8) if self.leq else torch.Tensor(real_num_exit).fill_(1e8)
        
        
        if self.exit_dist == 'exp':
            probs = exit_ratio ** torch.arange(1, real_num_exit+1) # n (including the last exit)
    
        elif self.exit_dist == 'gauss':
            # Gaussian (normal) distribution centered around `num_exit // 2`
            center = exit_ratio
            std_dev = 1.0  # Arbitrary standard deviation to cover a significant range
            probs = torch.tensor([math.exp(-(i - center) ** 2 / (2 * std_dev ** 2)) for i in range(real_num_exit)])
        
        elif self.exit_dist == 'gamma':
            # Gamma distribution
            x = torch.arange(1, real_num_exit + 1, dtype=torch.float32)
            shape = exit_ratio
            scale = 2.0
            probs = torch.tensor([scipy.stats.gamma.pdf(val, shape, scale=scale) for val in x], dtype=torch.float32)
        
        else:
            raise ValueError("Unsupported exit distribution")
        
        if 'mpt_9b' in model_name:
            probs[0] = 0 # only enable exits from at least 4th layer for very deep model
        
        probs /= probs.sum()
        
        if args.rank==0: print('Expected early exit rate ', probs)

        for k in range(real_num_exit - 1): # not include the last exit
            count = 0
            out_n = math.floor(n_sample * probs[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = pred_value_gathered[k][ori_idx]
                        break
            if self.leq:
                filtered.add_(pred_value_gathered[k].le(T[k]).type_as(filtered))
                # filtered.add_(pred_value_gathered[k].less(T[k]).type_as(filtered))
            else:
                filtered.add_(pred_value_gathered[k].ge(T[k]).type_as(filtered))
                # filtered.add_(pred_value_gathered[k].greater(T[k]).type_as(filtered))

        if self.leq:
            T[real_num_exit - 1] = 1e8
        else:
            T[real_num_exit - 1] = -1e8
        
        # verify
        # count = [0]
        # filtered = torch.zeros(n_sample)
        # for k in range(n_stage):
        #     filtered += pred_value_gathered[k] < T[k]
        #     count.append((filtered>0).sum()-sum(count))
        # percent = [c / sum(count) for c in count[1:]]    

        self.thresholds = {self.exit_id_list[i] : T[i]  for i in range(real_num_exit)}
        if args.rank == 0:
            print(f'Mean value for each layer:')
            for i in range(n_stage):
                print(f'{i+1} : {pred_value_gathered[i].mean():.5f}, {pred_value_gathered[i].std():.5f}, {pred_value_gathered[i].max():.5f}, {pred_value_gathered[i].min():.5f}')
            print(f'Find threshold on {n_sample} samples:')
            for i in range(real_num_exit):
                print(f'{i+1} : {T[i]:.5f}')
        return pred_value_gathered
    
    def set_timestep(self, t):
        self.cur_step = t

    @torch.no_grad()  
    def forward(self, x, i):
        assert self.thresholds is not None, "Please set thresholds before calling forward"
        assert isinstance(i, int), 'index muast be integer'
        if i not in self.exit_id_list:
            return False
        
        # if still in a stage just use previous exit id
        if self.cur_step % self.steps_per_stage != 0:
            return i >= self.cur_exit_id
        
        if isinstance(self.value_net, ActionValueNet):
            value = self.value_net(x, i)
        else:
            raise NotImplementedError

        # self.history_values[i].append(value)
        # if torch.distributed.get_rank() == 0:
        #     total = sum(len(h) for h in self.history_values)
        #     if total > 0 and total % 100 == 0:
        #         for layer, h in enumerate(self.history_values):
        #             print(f'{layer=}, count={len(h)}, mean value = {torch.tensor(h).mean():.5f}')
        
        if bool(value <= self.thresholds[i]) is self.leq or i >= self.max_layer: # both be true or both be false
            self.cur_exit_id = i
            return True 
        else:
            return False


@torch.no_grad()
def generate_action_values(
    args,
    model,
    value_net,
    calvin_loader,
    device_id,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_calvin

    cast_dtype = get_cast_dtype(args.precision)

    model.eval()
    value_net.eval()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=num_batches_per_epoch,
        initial=0,
    )
    t.set_description(f"generate values by similarity")
    mv_avg_loss = []
    pred_value_list = []
    
    for num_steps, batch_calvin in t:
        global_step = num_steps
        
        # put images and labels on device
        images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        if args.fusion_mode != 'vit_concat':
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1)
        else:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)
        # input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)

        # do the same to the attention mask 
        if args.fusion_mode != 'vit_concat':
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1)
        else:
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True)
        
        state_tensor = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        robot_obs = batch_calvin[5].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.clip_state:
            state_tensor = torch.cat([state_tensor[..., :6], state_tensor[..., [-1]]], dim=-1)
        labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        
        state_tensor = state_tensor.unsqueeze(2).unsqueeze(2)

        # merge the batch and the sequence dimension
        images = images.flatten(0, 1)
        gripper = gripper.flatten(0, 1)
        state_tensor = state_tensor.flatten(0, 1)
        if args.fusion_mode != 'vit_concat':
            input_ids = input_ids.flatten(0, 1)
            attention_mask = attention_mask.flatten(0, 1)

        # [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        if args.use_hist:
            labels = labels[:, [-1]]  # only calculate last step action
        if args.fusion_mode == 'vit_concat':
            labels = labels[:, -1]
        labels = [labels[..., :6], (labels[..., 6:] + 1) // 2]
        # print(f'{args.amp=}')
       
        with torch.cuda.amp.autocast(enabled=args.amp), torch.no_grad():
            if args.head_type == 'deterministic':
                final_output, exit_outputs, extra_exit_output, rand_layer_feat, _ = model(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    # labels=labels,  # loss计算放在外面
                    vision_gripper=gripper,
                    state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                    with_gripper_logits=True,
                    return_in_feat=True,
                    only_extra_exit=True,
                )


        feats = final_output.hidden_states # n_exit x (bs * action_seq_len, lang_len, d)
        rand_layer_feat = rand_layer_feat # (bs * action_seq_len, lang_len, d)
        sim = value_net(feats, mode='generate', rand_layer_feat=rand_layer_feat)
        
        # record
        pred_value_list.append(sim)  

    pred_value_list = torch.cat(pred_value_list, dim=1)
    # pred_value_list = pred_value_list.flatten(1, 2)
        
    # return pred_value_list, target_value_list
    return pred_value_list, None
