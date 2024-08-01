""" Main training script """

import argparse
import glob
import os
import gc
import re
import time
import random
from robot_flamingo.eval.eval_utils import eval_one_epoch_calvin_ddp
from torch.distributed.elastic.multiprocessing.errors import record

# please use EGL for GPU-accelerating rendering. Don't use osmesa (CPU-only software rendering), which causes texture discrepancy from GPU rendering.
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import numpy as np
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from robot_flamingo.data.data import get_data
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from eval_utils import eval_one_epoch_calvin, eval_one_epoch_calvin_ddp, check_loaded_parameters
from robot_flamingo.models.factory import create_model_and_transforms, mpt_dict
from models.value_net import LSTMValueHead, ExitController, MLPValueHead, DiffValueHead, SimValueNet, TimeValueNet, RandomValueNet, ActionValueNet


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=4,
        help="how often to add a cross-attention layer after each transformer layer",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="RobotFlamingo",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size_calvin", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--openflamingo_checkpoint", type=str, default="")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)  # 1e-4
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument(
        "--calvin_dataset",
        type=str,
        help="path to calvin_dataset",
    )
    
    
    parser.add_argument(
        "--validation_set",
        default=False,
        action="store_true",
        help="use validation set for finding threshold",
    )
    parser.add_argument("--loss_multiplier_calvin", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--evaluate_from_checkpoint",
        type=str,
        help="path to checkpoint to evaluate , this should contain model",
        default=None,
    )
    # data args
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples_calvin", type=int, default=100)
    parser.add_argument("--dataset_resampled", action="store_true")
    parser.add_argument("--calvin_conf_path", type=str, help="path to calvin configuration file")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )
    parser.add_argument(
        "--freeze_embed",
        default=False,
        action="store_true",
        help="freeze the parameters of embedding layer",
    )
    parser.add_argument(
        "--use_gripper",
        default=False,
        action="store_true",
        help="whether to use gripper image as input",
    )
    parser.add_argument(
        "--use_state",
        default=False,
        action="store_true",
        help="whether to use low-dim state as input",
    )
    parser.add_argument(
        "--fusion_mode",
        default="post",
        type=str,
        help="pre or post to fusion multi vision info",
    )
    parser.add_argument("--hist_window", type=int, default=1)  # input history window size for the model
    # history window size when evaluating, for FC head equals to hist_window, for LSTM head means refresh frequency
    parser.add_argument("--eval_hist_size", type=int, default=-1)
    parser.add_argument(
        "--sep_resampler",
        default=False,
        action="store_true",
        help="whether use separate resamplers for third party and gripper camera",
    )
    parser.add_argument("--train_params", type=int, default=-1)
    parser.add_argument('--n_timesteps', type=int, default=150, help="diffusion time steps")
    parser.add_argument(
        "--predict_epsilon",
        default=False,
        action="store_true",
        help="whether diffusion model should predict epsilon",
    )
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument('--head_type', type=str, default="deterministic")  # policy type: deterministic / gaussian / diffusion
    parser.add_argument(
        "--from_scratch",
        default=False,
        action="store_true",
        help="whether to train the model from scratch",
    )
    parser.add_argument("--n_obs_steps", default=6, type=int)
    parser.add_argument("--future_act_len", default=-1, type=int) # For diffusion head. Only use K predicted actions
    parser.add_argument("--diff_horizon", default=32, type=int)
    parser.add_argument(
        "--last_action",
        default=False,
        action="store_true",
        help="whether using last action as input",
    )
    parser.add_argument(
        "--use_hist",
        default=False,
        action="store_true",
        help="whether using multi-image encoder"
    )
    parser.add_argument(
        "--partial_data",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--data_percent", 
        type=float, 
        default=0.1,
        # default=1.0,
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--visualize",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--reset",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--sep_lm_head",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--clip_state",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--convert_rgb",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--diverse_inst",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--residual",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--tcp_rel",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--replan",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--freeze_sampler",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--fwd_pred",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--fwd_pred_hand",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--no_image_patch",
        default=False,
        action="store_true"
    )
    parser.add_argument("--global_latent", type=int, default=1)
    parser.add_argument("--save_every_iter", type=int, default=-1)
    parser.add_argument("--pad_length", type=int, default=-1)
    # For GPT decoder
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--decoder_type", type=str, default='lstm')

    parser.add_argument("--min_window_size", type=int, default=12)
    parser.add_argument("--max_window_size", type=int, default=24)
    parser.add_argument("--llm_name", type=str, default='llama_9b')
    parser.add_argument("--pooling", type=str, default='max')
    
    parser.add_argument(
        "--amp",
        default=0,
        type=int, help="enable amp during inference",
    )
    
    # multi-exit eval
    parser.add_argument("--eval_exit_mode", type=str, default='last') # [last/all/dynamic] eval the last exit / all exits / dynamic exit mechanism
    parser.add_argument("--layerwise_exit_eval", type=int, default=0) 
    # timestep dynamic
    parser.add_argument("--multi_execution", type=int, default=1, help="how many actions are executed in one time when predicting multiple actions; if only one predicted action, repeat it K times")
    # dynamic early-exit
    parser.add_argument("--value_type", type=str, default='loss') # loss / sim 
    parser.add_argument("--threshold_type", type=str, default='mean') # for action delta [mean / L2 / max]
    parser.add_argument("--exit_dist", type=str, default='') # for exit dist [exp / gauss / gamma]
    parser.add_argument("--value_net_ckpt", type=str, default=None) 
    parser.add_argument("--max_layer", type=int, default=None)
    parser.add_argument("--exit_ratio", type=float, default=1.0, help="decide the exit thresholds")
    parser.add_argument("--steps_per_stage", default=1, type=int)
    parser.add_argument("--use_action_ensemble", default=0, type=int)
    parser.add_argument("--load_threshold", default=1, type=int)
    parser.add_argument("--num_seq", default=1000, type=int)
    
    parser.add_argument("--thresholds", nargs='+', type=float, default=None, help="directly set thresholds API (for bayesian optimization)")
    
    args = parser.parse_args()
    args.amp = bool(args.amp)
    
    print(f'{args.amp=}')
    print(f'{args.precision=}')
    print(f'{args.eval_exit_mode=}')
    print(f'{args.load_threshold=}')
    print(f'{args.layerwise_exit_eval=}')
    
    if args.value_type == 'loss':
        args.batch_size_calvin = 32
    else:
        args.batch_size_calvin = 16
    
    args.real_data = True if 'real' in args.evaluate_from_checkpoint else False
    # Search for the pattern in args.evaluate_from_checkpoint
    match = re.search(r'aug_(\d+)_(\d+)', args.evaluate_from_checkpoint)
    if match:
        args.rgb_pad = int(match.group(1))
        args.gripper_pad = int(match.group(2))
    else:
        args.rgb_pad = -1
        args.gripper_pad = -1
    if args.head_type == "diffusion":
        args.pad_length = args.n_obs_steps
    if args.eval_hist_size == -1:
        args.eval_hist_size = args.window_size
        if args.head_type == "diffusion":
            args.eval_hist_size = args.n_obs_steps
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if 'sep' in args.evaluate_from_checkpoint:
        args.sep_resampler = True
    if 'lm_head' in args.evaluate_from_checkpoint:
        args.sep_lm_head = True
    if 'res_' in args.evaluate_from_checkpoint:
        args.residual = True
    if 'tcp' in args.evaluate_from_checkpoint:
        args.tcp_rel = True
    if 'step' in args.evaluate_from_checkpoint.split('_'):
        name_attrs = args.evaluate_from_checkpoint.split('_')
        args.multi_step_action = int(name_attrs[name_attrs.index('step')-1])
    else:
        args.multi_step_action = 1
    if 'text_aug' in args.evaluate_from_checkpoint:
        args.text_aug = True
    else:
        args.text_aug = False
    if 'traj_cons' in args.evaluate_from_checkpoint:
        args.traj_cons = True
    else:
        args.traj_cons = False
    if 'difws' in args.evaluate_from_checkpoint:
        args.dif_ws = True
        name_attrs = args.evaluate_from_checkpoint.split('_')
        ix = name_attrs.index('difws')
        min_ws = int(name_attrs[ix+1])
        max_ws = int(name_attrs[ix+2])
        args.min_window_size = min_ws
        args.max_window_size = max_ws
        args.window_size = max_ws
    else:
        args.dif_ws = False
    if 'latent' in args.evaluate_from_checkpoint:
        name_attrs = args.evaluate_from_checkpoint.split('_')
        ix = name_attrs.index('latent')
        args.global_latent = int(name_attrs[ix+1])
    if 'no_image_patch' in args.evaluate_from_checkpoint:
        args.no_image_patch = True
    if 'gpt' in args.evaluate_from_checkpoint:
        args.decoder_type = 'gpt'
        name_attrs = args.evaluate_from_checkpoint.split('_')
        hidden_size = int(name_attrs[name_attrs.index('gpt')+1])
        args.hidden_size = hidden_size
    for name in ['mpt_3b', 'mpt_4b', 'mpt_9b', 'mpt_dolly_3b', 'mpt_base_4b']:
        if name in args.evaluate_from_checkpoint:
            args.llm_name = name
            break
        
    print(f'Model class : {args.llm_name}')
    
    args.lm_path = mpt_dict[args.llm_name]["lang_encoder_path"]
    args.tokenizer_path = mpt_dict[args.llm_name]["tokenizer_path"]
    args.cross_attn_every_n_layers = mpt_dict[args.llm_name]["cross_attn_every_n_layers"]
    args.openflamingo_checkpoint = mpt_dict[args.llm_name]["openflamingo_checkpoint"]
    
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    print("world_size: ", torch.distributed.get_world_size())
    random_seed(args.seed)
    
    # if args.evaluate_from_checkpoint is specified, load checkpoint
    assert args.evaluate_from_checkpoint is not None, "Please specify a checkpoint to evaluate."
    if args.rank == 0:
        print(f"Loading robot-flamingo checkpoint from {args.evaluate_from_checkpoint}")
    checkpoint = torch.load(args.evaluate_from_checkpoint, map_location="cpu")

    # save memory by deleting useless optimizer state from processes except the rank 0
    try:
        if args.rank != 0:
            del checkpoint['optimizer_state_dict']
            del checkpoint['lr_scheduler_state_dict']
    except:
        pass
    
    def readout_args(args, ckpt, name, default):
        if name in ckpt:
            value = ckpt[name]
            
        else:
            value = default
        setattr(args, name, value)
        if args.rank==0: print(f'set {name} to {value}!')
        
    readout_args(args, checkpoint, 'head_type', 'deterministic')
    readout_args(args, checkpoint, 'tanh_squash_dist', False)
    readout_args(args, checkpoint, 'state_dependent_std', False)
    readout_args(args, checkpoint, 'early_exit_layer', -1)
    # readout_args(args, checkpoint, "precision", 'fp32')
    readout_args(args, checkpoint, "multi_exit", False)
    readout_args(args, checkpoint, "use_extra_exit", False)
    readout_args(args, checkpoint, "exit_interval", 1)
    readout_args(args, checkpoint, "exit_dropout", 0.0)
    readout_args(args, checkpoint, "lstm_dropout", 0.0)
    readout_args(args, checkpoint, "dropout_mode", "wo_last")
    readout_args(args, checkpoint, "mlp_layernorm", False)
    readout_args(args, checkpoint, "lstm_layernorm", False)
    readout_args(args, checkpoint, "mlp_num_hidden_layers", 3)
    readout_args(args, checkpoint, "lstm_num_layers", 4)
    readout_args(args, checkpoint, "pooling", 'max')
    readout_args(args, checkpoint, "use_layerwise_projection", False)
    readout_args(args, checkpoint, "num_projection_layers", 1)
    readout_args(args, checkpoint, "skip_connection", False)
    
    if 'layernorm' in checkpoint: # for compatibility with old code
        args.mlp_layernorm = checkpoint['layernorm']
        
    if args.early_exit_layer < 0:
        if name == 'mpt_3b' or name == 'mpt_dolly_3b':
            args.early_exit_layer += 24
        elif name == 'mpt_9b':
            args.early_exit_layer += 32
        else:
            raise NotImplementedError
    if args.max_layer is None or args.max_layer == -1:
        args.max_layer = args.early_exit_layer + 1

    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        use_media_placement_augmentation=args.use_media_placement_augmentation,
        window_size=args.eval_hist_size,
        freeze_embed=args.freeze_embed,
        train_params=args.train_params,
        sep_resampler=args.sep_resampler,
        last_action=args.last_action,
        use_diff=(args.head_type == "diffusion"),
        n_timesteps=args.n_timesteps,
        diff_horizon=args.diff_horizon,
        fusion_mode=args.fusion_mode,
        use_gripper=args.use_gripper,
        use_state=args.use_state,
        use_hist=args.use_hist,
        pad_length=args.pad_length,
        debug=args.debug,
        multi_step_action=args.multi_step_action,
        llm_name=args.llm_name,
        sep_lm_head=args.sep_lm_head,
        return_feature=True,
        residual=args.residual,
        tcp_rel=args.tcp_rel,
        replan=args.replan,
        decoder_type=args.decoder_type,
        hidden_size=args.hidden_size,
        freeze_sampler=args.freeze_sampler,
        fwd_pred=args.fwd_pred,
        fwd_pred_hand=args.fwd_pred_hand,
        no_image_patch=args.no_image_patch,
        global_latent=args.global_latent,
        # refresh=args.refresh
        head_type=args.head_type,
        tanh_squash_dist=args.tanh_squash_dist,
        state_dependent_std=args.state_dependent_std,
        early_exit_layer=min(args.early_exit_layer, args.max_layer),
        multi_exit=False if not args.layerwise_exit_eval else True, # save GPU memory by not creating auxiliary action heads 1,2,..,N
        exit_interval=args.exit_interval,
        exit_dropout=args.exit_dropout,
        lstm_dropout=args.lstm_dropout,
        dropout_mode=args.dropout_mode,
        mlp_layernorm=args.mlp_layernorm,
        lstm_layernorm=args.lstm_layernorm,
        lstm_num_layers=args.lstm_num_layers,
        mlp_num_hidden_layers=args.mlp_num_hidden_layers,
        use_extra_exit=args.use_extra_exit,
        use_layerwise_projection=args.use_layerwise_projection,
        num_projection_layers=args.num_projection_layers,
        skip_connection=args.skip_connection,
        layerwise_exit_eval=args.layerwise_exit_eval,
    )
    checkpoint_path = args.openflamingo_checkpoint
    print("Loading origin flamingo checkpoint from ", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    if args.sep_lm_head:
        model.lm_head.requires_grad_(True)
    else:
        model.lang_encoder.lm_head.requires_grad_(True)

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    else:
        model = model.float()
    model = model.to(device_id)
    model.eval()

    ddp_model = DDP(model, device_ids=[device_id])
    if args.residual:
        model.lang_encoder.clone_parameters()

    try:
        ckpt_dict = checkpoint["model_state_dict"]
    except:
        ckpt_dict = checkpoint  
    # check_loaded_parameters(ddp_model, ckpt_dict) # disabled due to max layer constrain
    ddp_model.load_state_dict(ckpt_dict, False)  # 只保存了求梯度的部分
    ddp_model.eval()
    
    value_net_ckpt = None
    if args.eval_exit_mode == 'dynamic':
        if args.value_type == 'loss':
            assert args.value_net_ckpt is not None, "Please specify a checkpoint for value net."
            if args.rank == 0:
                print(f"Loading value net checkpoint from {args.value_net_ckpt}")
            value_net_ckpt = torch.load(args.value_net_ckpt, map_location="cpu")
            
            readout_args(args, value_net_ckpt, "with_exit_embed", False)
            readout_args(args, value_net_ckpt, "discrete", False)
            readout_args(args, value_net_ckpt, "num_bin", 100)
            
            num_exit = model.get_exit_num()
            
            
            value_net = MLPValueHead(
                in_features=model.lm_head.in_features, 
                window_size=args.eval_hist_size,
                hidden_size=model.lm_head.hidden_size,
                fusion_mode=args.fusion_mode, 
                use_state=args.use_state, 
                pooling=args.pooling,
                with_exit_embed=args.with_exit_embed,
                num_exits=model.get_exit_num(),
                discrete=args.discrete,
                num_bin=args.num_bin,
            )
            # value_net = LSTMValueHead(
            #     in_features=model.lm_head.in_features, 
            #     window_size=args.eval_hist_size,
            #     dropout=0.0,
            #     hidden_size=model.lm_head.hidden_size,
            #     fusion_mode=args.fusion_mode, 
            #     use_state=args.use_state, 
            #     pooling=args.pooling,
            #     with_exit_embed=args.with_exit_embed,
            #     num_exits=num_exit,
            #     discrete=args.discrete,
            #     num_bin=args.num_bin,   
            # )
            # value_net = DiffValueHead(
            # in_features=1024, 
            # window_size=args.eval_hist_size,
            # dropout=0.0,
            # hidden_size=model.lm_head.hidden_size,
            # with_exit_embed=args.with_exit_embed,
            # # with_time_embed=args.with_time_embed,
            # with_time_embed=False,
            # num_exits=model.get_exit_num(),
            # discrete=args.discrete,
            # num_bin=args.num_bin,
            # )
            
            value_net_ckpt_dict = {k.replace('module.', ''): v for k, v in value_net_ckpt["model_state_dict"].items()} # remove ddp prefix
            value_net_ckpt_dict = {k.replace('value_net.', 'head.'): v for k, v in value_net_ckpt_dict.items()} # Be compatible with previous value_net code
            value_net.load_state_dict(value_net_ckpt_dict, True)
            
            exit_controller = ExitController(value_net, exit_id_list=model.get_all_exit_idx(), steps_per_stage=args.steps_per_stage, leq=True)
        
        elif args.value_type == 'sim':
            value_net = SimValueNet(pooling=False, exit_ids=model.get_all_exit_idx(), interval=args.exit_interval)
            exit_controller = ExitController(value_net, exit_id_list=model.get_all_exit_idx(), steps_per_stage=args.steps_per_stage, leq=False)
            # exit_controller = ExitController(value_net, exit_id_list=model.get_all_exit_idx(), leq=True)
        elif args.value_type == 'time':
            value_net = TimeValueNet(T=360, exit_ratio=args.exit_ratio, exit_list=model.get_all_exit_idx(), steps_per_stage=args.steps_per_stage)
            exit_controller = ExitController(value_net, exit_id_list=model.get_all_exit_idx(), steps_per_stage=args.steps_per_stage)
        elif args.value_type == 'random':
            value_net = RandomValueNet(exit_ratio=args.exit_ratio, exit_list=model.get_all_exit_idx(), steps_per_stage=args.steps_per_stage)
            exit_controller = ExitController(value_net, exit_id_list=model.get_all_exit_idx(), steps_per_stage=args.steps_per_stage)
        elif args.value_type == 'action':
            value_net = ActionValueNet(exit_list=model.get_all_exit_idx(), exit_head=ddp_model.module.extra_exit, interval=args.exit_interval, window_size=args.window_size, threshold_type=args.threshold_type)
            exit_controller = ExitController(value_net, exit_id_list=model.get_all_exit_idx(), steps_per_stage=args.steps_per_stage, leq=True, exit_dist=args.exit_dist, max_layer=args.max_layer)
        else:
            raise NotImplementedError
            
        exit_controller = exit_controller.to(device_id)
        exit_controller.eval()
        ddp_exit_controller = DDP(exit_controller, device_ids=[device_id])
        
        # setup dataloader for dynamically finding thresholds
        if args.debug:
            calvin_dataset = get_data(args, image_processor, tokenizer, "debug")
        # elif args.real_data:
        #     calvin_dataset = get_data(args, image_processor, tokenizer, "real")
        else:
            calvin_dataset = get_data(args, image_processor, tokenizer, "calvin")
        calvin_dataset.set_epoch(0)
        calvin_loader = calvin_dataset.dataloader
        
        # find threshold
        values = checkpoint['values'] if args.value_type == 'action' and checkpoint is not None and  "values" in checkpoint else None
        if not args.thresholds:
            if args.load_threshold and values is not None: # load cached value distribution
                print(f'load values for threshold')
                ddp_exit_controller.module.set_threshold(args, model, calvin_loader, args.exit_ratio, args.llm_name, values)
            else:
                values = ddp_exit_controller.module.set_threshold(args, model, calvin_loader, args.exit_ratio, args.llm_name)
                
                checkpoint["values"] = values
                if args.rank==0: 
                    print("save new values for threshold to ckpt.")
                    torch.save(checkpoint, args.evaluate_from_checkpoint)
        else:
            ddp_exit_controller.module._set_threshold_value(args.thresholds)
                
        del calvin_dataset
        del calvin_loader
        del checkpoint
            
    else:
        ddp_exit_controller = None
        calvin_loader = None
        
    # clear GPU memory used by finding thresholds        
    gc.collect()
    torch.cuda.empty_cache()

    
    eval_log_dir = None
    if args.visualize:
        eval_log_dir = 'evaluate/{}_{}_{}_{}'.format(args.evaluate_from_checkpoint.split('.')[0], args.eval_exit_mode, args.value_type, args.exit_ratio, )
    results = eval_one_epoch_calvin_ddp(
        args=args,
        model=ddp_model,
        image_processor=image_processor,
        tokenizer=tokenizer,
        dataset_path=args.calvin_dataset,
        future_act_len=args.future_act_len,
        eval_log_dir=eval_log_dir,
        debug=args.visualize,
        reset=args.reset,
        diverse_inst=args.diverse_inst,
        exit_controller=ddp_exit_controller,
    )
    
    torch.distributed.barrier()
    if args.rank == 0: 
        time.sleep(20) # wait all output are written the log file
        thresholds = ddp_exit_controller.module.thresholds
        thresholds = map(float, thresholds.values())
        threshold_str = ','.join(map(str, thresholds))
        print(threshold_str) # threshold
        print(results[0]) # avg len
        print(results[1]) # avg exit
    


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = '1'
    main()
