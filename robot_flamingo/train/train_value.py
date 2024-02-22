""" Main training script """

import argparse
import copy
import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
from huggingface_hub import hf_hub_download

from torch.nn.parallel import DistributedDataParallel as DDP

from robot_flamingo.data.data import get_data
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from train_utils import get_checkpoint, train_value_net_one_epoch_calvin_multi_exit, train_value_net_one_epoch_calvin_dynamic_exit, save_value_net_ckpt
from robot_flamingo.eval.eval_utils import check_loaded_parameters
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from robot_flamingo.models.factory import create_model_and_transforms, mpt_dict
from models.value_net import LSTMValueHead, MLPValueHead

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

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
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size_calvin", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # parser.add_argument("--openflamingo_checkpoint", type=str, default="")
    parser.add_argument(
        "--roboflamingo_checkpoint",
        type=str,
        help="path to roboFlamingo checkpoint",
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
    parser.add_argument("--loss_multiplier_calvin", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    # hot fix for torch.distributed.launch
    # parser.add_argument("--local-rank", type=int, default=1)

    # data args
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples_calvin", type=int, default=100)
    parser.add_argument("--dataset_resampled", action="store_true")
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
        default="pre", # pre / post / two way (use gripper view as extra input image)
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
    parser.add_argument('--rgb_pad', type=int, default=-1)
    parser.add_argument('--gripper_pad', type=int, default=-1)
    parser.add_argument('--n_timesteps', type=int, default=150, help="diffusion time steps")
    parser.add_argument(
        "--predict_epsilon",
        default=False,
        action="store_true",
        help="whether diffusion model should predict epsilon",
    )
    parser.add_argument(
        "--from_scratch",
        default=False,
        action="store_true",
        help="whether to train the model from scratch",
    )
    parser.add_argument("--n_obs_steps", default=6, type=int)
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
        action="store_true"
    )
    parser.add_argument(
        "--traj_cons",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--debug",
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
        "--unfreeze_vit",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--text_aug",
        default=False,
        action="store_true"
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
        "--dif_ws",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--partial_data",
        default=False,
        action="store_true"
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
        "--no_pretrain",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--real_data",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--no_image_patch",
        default=False,
        action="store_true"
    )
    # Co-Train settings
    parser.add_argument(
        "--cotrain",
        default=False,
        action="store_true"
    )
    parser.add_argument("--batch_size_vl", type=int, default=20)
    parser.add_argument("--vl_task_weights", type=float, default=0.005)

    parser.add_argument("--global_latent", type=int, default=1)
    parser.add_argument("--save_every_iter", type=int, default=-1)
    # For GPT decoder
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--decoder_type", type=str, default='lstm')
    
    parser.add_argument("--min_window_size", type=int, default=12)
    parser.add_argument("--max_window_size", type=int, default=24)
    parser.add_argument("--llm_name", type=str, default='llama_9b')
    parser.add_argument("--pooling", type=str, default='max')
    
    parser.add_argument('--head_type', type=str, default="deterministic")
    # for proxy task
    parser.add_argument("--data_percent", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    # for training value net
    parser.add_argument("--value_dropout", type=float, default=0.0, help='')
    parser.add_argument("--value_weight_decay", type=float, default=0.0, )
    parser.add_argument("--with_exit_embed", default=False, action="store_true")
    parser.add_argument("--with_time_embed", default=False, action="store_true")
    parser.add_argument("--discrete", default=False, action="store_true") # model value as discrete distribution and use cross-entropy loss
    parser.add_argument("--num_bin", type=int, default=100)
    
    args = parser.parse_args()
    
    # args.train_value = True
    if args.eval_hist_size == -1:
        args.eval_hist_size = args.window_size
        if args.head_type == "diffusion":
            args.eval_hist_size = args.n_obs_steps
    if args.tcp_rel:
        args.clip_state = True
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if 'sep' in args.roboflamingo_checkpoint:
        args.sep_resampler = True
    if 'lm_head' in args.roboflamingo_checkpoint:
        args.sep_lm_head = True
    if 'res_' in args.roboflamingo_checkpoint:
        args.residual = True
    if 'tcp' in args.roboflamingo_checkpoint:
        args.tcp_rel = True
    if 'step' in args.roboflamingo_checkpoint.split('_'):
        name_attrs = args.roboflamingo_checkpoint.split('_')
        args.multi_step_action = int(name_attrs[name_attrs.index('step')-1])
    else:
        args.multi_step_action = 1
    if 'bin_coef' in args.roboflamingo_checkpoint:
        name_attrs = args.roboflamingo_checkpoint.split('_')
        args.bin_coef = int(name_attrs[name_attrs.index('coef')+1])
    else:
        if args.real_data:
            args.bin_coef = 0.05
        else:
            args.bin_coef = 0.01
    if 'difws' in args.roboflamingo_checkpoint:
        args.dif_ws = True
        name_attrs = args.roboflamingo_checkpoint.split('_')
        ix = name_attrs.index('difws')
        min_ws = int(name_attrs[ix+1])
        max_ws = int(name_attrs[ix+2])
        args.min_window_size = min_ws
        args.max_window_size = max_ws
        args.window_size = max_ws
    if 'latent' in args.roboflamingo_checkpoint:
        name_attrs = args.roboflamingo_checkpoint.split('_')
        ix = name_attrs.index('latent')
        args.global_latent = int(name_attrs[ix+1])
    if 'no_image_patch' in args.roboflamingo_checkpoint:
        args.no_image_patch = True
    if 'gpt' in args.roboflamingo_checkpoint:
        args.decoder_type = 'gpt'
        name_attrs = args.roboflamingo_checkpoint.split('_')
        hidden_size = int(name_attrs[name_attrs.index('gpt')+1])
        args.hidden_size = hidden_size
    for name in ['mpt_3b', 'mpt_4b', 'mpt_9b', 'mpt_dolly_3b', 'mpt_base_4b']:
        if name in args.roboflamingo_checkpoint:
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
    
    # if args.roboflamingo_checkpoint is specified, load checkpoint
    assert args.roboflamingo_checkpoint is not None, "Please specify a checkpoint for RoboFlamingo."
    if args.rank == 0:
        print(f"Loading robot-flamingo checkpoint from {args.roboflamingo_checkpoint}")
    checkpoint = torch.load(args.roboflamingo_checkpoint, map_location="cpu")
    
    
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
        debug=args.debug,
        multi_step_action=args.multi_step_action,
        llm_name=args.llm_name,
        sep_lm_head=args.sep_lm_head,
        pooling=args.pooling,
        residual=args.residual,
        tcp_rel=args.tcp_rel,
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
        early_exit_layer=args.early_exit_layer,
        multi_exit=args.multi_exit,
        exit_interval=args.exit_interval,
        exit_dropout=args.exit_dropout,
        use_extra_exit=args.use_extra_exit,
    )

    checkpoint_path = args.openflamingo_checkpoint
    print("Loading origin flamingo checkpoint from ", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    
    if args.debug:
        calvin_dataset = get_data(args, image_processor, tokenizer, "debug")
    elif args.real_data:
        calvin_dataset = get_data(args, image_processor, tokenizer, "real")
    else:
        calvin_dataset = get_data(args, image_processor, tokenizer, "calvin")

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            # entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
    # value_net = MLPValueHead(
    #     in_features=model.lm_head.in_features, 
    #     window_size=args.eval_hist_size,
    #     dropout=args.value_dropout,
    #     hidden_size=model.lm_head.hidden_size,
    #     fusion_mode=args.fusion_mode, 
    #     use_state=args.use_state, 
    #     pooling=args.pooling,
    #     with_exit_embed=args.with_exit_embed,
    #     num_exits=model.get_exit_num(),
    #     discrete=args.discrete,
    #     num_bin=args.num_bin,
    #     )
    value_net = LSTMValueHead(
        in_features=model.lm_head.in_features, 
        window_size=args.eval_hist_size,
        dropout=args.value_dropout,
        hidden_size=model.lm_head.hidden_size,
        fusion_mode=args.fusion_mode, 
        use_state=args.use_state, 
        pooling=args.pooling,
        with_exit_embed=args.with_exit_embed,
        with_time_embed=args.with_time_embed,
        num_exits=model.get_exit_num(),
        discrete=args.discrete,
        num_bin=args.num_bin,
        )

    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
        value_net = value_net.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
        value_net = value_net.half()
    else:
        model = model.float()
        value_net = value_net.float()
        
    if args.head_type == "diffusion" and (not args.debug):
        normalizer = model.diffusion_model.normalizer
        all_actions = np.vstack([calvin_dataset.dataset.__getitem__((i,1),True)["actions"] for i in range(0,10000)])
        normalizer.fit(all_actions, last_n_dims=1, mode='limits')

    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    model.eval() # not train the VLM
    ddp_model.eval()
    
    value_net = value_net.to(device_id)
    ddp_value_net = DDP(value_net, device_ids=[device_id], find_unused_parameters=True)
    value_net.train()
    ddp_value_net.train()
    
    # load RoboFlamingo ckpt
    try:
        ckpt_dict = checkpoint["model_state_dict"]
    except:
        ckpt_dict = checkpoint  
    check_loaded_parameters(ddp_model, ckpt_dict)
    ddp_model.load_state_dict(ckpt_dict, False)  # 只保存了求梯度的部分

    args.learning_rate = args.learning_rate * args.batch_size_calvin / 6 # adaptive lr
    optimizer = torch.optim.AdamW(ddp_value_net.parameters(), lr=args.learning_rate, weight_decay=args.value_weight_decay)

    total_training_steps = (
        (args.train_num_samples_calvin) // (args.batch_size_calvin * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    use_diff = (args.head_type == "diffusion")

    # load value net if possible
    resume_from_epoch = 0 if not args.discrete else -1 # use one extra epoch to get the distribution of values
    # if args.from_scratch is False:
    
    for epoch in range(resume_from_epoch, args.num_epochs):
        calvin_dataset.set_epoch(epoch)
        calvin_loader = calvin_dataset.dataloader

        if args.head_type == "diffusion":
            raise NotImplementedError
        elif args.fusion_mode == 'two_way':
            raise NotImplementedError
        else:
            if args.multi_exit:
                if args.use_extra_exit:
                    train_value_net_one_epoch_calvin_dynamic_exit(
                        args=args,
                        model=ddp_model,
                        value_net=ddp_value_net,
                        epoch=epoch,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        calvin_loader=calvin_loader,
                        device_id=device_id,
                        wandb=wandb,
                    )
                else:
                    train_value_net_one_epoch_calvin_multi_exit(
                        args=args,
                        model=ddp_model,
                        value_net=ddp_value_net,
                        epoch=epoch,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        calvin_loader=calvin_loader,
                        device_id=device_id,
                        wandb=wandb,
                    )
            else:
                raise NotImplementedError

        if args.rank == 0 and epoch % args.save_freq == 0:
            save_value_net_ckpt(args, ddp_value_net, optimizer, lr_scheduler, epoch, epoch, args.roboflamingo_checkpoint)

if __name__ == "__main__":
    main()
