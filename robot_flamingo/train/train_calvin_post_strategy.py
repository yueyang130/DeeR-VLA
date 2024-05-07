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
from train_utils import get_checkpoint, train_one_epoch_calvin, train_one_epoch_calvin_diff, train_one_epoch_calvin_cotrain, train_one_epoch_calvin_two_way, \
train_one_epoch_calvin_multi_exit, get_ckpt_name, get_ckpt_name_pattern, save_ckpt, get_layerwise_lr_list, get_num_layer_for_flamingo
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from robot_flamingo.models.factory import create_model_and_transforms, mpt_dict

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
    parser.add_argument("--exit_strategy", type=str, default='post') # pre / joint / post
    parser.add_argument("--num_joint_epochs", type=int, default=1)
    parser.add_argument("--num_exit_epochs", type=int, default=1)
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
    parser.add_argument("--exit_learning_rate", default=1e-4, type=float)  # 1e-4
    parser.add_argument("--joint_learning_rate", default=1e-4, type=float)  # 1e-4
    parser.add_argument(
        "--joint_lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument(
        "--exit_lr_scheduler",
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
    parser.add_argument("--exit_warmup_steps", default=5000, type=int)
    parser.add_argument("--joint_warmup_steps", default=5000, type=int)
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
        "--wandb_note",
        type=str,
        default='',
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
    parser.add_argument("--multi_step_action", type=int, default=1, help="multiple step action prediction")
    
    # For policy
    # parser.add_argument('--head_type', type=str, default="lstm") # diffusion / gaussian
    parser.add_argument('--head_type', type=str, default="deterministic")  # policy type: deterministic / gaussian / diffusion
    parser.add_argument("--tanh_squash_dist", action="store_true", default=False)
    parser.add_argument("--state_dependent_std", action="store_true", default=False)
    parser.add_argument("--bin_coef", type=float, default=1.0)
    # for proxy task
    parser.add_argument("--data_percent", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    # for dynamic network
    # backbone
    parser.add_argument("--layer_decay", type=float, default=1.0, help='layerwise lr decay for flamingo layers')
    # exit
    parser.add_argument("--early_exit_layer", type=int, default=-1, help='remove all layers after it') 
    parser.add_argument("--multi_exit", action="store_true", default=False)
    parser.add_argument("--exit_interval", type=int, default=1, help='intervals between exits')
    parser.add_argument("--exit_weight", type=str, default='uniform', help='uniform/ascending/descending')
    parser.add_argument("--exit_lr_scale", type=float, default=1.0, help='scale learning rate for exits (only for joint training)')
    parser.add_argument("--exit_dropout", type=float, default=0.0, help='')
    parser.add_argument("--lstm_dropout", type=float, default=0.1, help='')
    parser.add_argument("--dropout_mode", default='wo_last', choices=['layerwise', 'last', 'wo_last'])
    parser.add_argument("--mlp_layernorm", default=False, action="store_true")
    parser.add_argument("--lstm_layernorm", default=False, action="store_true")
    parser.add_argument("--mlp_num_hidden_layers", type=int, default=3)
    parser.add_argument("--lstm_num_layers", type=int, default=4)
    parser.add_argument("--exit_decay", action="store_true", default=False)
    parser.add_argument("--use_extra_exit", action="store_true", default=False)
    parser.add_argument("--share_exit", action="store_true", default=False)
    parser.add_argument("--detach_extra_exit", type=int, default=1)
    parser.add_argument("--regularize_extra_exit", action="store_true", default=False)
    parser.add_argument("--use_layerwise_projection", action="store_true", default=False)
    parser.add_argument("--num_projection_layers", type=int, default=1)
    parser.add_argument("--skip_connection", action="store_true", default=False)
    parser.add_argument("--feat_distill_coef", type=float, default=0.0, help='use feature distillation if coef is greater than 0')
    # for value net
    # parser.add_argument("--with_value_net", action="store_true", default=False, help='jointly train value net')

    args = parser.parse_args()
    
    # print(f'{args.tanh_squash_dist=}')
    # if 'debug' in args.calvin_dataset:
    #     os.environ['WANDB_MODE'] = 'online'
    # else:
    #     # os.environ['WANDB_MODE'] = 'offline'
    #     os.environ['WANDB_MODE'] = 'online'
        
    
    if args.eval_hist_size == -1:
        args.eval_hist_size = args.window_size
        if args.head_type == "diffusion":
            args.eval_hist_size = args.n_obs_steps
    if args.tcp_rel:
        args.clip_state = True
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device_id = init_distributed_device(args)
    print("device_id: ", device_id)

    random_seed(args.seed)
    args.lm_path = mpt_dict[args.llm_name]["lang_encoder_path"]
    args.tokenizer_path = mpt_dict[args.llm_name]["tokenizer_path"]
    args.cross_attn_every_n_layers = mpt_dict[args.llm_name]["cross_attn_every_n_layers"]
    args.openflamingo_checkpoint = mpt_dict[args.llm_name]["openflamingo_checkpoint"]

    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_gripper=args.use_gripper,
        use_state=args.use_state,
        use_hist=args.use_hist,
        fusion_mode=args.fusion_mode,
        use_local_files=args.offline,
        use_media_placement_augmentation=args.use_media_placement_augmentation,
        window_size=args.eval_hist_size,
        freeze_embed=args.freeze_embed,
        train_params=args.train_params,
        sep_resampler=args.sep_resampler,
        last_action=args.last_action,
        use_diff=(args.head_type == "diffusion"), # Diff still have bugs of loaded data mismatch
        n_timesteps=args.n_timesteps,
        diff_horizon=args.diff_horizon,
        predict_epsilon=args.predict_epsilon,
        sep_lm_head=args.sep_lm_head,
        unfreeze_vit=args.unfreeze_vit,
        multi_step_action=args.multi_step_action,
        llm_name=args.llm_name,
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
        head_type=args.head_type,
        tanh_squash_dist=args.tanh_squash_dist,
        state_dependent_std=args.state_dependent_std,
        early_exit_layer=args.early_exit_layer,
        multi_exit=args.multi_exit,
        exit_interval=args.exit_interval,
        exit_dropout=args.exit_dropout,
        lstm_dropout=args.lstm_dropout,
        dropout_mode=args.dropout_mode,
        mlp_layernorm=args.mlp_layernorm,
        lstm_layernorm=args.lstm_layernorm,
        mlp_num_hidden_layers=args.mlp_num_hidden_layers,
        lstm_num_layers=args.lstm_num_layers,
        use_extra_exit=args.use_extra_exit,
        detach_extra_exit=args.detach_extra_exit,
        share_exit=args.share_exit,
        use_layerwise_projection=args.use_layerwise_projection,
        num_projection_layers=args.num_projection_layers,
        skip_connection=args.skip_connection,
    )
    
    if args.early_exit_layer < 0:
        args.early_exit_layer += model.lang_encoder.config.n_layers

    checkpoint_path = args.openflamingo_checkpoint
    if not args.debug and not args.no_pretrain:
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        if args.residual:
            model.lang_encoder.clone_parameters()

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
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

    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    else:
        model = model.float()
    if args.head_type == "diffusion" and (not args.debug):
        normalizer = model.diffusion_model.normalizer
        all_actions = np.vstack([calvin_dataset.dataset.__getitem__((i,1),True)["actions"] for i in range(0,10000)])
        normalizer.fit(all_actions, last_n_dims=1, mode='limits')

    model = model.to(device_id)

    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    def get_grouped_params(model, only_head=False):
        param_groups = {}
        def is_head(name):
            return 'lm_head' in name or 'lm_exit_modules' in name or 'extra_exit' in name

        def apply_decay(x):
            if not args.exit_decay:
                apply_decay_bool = "gated_cross_attn_layer" in x
            else:
                apply_decay_bool = ("gated_cross_attn_layer" in x) or is_head(x)
            return (
                apply_decay_bool
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x
            )

        def apply_lr_scale(n, p):
            if not only_head and is_head(n): # scale head lr only when joint training
                lr_scale = args.exit_lr_scale
            elif not only_head: # scale transformer layers lr when joint training
                layer_id = get_num_layer_for_flamingo(n, len(layerwsie_lr_scale_list), args.exit_interval)
                lr_scale = layerwsie_lr_scale_list[layer_id]
            else: # not scale when only train exit
                lr_scale = 1.0

            if lr_scale not in param_groups:
                param_groups[lr_scale] = []

            param_groups[lr_scale].append((n, p))

        # set lr_scale
        for n, p in model.named_parameters():
            if only_head and not is_head(n):
                continue
            apply_lr_scale(n, p)
            
        # set weight decay
        grouped_params = []
        for lr_scale, params in param_groups.items():
            params_with_wd, params_without_wd = [], []
            for n, p in params:
                if apply_decay(n):
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
            # Optimizer also support specifying per-parameter options. To do this, instead of passing an iterable of Variable s, pass in an iterable of dict s.
            #! Each of them will define a separate parameter group, and should contain a params key, containing a list of parameters belonging to it. 
            #! Other keys should match the keyword arguments accepted by the optimizers, and will be used as optimization options for this group.
            # Adamw doesn't natively support per-parameter-group learning rate scaling through the "lr_scale" key in the parameter group dictionary.
            # Unless update lr by lr_scale manually.
            # grouped_params.append({"params": [p for p in params_with_wd if p.requires_grad], "lr_scale": lr_scale, "weight_decay": args.weight_decay})
            # grouped_params.append({"params": [p for p in params_without_wd if p.requires_grad], "lr_scale": lr_scale, "weight_decay": 0.0})

            if only_head:
                lr = args.exit_learning_rate * lr_scale  # Calculate the learning rate for this group
            else:
                lr = args.joint_learning_rate * lr_scale  # Calculate the learning rate for this group
            grouped_params.append({"params": [p for p in params_with_wd if p.requires_grad], "lr": lr, "weight_decay": args.weight_decay})
            grouped_params.append({"params": [p for p in params_without_wd if p.requires_grad], "lr": lr, "weight_decay": 0.0})

        return grouped_params 


    # if args.rank == 0:
    #     print([n for n, p in model.named_parameters() if p.requires_grad])
    #     print([n for n, p in model.perceiver.named_parameters() if p.requires_grad])
    #     print([n for n, p in model.lang_encoder.named_parameters() if p.requires_grad])
    
    # adaptviely adjust learning rate with the base 8GPU and bs=6
    args.exit_learning_rate = args.exit_learning_rate * (args.batch_size_calvin / 6) * (args.world_size / 8) # adaptive lr
    args.joint_learning_rate = args.joint_learning_rate * (args.batch_size_calvin / 6) * (args.world_size / 8) # adaptive lr
    
    layerwsie_lr_scale_list = get_layerwise_lr_list(args)
    if args.rank == 0:
        print(layerwsie_lr_scale_list)
        print([{'lr': x['lr'], 'weight_decay': x['weight_decay'], 'num_params': sum(p.numel() for p in x['params']) / 1e6} for x in get_grouped_params(ddp_model, only_head=True)])
        print([{'lr': x['lr'], 'weight_decay': x['weight_decay'], 'num_params': sum(p.numel() for p in x['params']) / 1e6} for x in get_grouped_params(ddp_model, only_head=False)])
    
    exit_optimizer = torch.optim.AdamW(get_grouped_params(ddp_model, only_head=True), lr=args.exit_learning_rate)
    joint_optimizer = torch.optim.AdamW(get_grouped_params(ddp_model), lr=args.joint_learning_rate)

    # total_training_steps = (
    #     (args.train_num_samples_calvin) // (args.batch_size_calvin * args.world_size)
    # ) * args.num_epochs
    exit_training_steps = calvin_dataset.dataloader.num_batches * args.num_exit_epochs
    joint_training_steps = calvin_dataset.dataloader.num_batches * args.num_joint_epochs

    if args.rank == 0:
        print(f"Exit training steps: {exit_training_steps}")
        print(f"Joint training steps: {joint_training_steps}")

    if args.exit_lr_scheduler == "linear":
        exit_lr_scheduler = get_linear_schedule_with_warmup(
            exit_optimizer,
            num_warmup_steps=args.exit_warmup_steps,
            num_training_steps=exit_training_steps,
        )
    elif args.exit_lr_scheduler == "cosine":
        exit_lr_scheduler = get_cosine_schedule_with_warmup(
            exit_optimizer,
            num_warmup_steps=args.exit_warmup_steps,
            num_training_steps=exit_training_steps,
        )
    elif args.exit_lr_scheduler == 'cosine_restart':
        exit_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(exit_optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        exit_lr_scheduler = get_constant_schedule_with_warmup(
            exit_optimizer, num_warmup_steps=args.exit_warmup_steps
        )

        
    if args.joint_lr_scheduler == "linear":
        joint_lr_scheduler = get_linear_schedule_with_warmup(
            joint_optimizer,
            num_warmup_steps=args.joint_warmup_steps,
            num_training_steps=joint_training_steps,
        )
    elif args.joint_lr_scheduler == "cosine":
        joint_lr_scheduler = get_cosine_schedule_with_warmup(
            joint_optimizer,
            num_warmup_steps=args.joint_warmup_steps,
            num_training_steps=joint_training_steps,
        )
    elif args.joint_lr_scheduler == 'cosine_restart':
        joint_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(joint_optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        joint_lr_scheduler = get_constant_schedule_with_warmup(
            joint_optimizer, num_warmup_steps=args.joint_warmup_steps
        )

    use_diff = (args.head_type == "diffusion")
    # check if a checkpoint exists for this run

    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        ckpt_name = get_ckpt_name_pattern(args)
        checkpoint_list = glob.glob(f"{args.run_name}/{ckpt_name}")
        print(ckpt_name)
        checkpoint_list = [_ for _ in checkpoint_list if "__sep" not in _ and 'iter' not in _ and 'weights' not in _]
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
            )

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None and args.from_scratch is False:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        
        def filter_ckpt(checkpoint, skip_keys=[]):
            new_state_dict = OrderedDict()
            for key, value in checkpoint.items():
                flag = True
                for skip_key in skip_keys:
                    if skip_key in key:
                        flag = False
                        break
                if flag:
                    new_state_dict[key] = value
            return new_state_dict
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        if not args.real_data:
            if checkpoint["epoch"] < args.num_joint_epochs:
                joint_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                joint_lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            else:
                exit_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                exit_lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            resume_from_epoch = checkpoint["epoch"] + 1
    
            
    ddp_model.train()
    if args.real_data:
        resume_from_epoch = 0 
    
    print(f'{get_ckpt_name(args, 0)}')
    
    args.num_epochs = args.num_exit_epochs + args.num_joint_epochs
    
    for epoch in range(resume_from_epoch, args.num_epochs):
        calvin_dataset.set_epoch(epoch)
        calvin_loader = calvin_dataset.dataloader
        
        if epoch < args.num_joint_epochs:
            ddp_model.module.vision_encoder.train()
            ddp_model.module.perceiver.train()
            ddp_model.module.lang_encoder.train()  
            optimizer = joint_optimizer
            lr_scheduler = joint_lr_scheduler    
            only_train_head = False
        else:
            if epoch == args.num_joint_epochs:
                del joint_optimizer
                del joint_lr_scheduler
            ddp_model.module.vision_encoder.eval()
            ddp_model.module.perceiver.eval()
            ddp_model.module.lang_encoder.eval()
            optimizer = exit_optimizer
            lr_scheduler = exit_lr_scheduler
            only_train_head = True


        if args.head_type == "diffusion":
            train_one_epoch_calvin_diff(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
            )
        elif args.fusion_mode == 'two_way':
            train_one_epoch_calvin_two_way(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
            )
        else:
            if args.multi_exit:
                train_one_epoch_calvin_multi_exit(
                    args=args,
                    model=ddp_model,
                    epoch=epoch,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    calvin_loader=calvin_loader,
                    device_id=device_id,
                    wandb=wandb,
                    only_train_head=only_train_head,
                )
            else:
                train_one_epoch_calvin(
                    args=args,
                    model=ddp_model,
                    epoch=epoch,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    calvin_loader=calvin_loader,
                    device_id=device_id,
                    wandb=wandb,
                )

        if args.rank == 0 and epoch % args.save_freq == 0:
            save_ckpt(args, ddp_model, optimizer, lr_scheduler, epoch, epoch)

    # if args.rank == 0:
    #     if not os.path.exists(args.run_name):
    #         os.makedirs(args.run_name)

    #     ckpt_name = get_ckpt_name(args,)
    #     torch.save(get_checkpoint(ddp_model), f"{args.run_name}/{ckpt_name}")
    #     if args.report_to_wandb and args.save_checkpoints_to_wandb:
    #         wandb.save(f"{args.run_name}/{ckpt_name}")


if __name__ == "__main__":
    main()
