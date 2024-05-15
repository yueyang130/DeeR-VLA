import time
from contextlib import suppress

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from robot_flamingo.utils import world_to_tcp_frame, tcp_to_world_frame
import itertools
from einops import rearrange
from torch.cuda.amp import GradScaler
import os
from models.value_net import value_to_bin_index, get_bin_boundaries, get_similarity
import re

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
    
def save_ckpt(args, ddp_model, optimizer, lr_scheduler, epoch, step, extra_optimizer=None, extra_lr_scheduler=None):
    if not os.path.exists(args.run_name):
        os.makedirs(args.run_name)

    checkpoint_dict = {
        "epoch": epoch,
        "head_type": args.head_type,
        "tanh_squash_dist": args.tanh_squash_dist,
        "state_dependent_std": args.state_dependent_std,
        "early_exit_layer": args.early_exit_layer,
        "multi_exit": args.multi_exit,
        "share_exit": args.share_exit,
        "use_extra_exit": args.use_extra_exit,
        "exit_interval": args.exit_interval,
        "exit_weight": args.exit_weight,
        "exit_dropout": args.exit_dropout,
        "lstm_dropout": args.lstm_dropout,
        "dropout_mode": args.dropout_mode,
        "mlp_layernorm": args.mlp_layernorm,
        "lstm_layernorm": args.lstm_layernorm,
        "mlp_num_hidden_layers": args.mlp_num_hidden_layers,
        "lstm_num_layers": args.lstm_num_layers,
        "use_layerwise_projection": args.use_layerwise_projection,
        "num_projection_layers": args.num_projection_layers,
        "skip_connection": args.skip_connection,
        "pooling": args.pooling,
        "precision": args.precision,
        "model_state_dict": get_checkpoint(ddp_model),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }
    if extra_optimizer:
        checkpoint_dict['extra_optimizer_state_dict'] = extra_optimizer.state_dict(),
    if extra_lr_scheduler:
        checkpoint_dict['extra_lr_scheduler_state_dict'] = extra_lr_scheduler.state_dict()

    ckpt_name = get_ckpt_name(args, step)
    ckpt_path = os.path.join(args.run_name, ckpt_name)

    print(f"Saving checkpoint to {ckpt_path}")
    torch.save(checkpoint_dict, ckpt_path)
    if args.delete_previous_checkpoint:
        if epoch > 0:
            os.remove(ckpt_path)
            
def save_value_net_ckpt(args, ddp_value_net, optimizer, lr_scheduler, epoch, step, robo_ckpt_path):
    if not os.path.exists(args.run_name):
        os.makedirs(args.run_name)

    checkpoint_dict = {
        "epoch": epoch,
        "precision": args.precision,
        "with_exit_embed": args.with_exit_embed,
        "with_time_embed": args.with_time_embed,
        "discrete": args.discrete,
        "num_bin": args.num_bin,
        "value_net_type": args.value_net_type,
        "model_state_dict": ddp_value_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }
    
   

    ckpt_name = os.path.basename(robo_ckpt_path)[:-4]
    
    value_prefix = f'value_net_{args.value_net_type}'
    if args.discrete:
            value_prefix += f'_b{args.num_bin}'
    
    if epoch != -1:
        if epoch > 1000:
            ckpt_name += '_{}_{}_iter.pth'.format(value_prefix, epoch)
        else:
            ckpt_name += '_{}_{}.pth'.format(value_prefix, epoch)
    else:
        ckpt_name += '_{}_final_weights.pth'.format(value_prefix)
    
    ckpt_path = os.path.join(args.run_name, ckpt_name)
    if args.rank == 0:
        print(f"Saving checkpoint to {ckpt_path}")
    torch.save(checkpoint_dict, ckpt_path)
    if args.delete_previous_checkpoint:
        if epoch > 0:
            os.remove(ckpt_path)

def get_ckpt_prefix(args, train_value=False):
    ckpt_name = args.wandb_note + '_' if args.wandb_note else ''
    
    if hasattr(args, 'exit_strategy'):
        ckpt_name += 'stg={}_'.format(args.exit_strategy)
        if args.exit_strategy == 'joint':
            ckpt_name += 'frq={}_'.format(args.llm_update_freq)
        if args.exit_strategy == 'pre':
            ckpt_name += '{}+{}_'.format(args.num_exit_epochs, args.num_joint_epochs)
        if args.exit_strategy == 'post':
            ckpt_name += '{}+{}_'.format(args.num_joint_epochs, args.num_exit_epochs)           
    # if args.use_gripper:
    #     ckpt_name += '{}_checkpoint_gripper_{}_hist_{}_{}_exit_layer_{}_'.format(args.precision, args.fusion_mode, args.hist_window, '' if not args.sep_resampler else 'sep_', args.early_exit_layer)
    # else:
    #     ckpt_name += '{}_checkpoint_no_gripper_hist_{}_{}_exit_layer_{}_'.format(args.precision, args.hist_window, '' if not args.sep_resampler else 'sep_', args.early_exit_layer)
    ckpt_name += 'layer_{}_'.format(args.early_exit_layer)
    if args.multi_exit:
        ckpt_name += 'multie_'
        if args.share_exit:
            ckpt_name += 'share_'
        if args.no_auxiliary_action_head_loss:
            ckpt_name += 'noloss_'
        if args.exit_weight != 'uniform':
            ckpt_name += f'{args.exit_weight}_'
        ckpt_name += 'intv={}_'.format(args.exit_interval)
    if args.layer_decay != 1.0:
        ckpt_name += 'layerdecay={}_'.format(args.layer_decay)
    if args.feat_distill_coef > 0:
        ckpt_name += 'distill={}_'.format(args.feat_distill_coef)
    if args.use_extra_exit:
        ckpt_name += 'extrae_'
        if not args.detach_extra_exit:
            ckpt_name += 'nodth_'
        if args.regularize_extra_exit:
            ckpt_name += 'reg_'
        if args.use_layerwise_projection:
            ckpt_name += f'lwproj{args.num_projection_layers}L_'
            if args.skip_connection:
                ckpt_name += 'res_'
    # if args.mlp_num_hidden_layers != 2:
    #     ckpt_name += 'mlp{}L_'.format(args.mlp_num_hidden_layers)
    # if args.mlp_layernorm:
    #     ckpt_name += 'mlpln_'
    # if args.lstm_num_layers != 4:
    #     ckpt_name += 'lstm{}L_'.format(args.lstm_num_layers)
    # if args.lstm_layernorm:
    #     ckpt_name += 'lstmln_'
    if args.exit_dropout != 0:
        ckpt_name += 'mlpdrp={}_{}_'.format(args.exit_dropout, args.dropout_mode)    
    if args.lstm_dropout != 0:
        ckpt_name += 'lstmdrp={}_'.format(args.lstm_dropout)
    if args.exit_decay:
        ckpt_name += 'decay_'
    if args.data_percent < 1.0:
        ckpt_name += f'data_{args.data_percent}_'
    if args.real_data:
        ckpt_name += 'real_'
    if args.train_params != -1:
        ckpt_name += 'train_{}_'.format(args.train_params)
    if args.no_pretrain:
        ckpt_name += 'no_pretrain_'
    if args.fwd_pred:
        ckpt_name += 'pred_rgb_'
    if args.fwd_pred_hand:
        ckpt_name += 'pred_hand_'
    if args.freeze_sampler:
        ckpt_name += 'freeze_sam_'
    if args.use_state:
        ckpt_name += 'state_'
    if args.rgb_pad != -1 or args.gripper_pad != -1:
        ckpt_name += 'aug_{}_{}_'.format(args.rgb_pad, args.gripper_pad)
    if args.use_hist:
        ckpt_name += 'fc_'
    if args.multi_step_action != 1:
        ckpt_name += '{}_step_'.format(args.multi_step_action)
    if args.head_type == "diffusion":
        ckpt_name += 'diff_'
    if args.head_type == "gaussian":
        ckpt_name += 'gaussian_'
        ckpt_name += f'bin_coef_{args.bin_coef}_'
    if args.tanh_squash_dist:
        ckpt_name += 'ts_'
    if args.traj_cons:
        ckpt_name += 'traj_cons_'
    if args.sep_lm_head:
        ckpt_name += 'lm_head_'
    if args.dif_ws:
        ckpt_name += 'difws_{}_{}_'.format(args.min_window_size, args.max_window_size)
    elif args.window_size != 8:
        ckpt_name += 'ws_{}_'.format(args.window_size)
    if args.unfreeze_vit:
        ckpt_name += 'unfreeze_vit_'
    if args.llm_name != 'llama':
        ckpt_name += '{}_'.format(args.llm_name)
    if args.pooling != 'max':
        ckpt_name += '{}pool_'.format(args.pooling)
    if args.text_aug:
        ckpt_name += 'text_aug_'
    if args.residual:
        ckpt_name += 'res_'
    if args.freeze_embed:
        ckpt_name += 'freeze_emb_'
    if args.tcp_rel:
        ckpt_name += 'tcp_'
    if args.decoder_type != 'lstm':
        ckpt_name += '{}_{}_'.format(args.decoder_type, args.hidden_size)
    # ckpt_name += 'jointlr_{:.6f}_'.format(args.joint_learning_rate) 
    # if args.joint_lr_scheduler != 'constant':
    #     ckpt_name += '{}_'.format(args.joint_lr_scheduler) 
    # if args.exit_lr_scale != 1.0:
    #     ckpt_name += 'exitscale={}_'.format(args.exit_lr_scale) 
    # if args.exit_lr_scheduler != 'constant':
    #     ckpt_name += 'exitlr_{}_'.format(args.exit_lr_scheduler)  
    
    return ckpt_name

def get_ckpt_name(args, epoch=-1):
    
    ckpt_name = get_ckpt_prefix(args)

    if epoch != -1:
        if epoch > 1000:
            ckpt_name += '{}_iter.pth'.format(epoch)
        else:
            ckpt_name += '{}.pth'.format(epoch)
    else:
        ckpt_name += 'final_weights.pth'
    return ckpt_name

def get_ckpt_name_pattern(args):
    ckpt_name = get_ckpt_prefix(args)
    ckpt_name += '*.pth'
    return ckpt_name


def get_num_layer_for_flamingo(var_name, num_max_layer, exit_interval):
    """
    Divide [3, 3, 27, 3] layers into 12 groups; each group is three 
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    if var_name.startswith("module"):
        var_name = var_name.replace('module.', '')
    
    if var_name.startswith("lang_encoder.transformer.blocks"):
        match = re.search(r'blocks\.(\d+)', var_name)
        block_id = int(match.group(1))
        layer_id = int(block_id / exit_interval)
    elif var_name.startswith("lang_encoder.transformer.wte") or var_name.startswith('lang_encoder.transformer.ln_f'):
        layer_id = num_max_layer-2
    elif var_name.startswith("perceiver") or var_name.startswith("vision_encoder"):
        layer_id = num_max_layer-1
    else:
        raise NotImplementedError(f"Unknown parameter {var_name}")

    return layer_id
    
    
def get_layerwise_lr_list(args):
    num_layers = int((args.early_exit_layer + 1) / args.exit_interval)
    lr_scale_list = list(args.layer_decay ** (num_layers - 1 - i) for i in range(num_layers))
    embedding_lr_scale = lr_scale_list[0]
    perceiver_lr_scale = 1.0
    lr_scale_list.extend([embedding_lr_scale, perceiver_lr_scale])
    return lr_scale_list


def get_exit_weights(weight_mode, num, use_extra_exit, device):
    
    if weight_mode == 'uniform':
        weight = torch.ones(num, dtype=torch.float32, device=device)
    elif weight_mode == 'ascending':
        weight = torch.arange(1, num+1, dtype=torch.float32, device=device) # 1,2,3,4,..,N, placehold
    elif weight_mode == 'descending':
        weight = torch.arange(num-1, -1, step=-1, dtype=torch.float32, device=device) # N,...,2,1, placehold
    
        
    if use_extra_exit:
        weight[:-1] = weight[:-1] / weight[:-1].mean()
        weight[-1] = 1.0
    else:
        weight = weight / weight.mean()
    return weight

def train_one_epoch_calvin_diff(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
    normalizer=None,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    if isinstance(model, DistributedDataParallel):
        diffusion_model = model.module.diffusion_model
    else:
        diffusion_model = model.diffusion_model
    
    if normalizer is None:
        normalizer = diffusion_model.normalizer

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []

    if isinstance(model, DistributedDataParallel):
        action_dim = model.module.action_head.out_features + 1 # joint + gripper
    else:
        action_dim = model.action_head.out_features + 1 # joint + gripper
 
    class LowdimMaskGenerator(nn.Module):
        def __init__(self,
            action_dim, obs_dim,
            # obs mask setup
            max_n_obs_steps=3, 
            fix_obs_steps=True, 
            # action mask
            action_visible=True,
            return_one_mask=False
            ):
            super().__init__()
            self.action_dim = action_dim
            self.obs_dim = obs_dim
            self.max_n_obs_steps = max_n_obs_steps
            self.fix_obs_steps = fix_obs_steps
            self.action_visible = action_visible
            self.return_one_mask = return_one_mask

        @torch.no_grad()
        def forward(self, shape, device, seed=None):
            # device = self.device
            B, T, D = shape
            assert D == (self.action_dim + self.obs_dim)

            # create all tensors on this device
            rng = torch.Generator(device=device)
            if seed is not None:
                rng = rng.manual_seed(seed)

            # generate dim mask
            dim_mask = torch.zeros(size=shape, 
                dtype=torch.bool, device=device)
            is_action_dim = dim_mask.clone()
            is_action_dim[...,:self.action_dim] = True
            is_obs_dim = ~is_action_dim

            # generate obs mask
            if self.fix_obs_steps:
                obs_steps = torch.full((B,), 
                fill_value=self.max_n_obs_steps, device=device)
            else:
                obs_steps = torch.randint(
                    low=1, high=self.max_n_obs_steps+1, 
                    size=(B,), generator=rng, device=device)
                
            steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
            obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
            obs_mask = obs_mask

            # generate action mask
            if self.action_visible:
                action_steps = torch.maximum(
                    obs_steps - 1, 
                    torch.tensor(0,
                        dtype=obs_steps.dtype, 
                        device=obs_steps.device))
                action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
                action_mask = action_mask & is_action_dim


            if self.return_one_mask:
                mask = obs_mask & is_obs_dim
                if self.action_visible:
                    mask = mask | action_mask
            
                return mask
            if self.obs_dim <= 0:
                assert self.fix_obs_steps, "We require fix obs steps to obtain obs masks"
                obs_mask = obs_mask[0,:,0]
            return action_mask, obs_mask     

    mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=args.n_obs_steps,
            fix_obs_steps=True,
            action_visible=True,
    )

    act_mask, obs_mask = None, None
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        # put images and labels on device
        images = (batch_calvin[0].unsqueeze(2).unsqueeze(2))
        gripper = (batch_calvin[3].unsqueeze(2).unsqueeze(2))

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        input_ids = batch_calvin[1][0].unsqueeze(1).repeat(1, images.shape[1], 1)

        # do the same to the attention mask 
        attention_mask = batch_calvin[1][1].unsqueeze(1).repeat(1, images.shape[1], 1)
        state_tensor = batch_calvin[4].unsqueeze(2).unsqueeze(2)

        actions = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        actions = normalizer.normalize(actions) # labels normalization

        if act_mask is None or obs_mask is None:
            act_mask, obs_mask = mask_generator(actions.shape, images.device)

        batch_size = actions.shape[0]
        # Mask and leave history data for generating features
        images = images[:,obs_mask,...]
        gripper = gripper[:,obs_mask,...]
        input_ids = input_ids[:,obs_mask,...]
        attention_mask = attention_mask[:,obs_mask,...]
        state_tensor = state_tensor[:,obs_mask,...]

         # put images and labels on device
        images = images.to(device_id, dtype=cast_dtype, non_blocking=True)
        gripper = gripper.to(device_id, dtype=cast_dtype, non_blocking=True)

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        input_ids = input_ids.to(device_id, non_blocking=True)

        # do the same to the attention mask 
        attention_mask = attention_mask.to(device_id, non_blocking=True)
        state_tensor = state_tensor.to(device_id, dtype=cast_dtype, non_blocking=True)

        # print("test", images.shape, gripper.shape, input_ids.shape, attention_mask.shape, state_tensor.shape)
        # import pdb; pdb.set_trace()
        
        # merge the batch and the sequence dimension
        images = images.flatten(0, 1)
        gripper = gripper.flatten(0, 1)
        state_tensor = state_tensor.flatten(0, 1)
        input_ids = input_ids.flatten(0, 1)
        attention_mask = attention_mask.flatten(0, 1)

        with autocast():
            model_out = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=gripper,
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None
            ) # Features
            model_out = model_out.logits

        # compute loss
        tt = torch.randint(0, args.n_timesteps, (batch_size,), device=actions.device).long()
        noise = torch.randn_like(actions)
        
        action_noisy = diffusion_model.q_sample(x_start=actions, t=tt, noise=noise)
 
        # apply conditioning
        action_noisy[act_mask] = actions[act_mask]
        # pred = diffusion_model(action_noisy, tt, global_cond=None)
        pred = diffusion_model(action_noisy, tt, global_cond=model_out)
        pred[act_mask] = actions[act_mask] # So we remove the gradient
        assert noise.shape == pred.shape

        if args.predict_epsilon:
            loss = F.mse_loss(pred, noise, reduction='none')
        else:
            loss = F.mse_loss(pred, actions, reduction='none')

        loss_calvin = loss.mean()

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_calvin * args.loss_multiplier_calvin
        )
        mv_avg_loss.append(loss.item())
        loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_calvin": divided_loss_calvin.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )


        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item()})


def train_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    device_num = int(torch.distributed.get_world_size())
    scaler = GradScaler(enabled='amp' in args.precision, growth_interval=int(4000/device_num))

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
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
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        
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

        with autocast():
            if args.head_type == 'deterministic':
                output = model(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    # labels=labels,  # loss计算放在外面
                    vision_gripper=gripper,
                    state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                    with_gripper_logits=True,
                )
                num_actions, bin_gripper = output.logits[0], output.logits[1]
                bin_actions, bin_logits = bin_gripper
                if args.multi_step_action != 1:
                    bs, seq_len = num_actions.shape[:2]
                    num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                loss_mse = loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
                loss_mle = torch.tensor([.0])
                std = torch.tensor([.0])
            elif args.head_type == 'gaussian':
                output = model(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    # labels=labels,  # loss计算放在外面
                    vision_gripper=gripper,
                    state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                    with_gripper_logits=True,
                    act=labels[0]
                )
                mean,  bin_gripper, log_prob, std = output.logits[:4]
                bin_actions, bin_logits = bin_gripper
                loss_mse = torch.nn.functional.huber_loss(mean, labels[0])
                if args.multi_step_action != 1:
                    bs, seq_len = log_prob.shape[:2]
                    log_prob = log_prob.reshape(bs, seq_len, args.multi_step_action, -1)
                loss_mle = loss_calvin_num = - log_prob.mean()
            else:
                raise NotImplementedError(f'{args.head_type=}')

        # compute loss
        # if args.rank == 0:
            # print(len(output.hidden_states)) # number of attention layers (24 for MPT-1B)
            # Note: the dim of language tokens in a task is the token dim of Transformer since Flamingo is for visual understanding
            # Then the dim of action seqence is aggregated by LSTM head for decision.
            # print(output.hidden_states[0].shape, output.hidden_states[-1].shape)  # (bs * action_seq_len, lang_len, d)
            # print(output.hidden_states[0].requires_grad)
            # print(output.logits[0].shape) # (bs, action_seq_len, 6)
            # print(output.logits[0].requires_grad) # (bs, action_seq_len, 6)
            # print(output.logits[1].shape) # (bs, action_seq_len, 1)
            # print(labels[0].shape)
            # print(labels[1].shape)

        # reshape for loss calculation
        if args.multi_step_action != 1:
            bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            bin_logits = bin_logits.reshape(bs, seq_len, args.multi_step_action, -1)
        
        # print(f'{bin_actions.shape=}, {bin_logits.shape=}')

        with autocast():
            # loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
            loss_calvin_bin = torch.nn.functional.binary_cross_entropy_with_logits(bin_logits, labels[1])
        
            if args.head_type == 'deterministic':
                if args.real_data:
                    loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
                else:
                    loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
            elif args.head_type == 'gaussian':
                loss_calvin = loss_calvin_num + loss_calvin_bin * args.bin_coef

            divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

            #### BACKWARD PASS ####
            loss = (
                divided_loss_calvin * args.loss_multiplier_calvin
            )
        mv_avg_loss.append(loss.item())
        
        if 'amp' in args.precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad and m.weight.grad is not None:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        # model.apply(mask_embedding)

        # unscale grad. Thus clip with original threshold
        if 'amp' in args.precision:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            if 'amp' in args.precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                # wandb.log(
                #     {
                #         "data_time": data_time_m.avg,
                #         "step_time": step_time_m.avg,
                #         "calvin_samples_per_second": calvin_samples_per_second,
                #         "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                #     },
                #     commit=False,
                # )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                         "lr": optimizer.param_groups[0]["lr"],
                        "loss_calvin": divided_loss_calvin.item(),
                        "loss_calvin_bin": loss_calvin_bin.item(),
                        "loss_calvin_num": loss_calvin_num.item(),
                        "mse": loss_mse.item(),
                        "mle": loss_mle.item(),
                        "mean_std": std.mean().item(),
                        "global_step": global_step,
                        "scale_factor": scaler.get_scale(),
                    },
                    commit=True,
                )


        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f} (mle) {loss_mle.item():.3f} (mse){loss_mse.item():.3f} (bce){loss_calvin_bin.item():.3f} (std) {std.mean().item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "mle": loss_mle.item(), "mse": loss_mse.item(), "Lbin": loss_calvin_bin.item(), "std": std.mean().item()})

        if args.save_every_iter != -1 and global_step % args.save_every_iter == 0 and global_step > 0:
                
            if args.rank == 0:
                save_ckpt(args, model, optimizer, lr_scheduler, epoch, global_step)

def train_one_epoch_calvin_multi_exit(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
    only_train_head = False,
    value_net = None,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    device_num = int(torch.distributed.get_world_size())
    scaler = GradScaler(enabled='amp' in args.precision, growth_interval=int(4000/device_num))

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()
    if value_net:
        value_net.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
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
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        
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

        with autocast():
            if args.head_type == 'deterministic':
                o = model(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    # labels=labels,  # loss计算放在外面
                    vision_gripper=gripper,
                    state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                    with_gripper_logits=True,
                    # return_feature = True,
                    return_feature = False,
                    no_backbone_grad=only_train_head,
                )
                
                if args.use_extra_exit:
                    final_output, exit_outputs, extra_exit_output, extra_exit_output2 = o[0], o[1], o[2], o[3]
                    # features, exit_idx = o[3], o[4]
                    all_outputs = exit_outputs + [final_output.logits] + [extra_exit_output]
                    if args.regularize_extra_exit:
                        all_outputs.append(extra_exit_output2)
                    
                else:
                    final_output, exit_outputs = o[0], o[1]
                    # get joint outputs
                    all_outputs = exit_outputs + [final_output.logits]
                
                num_action_list, gripper_logit_list, proj_feat_list = [], [], []
                for output in all_outputs:
                    num_actions, bin_gripper = output[0], output[1]
                    # proj_feat =  output[2]
                    bin_actions, bin_logits = bin_gripper
                    if args.multi_step_action != 1:
                        bs, seq_len = num_actions.shape[:2]
                        num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                        # bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                        bin_logits = bin_logits.reshape(bs, seq_len, args.multi_step_action, -1)
                    num_action_list.append(num_actions)
                    gripper_logit_list.append(bin_logits)
                    # proj_feat_list.append(proj_feat)
                
                # if args.use_extra_exit:
                #     proj_feat_list = proj_feat_list[:-1]

                # get action loss per head type
                num_actions = torch.stack(num_action_list, dim=0)
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0][None], reduction='none').mean(-1)
                # print(f'{loss_calvin_num.shape=}')
                
                loss_mse = loss_calvin_num.mean()
                loss_mle = torch.tensor([.0])
                std = torch.tensor([.0])
            elif args.head_type == 'gaussian':
                raise NotImplementedError("Please fix the bug in gaussian policy in single exit before running multi-exit gaussian policy!")

        # compute loss
        # if args.rank == 0:
            # print(len(output.hidden_states)) # number of attention layers (24 for MPT-1B)
            # Note: the dim of language tokens in a task is the token dim of Transformer since Flamingo is for visual understanding
            # Then the dim of action seqence is aggregated by LSTM head for decision.
            # print(output.hidden_states[0].shape, output.hidden_states[-1].shape)  # (bs * action_seq_len, lang_len, d)
            # print(output.hidden_states[0].requires_grad)
            # print(output.logits[0].shape) # (bs, action_seq_len, 6)
            # print(output.logits[0].requires_grad) # (bs, action_seq_len, 6)
            # print(output.logits[1].shape) # (bs, action_seq_len, 1)
            # print(labels[0].shape)
            # print(labels[1].shape)

        
        # print(f'{bin_actions.shape=}, {bin_logits.shape=}')

        with autocast():
            # loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
            bin_logits = torch.stack(gripper_logit_list, dim=0)
            bin_targets = torch.stack([labels[1]] * len(all_outputs), dim=0)
            loss_calvin_bin = torch.nn.functional.binary_cross_entropy_with_logits(bin_logits, bin_targets, reduction='none').mean(-1)
            # print(f'{loss_calvin_num.shape=}')
        
            if args.head_type == 'deterministic':
                if args.real_data:
                    loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
                else:
                    loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
            elif args.head_type == 'gaussian':
                loss_calvin = loss_calvin_num + loss_calvin_bin * args.bin_coef
            
            # get mean for every exit
            dim = loss_calvin.dim()
            loss_calvin = loss_calvin.mean(dim=tuple(range(1, dim)))
            weights = get_exit_weights(args.exit_weight, len(all_outputs), args.use_extra_exit, device=loss_calvin.device)
            if args.no_auxiliary_action_head_loss:
                if args.regularize_extra_exit:
                    weights[:-2] = 0
                else:
                    weights[:-1] = 0
            if args.rank == 0 and num_steps <= 1:
                print(weights)
            # print(loss_calvin)
            loss_calvin *= weights
            # print(loss_calvin)
            loss_calvin = loss_calvin.sum() # since weights are normalzied, thus sum losses of all exits
            
            # feature distillation
            #! take lots of GPU memory dut to huge size of hidden states
            # feats = final_output.hidden_states # n_exit x (bs * action_seq_len, lang_len, d)
            # last_feat = feats[-1].unsqueeze(0) # (1, bs * action_seq_len, lang_len, d)
            # prev_feats = torch.stack(feats[:-1], dim=0) # (n_exit - 1, bs * action_seq_len, lang_len, d)
            
            # last_feat = torch.max(last_feat, dim=-2)[0]
            # prev_feats = torch.max(prev_feats, dim=-2)[0] # (n_exit - 1, bs * action_seq_len, d)
            
            # last_feat = proj_feat_list[-1].unsqueeze(0)
            # prev_feats = torch.stack(proj_feat_list[:-1], dim=0)
            
            # sim = get_similarity(last_feat, prev_feats, detach_f1=True) # (n_exit - 1, bs * action_seq_len)
            # sim = sim.mean(dim=(1,2)) # (n_exit - 1,)
            # loss_distill = - sim.mean()
            
            # loss_distill = nn.functional.mse_loss(prev_feats, last_feat.detach(), reduction='none').mean(dim=(1,2))
            
            if args.feat_distill_coef > 0:
                loss_calvin += loss_distill * args.feat_distill_coef
            else:
                 loss_distill = 0
             
            divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

            #### BACKWARD PASS ####
            loss = (
                divided_loss_calvin * args.loss_multiplier_calvin
            )
            
        #### LOG #####
        if args.use_extra_exit:
            if args.regularize_extra_exit:
                loss_calvin_bin_list = loss_calvin_bin[:-2].mean(dim=tuple(range(1, dim)))
                loss_calvin_num_list = loss_calvin_num[:-2].mean(dim=tuple(range(1, dim)))
                extra_exit_loss_bin = loss_calvin_bin[-2].mean()
                extra_exit_loss_num = loss_calvin_num[-2].mean()
                extra_exit_loss2_bin = loss_calvin_bin[-1].mean()
                extra_exit_loss2_num = loss_calvin_num[-1].mean()
            else:
                loss_calvin_bin_list = loss_calvin_bin[:-1].mean(dim=tuple(range(1, dim)))
                loss_calvin_num_list = loss_calvin_num[:-1].mean(dim=tuple(range(1, dim)))
                extra_exit_loss_bin = loss_calvin_bin[-1].mean()
                extra_exit_loss_num = loss_calvin_num[-1].mean()
                extra_exit_loss2_bin = torch.tensor([.0])
                extra_exit_loss2_num =  torch.tensor([.0])
        else:
            loss_calvin_bin_list = loss_calvin_bin.mean(dim=tuple(range(1, dim)))
            loss_calvin_num_list = loss_calvin_num.mean(dim=tuple(range(1, dim)))
            extra_exit_loss_bin = extra_exit_loss2_bin = torch.tensor([.0])
            extra_exit_loss_num = extra_exit_loss2_num =  torch.tensor([.0])
        
            
        mv_avg_loss.append(loss.item())
        
        if 'amp' in args.precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad and m.weight.grad is not None:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        # model.apply(mask_embedding)

        # unscale grad. Thus clip with original threshold
        if 'amp' in args.precision:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            if 'amp' in args.precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                # wandb.log(
                #     {
                #         "data_time": data_time_m.avg,
                #         "step_time": step_time_m.avg,
                #         "calvin_samples_per_second": calvin_samples_per_second,
                #         "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                #     },
                #     commit=False,
                # )
                step_time_m.reset()
                data_time_m.reset()
                
                log_dict = {
                        "lr": 0 if only_train_head else optimizer.param_groups[0]["lr"],
                        "exit_lr": optimizer.param_groups[0]["lr"] if only_train_head else optimizer.param_groups[-1]["lr"],
                        "loss_calvin": divided_loss_calvin.item(),
                        "loss_calvin_bin": loss_calvin_bin.mean().item(),
                        "loss_calvin_num": loss_calvin_num.mean().item(),
                        **{f"loss_calvin_bin_{i}": x.item() for i, x in enumerate(loss_calvin_bin_list)},
                        **{f"loss_calvin_num_{i}": x.item() for i, x in enumerate(loss_calvin_num_list)},
                        # **{f"feat_sim_{i}": x.item() for i, x in enumerate(sim)},
                        "extra_exit_loss_bin": extra_exit_loss_bin.item(),
                        "extra_exit_loss_num": extra_exit_loss_num.item(),
                        "extra_exit_loss2_bin": extra_exit_loss2_bin.item(),
                        "extra_exit_loss2_num": extra_exit_loss2_num.item(),
                        "mse": loss_mse.item(),
                        "mle": loss_mle.item(),
                        "mean_std": std.mean().item(),
                        "global_step": global_step,
                        "scale_factor": scaler.get_scale(),
                    }
                
                if args.feat_distill_coef > 0:
                    log_dict['loss_distill'] = loss_distill.mean().item()

                wandb.log(
                    log_dict,
                    commit=True,
                )


        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f} (mle) {loss_mle.item():.3f} (mse){loss_mse.item():.3f} (bce){loss_calvin_bin.mean().item():.3f} (std) {std.mean().item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "mle": loss_mle.item(), "mse": loss_mse.item(), "Lbin": loss_calvin_bin.mean().item(), "std": std.mean().item()})

        if args.save_every_iter != -1 and global_step % args.save_every_iter == 0 and global_step > 0:
            if args.rank == 0:
                save_ckpt(args, model, optimizer, lr_scheduler, epoch, global_step)


def train_one_epoch_calvin_multi_exit_joint_strategy(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    llm_optimizer,
    exit_optimizer,
    llm_lr_scheduler,
    exit_lr_scheduler,
    device_id,
    wandb,
    value_net = None,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    device_num = int(torch.distributed.get_world_size())
    scaler = GradScaler(enabled='amp' in args.precision, growth_interval=int(4000/device_num))

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()
    if value_net:
        value_net.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    for num_steps, batch_calvin in t:
        if num_steps % args.llm_update_freq == 0:
            only_train_head = False # update llm and exits
        else:
            only_train_head = True # only update exits
        
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
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
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        
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

        with autocast():
            if args.head_type == 'deterministic':
                o = model(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    # labels=labels,  # loss计算放在外面
                    vision_gripper=gripper,
                    state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                    with_gripper_logits=True,
                    # return_feature = True,
                    return_feature = False,
                    no_backbone_grad=only_train_head,
                )
                
                if args.use_extra_exit:
                    final_output, exit_outputs, extra_exit_output = o[0], o[1], o[2]
                    # features, exit_idx = o[3], o[4]
                    all_outputs = exit_outputs + [final_output.logits] + [extra_exit_output]
                else:
                    final_output, exit_outputs = o[0], o[1]
                    # get joint outputs
                    all_outputs = exit_outputs + [final_output.logits]
                
                num_action_list, gripper_logit_list, proj_feat_list = [], [], []
                for output in all_outputs:
                    num_actions, bin_gripper = output[0], output[1]
                    # proj_feat =  output[2]
                    bin_actions, bin_logits = bin_gripper
                    if args.multi_step_action != 1:
                        bs, seq_len = num_actions.shape[:2]
                        num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                        # bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                        bin_logits = bin_logits.reshape(bs, seq_len, args.multi_step_action, -1)
                    num_action_list.append(num_actions)
                    gripper_logit_list.append(bin_logits)
                    # proj_feat_list.append(proj_feat)
                
                # if args.use_extra_exit:
                #     proj_feat_list = proj_feat_list[:-1]

                # get action loss per head type
                num_actions = torch.stack(num_action_list, dim=0)
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0][None], reduction='none').mean(-1)
                # print(f'{loss_calvin_num.shape=}')
                
                loss_mse = loss_calvin_num.mean()
                loss_mle = torch.tensor([.0])
                std = torch.tensor([.0])
            elif args.head_type == 'gaussian':
                raise NotImplementedError("Please fix the bug in gaussian policy in single exit before running multi-exit gaussian policy!")

        # compute loss
        # if args.rank == 0:
            # print(len(output.hidden_states)) # number of attention layers (24 for MPT-1B)
            # Note: the dim of language tokens in a task is the token dim of Transformer since Flamingo is for visual understanding
            # Then the dim of action seqence is aggregated by LSTM head for decision.
            # print(output.hidden_states[0].shape, output.hidden_states[-1].shape)  # (bs * action_seq_len, lang_len, d)
            # print(output.hidden_states[0].requires_grad)
            # print(output.logits[0].shape) # (bs, action_seq_len, 6)
            # print(output.logits[0].requires_grad) # (bs, action_seq_len, 6)
            # print(output.logits[1].shape) # (bs, action_seq_len, 1)
            # print(labels[0].shape)
            # print(labels[1].shape)

        
        # print(f'{bin_actions.shape=}, {bin_logits.shape=}')

        with autocast():
            # loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
            bin_logits = torch.stack(gripper_logit_list, dim=0)
            bin_targets = torch.stack([labels[1]] * len(all_outputs), dim=0)
            loss_calvin_bin = torch.nn.functional.binary_cross_entropy_with_logits(bin_logits, bin_targets, reduction='none').mean(-1)
            # print(f'{loss_calvin_num.shape=}')
        
            if args.head_type == 'deterministic':
                if args.real_data:
                    loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
                else:
                    loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
            elif args.head_type == 'gaussian':
                loss_calvin = loss_calvin_num + loss_calvin_bin * args.bin_coef
            
            # get mean for every exit
            dim = loss_calvin.dim()
            loss_calvin = loss_calvin.mean(dim=tuple(range(1, dim)))
            weights = get_exit_weights(args.exit_weight, len(all_outputs), args.use_extra_exit, device=loss_calvin.device)
            loss_calvin *= weights
            loss_calvin = loss_calvin.sum() # since weights are normalzied, thus sum losses of all exits
            
            # feature distillation
            #! take lots of GPU memory dut to huge size of hidden states
            # feats = final_output.hidden_states # n_exit x (bs * action_seq_len, lang_len, d)
            # last_feat = feats[-1].unsqueeze(0) # (1, bs * action_seq_len, lang_len, d)
            # prev_feats = torch.stack(feats[:-1], dim=0) # (n_exit - 1, bs * action_seq_len, lang_len, d)
            
            # last_feat = torch.max(last_feat, dim=-2)[0]
            # prev_feats = torch.max(prev_feats, dim=-2)[0] # (n_exit - 1, bs * action_seq_len, d)
            
            # last_feat = proj_feat_list[-1].unsqueeze(0)
            # prev_feats = torch.stack(proj_feat_list[:-1], dim=0)
            
            # sim = get_similarity(last_feat, prev_feats, detach_f1=True) # (n_exit - 1, bs * action_seq_len)
            # sim = sim.mean(dim=(1,2)) # (n_exit - 1,)
            # loss_distill = - sim.mean()
            
            # loss_distill = nn.functional.mse_loss(prev_feats, last_feat.detach(), reduction='none').mean(dim=(1,2))
            
            if args.feat_distill_coef > 0:
                loss_calvin += loss_distill * args.feat_distill_coef
            else:
                 loss_distill = 0
             
            divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

            #### BACKWARD PASS ####
            loss = (
                divided_loss_calvin * args.loss_multiplier_calvin
            )
            
        #### LOG #####
        if args.use_extra_exit:
            loss_calvin_bin_list = loss_calvin_bin[:-1].mean(dim=tuple(range(1, dim)))
            loss_calvin_num_list = loss_calvin_num[:-1].mean(dim=tuple(range(1, dim)))
            extra_exit_loss_bin = loss_calvin_bin[-1].mean()
            extra_exit_loss_num = loss_calvin_num[-1].mean()
        else:
            loss_calvin_bin_list = loss_calvin_bin.mean(dim=tuple(range(1, dim)))
            loss_calvin_num_list = loss_calvin_num.mean(dim=tuple(range(1, dim)))
            extra_exit_loss_bin = torch.tensor([.0])
            extra_exit_loss_num = torch.tensor([.0])
        
            
        mv_avg_loss.append(loss.item())
        
        if 'amp' in args.precision:
            # scaler.scale(loss).backward(retain_graph=True)
            scaler.scale(loss).backward()
        else:
            loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad and m.weight.grad is not None:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        # model.apply(mask_embedding)

        # unscale grad. Thus clip with original threshold
        if 'amp' in args.precision:
            scaler.unscale_(exit_optimizer)
            if not only_train_head:
                scaler.unscale_(llm_optimizer)
                
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            if 'amp' in args.precision:
                scaler.step(exit_optimizer)
                if not only_train_head:
                    scaler.step(llm_optimizer)
                scaler.update()
            else:
                exit_optimizer.step()
                if not only_train_head:
                    exit_optimizer.step()
                
            exit_lr_scheduler.step()
            exit_optimizer.zero_grad()
            if not only_train_head:
                llm_lr_scheduler.step()
                llm_optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                # wandb.log(
                #     {
                #         "data_time": data_time_m.avg,
                #         "step_time": step_time_m.avg,
                #         "calvin_samples_per_second": calvin_samples_per_second,
                #         "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                #     },
                #     commit=False,
                # )
                step_time_m.reset()
                data_time_m.reset()
                
                log_dict = {
                        "lr": llm_optimizer.param_groups[0]["lr"],
                        "exit_lr": exit_optimizer.param_groups[0]["lr"],
                        "loss_calvin": divided_loss_calvin.item(),
                        "loss_calvin_bin": loss_calvin_bin.mean().item(),
                        "loss_calvin_num": loss_calvin_num.mean().item(),
                        **{f"loss_calvin_bin_{i}": x.item() for i, x in enumerate(loss_calvin_bin_list)},
                        **{f"loss_calvin_num_{i}": x.item() for i, x in enumerate(loss_calvin_num_list)},
                        # **{f"feat_sim_{i}": x.item() for i, x in enumerate(sim)},
                        "extra_exit_loss_bin": extra_exit_loss_bin.item(),
                        "extra_exit_loss_num": extra_exit_loss_num.item(),
                        "mse": loss_mse.item(),
                        "mle": loss_mle.item(),
                        "mean_std": std.mean().item(),
                        "global_step": global_step,
                        "scale_factor": scaler.get_scale(),
                    }
                
                if args.feat_distill_coef > 0:
                    log_dict['loss_distill'] = loss_distill.mean().item()

                wandb.log(
                    log_dict,
                    commit=True,
                )


        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f} (mle) {loss_mle.item():.3f} (mse){loss_mse.item():.3f} (bce){loss_calvin_bin.mean().item():.3f} (std) {std.mean().item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "mle": loss_mle.item(), "mse": loss_mse.item(), "Lbin": loss_calvin_bin.mean().item(), "std": std.mean().item()})

        if args.save_every_iter != -1 and global_step % args.save_every_iter == 0 and global_step > 0:
            if args.rank == 0:
                save_ckpt(args, model, llm_optimizer, llm_lr_scheduler, epoch, global_step)


def train_one_epoch_calvin_cotrain(
    args,
    model,
    epoch,
    calvin_loader,
    coco_loader,
    vqa_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    # setup loaders
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    t = tqdm(
        enumerate(zip(coco_loader, vqa_loader, calvin_loader)),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")

    mv_avg_loss = []
    mv_avg_loss_coco = []
    mv_avg_loss_vqa = []
    for num_steps, (batch_coco, batch_vqa, batch_calvin) in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        #### COCO FORWARD PASS ####
        images = batch_coco[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
        input_ids = batch_coco[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_coco[1][1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        def calculate_vl_cross_entropy(logits, labels, mask=None):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if mask is None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(
                        -1, logits.shape[-1]
                    ),
                    shift_labels.view(-1),
                )
            else:
                # TODO: mask is with the same shape of labels, 
                # 1 represents valid, 0 for non-valid, only calculate loss for valid tokens
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                shift_logits.view(
                        -1, logits.shape[-1]
                    ),
                shift_labels.view(-1),
                )
                # mask the loss
                mask = mask[..., 1:].contiguous()
                loss = loss * mask.reshape(-1)
                # mean
                loss = loss.mean()
            return loss

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                mode = 'vision_lang'
            )
        
        logits = output.logits
        loss_coco = calculate_vl_cross_entropy(logits, labels)
        mv_avg_loss_coco.append(loss_coco.item())
        divided_loss_coco = loss_coco * args.vl_task_weights
        divided_loss_coco = divided_loss_coco / args.gradient_accumulation_steps
        
        (divided_loss_coco * args.loss_multiplier_calvin).backward()

        #### VQA FORWARD PASS ####
        images = batch_vqa[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
        input_ids = batch_vqa[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_vqa[1][1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )
        ques_mask = batch_vqa[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids.to(device_id),
                attention_mask=attention_mask.to(device_id),
                # labels=labels,
                mode = 'vision_lang'
            )
        
        logits = output.logits
        loss_vqa = calculate_vl_cross_entropy(logits, labels, ques_mask)
        mv_avg_loss_vqa.append(loss_vqa.item())
        divided_loss_vqa = loss_vqa * 0.5
        divided_loss_vqa = divided_loss_vqa / args.gradient_accumulation_steps
        (divided_loss_vqa * args.loss_multiplier_calvin).backward()
        
        #### CALVIN FORWARD PASS ####
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
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        
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

        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=gripper,
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None
            )

        # compute loss
        num_actions, bin_actions = output.logits[0], output.logits[1]

        def discretize_actions(pose_action):
            action_min = -1.001
            action_max = 1.001
            action_len = (action_max - action_min) / args.act_disc
            pose_action = (pose_action - action_min) / action_len
            pose_action = torch.floor(pose_action).long()
            return pose_action
        
        if args.act_disc != -1:
            # assert labels[0].max() < 1.0, f"{labels[0].max()} >= 1.0"
            # assert labels[0].min() > -1.0, f"{labels[0].min()} <= -1.0"
            labels[0] = discretize_actions(labels[0])
            assert labels[0].max() < args.act_disc, f"{labels[0].max()} >= {args.act_disc}"
            assert labels[0].min() >= 0, f"{labels[0].min()} < 0"
        # reshape for loss calculation
        if args.multi_step_action != 1:
            bs, seq_len = num_actions.shape[:2]
            num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)

        loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        if args.act_disc == -1:
            loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
            if args.real_data:
                loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
            else:
                loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
        else:
            bs, seq_len = num_actions.shape[:2]
            num_actions = num_actions.view(bs, seq_len, -1, args.act_disc).permute(0, 3, 1, 2)
            labels[0] = labels[0].view(bs, seq_len, -1)
            # print('-'*100)
            # print(num_actions, labels[0])
            loss_calvin_num = torch.nn.functional.cross_entropy(num_actions, labels[0])
            if args.real_data:
                loss_calvin = loss_calvin_num + loss_calvin_bin * 0.2
            else:
                loss_calvin = loss_calvin_num + loss_calvin_bin * 0.1
        

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_calvin * args.loss_multiplier_calvin
        )
        mv_avg_loss.append(loss.item())
        loss.backward()
        
        # clip gradient norm
        if args.fsdp:
            """
            The way we clip gradients with FSDP is different than the non-FSDP case,
            because during FSDP, gradient norms are computed over certain submodules,
            rather than the entire model.
            At least for OPT-125M, this didn't seem to make a difference in performance.
            """
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0 and args.report_to_wandb:
                coco_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_vl
                    * args.world_size
                    / step_time_m.val
                )
                coco_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_vl
                    / step_time_m.val
                )
                vqa_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_vl
                    * args.world_size
                    / step_time_m.val
                )
                vqa_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_vl
                    / step_time_m.val
                )
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "coco_samples_per_second": coco_samples_per_second,
                        "coco_samples_per_second_per_gpu": coco_samples_per_second_per_gpu,
                        "vqa_samples_per_second": vqa_samples_per_second,
                        "vqa_samples_per_second_per_gpu": vqa_samples_per_second_per_gpu,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_coco": loss_coco.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )
                wandb.log(
                    {"loss_vqa": loss_vqa.item(), "global_step": global_step},
                    commit=False,
                )

                wandb.log(
                    {
                        "loss_calvin": divided_loss_calvin.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete.  Loss coco: {loss_coco.item():.3f} // Loss vqa: {loss_vqa.item():.3f} // Loss CALVIN: (all){loss_calvin.item():.3f} (mse){loss_calvin_num.item():.3f} (bce){loss_calvin_bin.item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg calvin loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "avg coco loss": sum(mv_avg_loss_coco[-avg_horizon:]) / avg_horizon, "avg vqa loss": sum(mv_avg_loss_vqa[-avg_horizon:]) / avg_horizon,
                        "loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item()})


def train_one_epoch_calvin_two_way(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        # put images and labels on device
        images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        # images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(1).unsqueeze(1))
        vision_x = torch.cat([images, gripper], dim=0)
        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(2, images.shape[1], 1)

        # input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)

        # do the same to the attention mask 
        attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(2, images.shape[1], 1)
        state_tensor = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True).repeat(2, 1, 1).unsqueeze(2).unsqueeze(2)
        # import pdb; pdb.set_trace()
        # merge the batch and the sequence dimension
        # images = images.flatten(0, 1)
        # gripper = gripper.flatten(0, 1)
        images = images.detach().cpu()
        gripper = gripper.detach().cpu()
        vision_x = vision_x.flatten(0, 1)
        state_tensor = state_tensor.flatten(0, 1)
        input_ids = input_ids.flatten(0, 1)
        attention_mask = attention_mask.flatten(0, 1)

        # attention_mask = batch_calvin[1][1].to(device_id, dtype=cast_dtype, non_blocking=True)
        # attention_mask = None

        labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        # [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        if args.use_hist:
            labels = labels[:, [-1]]  # only calculate last step action
        labels = [labels[..., :6], (labels[..., 6:] + 1) // 2]
        # labels = [labels[..., :6], labels[..., 6:]]

        with autocast():
            output = model(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=None,
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None
            )

        # compute loss
        num_actions, bin_actions = output.logits
        loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
        loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        # loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
        loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_calvin * args.loss_multiplier_calvin
        )
        mv_avg_loss.append(loss.item())
        loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_calvin": divided_loss_calvin.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )


        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f} (mse){loss_calvin_num.item():.3f} (bce){loss_calvin_bin.item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item()})


def train_one_epoch(
    args,
    model,
    epoch,
    laion_loader,
    mmc4_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    num_batches_per_epoch_laion = laion_loader.num_batches
    num_batches_per_epoch_mmc4 = mmc4_loader.num_batches

    assert (
        num_batches_per_epoch_laion == num_batches_per_epoch_mmc4
    ), "Number of batches in laion and mmc4 datasets must be the same"
    num_batches_per_epoch = num_batches_per_epoch_mmc4
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    for num_steps, (batch_laion, batch_mmc4) in tqdm(
        enumerate(zip(laion_loader, mmc4_loader)),
        # disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)

        global_step = num_steps + epoch * num_batches_per_epoch

        #### LAION FORWARD PASS ####
        images = (
            batch_laion[0]
            .to(device_id, dtype=cast_dtype, non_blocking=True)
            .unsqueeze(1)
            .unsqueeze(1)
        )

        input_ids = batch_laion[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_laion[1][1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == media_token_id] = -100
        labels.to(device_id)

        with autocast():
            loss_laion = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
        divided_loss_laion = loss_laion / args.gradient_accumulation_steps

        #### C4 FORWARD PASS ####
        images = (
            batch_mmc4[0]
            .to(device_id, dtype=cast_dtype, non_blocking=True)
            .unsqueeze(2)
        )
        input_ids = torch.stack([x[0] for x in batch_mmc4[1]]).squeeze(1)
        attention_mask = torch.stack([x[1] for x in batch_mmc4[1]]).squeeze(1)

        # NOTE: irena: expected shape of clip_text_input_ids / attention_mask is (N, I, max_seq_len)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100

        for i in range(labels.shape[0]):
            # remove loss for any token before the first <image> token
            label_idx = 0
            while (
                label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id
            ):
                labels[i][label_idx] = -100
                label_idx += 1

            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while (
                    token_idx < labels.shape[1]
                    and labels[i][token_idx] != media_token_id
                ):
                    labels[i][token_idx] = -100
                    token_idx += 1

        labels[labels == media_token_id] = -100
        labels.to(device_id)

        with autocast():
            loss_mmc4 = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

            # if loss is nan, skip this batch
            if torch.isnan(loss_mmc4):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad()
                continue

        divided_loss_mmc4 = loss_mmc4 / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_laion * args.loss_multiplier_laion
            + divided_loss_mmc4 * args.loss_multiplier_mmc4
        )
        loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                laion_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    * args.world_size
                    / step_time_m.val
                )
                laion_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    / step_time_m.val
                )

                c4_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_mmc4
                    * args.world_size
                    / step_time_m.val
                )
                c4_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_mmc4
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "laion_samples_per_second": laion_samples_per_second,
                        "laion_samples_per_second_per_gpu": laion_samples_per_second_per_gpu,
                        "c4_samples_per_second": c4_samples_per_second,
                        "c4_samples_per_second_per_gpu": c4_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_laion": divided_loss_laion.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )
                wandb.log(
                    {"loss_mmc4": divided_loss_mmc4.item(), "global_step": global_step},
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0):
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss LAION: {loss_laion.item():.3f} // Loss MMC4: {loss_mmc4.item():.3f}"
            )


def train_value_net_one_epoch_calvin_multi_exit(
    args,
    model,
    value_net,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    device_num = int(torch.distributed.get_world_size())
    scaler = GradScaler(enabled='amp' in args.precision, growth_interval=int(4000/device_num))

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.eval()
    value_net.train()
    
    

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
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
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        
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

        with autocast():
            # get loss for each layer as target label
            with torch.no_grad():
                if args.head_type == 'deterministic':
                    final_output, exit_outputs = model(
                        vision_x=images,
                        lang_x=input_ids,
                        attention_mask=attention_mask,
                        # labels=labels,  # loss计算放在外面
                        vision_gripper=gripper,
                        state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                        with_gripper_logits=True,
                    )
                    
                    # get joint outputs
                    all_outputs = exit_outputs + [final_output.logits]
                    
                    num_action_list, gripper_logit_list = [], []
                    for output in all_outputs:
                        num_actions, bin_gripper = output[0], output[1]
                        bin_actions, bin_logits = bin_gripper
                        if args.multi_step_action != 1:
                            bs, seq_len = num_actions.shape[:2]
                            num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                            # bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                            bin_logits = bin_logits.reshape(bs, seq_len, args.multi_step_action, -1)
                        num_action_list.append(num_actions)
                        gripper_logit_list.append(bin_logits)

                    # get action loss per head type
                    num_actions = torch.stack(num_action_list, dim=0)
                    loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0][None], reduction='none').mean(-1)
                    # print(f'{loss_calvin_num.shape=}')
                    
                    loss_mse = loss_calvin_num.mean()
                    loss_mle = torch.tensor([.0])
                    std = torch.tensor([.0])
                elif args.head_type == 'gaussian':
                    raise NotImplementedError("Please fix the bug in gaussian policy in single exit before running multi-exit gaussian policy!")

                # loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
                bin_logits = torch.stack(gripper_logit_list, dim=0)
                bin_targets = torch.stack([labels[1]] * len(all_outputs), dim=0)
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy_with_logits(bin_logits, bin_targets, reduction='none').mean(-1)
                # print(f'{loss_calvin_num.shape=}')
                
                # loss for each layer and each sample
                if args.head_type == 'deterministic':
                    if args.real_data:
                        loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
                    else:
                        loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
                elif args.head_type == 'gaussian':
                    loss_calvin = loss_calvin_num + loss_calvin_bin * args.bin_coef
                loss_calvin *= args.loss_multiplier_calvin
                # weights = get_exit_weights('uniform', len(all_outputs), device=loss_calvin.device)
                # weights = weights * weights.shape[0] 
                # loss_calvin *= weights
                
                #### MASK GRADIENTS FOR EMBEDDINGS ####
                # Note (anas): Do not apply weight decay to embeddings as it will break this function.
                # def mask_embedding(m):
                #     if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad and m.weight.grad is not None:
                #         zero_mask = torch.zeros_like(m.weight.grad)
                #         zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                #         zero_mask[endofchunk_token_id] = torch.ones_like(
                #             zero_mask[endofchunk_token_id]
                #         )
                #         m.weight.grad = m.weight.grad * zero_mask
                # model.apply(mask_embedding)

            # train value net
            features = torch.stack(final_output.hidden_states, dim=0) 
            values = value_net(features) 
            values = values.reshape(len(all_outputs), -1, values.shape[1])
            target_values = loss_calvin.detach()
            loss_value = torch.nn.functional.huber_loss(values, target_values, reduction='none')
            dim = loss_value.dim()
            loss_value_layer = loss_value.mean(dim=tuple(range(1, dim))).detach() # only for log
            loss_value = loss_value.mean()
            mv_avg_loss.append(loss_value.item())
            
        # backward
        if 'amp' in args.precision:
            scaler.scale(loss_value).backward()
        else:
            loss_value.backward()
        
        # unscale grad. Thus clip with original threshold
        if 'amp' in args.precision:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
 
        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            if 'amp' in args.precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            lr_scheduler.step()
            optimizer.zero_grad()

            if args.rank == 0 and args.report_to_wandb:
                wandb.log(
                    {
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss_value": loss_value.item(),
                        **{f"loss_value_{i}": x.item() for i, x in enumerate(loss_value_layer)},
                        "global_step": global_step,
                        "scale_factor": scaler.get_scale(),
                    },
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Value Loss: ({loss_value.item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon})

        if args.save_every_iter != -1 and global_step % args.save_every_iter == 0 and global_step > 0:
            if args.rank == 0:
                save_value_net_ckpt(args, value_net, optimizer, lr_scheduler, epoch, global_step, args.roboflamingo_checkpoint)



def train_value_net_one_epoch_calvin_dynamic_exit(
    args,
    model,
    value_net,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    device_num = int(torch.distributed.get_world_size())
    scaler = GradScaler(enabled='amp' in args.precision, growth_interval=int(4000/device_num))

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.eval()
    value_net.train()
    
    target_value_list = []

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
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
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        
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

        with autocast():
            # get loss for each layer as target label
            with torch.no_grad():
                if args.head_type == 'deterministic':
                    o = model(
                        vision_x=images,
                        lang_x=input_ids,
                        attention_mask=attention_mask,
                        # labels=labels,  # loss计算放在外面
                        vision_gripper=gripper,
                        state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                        with_gripper_logits=True,
                        return_in_feat=True,
                    )
                    # only need extra exit loss
                    extra_exit_output = o[2]
                    features, exit_idx = o[3], o[4]
                    num_actions, bin_gripper = extra_exit_output[0], extra_exit_output[1]
                    bin_actions, bin_logits = bin_gripper
                    if args.multi_step_action != 1:
                        bs, seq_len = num_actions.shape[:2]
                        num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                        # bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                        bin_logits = bin_logits.reshape(bs, seq_len, args.multi_step_action, -1)
                    loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0], reduction='none').mean(-1)
                    
                    loss_mse = loss_calvin_num.mean()
                    loss_mle = torch.tensor([.0])
                    std = torch.tensor([.0])
                elif args.head_type == 'gaussian':
                    raise NotImplementedError("Please fix the bug in gaussian policy in single exit before running multi-exit gaussian policy!")

                # loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
                bin_targets = labels[1]
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy_with_logits(bin_logits, bin_targets, reduction='none').mean(-1)
                # print(f'{loss_calvin_num.shape=}')
                
                # loss for each layer and each sample
                if args.head_type == 'deterministic':
                    if args.real_data:
                        loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
                    else:
                        loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
                elif args.head_type == 'gaussian':
                    loss_calvin = loss_calvin_num + loss_calvin_bin * args.bin_coef
                loss_calvin *= args.loss_multiplier_calvin
                # weights = get_exit_weights('uniform', len(all_outputs), device=loss_calvin.device)
                # weights = weights * weights.shape[0] 
                # loss_calvin *= weights
                
                
                #### MASK GRADIENTS FOR EMBEDDINGS ####
                # Note (anas): Do not apply weight decay to embeddings as it will break this function.
                # def mask_embedding(m):
                #     if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad and m.weight.grad is not None:
                #         zero_mask = torch.zeros_like(m.weight.grad)
                #         zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                #         zero_mask[endofchunk_token_id] = torch.ones_like(
                #             zero_mask[endofchunk_token_id]
                #         )
                #         m.weight.grad = m.weight.grad * zero_mask
                # model.apply(mask_embedding)

            if epoch == -1:
                target_value_list.append(loss_calvin.detach())
            else:
                # train value net
                target_values = loss_calvin.detach()
                if args.discrete:
                    logits = value_net(features, exit_idx=exit_idx)
                    # logits = value_net(features, exit_idx=exit_idx)
                    bin_label = value_to_bin_index(target_values, value_net.module.boundaries)
                    
                    # logits = logits[:, 4:].flatten(0, 1)
                    # bin_label = bin_label[:, 4:].flatten(0, 1)
                    
                    # loss_value = torch.nn.functional.cross_entropy(logits, bin_label, reduction='none')
                    
                    # loss_value = cumulative_link_loss(logits, bin_label)
                    
                    loss_value = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), bin_label.float(), reduction='none')
                    
                    # test1
                    # t_label = torch.arange(args.window_size, device=logits.device).unsqueeze(0).expand(args.batch_size_calvin, -1).flatten(0, 1)
                    # loss_value = torch.nn.functional.cross_entropy(logits, t_label, reduction='none')
                    
                    # test2
                    # loss_value = torch.nn.functional.binary_cross_entropy_with_logits(logits, bin_targets, reduction='none')
                    
                    # test3
                    # loss_value = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), torch.zeros_like(bin_label.float()), reduction='none')
                    
                    loss_value = loss_value.mean()
                else:
                    values = value_net(features, exit_idx=exit_idx).squeeze(-1)
                    target_values *= 1000
                    loss_value = torch.nn.functional.huber_loss(values, target_values, reduction='none')
                    loss_value = loss_value.mean()
                mv_avg_loss.append(loss_value.item())
        
        if epoch >= 0: 
            # backward
            if 'amp' in args.precision:
                scaler.scale(loss_value).backward()
            else:
                loss_value.backward()
            
            # unscale grad. Thus clip with original threshold
            if 'amp' in args.precision:
                scaler.unscale_(optimizer)
            if args.discrete:
                grad_norm = torch.nn.utils.clip_grad_norm_(value_net.parameters(), 10.0)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
    
            # step optimizer and log
            if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
                num_steps == num_batches_per_epoch - 1
            ):
                if 'amp' in args.precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.rank == 0 and args.report_to_wandb:
                    wandb.log(
                        {
                            "lr": optimizer.param_groups[0]["lr"],
                            "loss_value": loss_value.item(),
                            "global_step": global_step,
                            "scale_factor": scaler.get_scale(),
                            "grad_norm": grad_norm.item(),
                        },
                        commit=True,
                    )

            # Log loss to console
            if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
                print(
                    f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Value Loss: ({loss_value.item():.3f}"
                )
            avg_horizon = min(100, len(mv_avg_loss))
            t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon})

            if args.save_every_iter != -1 and global_step % args.save_every_iter == 0 and global_step > 0:
                if args.rank == 0:
                    save_value_net_ckpt(args, value_net, optimizer, lr_scheduler, epoch, global_step, args.roboflamingo_checkpoint)
    
    if epoch == -1:
        device_id = torch.distributed.get_rank()
        num_devices = torch.distributed.get_world_size()
        target_value_list = torch.cat(target_value_list, dim=0) # (bs , action_seq_len)
        target_value_gathered = [torch.zeros_like(target_value_list) for _ in range(num_devices)]
        torch.distributed.all_gather(target_value_gathered, target_value_list) 
        target_value_gathered = torch.cat(target_value_gathered, dim=0) # (bs , action_seq_len)
        if args.rank == 0:
            print(f'{target_value_list.shape[0]} samples on device 0, {target_value_gathered.shape[0]} samples on all devices.')
        
        # Calculate quantiles to determine bin edges
        boundaries = get_bin_boundaries(target_value_gathered.flatten(0, 1), args.num_bin)
        value_net.module.set_bin_boundaries(boundaries)
        if args.rank == 0:
            print(f'min value: {target_value_gathered.min():.5f}')                                
            print(f'mean value: {target_value_gathered.mean():.5f}')                                
            print(f'median value: {target_value_gathered.median():.5f}')                                
            print(f'max value: {target_value_gathered.max():.5f}')
            
            import matplotlib.pyplot as plt
            for t in range(target_value_gathered.shape[1]):
                data = target_value_gathered[:, t].cpu().numpy()
                plt.hist(data, bins=50, range=(0, 0.07))
                plt.savefig(f'vis/value_dist_{os.path.basename(args.calvin_dataset)}_{t=}_bin50.jpg')
                plt.close()
                # plot the distribution                                


def train_value_net_one_epoch_calvin_dynamic_exit_debug(
    args,
    model,
    value_net,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    device_num = int(torch.distributed.get_world_size())
    scaler = GradScaler(enabled='amp' in args.precision, growth_interval=int(4000/device_num))

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()
    value_net.train()
    
    target_value_list = []

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
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
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        
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

        with autocast():
            with torch.no_grad():
                # get loss for each layer as target label
                if args.head_type == 'deterministic':
                    o = model(
                        vision_x=images,
                        lang_x=input_ids,
                        attention_mask=attention_mask,
                        # labels=labels,  # loss计算放在外面
                        vision_gripper=gripper,
                        state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
                        with_gripper_logits=True,
                        return_in_feat=True,
                        # return_aggregate_feature=True,
                        only_extra_exit=True,
                    )
                    # only need extra exit loss
                    extra_exit_output = o[2]
                    in_features, exit_idx = o[3], o[4]
                    # num_actions, bin_gripper, agg_features = extra_exit_output[0], extra_exit_output[1], extra_exit_output[2]
                    num_actions, bin_gripper = extra_exit_output[0], extra_exit_output[1]
                    bin_actions, bin_logits = bin_gripper
                    if args.multi_step_action != 1:
                        bs, seq_len = num_actions.shape[:2]
                        num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                        # bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                        bin_logits = bin_logits.reshape(bs, seq_len, args.multi_step_action, -1)
                    loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0], reduction='none').mean(-1)
                    
                    loss_mse = loss_calvin_num.mean()
                    loss_mle = torch.tensor([.0])
                    std = torch.tensor([.0])
                elif args.head_type == 'gaussian':
                    raise NotImplementedError("Please fix the bug in gaussian policy in single exit before running multi-exit gaussian policy!")

                # loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
                bin_targets = labels[1]
                loss_calvin_bin = torch.nn.functional.binary_cross_entropy_with_logits(bin_logits, bin_targets, reduction='none').mean(-1)
                # print(f'{loss_calvin_num.shape=}')
                
                # loss for each layer and each sample
                if args.head_type == 'deterministic':
                    if args.real_data:
                        loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
                    else:
                        loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
                elif args.head_type == 'gaussian':
                    loss_calvin = loss_calvin_num + loss_calvin_bin * args.bin_coef
                loss_calvin *= args.loss_multiplier_calvin
                # weights = get_exit_weights('uniform', len(all_outputs), device=loss_calvin.device)
                # weights = weights * weights.shape[0] 
                # loss_calvin *= weights
                
                
                #### MASK GRADIENTS FOR EMBEDDINGS ####
                # Note (anas): Do not apply weight decay to embeddings as it will break this function.
                # def mask_embedding(m):
                #     if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad and m.weight.grad is not None:
                #         zero_mask = torch.zeros_like(m.weight.grad)
                #         zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                #         zero_mask[endofchunk_token_id] = torch.ones_like(
                #             zero_mask[endofchunk_token_id]
                #         )
                #         m.weight.grad = m.weight.grad * zero_mask
                # model.apply(mask_embedding)

            if epoch == -1:
                target_value_list.append(loss_calvin.detach())
            else:
                # train value net
                target_values = loss_calvin.detach()
                if args.discrete:
                    # prev_agg_features = agg_features[:-1].detach()
                    # agg_features = agg_features[1:].detach()
                    # target_values = target_values[1:]
                    # logits = value_net(agg_features, prev_agg_features, exit_idx=exit_idx)
                    
                    logits = value_net(in_features, exit_idx=exit_idx)
                    bin_label = value_to_bin_index(target_values, value_net.module.boundaries)
                    
                    # logits = logits[:, :, 4:]
                    # bin_label = bin_label[:, :, 4:]
                    logits = logits[:, 4:].flatten(0, 1)
                    bin_label = bin_label[:, 4:].flatten(0, 1)
                    
                    loss_value = torch.nn.functional.cross_entropy(logits, bin_label, reduction='none')
                    
                    # loss_value = cumulative_link_loss(logits, bin_label)
                    
                    # loss_value = torch.nn.functional.binary_cross_entropy_with_logits(logits, bin_label.float().unsqueeze(-1), reduction='none')
                    
                    # test1
                    # t_label = torch.arange(args.window_size, device=logits.device).unsqueeze(0).expand(args.batch_size_calvin, -1).flatten(0, 1)
                    # loss_value = torch.nn.functional.cross_entropy(logits, t_label, reduction='none')
                    
                    # test2
                    # loss_value = torch.nn.functional.binary_cross_entropy_with_logits(logits, bin_targets, reduction='none')
                    
                    # test3
                    # loss_value = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), torch.zeros_like(bin_label.float()), reduction='none')
                    
                    loss_value = loss_value.mean()
                else:
                    values = value_net(in_features, exit_idx=exit_idx).squeeze(-1)
                    target_values *= 1000
                    loss_value = torch.nn.functional.huber_loss(values, target_values, reduction='none')
                    loss_value = loss_value.mean()
                mv_avg_loss.append(loss_value.item())
                
                # loss = loss_value + loss_calvin.mean()
                loss = loss_value
        
        if epoch >= 0: 
            # backward
            if 'amp' in args.precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # unscale grad. Thus clip with original threshold
            if 'amp' in args.precision:
                scaler.unscale_(optimizer)
            value_grad_norm = torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
            model_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            bin_rate = torch.sum(bin_label) / torch.numel(bin_label)
    
            # step optimizer and log
            if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
                num_steps == num_batches_per_epoch - 1
            ):
                if 'amp' in args.precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.rank == 0 and args.report_to_wandb:
                    wandb.log(
                        {
                            "lr": optimizer.param_groups[0]["lr"],
                            "loss_value": loss_value.item(),
                            "loss_calvin": loss_calvin.mean().item(),
                            "global_step": global_step,
                            "scale_factor": scaler.get_scale(),
                            "model_grad_norm": model_grad_norm.item(),
                            "value_grad_norm": value_grad_norm.item(),
                            "bin_rate": bin_rate,
                        },
                        commit=True,
                    )

            # Log loss to console
            if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
                print(
                    f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Value Loss: ({loss_value.item():.3f}"
                )
            avg_horizon = min(100, len(mv_avg_loss))
            t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon})

            if args.save_every_iter != -1 and global_step % args.save_every_iter == 0 and global_step > 0:
                if args.rank == 0:
                    save_value_net_ckpt(args, value_net, optimizer, lr_scheduler, epoch, global_step, args.roboflamingo_checkpoint)
    
    if epoch == -1:
        device_id = torch.distributed.get_rank()
        num_devices = torch.distributed.get_world_size()
        target_value_list = torch.cat(target_value_list, dim=0) # (bs , action_seq_len)
        target_value_gathered = [torch.zeros_like(target_value_list) for _ in range(num_devices)]
        torch.distributed.all_gather(target_value_gathered, target_value_list) 
        target_value_gathered = torch.cat(target_value_gathered, dim=0) # (bs , action_seq_len)
        if args.rank == 0:
            # print(f'{target_value_list.shape[1]} samples on device 0, {target_value_gathered.shape[1]} samples on all devices.')
            print(f'{target_value_list.shape[0]} samples on device 0, {target_value_gathered.shape[0]} samples on all devices.')
        
        # Calculate quantiles to determine bin edges
        # boundaries = get_bin_boundaries(target_value_gathered[:, :, 4:].flatten(0, 2), args.num_bin)
        boundaries = get_bin_boundaries(target_value_gathered[:, 4:].flatten(0, 1), args.num_bin)
        value_net.module.set_bin_boundaries(boundaries)
        if args.rank == 0:
            print(f'min value: {target_value_gathered.min():.5f}')                                
            print(f'mean value: {target_value_gathered.mean():.5f}')                                
            print(f'median value: {target_value_gathered.median():.5f}')                                
            print(f'max value: {target_value_gathered.max():.5f}')
            print(f'boundary: ', boundaries)
            
            # import matplotlib.pyplot as plt
            # for t in range(target_value_gathered.shape[2]):
            #     data = target_value_gathered[:, :, t].cpu().numpy()
            #     plt.hist(data, bins=50, range=(0, 0.07))
            #     plt.savefig(f'vis/value_dist_{os.path.basename(args.calvin_dataset)}_{t=}_bin50.jpg')
            #     plt.close()
                # plot the distribution     


def cumulative_link_loss(y_pred, y_true):
    # y_pred: predictions, a tensor of shape (batch_size, num_classes)
    # y_true: true labels, a tensor of shape (batch_size,)
    
    # Get the number of classes
    num_classes = y_pred.size(1)
    
    # Create a tensor with cumulative probabilities
    y_cum_prob = torch.cumsum(F.softmax(y_pred, dim=1), dim=1)
    
    # Create a tensor with binary labels for each binary classification problem
    y_binary = torch.zeros_like(y_pred)
    y_binary[torch.arange(y_true.size(0)), y_true] = 1
    y_binary = torch.cumsum(y_binary, dim=1)
    y_binary = y_binary[:, :-1]
    
    def logit(y):
        eps = 1e-6  # Small constant
        return torch.log((y + eps) / (1 - y + eps))
    y_cum_logit = logit(y_cum_prob[:, :-1])
    # Calculate the binary cross-entropy for each binary classification problem
    losses = F.binary_cross_entropy_with_logits(y_cum_logit, y_binary, reduction='none')
    
    # Return the average loss
    return losses.mean()


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad and 'normalizer' not in name:
            del state_dict[name]

    return state_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
