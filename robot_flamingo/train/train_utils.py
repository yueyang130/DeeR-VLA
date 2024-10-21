import time
from contextlib import suppress
import torch
from tqdm import tqdm
from robot_flamingo.utils import world_to_tcp_frame, tcp_to_world_frame
from torch.cuda.amp import GradScaler
import os

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
        "early_exit_layer": args.early_exit_layer,
        "multi_exit": args.multi_exit,
        "share_exit": args.share_exit,
        "exit_interval": args.exit_interval,
        "exit_dropout": args.exit_dropout,
        "lstm_dropout": args.lstm_dropout,
        "dropout_mode": args.dropout_mode,
        "mlp_layernorm": args.mlp_layernorm,
        "lstm_layernorm": args.lstm_layernorm,
        "mlp_num_hidden_layers": args.mlp_num_hidden_layers,
        "lstm_num_layers": args.lstm_num_layers,
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
        ckpt_name += 'intv={}_'.format(args.exit_interval)
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
        ckpt_name += f'bin_coef_{args.bin_coef}_'
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
    

def get_exit_weights(num, device):
    weight = torch.ones(num, dtype=torch.float32, device=device)
    return weight


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
            else:
                raise NotImplementedError(f'{args.head_type=}')

        # reshape for loss calculation
        if args.multi_step_action != 1:
            bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            bin_logits = bin_logits.reshape(bs, seq_len, args.multi_step_action, -1)
        
        with autocast():
            # loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
            loss_calvin_bin = torch.nn.functional.binary_cross_entropy_with_logits(bin_logits, labels[1])
        
            if args.head_type == 'deterministic':
                if args.real_data:
                    loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
                else:
                    loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01

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
                
                final_output, exit_outputs, extra_exit_output, extra_exit_output2 = o[0], o[1], o[2], o[3]
                all_outputs = exit_outputs + [final_output.logits, extra_exit_output, extra_exit_output2]
                
                num_action_list, gripper_logit_list = [], []
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

                # get action loss per head type
                num_actions = torch.stack(num_action_list, dim=0)
                loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0][None], reduction='none').mean(-1)
                # print(f'{loss_calvin_num.shape=}')
                
                loss_mse = loss_calvin_num.mean()
                loss_mle = torch.tensor([.0])
                std = torch.tensor([.0])

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

            # get mean for every exit
            dim = loss_calvin.dim()
            loss_calvin = loss_calvin.mean(dim=tuple(range(1, dim)))
            weights = get_exit_weights(len(all_outputs), device=loss_calvin.device)
            loss_calvin *= weights
            loss_calvin = loss_calvin.sum() # since weights are normalzied, thus sum losses of all exits
             
            divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

            #### BACKWARD PASS ####
            loss = (
                divided_loss_calvin * args.loss_multiplier_calvin
            )
            
        #### LOG #####
        loss_calvin_bin_list = loss_calvin_bin[:-2].mean(dim=tuple(range(1, dim)))
        loss_calvin_num_list = loss_calvin_num[:-2].mean(dim=tuple(range(1, dim)))
        extra_exit_loss_bin = loss_calvin_bin[-2].mean()
        extra_exit_loss_num = loss_calvin_num[-2].mean()
        extra_exit_loss2_bin = loss_calvin_bin[-1].mean()
        extra_exit_loss2_num = loss_calvin_num[-1].mean()
            
        mv_avg_loss.append(loss.item())
        
        if 'amp' in args.precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

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
