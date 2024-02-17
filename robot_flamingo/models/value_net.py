import torch
import torch.nn as nn
import torch.nn.functional as F
from robot_flamingo.models.action_head import MLPNohHead, lstm_decoder
import copy
from typing import Optional, Tuple
from tqdm.auto import tqdm
import math

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype

class ValueNet(nn.Module):
    def __init__(self, hidden_size, feat_size, dropout, with_layer_embed=False):
        super().__init__()
        self.value_net = MLPNohHead(hidden_size+feat_size, 1, dropout)
        self.with_layer_embed = with_layer_embed
        if with_layer_embed:
            self.layer_embed = None
            raise NotImplementedError
        
    def forward(self, hidden_state, feat):
        if self.with_layer_embed:
            raise NotImplementedError
        if hidden_state.dim > 2:
            raise NotImplementedError
        x = torch.concat([hidden_state, feat], dim=1)
        x = self.value_net(x)
        return x
    

class LSTMValueHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        window_size: int,
        dropout: float,
        history_len = None,
        out_features: int = 6,
        hidden_size: int = 1024,
        num_layers: int = 4,
        policy_rnn_dropout_p: float = 0.1,
        fusion_mode='',
        use_state=False,
        pooling='max',
        with_exit_embed=False,
        num_exits=1,
    ):
        super().__init__()
        
        self.fc_state = None
        self.use_state = use_state
        self.with_exit_embed = with_exit_embed
        if use_state:
            print('Using state in LSTM value net')
            state_in_dim = 7
            self.embed_arm_state = nn.Sequential(torch.nn.Linear(state_in_dim-1, in_features), nn.ReLU())
            self.embed_gripper_state = nn.Sequential(torch.nn.Embedding(2, in_features), nn.ReLU()) # one-hot gripper state
            self.embed_state = torch.nn.Linear(2*in_features, in_features)
        if with_exit_embed:
            self.embed_exit = torch.nn.Embedding(num_exits, in_features)
        
        if fusion_mode == 'two_way':
            in_features *= 2
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        self.rnn = lstm_decoder
        self.rnn = self.rnn(in_features, hidden_size, num_layers, policy_rnn_dropout_p)
        self.fusion_mode = fusion_mode
        
        self.head = MLPNohHead(hidden_size, 1, dropout)

        self.hidden_state = None
        self.hidden_size = hidden_size
        self.rnn_out = None

        if pooling == 'max':
            self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        
        if self.fusion_mode == 'two_way':
            if pooling == 'max':
                self.gripper_1d_max_pool = nn.AdaptiveMaxPool1d(1)
            else:
                self.gripper_1d_max_pool = nn.AdaptiveAvgPool1d(1)

    def clear_hidden_state(self) -> None:
        self.hidden_state = None
        
    def update_memory(self):
        self.history_memory.append(self.tmp_input_feature)
        self.hidden_state = self.tmp_hidden_state

    def forward(  # type: ignore
        self,
        input_feature: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        state_tensor=None,
    ):
        """
        The key difference between LSTMActionHead and LSTMValueHead lies in hidden states with inference.
        For LSTMActionHead, we update history_memory and hidden_state at every inference.
        For LSTMValueHead, we infer the values of exits from lowest and highest unitl early exit, and only use the hidden state of the early exit one.
        """
        
        # (num_layer, bs * action_seq_len, lang_len, d) --> (num_exit * bs * action_seq_len, lang_len, d)
        if input_feature.dim() == 4:
            input_feature = input_feature.reshape(-1, *input_feature.shape[2:]) 
        
        if self.with_exit_embed:
            raise NotImplementedError
        
        # reshape
        if input_feature.dim() == 3: # (num_exit * bs * action_seq_len, lang_len, d)
            if self.fusion_mode == 'two_way':
                input_feature = input_feature.reshape(-1, self.window_size, *input_feature.shape[1:]) 
                
                bs = int(input_feature.shape[0] // 2)
                
                rgb_feat = input_feature[:bs].view(bs*self.window_size, *input_feature.shape[2:])
                rgb_feat = self.global_1d_pool(rgb_feat.permute(0, 2, 1)).squeeze(-1)
                
                gripper_feat = input_feature[bs:].view(bs*self.window_size, *input_feature.shape[2:])
                gripper_feat = self.global_1d_pool(gripper_feat.permute(0, 2, 1)).squeeze(-1)
                
                input_feature = torch.cat([rgb_feat, gripper_feat], dim=-1)
            else:
                input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1) # (num_exit * bs * seq_len, d) maxpooling along lang_seq
        input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1]) # (num_exit * bs, seq_len, d)

        if state_tensor is not None and self.use_state:
            assert NotImplementedError("The Reshape operation likely needs to be rewritten.")
            arm_state = state_tensor[..., :6] # b,len,state_dim-1
            arm_state_embeddings = self.embed_arm_state(arm_state)
            arm_state_embeddings = arm_state_embeddings.view(-1, self.window_size, arm_state_embeddings.shape[-1]) # b,len,h
            gripper_state = ((state_tensor[..., -1]+1.0) / 2).long() # b,len,1
            gripper_state_embeddings = self.embed_gripper_state(gripper_state)
            gripper_state_embeddings = gripper_state_embeddings.view(-1, self.window_size, gripper_state_embeddings.shape[-1]) # b,len,h
            state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2) # b,len,2h
            state_embeddings = self.embed_state(state_embeddings) # b,len,h

            # input_feature = torch.cat([input_feature, state_embeddings], dim=-1)
            input_feature = input_feature + state_embeddings
        
        if not isinstance(self.rnn, nn.Sequential) and isinstance(self.rnn, nn.RNNBase):
            if input_feature.shape[1] == 1:  # the first frame of an action sequence
                # infernece
                self.tmp_input_feature = input_feature # used for inference when len <= window_size
                tmp_history_memory = self.history_memory + [input_feature] # used for inference when len > window_size
                if len(tmp_history_memory) <= self.history_len:
                    # print('cur hist_mem len: {}'.format(len(self.history_memory)))
                    x, h_n = self.rnn(input_feature, self.hidden_state)
                    self.tmp_hidden_state = h_n
                    x = x[:, -1].unsqueeze(1)
                    self.rnn_out = x.squeeze(1)
                else:
                    # the hidden state need to be refreshed based on the history window
                    # print('hist_mem exceeded, refresh hidden state')
                    cur_len = len(tmp_history_memory)
                    for _ in range(cur_len - self.history_len):
                        tmp_history_memory.pop(0)
                    assert len(tmp_history_memory) == self.history_len
                    hist_feature = torch.cat(tmp_history_memory, dim=1)
                    self.hidden_state = None
                    x, h_n = self.rnn(hist_feature, self.hidden_state)
                    x = x[:, -1].unsqueeze(1)
                    self.rnn_out = x.squeeze(1)
            else:
                # train
                # print('input feature lenght > 1', input_feature.shape)
                self.hidden_state = h_0
                x, h_n = self.rnn(input_feature, self.hidden_state) # (exits * bs, seq_len, d) --> (exits * bs, seq_len, d) LSTM go along action seqence
                self.hidden_state = h_n
                self.rnn_out = x.squeeze(1)
        else:
            raise NotImplementedError

        values = self.head(x) # (exits * bs, seq_len, 1)
        return values
    

class ExitController(torch.nn.Module):
    def __init__(self, value_net, num_exit, leq=True):
        super().__init__()
        self.value_net = value_net
        self.thresholds = None
        self.leq = leq
        self.num_exit = num_exit
        # for debug
        self.history_values = [[] for i in range(num_exit)]
        
    def set_threshold(self, args, model, dataloader, exit_ratio, values=None):  
        if values is None:  
            device_id = torch.distributed.get_rank()
            num_devices = torch.distributed.get_world_size()
            pred_value_list, target_value_list = generate_values(args, model, self.value_net, dataloader, device_id=device_id)

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
        T = torch.Tensor(n_stage).fill_(-1e8) if self.leq else torch.Tensor(n_stage).fill_(1e8)
        probs = exit_ratio ** torch.arange(1, self.num_exit+1) # n (including the last exit)
        # probs[0] = 0.0
        probs /= probs.sum()
        if args.rank==0: print('Expected early exit rate ', probs)

        for k in range(n_stage - 1): # not include the last exit
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
            T[n_stage - 1] = 1e8
        else:
            T[n_stage - 1] = -1e8
        
        # verify
        # count = [0]
        # filtered = torch.zeros(n_sample)
        # for k in range(n_stage):
        #     filtered += pred_value_gathered[k] < T[k]
        #     count.append((filtered>0).sum()-sum(count))
        # percent = [c / sum(count) for c in count[1:]]    

        self.thresholds = T
        if args.rank == 0:
            print(f'Mean value for each layer:')
            for i in range(n_stage):
                print(f'{i+1} : {pred_value_gathered[i].mean():.5f}, {pred_value_gathered[i].std():.5f}')
            print(f'Find threshold on {n_sample} samples:')
            for i in range(n_stage):
                print(f'{i+1} : {T[i]:.5f}')
        return pred_value_gathered
                

    @torch.no_grad()  
    def forward(self, x, i):
        assert self.thresholds is not None, "Please set thresholds before calling forward"
        
        value = self.value_net(x)
        
        # self.history_values[i].append(value)
        # if torch.distributed.get_rank() == 0:
        #     total = sum(len(h) for h in self.history_values)
        #     if total > 0 and total % 100 == 0:
        #         for layer, h in enumerate(self.history_values):
        #             print(f'{layer=}, count={len(h)}, mean value = {torch.tensor(h).mean():.5f}')
        
        if bool(value <= self.thresholds[i]) is self.leq: # both be true or both be false
            # Already find the dynamic exit. We need to update hidden state
            self.value_net.update_memory()
            return True 
        else:
            return False

@torch.no_grad()
def generate_values(
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
    t.set_description(f"generate values ")
    mv_avg_loss = []
    pred_value_list, target_value_list = [], []
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
        # print(f'{args.amp=}')
        if args.use_extra_exit:
            with torch.cuda.amp.autocast(enabled=args.amp), torch.no_grad():
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
                        only_extra_exit=True,
                    )
                    # only need extra exit loss
                    # extra_exit_output = o[2]
                    features = o[3]
                #     num_actions, bin_gripper = extra_exit_output[0], extra_exit_output[1]
                #     bin_actions, bin_logits = bin_gripper
                #     if args.multi_step_action != 1:
                #         bs, seq_len = num_actions.shape[:2]
                #         num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                #         # bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)
                #         bin_logits = bin_logits.reshape(bs, seq_len, args.multi_step_action, -1)
                #     loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0], reduction='none').mean(-1)
                    
                #     loss_mse = loss_calvin_num.mean()
                #     loss_mle = torch.tensor([.0])
                #     std = torch.tensor([.0])
                # elif args.head_type == 'gaussian':
                #     raise NotImplementedError("Please fix the bug in gaussian policy in single exit before running multi-exit gaussian policy!")

                # # loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
                # bin_targets = labels[1]
                # loss_calvin_bin = torch.nn.functional.binary_cross_entropy_with_logits(bin_logits, bin_targets, reduction='none').mean(-1)
                # # print(f'{loss_calvin_num.shape=}')
                
                # # loss for each layer and each sample
                # if args.head_type == 'deterministic':
                #     if args.real_data:
                #         loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
                #     else:
                #         loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
                # elif args.head_type == 'gaussian':
                #     loss_calvin = loss_calvin_num + loss_calvin_bin * args.bin_coef
                # loss_calvin *= args.loss_multiplier_calvin
                action_seq_len, exit_num, bs = features.shape[:3]  # (action_seq_len, n_exit, bs, action_seq_len, lang_len, d)
                features = features.flatten(2, 3).flatten(0, 1)
                values = value_net(features).squeeze(-1) # (action_seq_len * exits, bs * seq_len, lang_len, d) -> (action_seq_len * exits * bs, seq_len)
                values = values.reshape(action_seq_len, exit_num, bs, action_seq_len)
                value_list = [values[i, :, :, i:i+1]  for i in range(action_seq_len)] # action_seq_len * (exit_num, bs, 1)
                values = torch.cat(value_list, dim=2) #  (exit_num, bs, action_seq_len)
                
                if args.rank==0:
                    for i in range(action_seq_len):
                        print(f'Timestep {i+1} mean value {values[:, :, i].mean():.5f}')
                
                # remove few timesteps at beginning because they have statistically larger loss
                values = values[:, :, 4:] 
        
        else:
            with torch.cuda.amp.autocast(enabled=args.amp), torch.no_grad():
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

                features = torch.stack(final_output.hidden_states, dim=0) 
                values = value_net(features) # (exits, bs * seq_len, lang_len, 1) -> (exits * bs, seq_len, 1)
                values = values.reshape(len(all_outputs), -1, values.shape[1]) # (exits, bs, seq_len)
                target_values = loss_calvin.detach()
            
        # record
        pred_value_list.append(values)  
        # target_value_list.append(target_values)


    pred_value_list = torch.cat(pred_value_list, dim=1)
    # target_value_list = torch.cat(target_value_list, dim=1)
    # flatten bs and action length
    pred_value_list = pred_value_list.flatten(1, 2)
        
    # return pred_value_list, target_value_list
    return pred_value_list, None
