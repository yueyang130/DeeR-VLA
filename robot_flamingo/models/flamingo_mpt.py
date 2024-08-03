import torch
from einops import rearrange, repeat
from torch import nn
import copy
from open_flamingo.src.helpers import PerceiverResampler
from robot_flamingo.models.action_head import DeterministicDecoder, DiffusionDecoder, FCDecoder, GPTDecoder, GaussianDecoder
from collections import namedtuple
from copy import deepcopy
import time
import shutil
from fvcore.nn import FlopCountAnalysis
from thop import profile
from contextlib import suppress
import random
import numpy as np

class MPTFlamingo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        use_media_placement_augmentation: bool = False,
        # this is the window size sampled from the episode
        window_size: int = 8,
        use_gripper=False,
        fusion_mode='',
        sep_resampler=False,
        use_state=False,
        use_diff=False,
        diff_horizon=32,
        last_action=False,
        n_timesteps=150,
        state_dim=15,
        use_hist=False,
        debug=False,
        predict_epsilon=True,
        pad_length=-1,
        multi_step_action=1,
        sep_lm_head=False,
        return_feature = False,
        llm='llama',
        pooling='max',
        residual=False,
        tcp_rel=False,
        replan=-1,
        decoder_type='lstm',
        head_type='deterministic',
        tanh_squash_dist=True, 
        state_dependent_std=True,
        hidden_size=None,
        fwd_pred=False,
        fwd_pred_hand=False,
        global_latent=10,
        no_image_patch=False,
        refresh=-1,
        early_exit_layer=-1,
        multi_exit=False,
        share_exit=False,
        exit_interval=1,
        exit_dropout=0.0,
        lstm_dropout=0.0,
        dropout_mode='layerwise',
        # for dynamic exit
        use_extra_exit=False,
        detach_extra_exit=True,
        layerwise_exit_eval=False,
        mlp_layernorm=False,
        lstm_layernorm=False,
        mlp_num_hidden_layers=3,
        lstm_num_layers=4,
        use_layerwise_projection=False,
        num_projection_layers : int = 1,
        skip_connection=False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
            use_media_placement_augmentation (bool, optional): Whether to randomly assign images to the preceding or following text in training. Defaults to False.
        """
        super().__init__()

        self.use_gripper = use_gripper
        self.use_state = use_state
        self.fusion_mode = fusion_mode
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = use_media_placement_augmentation
        self.vis_dim = vis_dim
        self.window_size = window_size
        self.tcp_rel = tcp_rel
        self.act_step = multi_step_action
        print('window size: {}'.format(window_size))
        self.vision_encoder = vision_encoder
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.sep_resampler = sep_resampler
        self.use_hist = use_hist
        self.lang_encoder = lang_encoder
        self.pad_length = pad_length
        self.replan = replan
        if self.replan != -1:
            self.replan = min(int(replan * self.window_size), 180)
        self.refresh = refresh
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size

        self.residual = residual
        print(self.vis_dim, self.lang_dim)
        # print(lang_encoder.config)
        if not debug:
            if 'llama' in llm:
                self.lang_encoder.init_flamingo(
                    media_token_id=media_token_id,
                    vis_hidden_size=self.vis_dim,
                    cross_attn_every_n_layers=cross_attn_every_n_layers,
                    use_media_placement_augmentation=self.use_media_placement_augmentation,
                    residual=residual,
                )
            else:
                self.lang_encoder.init_flamingo(
                    media_token_id=media_token_id,
                    lang_hidden_size=self.lang_dim,
                    vis_hidden_size=self.vis_dim,
                    cross_attn_every_n_layers=cross_attn_every_n_layers,
                    gradient_checkpointing=False,
                )

        if sep_resampler:
            self.perceiver_gripper = PerceiverResampler(dim=self.vis_dim)
            self.perceiver_gripper.load_state_dict(copy.deepcopy(self.perceiver.state_dict()))
        if use_state:
            self.state_fc = nn.Linear(state_dim, self.vis_dim)
        if use_hist:
            self.frame_embs = nn.Parameter(torch.randn(self.window_size, self.vis_dim))
        # To-do: nn archiecture for actor
        self.llm = llm
        if llm=='llama':
            in_features = lang_encoder.lm_head.in_features
        else:
            in_features = self.lang_dim
        self.use_diff = use_diff
        
        self.decoder_type = decoder_type
        self.head_type = head_type
        if decoder_type == 'lstm':
            print(f'{head_type=}')
            if head_type == 'deterministic':
                lm_head = DeterministicDecoder(in_features, self.window_size, exit_dropout, lstm_dropout, dropout_mode, mlp_layernorm, lstm_layernorm, mlp_num_hidden_layers, lstm_num_layers=lstm_num_layers,
                    use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action, pooling=pooling,
                    )
            elif head_type == 'gaussian':
                lm_head = GaussianDecoder(in_features, self.window_size, exit_dropout,
                    use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action, pooling=pooling,
                    tanh_squash_dist=tanh_squash_dist, state_dependent_std=state_dependent_std)
                print(f'Use guassian policy! {tanh_squash_dist=}, {state_dependent_std=}')
                
            self.lang_encoder.lm_head = lm_head
        elif decoder_type == 'fc':
            if use_hist:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
            elif 'vit_concat' in fusion_mode:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
            else:
                raise NotImplementedError
        elif decoder_type == 'diffusion':
            if use_diff:
                self.diffusion_model = DiffusionDecoder(
                    self.action_head.hidden_size, 
                    self.window_size,
                    input_dim=self.action_head.out_features+1,
                    n_timesteps=n_timesteps,
                    horizon=diff_horizon,
                    predict_epsilon=predict_epsilon,
                )
            else:
                raise NotImplementedError
        elif decoder_type=='gpt':
            lm_head = GPTDecoder(in_features, self.window_size, use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, multi_step_action=multi_step_action, pooling=pooling, hidden_size=hidden_size)
            self.lang_encoder.lm_head = self.action_head = lm_head
        else:
            raise NotImplementedError
        
        sep_lm_head = True
        self.sep_lm_head = sep_lm_head
        if sep_lm_head:
            self.lm_head = self.lang_encoder.lm_head
            self.lang_encoder.lm_head = nn.Identity()
        self.in_features = in_features
        
        if early_exit_layer < 0:
            early_exit_layer += lang_encoder.config.n_layers
          
        assert 0 <= early_exit_layer < lang_encoder.config.n_layers
        print(f'Early Exit from Layer{early_exit_layer}')
        self.early_exit_layer = early_exit_layer

        self.lang_encoder._delete_decoder_layers(list(range(early_exit_layer+1, lang_encoder.config.n_layers)))
        
        
        def get_encoder(is_extra_exit, exit_list=None):
            if decoder_type == 'lstm':
                if head_type == 'deterministic':
                    lm_head = DeterministicDecoder(in_features, self.window_size, exit_dropout, lstm_dropout, dropout_mode, mlp_layernorm, lstm_layernorm, mlp_num_hidden_layers, lstm_num_layers=lstm_num_layers,
                        use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action, pooling=pooling,
                        is_extra_exit=is_extra_exit, use_layerwise_projection=use_layerwise_projection, num_projection_layers=num_projection_layers, skip_connection=skip_connection, exit_list=exit_list)
                elif head_type == 'gaussian':
                    lm_head = GaussianDecoder(in_features, self.window_size, exit_dropout,
                        use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action, pooling=pooling,
                        tanh_squash_dist=tanh_squash_dist, state_dependent_std=state_dependent_std)
            elif decoder_type == 'fc':
                if use_hist:
                    lm_head = FCDecoder(in_features, self.window_size, 
                    use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
                elif 'vit_concat' in fusion_mode:
                    lm_head = FCDecoder(in_features, self.window_size, 
                    use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
                else:
                    raise NotImplementedError
            elif decoder_type == 'diffusion':
                if use_diff:
                    lm_head = DiffusionDecoder(
                        self.action_head.hidden_size, 
                        self.window_size,
                        input_dim=self.action_head.out_features+1,
                        n_timesteps=n_timesteps,
                        horizon=diff_horizon,
                        predict_epsilon=predict_epsilon,
                    )
                else:
                    raise NotImplementedError
            elif decoder_type=='gpt':
                lm_head = GPTDecoder(in_features, self.window_size, use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, multi_step_action=multi_step_action, pooling=pooling, hidden_size=hidden_size)
            else:
                raise NotImplementedError
            return lm_head
        
        # multi-exit
        self.lm_exits = {}
        if multi_exit:
            print("Enable multi-exit!")
            
            for i in range(exit_interval-1, early_exit_layer, exit_interval):
                if share_exit:
                    self.lm_exits[i] = self.lm_head
                else:
                    self.lm_exits[i] = get_encoder(is_extra_exit=False)
            self.lm_exit_modules = nn.ModuleList(self.lm_exits.values()) # make exits on gpu  device automatically

            print(f'{len(self.lm_exits)} internal exits {list(self.lm_exits.keys())} and one internal exit!')  
        else:
            for i in range(exit_interval-1, early_exit_layer, exit_interval):
                    self.lm_exits[i] = nn.Identity()
            self.lm_exit_modules = nn.ModuleList(self.lm_exits.values())

                    
        # extra one exit
        self.use_extra_exit = use_extra_exit
        self.detach_extra_exit = detach_extra_exit
        self.layerwise_exit_eval = layerwise_exit_eval

        if use_extra_exit:
            if share_exit:
                self.extra_exit = self.lm_head
            else:
                self.extra_exit = get_encoder(is_extra_exit=True, exit_list=self.get_all_exit_idx())
        
            if not self.layerwise_exit_eval:
                print('eval the extra exit!') 
            else:
                print('eval layerwise exits!') 

        self.llm_inference_time = -1.0
        
    def get_all_exit_idx(self):
        # not include the extra exit
        return list(self.lm_exits.keys()) + [self.lang_encoder.config.n_layers - 1]
    
    def get_exit_num(self):
        return len(self.get_all_exit_idx())
    
    def set_all_exit_window_size(self, new_window_size):
        # include the extra exit if it exists
        if self.sep_lm_head:
            old_window_size = self.lm_head.window_size
            self.lm_head.window_size = new_window_size
        else:
            old_window_size = self.lang_encoder.lm_head.window_size
            self.lang_encoder.lm_head.window_size = new_window_size
        
        if len(self.lm_exits) > 0:
            for exit in self.lm_exit_modules:
                exit.window_size = new_window_size
                
        if self.use_extra_exit:
            self.extra_exit.window_size = new_window_size
                
        return old_window_size
    
    def clear_all_exit_memory(self):
        if self.sep_lm_head:
            self.lm_head.hidden_state = None
            self.lm_head.history_memory = []
        else:
            self.lang_encoder.lm_head.hidden_state = None
            self.lang_encoder.lm_head.history_memory = []
            
        if len(self.lm_exits) > 0:
            for exit in self.lm_exit_modules:
                exit.hidden_state = None
                exit.history_memory = []
        
        if self.use_extra_exit:
            self.extra_exit.hidden_state = None
            self.extra_exit.history_memory = []
    
    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper = None,
        state_tensor = None,
        return_feature = False,
        policy_mask=None,
        act=None,
        deterministic=False,
        with_gripper_logits=False,
        exit_id=None,
        dynamic_early_exit=False,
        exit_controller=None,
        return_in_feat=False,
        return_aggregate_feature=False,
        only_extra_exit=False,
        eval_time=False,
        no_backbone_grad=False,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        raw_rgb = vision_x.clone()
        raw_gripper = vision_gripper.clone()
        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            if self.use_hist:
                self._encode_history_vision_post_fusion(vision_x, vision_gripper)
            else:
                if not self.use_gripper or self.fusion_mode == 'two_way':
                    vision_x = self._encode_vision_x(vision_x=vision_x)
                else:
                    if self.fusion_mode == 'pre':
                        self._encode_multi_vision_pre_fusion(vision_x, vision_gripper)
                    elif self.fusion_mode == 'post':
                        self._encode_multi_vision_post_fusion(vision_x, vision_gripper)
                    elif self.fusion_mode == 'vit_concat':
                        self._encode_history_vision_fc_post(vision_x, vision_gripper)
        

        if eval_time:
            torch.cuda.synchronize()
            cur_time = time.time()
            
        grad_manager = torch.no_grad if no_backbone_grad else suppress

        with grad_manager():
            if dynamic_early_exit and exit_controller is not None:
                output = self.lang_encoder(
                    input_ids=lang_x,
                    attention_mask=attention_mask.bool(),
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_hidden_states=True,
                    exit_controller = exit_controller,
                )
            else:
                # when exit_id is None, it behaves as the static LLM forward            
                output = self.lang_encoder(
                    input_ids=lang_x,
                    attention_mask=attention_mask.bool(),
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_hidden_states=True,
                    exit_id=exit_id,
                )

        if eval_time:
            torch.cuda.synchronize()
            llm_time = time.time()-cur_time
            # print(f"LLM total time: {llm_time:.4f} seconds")
            self.llm_inference_time = llm_time
            # if len(self.llm_inference_time_list) > 1000:
            #     print('AVG LLM INFERENCE TIME: ', np.mean(self.llm_inference_time_list))
            #     exit()
            
        
        def get_action(head, in_feat, in_state, return_aggregate_feature=False, eval_flop=False, layer_indices=None):
            
            if eval_flop:
                vis_per_flop = profile(head, inputs=(in_feat, in_state, return_feature, return_aggregate_feature, with_gripper_logits, layer_indices))[0]
                print(f'thop action head flops = {vis_per_flop/1e9:.2f}G, ')
            
            if eval_time:
                torch.cuda.synchronize()
                cur_time = time.time()
                
            if isinstance(head, GaussianDecoder):
                o = head(in_feat, state_tensor=in_state, return_feature=return_feature, return_aggregate_feature=return_aggregate_feature, with_gripper_logits=with_gripper_logits, act=act, deterministic=deterministic)
            else:
                o = head(in_feat, state_tensor=in_state, return_feature=return_feature, return_aggregate_feature=return_aggregate_feature, with_gripper_logits=with_gripper_logits, layer_indices=layer_indices)
            
            if eval_time:
                torch.cuda.synchronize()
                # print(f"action head time: {time.time()-cur_time:.4f} seconds")
                cur_time = time.time()
            return o
        
        
        # inference: only output action by one specified exit
        if dynamic_early_exit and exit_controller is not None:
            exit_id = output.exit_layer
        # exit with exit id (specified by dynamic exit or munual set)
        if exit_id is not None: 
            if exit_id < 0:
                exit_id += self.lang_encoder.config.n_layers
            assert 0 <= exit_id < self.lang_encoder.config.n_layers
            if self.use_extra_exit and not self.layerwise_exit_eval:
                # only use the extra exit for inference
                exit_head = self.extra_exit
            else:
                if exit_id == self.lang_encoder.config.n_layers - 1:
                    exit_head = self.lm_head
                else:
                    exit_head = self.lm_exits[exit_id]
            assert len(output.hidden_states) == exit_id + 1, f'{len(output.hidden_states)}, {exit_id+1}'
            exit_action_output = get_action(exit_head, output.hidden_states[exit_id], state_tensor, layer_indices=exit_id)
            output.logits = exit_action_output
            return output
        
        # training: output actions by all exits
        last_feat = output.hidden_states[-1] # (bs * action_seq_len, lang_len, d)
        last_action_output = get_action(self.lm_head, last_feat, state_tensor)
        output.logits = last_action_output
        
        exit_outputs = []
        if len(self.lm_exits) > 0 and not only_extra_exit: # when generate value for finding threshold, we don't need actions from layer exits
            for exit_layer, exit_head in self.lm_exits.items():
                internal_feat = output.hidden_states[exit_layer]
                exit_action_output = get_action(exit_head, internal_feat, state_tensor)
                exit_outputs.append(exit_action_output)
            
        
        if self.use_extra_exit:
            all_feats = torch.stack(output.hidden_states, dim=1) # (bs * action_seq_len, n_exit, lang_len, d)
            # (bs * action_seq_len, n_exit, lang_len, d) -> (bs, action_seq_len, n_exit, lang_len, d)
            all_feats = all_feats.reshape(-1, self.window_size, *all_feats.shape[1:]) 
            # (bs, action_seq_len, n_exit, lang_len, d) -> (bs, action_seq_len, lang_len, d)
            bs, action_seq_len, _ = all_feats.shape[:3]
            exit_ids = self.get_all_exit_idx()
            exit_num = self.get_exit_num()
            
            # use features from random layers as LSTM input
            # rand_layer_indices = torch.randint(0, n_exit, size=(bs, action_seq_len, 1, 1, 1), device=all_feats.device)
            indices = torch.randint(0, exit_num, size=(bs, action_seq_len), device=all_feats.device)
            in_indices1 = rand_layer_indices = torch.tensor([exit_ids[idx] for idx in indices.reshape(-1)], device=all_feats.device).reshape(bs, action_seq_len)
            
            rand_layer_indices = rand_layer_indices.reshape(bs, action_seq_len, 1, 1, 1).expand(-1, -1, -1, all_feats.shape[3], all_feats.shape[4])
            rand_layer_feat = torch.gather(all_feats, 2, rand_layer_indices).squeeze(2)     
            # if not only_extra_exit:
            # (bs, action_seq_len, lang_len, d) -> (bs * action_seq_len, lang_len, d)
            rand_layer_feat = rand_layer_feat.flatten(0, 1)
            extra_exit_output = get_action(self.extra_exit, 
                                            rand_layer_feat.detach() if self.detach_extra_exit else rand_layer_feat,
                                            state_tensor, 
                                            layer_indices=in_indices1
                                            )
            
            # use features from two layers as LSTM input
            prev_len = random.randint(1, action_seq_len)
            indices = torch.randint(0, exit_num, size=(bs, 2), device=all_feats.device)
            indices_list = [indices[:, 0] for _ in range(prev_len)] + [indices[:, 1] for _ in range(action_seq_len-prev_len)]
            in_indices2 = indices = torch.tensor([exit_ids[idx] for idx in torch.stack(indices_list, dim=1).reshape(-1)], device=all_feats.device).reshape(bs, action_seq_len)
            indices = indices.reshape(bs, action_seq_len, 1, 1, 1).expand(-1, -1, -1, all_feats.shape[3], all_feats.shape[4])
            two_layer_feat = torch.gather(all_feats, 2, indices).squeeze(2)     
            # if not only_extra_exit:
            # (bs, action_seq_len, lang_len, d) -> (bs * action_seq_len, lang_len, d)
            two_layer_feat = two_layer_feat.flatten(0, 1)
            extra_exit_output2 = get_action(self.extra_exit, 
                                            two_layer_feat.detach() if self.detach_extra_exit else two_layer_feat,
                                            state_tensor,
                                            layer_indices=in_indices2
                                            )
            
            
                # extra_exit_output = get_action(self.extra_exit, rand_layer_feat, state_tensor)
            # else:
            #     # we only get the predicted values at the t timestep with input feature from all layers.
            #     all_layer_feat = []
            #     for t in range(self.window_size):
            #         all_layer_feat_t = []
            #         for i in range(len(self.lm_exits)+1):
            #             rand_layer_feat_i = rand_layer_feat # (bs, action_seq_len, lang_len, d)
            #             rand_layer_feat_i[:, t] = all_feats[:, t, i] # (bs, action_seq_len, n_exit, lang_len, d)
            #             all_layer_feat_t.append(rand_layer_feat_i)
            #         all_layer_feat.append(torch.stack(all_layer_feat_t, dim=0))  # (n_exit, bs, action_seq_len, lang_len, d)
            #     rand_layer_feat = torch.stack(all_layer_feat, dim=0)  # (action_seq_len, n_exit, bs, action_seq_len, lang_len, d)
                
            #     # input_feat = rand_layer_feat.flatten(2, 3).flatten(0, 1).detach()
            #     input_feat = rand_layer_feat.flatten(2, 3).flatten(0, 1)
            #     extra_exit_output = get_action(self.extra_exit, input_feat, state_tensor, return_aggregate_feature=return_aggregate_feature) # (action_seq_len * exits * bs, seq_len)
                
                # def get_output(x):
                #     x = x.reshape(action_seq_len, len(self.lm_exits)+1, bs, action_seq_len, -1)
                #     x = [x[i, :, :, i:i+1]  for i in range(action_seq_len)] # action_seq_len * (exit_num, bs, 1)
                #     x = torch.cat(x, dim=2) #  (exit_num, bs, action_seq_len)
                #     return x
                # extra_exit_output = (
                #     get_output(extra_exit_output[0]), 
                #     (get_output(extra_exit_output[1][0]),  get_output(extra_exit_output[1][1])),
                #     get_output(extra_exit_output[2]), 
                # )
            
            if return_in_feat:
                return output, exit_outputs, extra_exit_output, rand_layer_feat, rand_layer_indices[:, :, 0, 0, 0]
            else:
                return output, exit_outputs, extra_exit_output, extra_exit_output2
        else:
            if len(self.lm_exits) > 0:
                return output, exit_outputs            
        
        return output
        
    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]
            if eval_flop:
                vis_enc_flop = FlopCountAnalysis(self.vision_encoder.visual, vision_x)
                print(f'vis encoder flops = {vis_enc_flop.total()/1e9:.1f}G, ')
                
            
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)


        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_vision(self, vision_x: torch.Tensor, state_tensor=None, eval_flop=False):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            if eval_flop:
                vis_enc_flop = FlopCountAnalysis(self.vision_encoder.visual, vision_x).total()
                print(f'fvcore vis encoder flops = {vis_enc_flop/1e9:.1f}G, ')
                vis_enc_flop = profile(self.vision_encoder.visual, inputs=(vision_x,))[0]
                print(f'thop vis encoder flops = {vis_enc_flop/1e9:.1f}G, ')
            
            vision_x = self.vision_encoder.visual(vision_x)[1]
            
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        return vision_x

    def _encode_multi_vision_pre_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)
        vision_x = torch.cat([vision_rgb, vision_gripper], dim=3)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_multi_vision_post_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None, eval_flop=False, eval_time=False):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        if eval_time:
            torch.cuda.synchronize()
            cur_time = time.time()
        
        vision_rgb = self._encode_vision(vision_rgb)
        
        if eval_time:
            torch.cuda.synchronize()
            print(f"vis rgb encoder time: {cur_time-time.time():.4f} seconds")
            cur_time = time.time()
        
        vision_gripper = self._encode_vision(vision_gripper)
        
        if eval_time:
            torch.cuda.synchronize()
            print(f"vis gripper encoder time: {cur_time-time.time():.4f} seconds")
            cur_time = time.time()

        if eval_flop:
            vis_per_flop = FlopCountAnalysis(self.perceiver, vision_rgb).total()
            print(f'fvflop vis perceiver flops = {vis_per_flop/1e9:.1f}G, ')
            vis_per_flop = profile(self.perceiver, inputs=(vision_rgb,))[0]
            print(f'thop vis perceiver flops = {vis_per_flop/1e9:.1f}G, ')

        if eval_time:
            torch.cuda.synchronize()
            cur_time = time.time()

        vision_rgb = self.perceiver(vision_rgb)
        
        if eval_time:
            torch.cuda.synchronize()
            print(f"vis rgb perceiver time: {cur_time-time.time():.4f} seconds")
        
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2)  # reshapes to (b, T, 2*n, d)
        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=2)  # reshapes to (b, T, 2*n+1, d)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_multi_vision_two_way(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)
        vision_rgb = self.perceiver(vision_rgb)
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=0)  # reshapes to (b, T, 2*n, d)
        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=0)  # reshapes to (b, T, 2*n+1, d)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_history_vision_post_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)
        bs = int(vision_rgb.shape[0] // self.window_size)
        vision_rgb = vision_rgb.view(bs, self.window_size, *vision_rgb.shape[1:])
        _, _, T, p, v_tok, dim = vision_rgb.shape[:6]
        frame_embs = repeat(self.frame_embs, 'F d -> b F T p v d', b=bs, T=T, p=p, v=v_tok)
        vision_rgb = vision_rgb + frame_embs
        vision_rgb = rearrange(vision_rgb, 'b F T p v d -> (b F) T p v d')
        vision_rgb = self.perceiver(vision_rgb)

        vision_gripper = vision_gripper.view(vision_gripper.shape[0] // self.window_size, self.window_size,
                                             *vision_gripper.shape[1:])
        frame_embs = repeat(self.frame_embs, 'F d -> b F T p v d', b=bs, T=T, p=p, v=v_tok)
        vision_gripper = vision_gripper + frame_embs
        vision_gripper = rearrange(vision_gripper, 'b F T p v d -> (b F) T p v d')
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2)  # reshapes to (b, T, 2*n, d)
        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=2)  # reshapes to (b, T, 2*n+1, d)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x
    
    def _encode_history_vision_fc_post(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        bs = int(vision_rgb.shape[0] // self.window_size)
        vision_rgb = self._encode_vision(vision_rgb)
        vision_rgb = self.perceiver(vision_rgb) # BxL, T, n, d
        vision_rgb = vision_rgb.view(-1, self.window_size, *vision_rgb.shape[1:]) # B, L, T, n, d
        vision_rgb = rearrange(vision_rgb, 'b L T n d -> b T (n L) d')

        vision_gripper = self._encode_vision(vision_gripper)
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)
        vision_gripper = vision_gripper.view(-1, self.window_size, *vision_gripper.shape[1:]) # B, L, T, n, d
        vision_gripper = rearrange(vision_gripper, 'b L T n d -> b T (n L) d')

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2)

        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=2)  # reshapes to (b, T, 2*n+1, d)
        
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x