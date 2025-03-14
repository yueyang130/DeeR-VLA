from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from open_flamingo.src.helpers import PerceiverResampler
from robot_flamingo.models.normalizer import LinearNormalizer
from robot_flamingo.models.trajectory_gpt2 import get_gpt_model
# from .unets import *
import copy

    

class LayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, batch_first):
        super(LayerNormLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        
        # Create layers of LSTM followed by LayerNorm
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_last_layer = i == num_layers - 1
            self.layers.append(nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                bidirectional=False,
                batch_first=batch_first,
                # dropout=dropout_value, # doesn't work here because dropout not applied to single-layer LSTM
            ))
            self.layers.append(nn.LayerNorm(hidden_size))
            
            # Only add dropout layers between LSTMs and not after the last LSTM
            if not is_last_layer:
                self.layers.append(nn.Dropout(dropout))
            
            input_size = hidden_size  # Next layer's input is current layer's output

    def forward(self, x, hidden=None):
        # Split hidden state into h and c for each layer, if provided
        if hidden:
            hidden_states = [(hidden[0][i:i+1], hidden[1][i:i+1]) for i in range(self.num_layers)]
        else:
            hidden_states = [None] * self.num_layers
        
        # Process each LSTM layer followed by LayerNorm
        for i in range(0, len(self.layers), 3):
            lstm = self.layers[i]
            layer_norm = self.layers[i + 1]
            dropout_layer = self.layers[i + 2] if i + 2 < len(self.layers) else None
            
            x, new_hidden_state = lstm(x, hidden_states[i // 3])
            x = layer_norm(x)
            if dropout_layer:
                x = dropout_layer(x)
            hidden_states[i // 3] = new_hidden_state
        
        # Prepare the final hidden state tuple for return
        final_hidden = (torch.cat([h[0] for h in hidden_states], dim=0),
                        torch.cat([h[1] for h in hidden_states], dim=0))
        return x, final_hidden

def lstm_decoder(
    in_features: int, hidden_size: int, num_layers: int, policy_rnn_dropout_p: float, layernorm: bool = False
) -> torch.nn.Module:
    if layernorm:
        return LayerNormLSTM(in_features, hidden_size, num_layers, policy_rnn_dropout_p, batch_first=True)
    else:
        return nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=policy_rnn_dropout_p,
        )


class MLPTanhHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size, dropout, layernorm=False, dropout_mode='layerwise', num_hidden_layers=3):
        super().__init__()
        
        if dropout_mode in ['last', 'layerwise']:
            hidden_dims=[1024, 512, 256]
            assert len(hidden_dims) >= num_hidden_layers
            hidden_dims = hidden_dims[:num_hidden_layers]
            
            layers = []
            current_size = hidden_size
            
            if dropout_mode == 'layerwise':
                layers.append(nn.Dropout(dropout))
            
            for i, dim in enumerate(hidden_dims):
                layers.append(nn.Linear(current_size, dim))
                if layernorm:
                    layers.append(nn.LayerNorm(dim))
                else:
                    layers.append(nn.Identity())
                layers.append(nn.ReLU())

                # Apply dropout according to the specified mode
                if dropout_mode == 'layerwise' or (dropout_mode == 'last' and i == num_hidden_layers - 1):
                    layers.append(nn.Dropout(dropout))
                
                current_size = dim

            # Output layer
            layers.append(nn.Linear(hidden_dims[-1], output_size))
            layers.append(torch.nn.Tanh())

            # Register the sequential model
            self.mlp = nn.Sequential(*layers)
        elif layernorm:
            self.mlp = torch.nn.Sequential(
                torch.nn.Dropout(dropout), 
                torch.nn.Linear(hidden_size, 1024),
                nn.LayerNorm(1024) if layernorm else nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout), 
                torch.nn.Linear(1024, 512),
                nn.LayerNorm(512) if layernorm else nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout), 
                torch.nn.Linear(512, 256),
                nn.LayerNorm(256) if layernorm else nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(256, output_size),
                torch.nn.Tanh(),
            )            
        elif dropout > 0:
            self.mlp = torch.nn.Sequential(
                torch.nn.Dropout(dropout), 
                torch.nn.Linear(hidden_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout), 
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout), 
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, output_size),
                torch.nn.Tanh(),
            )
        else: # workaround for loading old checkpoints
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, output_size),
                torch.nn.Tanh(),
            )

    def forward(self, x):
        return self.mlp(x)
    

class MLPNohHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size, dropout, layernorm):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Dropout(dropout), 
            torch.nn.Linear(hidden_size, 1024),
            nn.LayerNorm(1024) if layernorm else nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout), 
            torch.nn.Linear(1024, 512),
            nn.LayerNorm(512) if layernorm else nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout), 
            torch.nn.Linear(512, output_size),
        )            

    def forward(self, x):
        return self.mlp(x)

class MLPSigmoidHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size, dropout, layernorm, dropout_mode='layerwise', num_hidden_layers=3):
        super().__init__()
        
        if dropout_mode in ['last', 'layerwise']:
            hidden_dims=[1024, 512, 256]
            assert len(hidden_dims) >= num_hidden_layers
            hidden_dims = hidden_dims[:num_hidden_layers]
            
            layers = []
            current_size = hidden_size
            
            if dropout_mode == 'layerwise':
                layers.append(nn.Dropout(dropout))
            
            for i, dim in enumerate(hidden_dims):
                layers.append(nn.Linear(current_size, dim))
                if layernorm:
                    layers.append(nn.LayerNorm(dim))
                else:
                    layers.append(nn.Identity())
                layers.append(nn.ReLU())

                # Apply dropout according to the specified mode
                if dropout_mode == 'layerwise' or (dropout_mode == 'last' and i == num_hidden_layers - 1):
                    layers.append(nn.Dropout(dropout))
                
                current_size = dim

            # Output layer
            layers.append(nn.Linear(hidden_dims[-1], output_size))
            layers.append(nn.Sigmoid())

            # Register the sequential model
            self.mlp = nn.Sequential(*layers)

        elif layernorm:
            if num_hidden_layers != 3: raise NotImplementedError
            self.mlp = torch.nn.Sequential(
                torch.nn.Dropout(dropout), 
                torch.nn.Linear(hidden_size, 1024),
                nn.LayerNorm(1024) if layernorm else nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout), 
                torch.nn.Linear(1024, 512),
                nn.LayerNorm(512) if layernorm else nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout), 
                torch.nn.Linear(512, 256),
                nn.LayerNorm(256) if layernorm else nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(256, output_size),
                torch.nn.Sigmoid(),
            )
        elif dropout > 0:
            self.mlp = torch.nn.Sequential(
                torch.nn.Dropout(dropout), 
                torch.nn.Linear(hidden_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout), 
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout), 
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, output_size),
                torch.nn.Sigmoid(),
            )
        else:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, output_size),
                torch.nn.Sigmoid(),
            )

    def forward(self, x, with_logits=False):
        for layer in self.mlp[:-1]:
            x = layer(x)
        if with_logits:
            return self.mlp[-1](x), x # return
        else:
            return self.mlp[-1](x)


class ActionDecoder(nn.Module):
    def act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss_and_act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _sample(self, *args, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def clear_hidden_state(self) -> None:
        pass
    

class FCDecoder(ActionDecoder):
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
        use_diff=False,
        last_action=False,
        fusion_mode='',
        use_state=False,
        return_feature=False,
        multi_step_action=1
    ):
        super(FCDecoder, self).__init__()
        self.return_feature = return_feature
        if use_state:
            state_in_dim = 7
            state_out_dim = 128
            self.fc_state = MLPNohHead(state_in_dim, state_out_dim)
            in_features += state_out_dim
        
        if fusion_mode == 'two_way':
            in_features *= 2
        
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []

        self.use_diff = use_diff
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Dropout(dropout), 
            torch.nn.Linear(in_features, in_features//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout), 
            torch.nn.Linear(in_features//2, hidden_size),
        )
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features)
            self.gripper = MLPSigmoidHead(hidden_size, 1)
        self.hidden_state = None
        self.hidden_size = hidden_size * history_len
        
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        # self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        self.global_1d_pool = nn.AdaptiveMaxPool1d(1)

    def forward(  # type: ignore
            self,
            input_feature: torch.Tensor,
            h_0: Optional[torch.Tensor] = None,
            state_tensor = None,
    ):
        if self.return_feature:
            org_feat = copy.deepcopy(input_feature) 
            org_feat = org_feat.view(self.window_size, *org_feat.shape[1:])
        # reshape
        input_feature = self.mlp(input_feature)
        input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)
        if self.use_diff:
            input_feature = input_feature.reshape(-1, self.window_size * input_feature.shape[1])
            return input_feature

        input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1])
        if state_tensor is not None:
            state_tensor = self.fc_state(state_tensor)
            state_tensor = state_tensor.reshape(-1, self.window_size, state_tensor.shape[-1])
            input_feature = torch.cat([input_feature, state_tensor], dim=-1)

        actions = self.actions(input_feature)
        gripper = self.gripper(input_feature)

        if self.return_feature:
            return actions, gripper, org_feat
        else:
            return actions, gripper


class DeterministicDecoder(ActionDecoder):
    def __init__(
        self,
        in_features: int,
        window_size: int,
        dropout: float,
        policy_rnn_dropout_p: float,
        dropout_mode: str,
        mlp_layernorm: bool,
        lstm_layernorm: bool,
        mlp_num_hidden_layers: int,
        history_len = None,
        out_features: int = 6,
        hidden_size: int = 1024,
        lstm_num_layers: int = 4, # move two layers to projection
        use_diff=False,
        last_action=False,
        fusion_mode='',
        use_state=False,
        multi_step_action=1,
        return_feature=False,
        pooling='max',
        is_extra_exit=False,
        exit_list=None,
        refresh_window=False, # enabling it would result in more computational cost
    ):
        super(DeterministicDecoder, self).__init__()
        self.fc_state = None
        self.use_state = use_state
        self.refresh_window = refresh_window
        if use_state:
            print('Using state in decoder')
            state_in_dim = 7
            # state_out_dim = 256
            # in_features += state_out_dim
            # self.embed_arm_state = nn.Sequential(torch.nn.Linear(state_in_dim-1, state_out_dim), nn.ReLU())
            # self.embed_gripper_state = nn.Sequential(torch.nn.Embedding(2, state_out_dim), nn.ReLU()) # one-hot gripper state
            # self.embed_state = torch.nn.Linear(2*state_out_dim, state_out_dim)

            self.embed_arm_state = nn.Sequential(torch.nn.Linear(state_in_dim-1, in_features), nn.ReLU())
            self.embed_gripper_state = nn.Sequential(torch.nn.Embedding(2, in_features), nn.ReLU()) # one-hot gripper state
            self.embed_state = torch.nn.Linear(2*in_features, in_features)
        
        if fusion_mode == 'two_way':
            in_features *= 2
        self.return_feature = return_feature
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        
        self.is_extra_exit = is_extra_exit
        
        self.rnn = lstm_decoder
        self.rnn = self.rnn(in_features, hidden_size, lstm_num_layers, policy_rnn_dropout_p, lstm_layernorm)
        # self.rnn = self.rnn(hidden_size, hidden_size, num_layers, policy_rnn_dropout_p)
        self.use_diff = use_diff
        self.fusion_mode = fusion_mode
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features*multi_step_action, dropout, mlp_layernorm, dropout_mode, mlp_num_hidden_layers)
            self.gripper = MLPSigmoidHead(hidden_size, 1*multi_step_action, dropout, mlp_layernorm, dropout_mode, mlp_num_hidden_layers)
        self.hidden_state = None
        self.hidden_size = hidden_size
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
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
        
    def update_hidden_state(self):
        assert self.tmp_hidden_state is not None
        self.hidden_state = self.tmp_hidden_state
        self.tmp_hidden_state = None

    def forward(  # type: ignore
        self,
        input_feature: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        state_tensor=None,
        return_feature=False,
        return_aggregate_feature=False,
        with_gripper_logits=False,
        layer_indices=None,
        update_hidden_state=True,
    ):
        self.return_feature = return_feature
        if input_feature.dim() == 4:
            cur_window_size = input_feature.shape[1]
            input_feature = input_feature.reshape(-1, *input_feature.shape[2:]) 
        else:
            cur_window_size = self.window_size
        
        
        # reshape
        if input_feature.dim() == 3: # (bs * action_seq_len, lang_len, d)
            input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1) # (bs * seq_len, d) maxpooling along lang_seq
        
        input_feature = input_feature.reshape(-1, cur_window_size, input_feature.shape[1]) # (bs, seq_len, d)
        
        if self.use_state:
            # print('use_state, ', state_tensor.shape)
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
        

        if self.return_feature:
            # org_feat = copy.deepcopy(input_feature)
            org_feat = input_feature
            if org_feat.dim() == 2 or org_feat.dim() == 3 and org_feat.shape[1] == 1:
                org_feat = org_feat.view(cur_window_size, org_feat.shape[-1])
        
        
        if (not isinstance(self.rnn, nn.Sequential) and isinstance(self.rnn, nn.RNNBase)) \
            or isinstance(self.rnn, LayerNormLSTM):
            if input_feature.shape[1] == 1:  # inference, only input current frame
                
                self.history_memory.append(input_feature)
                # print('history mem len:', len(self.history_memory))
                
                if not self.refresh_window:
                    x, h_n = self.rnn(input_feature, self.hidden_state)
                    if update_hidden_state:
                        self.hidden_state = h_n
                    else:
                        self.tmp_hidden_state = h_n
                    x = x[:, -1].unsqueeze(1)
                    self.rnn_out = x.squeeze(1)
                else:   # inference code for original Robofalmingo; disabled in DeerVLM for efficiency
                    if len(self.history_memory) <= self.history_len:  # timesteps less than wondow size
                    
                        x, h_n = self.rnn(input_feature, self.hidden_state)
                        self.hidden_state = h_n
                        x = x[:, -1].unsqueeze(1)
                        self.rnn_out = x.squeeze(1)
                        # print(f'{x=}')
                        # print(f'{self.actions(x)=}')
                        # print(f'{self.gripper(x)=}')
                            
                    else:
                        # timesteps greater than window size
                        # the hidden state need to be refreshed based on the history window
                        if len(self.history_memory) > self.history_len:
                            cur_len = len(self.history_memory)
                            for _ in range(cur_len - self.history_len):
                                self.history_memory.pop(0)
                            assert len(self.history_memory) == self.history_len
                        hist_feature = torch.cat(self.history_memory, dim=1)
                        x2, h_n2 = self.rnn(hist_feature, None)
                        x2 = x2[:, -1].unsqueeze(1)
                        x = x2
                        # print(f'{x2=}')
                        # print(f'{self.actions(x2)=}')
                        # print(f'{self.gripper(x2)=}')
                
            else:
                # print('input feature lenght > 1', input_feature.shape)
                self.hidden_state = h_0
                x, h_n = self.rnn(input_feature, self.hidden_state) # (bs, seq_len, d) --> (bs, seq_len, d) LSTM go along action seqence
                self.hidden_state = h_n
                if self.last_action:
                    x = x[:, -1].unsqueeze(1)
                self.rnn_out = x.squeeze(1)
        else:
            raise NotImplementedError
        
        if return_aggregate_feature:
            agg_feat = x
        
        if self.use_diff:
            return self.rnn_out
        actions = self.actions(x)
        gripper = self.gripper(x, with_logits=with_gripper_logits)
        if return_aggregate_feature:
            return actions, gripper, agg_feat
        if self.return_feature:
            return actions, gripper, org_feat
        else:
            return actions, gripper

    # def act(
    #     self,
    #     input_feature: torch.Tensor,
    # ) -> torch.Tensor:
    #     pred_actions, self.hidden_state = self(
    #         input_feature, self.hidden_state
    #     )

    #     return pred_actions


class GPTDecoder(ActionDecoder):
    def __init__(
        self,
        in_features: int,
        window_size: int,
        history_len = None,
        out_features: int = 6,
        hidden_size = None,
        num_layers: int = 4,
        policy_rnn_dropout_p: float = 0.1,
        last_action=False,
        use_diff=False,
        fusion_mode='',
        use_state=False,
        multi_step_action=1,
        return_feature=False,
        pooling='max',
        **kwargs
    ):
        super(GPTDecoder, self).__init__()
        
        if use_state:
            state_in_dim = 7
            state_out_dim = 128
            self.fc_state = MLPNohHead(state_in_dim, state_out_dim)
            in_features += state_out_dim
        
        if fusion_mode == 'two_way':
            in_features *= 2
        self.return_feature = return_feature
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        
        if hidden_size is None:
            hidden_size = in_features
        
        self.gpt = get_gpt_model(hidden_size, history_len)
        self.use_diff = use_diff
        self.fusion_mode = fusion_mode
        
        self.hidden_size = hidden_size
        if hidden_size != in_features:
            self.fc = nn.Linear(in_features, hidden_size)
        else:
            self.fc = nn.Identity()
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features*multi_step_action)
            self.gripper = MLPSigmoidHead(hidden_size, 1*multi_step_action)
        self.hidden_state = None
        self.hidden_size = hidden_size
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        if pooling == 'max':
            self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        
        if self.fusion_mode == 'two_way':
            if pooling == 'max':
                self.gripper_1d_max_pool = nn.AdaptiveMaxPool1d(1)
            else:
                self.gripper_1d_max_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_feature: torch.Tensor):
        time_step=None
        attention_mask=None
        if input_feature.dim() == 3:
            input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)
        input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1]) # bs, seq_len, feat_dim
        input_feature = self.fc(input_feature)
        if input_feature.shape[1] == 1:
            self.history_memory.append(input_feature)
            
            if len(self.history_memory) <= self.history_len:
                hist_feature = torch.cat(self.history_memory, dim=1)
                x = self.gpt(hist_feature, time_step ,attention_mask)
                x = x[:, -1].unsqueeze(1)
                
            else:
                # the hidden state need to be refreshed based on the history window
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                assert len(self.history_memory) == self.history_len
                hist_feature = torch.cat(self.history_memory, dim=1)
                x= self.gpt(hist_feature, time_step, attention_mask)
                x = x[:, -1].unsqueeze(1)
                
        else:
            x = self.gpt(input_feature, time_step, attention_mask)
            if self.last_action:
                x = x[:, -1].unsqueeze(1)
        actions = self.actions(x)
        gripper = self.gripper(x)
        return actions, gripper
    
    def get_pattern_name(self):
        return 'gpt_{}_'.format(self.hidden_size, )

class GPTDecoderActPad(ActionDecoder):
    def __init__(
        self,
        in_features: int,
        window_size: int,
        use_vision = False,
        history_len = None,
        out_features: int = 6,
        hidden_size = None,
        last_action=False,
        use_diff=False,
        fusion_mode='',
        use_state=False,
        multi_step_action=1,
        return_feature=False,
        pooling='sampler',
        global_latent=10,
        **kwargs
    ):
        super(GPTDecoderActPad, self).__init__()
        
        if use_state:
            state_in_dim = 7
            state_out_dim = 128
            self.fc_state = MLPNohHead(state_in_dim, state_out_dim)
            in_features += state_out_dim
        
        if fusion_mode == 'two_way':
            in_features *= 2
        
        self.return_feature = return_feature
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        
        if hidden_size is None:
            hidden_size = in_features
        
        self.gpt = get_gpt_model(hidden_size, history_len, use_pe=False)
        self.use_diff = use_diff
        self.fusion_mode = fusion_mode
        
        self.hidden_size = hidden_size
        if hidden_size != in_features:
            self.fc = nn.Linear(in_features, hidden_size)
        else:
            self.fc = nn.Identity()
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features*multi_step_action)
            self.gripper = MLPSigmoidHead(hidden_size, 1*multi_step_action)
        self.hidden_state = None
        self.hidden_size = hidden_size
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        self.global_latent = global_latent
        self.use_vision = use_vision
        if self.use_vision:
            self.vision_resampler = PerceiverResampler(dim=hidden_size)
        if pooling == 'sampler':
            self.global_1d_pool = PerceiverResampler(dim=hidden_size, depth=2, num_latents=global_latent)
        if pooling == 'max':
            self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        
        if self.fusion_mode == 'two_way':
            if pooling == 'max':
                self.gripper_1d_max_pool = nn.AdaptiveMaxPool1d(1)
            else:
                self.gripper_1d_max_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_feature: torch.Tensor, rgb=None):
        time_step=None
        attention_mask=None
        input_feature = self.global_1d_pool(input_feature.unsqueeze(1)).squeeze(1)
        input_feature = input_feature.view(-1, self.window_size, self.global_latent, input_feature.shape[-1]) # bs, seq_len, n_tok, feat_dim
        bs, seq_len, n_tok = input_feature.shape[:3]
        input_feature = self.fc(input_feature) # # bs, seq_len, n_tok, feat_dim
        attention_mask = torch.ones((bs, n_tok, seq_len), dtype=torch.long).to(input_feature.device)
        
        if input_feature.shape[1] == 1:
            self.history_memory.append(input_feature)
            
            if len(self.history_memory) <= self.history_len:
                hist_feature = torch.cat(self.history_memory, dim=1)
                x = self.gpt(hist_feature, time_step ,attention_mask)
                x = x[:, -1].unsqueeze(1)
                
            else:
                # the hidden state need to be refreshed based on the history window
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                assert len(self.history_memory) == self.history_len
                hist_feature = torch.cat(self.history_memory, dim=1)
                x= self.gpt(hist_feature, time_step, attention_mask)
                x = x[:, -1].unsqueeze(1)
                
        else:
            x = self.gpt(input_feature, time_step, attention_mask)
            if self.last_action:
                x = x[:, -1].unsqueeze(1)
        actions = self.actions(x)
        gripper = nn.functional.sigmoid(self.gripper(x))
        return actions, gripper
    
    def get_pattern_name(self):
        return 'gpt_{}_'.format(self.hidden_size, )


class DiffusionDecoder(ActionDecoder):
    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        history_len = None,
        horizon = 32,
        input_dim: int = 7, # dim of vectors to be diffused
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        n_timesteps=150,
        clip_denoised=False,
        predict_epsilon=True,
        normalizer = LinearNormalizer()
    ):
        super(DiffusionDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.window_size = window_size
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        self.normalizer = normalizer
        self.data_dim = input_dim

        self.model = ConditionalUnet1D(
            input_dim,
            global_cond_dim=feature_dim,
            # global_cond_dim=None,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )


    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        sets loss coefficients for trajectory
        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c
        """
        self.action_weight = action_weight

        dim_weights = torch.ones(self.action_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights = loss_weights.unsqueeze(1).clone()

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, local_cond=None, global_cond=None, returns=None):

        if returns is not None: 
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, t, local_cond, global_cond, returns, use_dropout=False)
            epsilon_uncond = self.model(x, t, local_cond, global_cond, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )
        else:
            epsilon = self.model(x, t, local_cond, global_cond)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, local_cond=None, global_cond=None, returns=None):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, local_cond=local_cond, global_cond=global_cond, returns=returns
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self, cond_data, cond_mask, local_cond=None, global_cond=None, returns=None, verbose=False, return_diffusion=False, **kwargs
    ):
        device = self.betas.device

        batch_size = cond_data.shape[0]
        x = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device
        )

        if return_diffusion:
            diffusion = [x]

        x[cond_mask] = cond_data[cond_mask]
        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):

            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # 1. predict model output and replace sample
            x = self.p_sample(x, timesteps, local_cond, global_cond, returns)
            
            # 2. apply conditioning
            x[cond_mask] = cond_data[cond_mask]

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond_data, cond_mask, local_cond=None, global_cond=None, returns=None, action_seq_len=None, *args, **kwargs):
        """
        conditions : [ (time, state), ... ]
        """

        # horizon = action_seq_len or self.action_seq_len
        # batch_size = len(list(cond_data.values())[0])
        # shape = (batch_size, horizon, self.action_dim) # cond_data.shape
        return self.p_sample_loop(cond_data, cond_mask, local_cond, global_cond, returns, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample
    
    def forward(
        self,
        x,
        t,
        local_cond=None,
        global_cond=None,
        **kwargs
    ):
        return self.model(x, t, local_cond, global_cond)

    def act(
        self,
        input_feature: torch.Tensor,
    ) -> torch.Tensor:
        pred_actions, self.hidden_state = self(
            input_feature, self.hidden_state
        )

        raise NotImplementedError

if __name__ == "__main__":
    model = GPTDecoder(128, 24)
    in_feat = torch.randn((4*24, 12, 128))
    out = model(in_feat)
    print(out[0].shape, out[1].shape)
    pass