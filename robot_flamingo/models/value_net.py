import torch
import torch.nn as nn
import torch.nn.functional as F
from robot_flamingo.models.action_head import MLPNohHead, lstm_decoder
import copy
from typing import Optional, Tuple

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
        
        self.value_net = MLPNohHead(hidden_size, 1, dropout)

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

    def forward(  # type: ignore
        self,
        input_feature: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        state_tensor=None,
    ):
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
            # print('history len:',self.history_len)
            if input_feature.shape[1] == 1:  # the first frame of an action sequence
                self.history_memory.append(input_feature)
                if len(self.history_memory) <= self.history_len:
                    # print('cur hist_mem len: {}'.format(len(self.history_memory)))
                    x, h_n = self.rnn(input_feature, self.hidden_state)
                    self.hidden_state = h_n
                    x = x[:, -1].unsqueeze(1)
                    self.rnn_out = x.squeeze(1)
                else:
                    # the hidden state need to be refreshed based on the history window
                    # print('hist_mem exceeded, refresh hidden state')
                    cur_len = len(self.history_memory)
                    for _ in range(cur_len - self.history_len):
                        self.history_memory.pop(0)
                    assert len(self.history_memory) == self.history_len
                    hist_feature = torch.cat(self.history_memory, dim=1)
                    self.hidden_state = None
                    x, h_n = self.rnn(hist_feature, self.hidden_state)
                    x = x[:, -1].unsqueeze(1)
                    self.rnn_out = x.squeeze(1)
            else:
                # print('input feature lenght > 1', input_feature.shape)
                self.hidden_state = h_0
                x, h_n = self.rnn(input_feature, self.hidden_state) # (exits * bs, seq_len, d) --> (exits * bs, seq_len, d) LSTM go along action seqence
                self.hidden_state = h_n
                self.rnn_out = x.squeeze(1)
        else:
            raise NotImplementedError

        values = self.value_net(x) # (exits * bs, seq_len, 1)
        return values