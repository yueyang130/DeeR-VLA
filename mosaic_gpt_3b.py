# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A simple, flexible implementation of a GPT model.

Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput, dataclass
from .attention import attn_bias as module_attn_bias, attn_bias_shape as module_attn_bias_shape
from .gpt_blocks import GPTBlock
from .configuration_mosaic_gpt import \
    MosaicGPTConfig
from .param_init_fns import MODEL_INIT_REGISTRY
from .low_precision_layernorm import LPLayerNorm
import time


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    exit_layer: Optional[int] = None


class MosaicGPT(PreTrainedModel):
    config_class = MosaicGPTConfig
    base_model_prefix = 'mosaic_gpt'

    def __init__(self, config: MosaicGPTConfig):
        
        super().__init__(config)

        if config.attn_impl == 'flash' and config.alibi:
            raise RuntimeError("ALiBi is not supported with flash attention. Please use triton or torch.")

        self.attn_impl = config.attn_impl
        self.prefix_lm = config.prefix_lm
        self.attn_uses_sequence_id = config.attn_uses_sequence_id
        self.alibi = config.alibi
        self.alibi_bias_max = config.alibi_bias_max
        
        layernorm_class = LPLayerNorm if config.low_precision_layernorm else nn.LayerNorm

        # CogView (https://arxiv.org/abs/2105.13290) and GLM-130B (https://arxiv.org/abs/2210.02414)
        # both report this helping with stabilizing training
        self.embedding_fraction = config.embedding_fraction

        self.transformer = nn.ModuleDict({
            'wte':
                nn.Embedding(config.vocab_size,
                             config.d_model,
                             device=config.init_device)
        })
        if not self.alibi:
            self.transformer.update({
                'wpe':
                    nn.Embedding(config.max_seq_len,
                                 config.d_model,
                                 device=config.init_device)
            })
        self.transformer.update({'emb_drop': nn.Dropout(config.emb_pdrop)})
        self.transformer.update({
            'blocks':
                nn.ModuleList([
                    GPTBlock(device=config.init_device,
                                        **config.to_dict())
                    for _ in range(config.n_layers)
                ])
        })
        self.transformer.update({
            'ln_f': layernorm_class(config.d_model, device=config.init_device)
        })

        # enables scaling output logits; similar to a softmax "temperature"
        # PaLM paper uses scale 1/sqrt(config.d_model)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(
                        f"{logit_scale=} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
                    )
            self.logit_scale = logit_scale

        if config.init_device != 'meta':
            print(
                f'You are using {config.init_device=}, but you can also use config.init_device="meta" with Composer + FSDP for fast initialization.'
            )
            self.apply(self.param_init_fn)

        self.is_causal = not self.prefix_lm

        # define attn mask
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = module_attn_bias_shape(
            self.attn_impl,
            config.n_heads,
            config.max_seq_len,
            self.alibi,
            prefix_lm=self.prefix_lm,
            causal=self.is_causal,
            use_sequence_id=self.attn_uses_sequence_id)

        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(
                        module.bias, nn.Parameter):
                    if config.verbose:
                        print(f'Removing bias ({module.bias}) from {module}.')
                    module.register_parameter('bias', None)

        if config.verbose and config.verbose > 2:
            print(self)

    @torch.no_grad()
    def _attn_bias(self,
                   device,
                   dtype,
                   attention_mask: Optional[torch.ByteTensor] = None,
                   prefix_mask: Optional[torch.ByteTensor] = None,
                   sequence_id: Optional[torch.LongTensor] = None):
        if not self._attn_bias_initialized:
            if self.attn_bias_shape:
                self.attn_bias = torch.zeros(self.attn_bias_shape,
                                             device=device,
                                             dtype=dtype)
                self.attn_bias = module_attn_bias(
                    self.attn_impl,
                    self.attn_bias,
                    self.config.n_heads,
                    self.config.max_seq_len,
                    causal=self.is_causal,
                    alibi=self.alibi,
                    alibi_bias_max=self.alibi_bias_max)
            self._attn_bias_initialized = True

        # flash does not support prefix_lm and will incorporate any
        # attention_mask inside the attention module
        if self.attn_impl == 'flash':
            return self.attn_bias, attention_mask

        attn_bias = self.attn_bias

        # If using torch or triton, we incorporate the prefix_mask (if appropriate)
        if self.prefix_lm:
            assert isinstance(attn_bias, torch.Tensor)  # pyright
            assert isinstance(prefix_mask, torch.Tensor)  # pyright
            attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)

        # If using torch or triton, we incorporate sequence_id (if appropriate)
        if self.attn_uses_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)  # pyright
            attn_bias = self._apply_sequence_id(attn_bias, sequence_id)

        # If using torch or triton, we incorporate attention_mask. This will output
        # None in place of attention_mask since it will not be further needed in the
        # attention modules.
        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k),
                                        device=device,
                                        dtype=dtype)
            else:
                attn_bias = attn_bias[:, :, :, -s_k:]
            if prefix_mask is not None and (attention_mask.shape !=
                                            prefix_mask.shape):
                raise ValueError(
                    f'attention_mask shape={attention_mask.shape} ' +\
                    f'and prefix_mask shape={prefix_mask.shape} are not equal.'
                )
            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(
                ~attention_mask.view(-1, 1, 1, s_k), min_val)

        return attn_bias, None

    def _apply_prefix_mask(self, attn_bias: torch.Tensor,
                           prefix_mask: torch.Tensor):
        s_k, s_q = attn_bias.shape[-2:]
        if (s_k != self.config.max_seq_len) or (s_q != self.config.max_seq_len):
            raise ValueError(
                'attn_bias does not match the expected shape. ' +\
                f'The last two dimensions should both be {self.config.max_length} ' +\
                f'but are {s_k} and {s_q}.'
            )
        seq_len = prefix_mask.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f'prefix_mask sequence length cannot exceed max_seq_len={self.config.max_seq_len}'
            )

        # select seq_len subset of attn mask
        attn_bias = attn_bias[..., :seq_len, :seq_len]

        # Mix the causal max and the bidirectional mask to get the full
        # allowable attention (i.e. full = not accounting for padding yet)
        causal = torch.tril(
            torch.ones((seq_len, seq_len),
                       dtype=torch.bool,
                       device=prefix_mask.device)).view(1, 1, seq_len, seq_len)
        prefix = prefix_mask.view(-1, 1, 1, seq_len)
        cannot_attend = ~torch.logical_or(causal, prefix.bool())

        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)

        return attn_bias

    def _apply_sequence_id(self, attn_bias: torch.Tensor,
                           sequence_id: torch.LongTensor):
        seq_len = sequence_id.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f'sequence_id sequence length cannot exceed max_seq_len={self.config.max_seq_len}'
            )

        # select seq_len subset of attn mask
        attn_bias = attn_bias[..., :seq_len, :seq_len]

        # Restrict attention to tokens that share the same value
        # in sequence_id
        cannot_attend = torch.logical_not(
            torch.eq(sequence_id.view(-1, seq_len, 1),
                     sequence_id.view(-1, 1, seq_len))).unsqueeze(1)
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)

        return attn_bias

    def forward(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
            attention_mask: Optional[torch.ByteTensor] = None,
            prefix_mask: Optional[torch.ByteTensor] = None,
            sequence_id: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            use_cache: Optional[bool] = None,
            exit_controller = None,
            exit_id = None,
            all_hidden_states = None,
            eval_flop = False,
            eval_time = False,
            ):
        

        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # These args are passed in by keyword in huggingface's generate function
        # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/generation/utils.py#L2201-L2206
        # but have not yet been fully implemented in MosaicGPT
        
        assert exit_controller is None or exit_id is None, "Only one exit indicator can be sepcified!"
        if exit_id is not None and exit_id < 0:
            exit_id += self.config.n_layers
        
        # adapted to dynamic early-exit
        if not return_dict:
            raise NotImplementedError(
                'return_dict False is not implemented yet for MosaicGPT')
        if output_attentions:
            raise NotImplementedError(
                'output_attentions is not implemented yet for MosaicGPT')

        if attention_mask is not None and attention_mask[:, 0].sum(
        ) != attention_mask.shape[0] and self.training:
            raise NotImplementedError(
                'MosaicGPT does not support training with left padding.')

        if self.prefix_lm and prefix_mask is None:
            raise ValueError(
                'prefix_mask is a required argument when MosaicGPT is configured with prefix_lm=True.'
            )

        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError(
                    'sequence_id is a required argument when MosaicGPT is configured with attn_uses_sequence_id=True ' +\
                    'and the model is in train mode.'
                )
            elif (self.attn_uses_sequence_id is False) and (sequence_id
                                                            is not None):
                warnings.warn(
                    'MosaicGPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. ' +\
                    'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.'
                )

        S = input_ids.size(1)

        assert (
            S <= self.config.max_seq_len
        ), f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'

        tok_emb = self.transformer.wte(input_ids)  # type: ignore
        if self.alibi:
            x = tok_emb
        else:
            past_position = 0
            if past_key_values is not None:
                if len(past_key_values) != self.config.n_layers:
                    raise ValueError(
                        f'past_key_values must provide a past_key_value for each attention ' +\
                        f'layer in the network ({len(past_key_values)=}; {self.config.n_layers=}).'
                    )
                # get the key tensor whose spec should be (batch, seq, dim), and
                # collect the `seq`, so that the position embedding is shifted
                past_position = past_key_values[0][0].size(1)

            if S + past_position > self.config.max_seq_len:
                raise ValueError(
                    f'Cannot forward input with past sequence length {past_position} and current sequence length '
                    f'{S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.'
                )
            pos = torch.arange(past_position,
                               S + past_position,
                               dtype=torch.long,
                               device=input_ids.device).unsqueeze(0)
            if attention_mask is not None:
                # adjust the position indices to account for padding tokens
                pos = torch.clamp(pos - torch.cumsum(
                    (~attention_mask).to(torch.int32), dim=1)[:,
                                                              past_position:],
                                  min=0)

            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = tok_emb + pos_emb

        if self.embedding_fraction == 1:
            x = self.transformer.emb_drop(x)  # type: ignore
        else:
            # this implementation is proposed on page 7 of the GLM-130B paper https://arxiv.org/abs/2210.02414
            x_shrunk = (x * self.embedding_fraction) + (
                x.detach() * (1 - self.embedding_fraction))
            assert isinstance(self.transformer.emb_drop, nn.Module)  # pyright
            x = self.transformer.emb_drop(x_shrunk)

        attn_bias, attention_mask = self._attn_bias(
            device=x.device,
            dtype=x.dtype,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id)

        # initialize the past key values cache if it should be used
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)
                              ]  # type: ignore

        all_hidden_states = () if output_hidden_states else None
        for b_idx, block in enumerate(self.transformer.blocks):  # type: ignore
            past_key_value = past_key_values[
                b_idx] if past_key_values is not None else None
            
            if eval_flop:
                from fvcore.nn import FlopCountAnalysis
                from thop import profile
                vis_per_flop = FlopCountAnalysis(block, inputs=(x, past_key_value, attn_bias, attention_mask, self.is_causal)).total()
                print(f'fvflop LLM layer {b_idx} flops = {vis_per_flop/1e9:.1f}G, ')
                vis_per_flop = profile(block, inputs=(x, past_key_value, attn_bias, attention_mask, self.is_causal))[0]
                print(f'thop LLM layer {b_idx} flops = {vis_per_flop/1e9:.1f}G, ')
            
            if eval_time:
                torch.cuda.synchronize()
                cur_time = time.time()
            
            x, past_key_value = block(x,
                                      past_key_value=past_key_value,
                                      attn_bias=attn_bias,
                                      attention_mask=attention_mask,
                                      is_causal=self.is_causal)
            
            if eval_time:
                torch.cuda.synchronize()
                print(f"LLM layer time: {cur_time-time.time():.4f} seconds")
                cur_time = time.time()
            
            # make i-th hidden_state is the output of i-th layer instead of the input of i-th layer in original MPT
            if output_hidden_states:
                assert all_hidden_states is not None  # pyright
                all_hidden_states = all_hidden_states + (x,)
            if past_key_values is not None:
                past_key_values[b_idx] = past_key_value
                
            if exit_id is not None:
                if exit_id == b_idx:
                    return CausalLMOutputWithPast(logits=[1.0],
                                past_key_values=past_key_values,
                                hidden_states=all_hidden_states,
                                exit_layer=b_idx)

            if exit_controller is not None:
                if exit_controller(all_hidden_states, b_idx):
                    return CausalLMOutputWithPast(logits=[1.0],
                                past_key_values=past_key_values,
                                hidden_states=all_hidden_states,
                                exit_layer=b_idx)

        # for not-early-exit model
        return CausalLMOutputWithPast(logits=[1.0],
                                      past_key_values=past_key_values,
                                      hidden_states=all_hidden_states,
                                      exit_layer=b_idx)

    # Param Initialization, needed for device='meta' fast initialization
    def param_init_fn(self, module):
        init_fn_name = self.config.param_init_fn
        if self.config.verbose > 1:
            warnings.warn(f'Using {init_fn_name} initialization.')
        MODEL_INIT_REGISTRY[init_fn_name](module=module,
                                          **self.config.to_dict())

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module):
        return isinstance(module, GPTBlock)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module):
        return isinstance(module, GPTBlock)

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      inputs_embeds=None,
                                      **kwargs):
        if inputs_embeds is not None:
            raise NotImplementedError(
                'inputs_embeds is not implemented for MosaicGPT yet')

        attention_mask = kwargs['attention_mask'].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError(
                'MosaicGPT does not support generation with right padding.')

        if self.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None

        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if self.prefix_lm:
            # Leverage a convenience of sequential generation!
            prefix_mask = torch.ones_like(attention_mask)
            # This requires that we're using the cache
            if kwargs.get('use_cache') == False:
                raise NotImplementedError(
                    'MosaicGPT with prefix_lm=True does not support use_cache=False.'
                )
        else:
            prefix_mask = None

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prefix_mask': prefix_mask,
            'sequence_id': sequence_id,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache', True),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Used by HuggingFace generate when using beam search with kv-caching.

        See https://github.com/huggingface/transformers/blob/3ec7a47664ebe40c40f4b722f6bb1cd30c3821ec/src/transformers/models/gpt2/modeling_gpt2.py#L1122-L1133
        for an example in transformers.
        """
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past += [
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past)
            ]
        return reordered_past
