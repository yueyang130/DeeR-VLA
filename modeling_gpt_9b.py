"""A simple, flexible implementation of a GPT model.

Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""
from __future__ import annotations
import math
import warnings
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import is_flash_v1_installed, is_flash_v2_installed
if is_flash_v2_installed():
    try:
        from flash_attn import bert_padding
        from flash_attn.layers.rotary import RotaryEmbedding as DAILRotaryEmbedding
    except Exception as e:
        raise e
if is_flash_v1_installed():
    try:
        from flash_attn import bert_padding
    except Exception as e:
        raise e
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import ModelOutput, dataclass
from transformers.models.llama.modeling_llama import LlamaDynamicNTKScalingRotaryEmbedding as HFDynamicNTKScalingRotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaLinearScalingRotaryEmbedding as HFLinearScalingRotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFRotaryEmbedding
from .attention import ATTN_CLASS_REGISTRY, attn_bias_shape, build_attn_bias, gen_slopes
from .blocks import MPTBlock
from .custom_embedding import SharedEmbedding
from .fc import FC_CLASS_REGISTRY as FC_CLASS_REGISTRY
from .ffn import FFN_CLASS_REGISTRY as FFN_CLASS_REGISTRY
from .ffn import MPTMLP as MPTMLP
from .ffn import build_ffn as build_ffn
from .norm import NORM_CLASS_REGISTRY
from .configuration_mpt import MPTConfig
from .adapt_tokenizer import AutoTokenizerForMOD, adapt_tokenizer_for_denoising
from .hf_prefixlm_converter import add_bidirectional_mask_if_missing, convert_hf_causal_lm_to_prefix_lm
from .meta_init_context import init_empty_weights
from .param_init_fns import generic_param_init_fn_, MODEL_INIT_REGISTRY
try:
    from .flash_attn_triton import flash_attn_func as flash_attn_func
except:
    pass
import logging
log = logging.getLogger(__name__)

def gen_rotary_embedding(rope_head_dim: int, rope_impl: str, rope_theta: int, rope_dail_config: dict, rope_hf_config: dict, max_seq_len: int):
    if rope_impl == 'dail':
        return DAILRotaryEmbedding(dim=rope_head_dim, base=rope_theta, interleaved=False, scale_base=rope_dail_config['xpos_scale_base'] if rope_dail_config['type'] == 'xpos' else None, pos_idx_in_fp32=rope_dail_config['pos_idx_in_fp32'], device='cpu')
    elif rope_impl == 'hf':
        if rope_hf_config['type'] == 'no_scaling':
            return HFRotaryEmbedding(rope_head_dim, max_position_embeddings=max_seq_len, base=rope_theta, device='cpu')
        elif rope_hf_config['type'] == 'linear':
            return HFLinearScalingRotaryEmbedding(rope_head_dim, max_position_embeddings=max_seq_len, base=rope_theta, scaling_factor=rope_hf_config['factor'], device='cpu')
        elif rope_hf_config['type'] == 'dynamic':
            return HFDynamicNTKScalingRotaryEmbedding(rope_head_dim, max_position_embeddings=max_seq_len, base=rope_theta, scaling_factor=rope_hf_config['factor'], device='cpu')
    raise ValueError('rope_impl needs to be either dail or hf')

def gen_attention_mask_in_length(sequence_id: Union[None, torch.Tensor], S: int, attn_uses_sequence_id: bool, attn_impl: str, attention_mask: Union[torch.Tensor, None]):
    """Generates the attention mask used for sequence masking in FA v2.

    Only supports sequence id based sparse attention for no attention masking or attention masking with right padding.
    In case of left padding:
        1. Training with left padding is not supported in MPT (see https://github.com/mosaicml/llm-foundry/blob/1eecd4cb8e734499f77f6a35f657b8b20c0adfcb/llmfoundry/models/mpt/modeling_mpt.py#L407).
        2. For generation with left padding, we only have a single sequence id per sample, so we don't need sequence id based sparse attention.

    Args:
        sequence_id (Union[None, torch.Tensor]): Tensor containing the sequence id for each token. Shape (batch_size, seq_len).
        S (int): Sequence length
        attn_uses_sequence_id (bool): Whether the attention uses sequence id based masking.
        attn_impl (str): Attention implementation. This function is only creates attention_mask_in_length for flash attention.
        attention_mask (Union[torch.Tensor, None]): Attention mask tensor of shape (batch_size, seq_len)

    Returns:
        attention_mask_in_length: (batch, seqlen), int, a nonzero number (e.g., 1, 2, 3, etc.) means length of concatenated sequence in b-th batch, and 0 means none. For example, if batch = 3 and seqlen = 6, the attention_mask_in_length is:
            ```
            [
            [2, 3, 0, 0, 0, 0],
            [3, 2, 0, 0, 0, 0],
            [6, 0, 0, 0, 0, 0]
            ]
            ```
        , which refers to the 3D-attention mask:
            ```
            [
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ],
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ],
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1]
            ]
            ]
            ```.
            (The description above is taken verbatim from https://github.com/Dao-AILab/flash-attention/blob/9356a1c0389660d7e231ff3163c1ac17d9e3824a/flash_attn/bert_padding.py#L125 .)
    """
    attention_mask_in_length = None
    if sequence_id is not None and attn_uses_sequence_id and (attn_impl == 'flash'):
        if attention_mask is not None and attention_mask[:, 0].sum() != attention_mask.shape[0]:
            raise NotImplementedError('Left padding is not supported with flash attention when attn_uses_sequence_id is set to True.')
        if S != sequence_id.shape[-1]:
            raise ValueError(f'Sequence length ({S}) does not match length of sequences in sequence_id ({sequence_id.shape[-1]}).')
        if attention_mask is not None:
            sequence_id = sequence_id.masked_fill(~attention_mask, 0)
        attention_mask_in_length = torch.nn.functional.one_hot(sequence_id)
        if attention_mask is not None:
            attention_mask_in_length = attention_mask_in_length.masked_fill(~attention_mask.unsqueeze(-1), 0)
        attention_mask_in_length = attention_mask_in_length.sum(dim=1)
        attention_mask_in_length = torch.nn.functional.pad(attention_mask_in_length, (0, S - attention_mask_in_length.shape[-1]), mode='constant', value=0)
    return attention_mask_in_length

def gen_flash_attn_padding_info(bsz: int, S: int, past_key_len: int, device: torch.device, attention_mask_in_length: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None):
    flash_attn_padding_info = {}
    if attention_mask_in_length is None:
        key_padding_mask = attention_mask
        if key_padding_mask is None:
            key_padding_mask = torch.ones((bsz, past_key_len + S), dtype=torch.bool, device=device)
        query_padding_mask = key_padding_mask[:, -S:]
        unpadding_function = bert_padding.unpad_input
    else:
        key_padding_mask = attention_mask_in_length
        query_padding_mask = attention_mask_in_length
        unpadding_function = bert_padding.unpad_input_for_concatenated_sequences
    (_, indices_q, cu_seqlens_q, max_seqlen_q) = unpadding_function(torch.empty(bsz, S, 1, device=device), query_padding_mask)
    (_, indices_k, cu_seqlens_k, max_seqlen_k) = unpadding_function(torch.empty(bsz, past_key_len + S, 1, device=device), key_padding_mask)
    (_, indices_v, _, _) = unpadding_function(torch.empty(bsz, past_key_len + S, 1, device=device), key_padding_mask)
    flash_attn_padding_info['indices_q'] = indices_q
    flash_attn_padding_info['indices_k'] = indices_k
    flash_attn_padding_info['indices_v'] = indices_v
    flash_attn_padding_info['cu_seqlens_q'] = cu_seqlens_q
    flash_attn_padding_info['cu_seqlens_k'] = cu_seqlens_k
    flash_attn_padding_info['max_seqlen_q'] = max_seqlen_q
    flash_attn_padding_info['max_seqlen_k'] = max_seqlen_k
    return flash_attn_padding_info

def apply_sequence_id(attn_bias: torch.Tensor, sequence_id: torch.LongTensor, max_seq_len: int) -> torch.Tensor:
    seq_len = sequence_id.shape[-1]
    if seq_len > max_seq_len:
        raise ValueError(f'sequence_id sequence length cannot exceed max_seq_len={max_seq_len}')
    attn_bias = attn_bias[..., :seq_len, :seq_len]
    cannot_attend = torch.logical_not(torch.eq(sequence_id.view(-1, seq_len, 1), sequence_id.view(-1, 1, seq_len))).unsqueeze(1)
    min_val = torch.finfo(attn_bias.dtype).min
    attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
    return attn_bias

@dataclass
class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
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

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    exit_layer: Optional[int] = None
    
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


class MPTPreTrainedModel(PreTrainedModel):
    config_class = MPTConfig
    base_model_prefix = 'model'
    _no_split_modules = ['MPTBlock']

def _fsdp_wrap_fn(self: Union[MPTModel, MPTForCausalLM], module: nn.Module) -> bool:
    return isinstance(module, MPTBlock)

class MPTModel(MPTPreTrainedModel):

    def __init__(self, config: MPTConfig):
        config._validate_config()
        super().__init__(config)
        self.attn_impl = config.attn_config['attn_impl']
        self.prefix_lm = config.attn_config['prefix_lm']
        self.attn_uses_sequence_id = config.attn_config['attn_uses_sequence_id']
        self.alibi = config.attn_config['alibi']
        self.alibi_bias_max = config.attn_config['alibi_bias_max']
        self.learned_pos_emb = config.learned_pos_emb
        if config.init_device == 'mixed':
            if dist.get_local_rank() == 0:
                config.init_device = 'cpu'
            else:
                config.init_device = 'meta'
        if config.norm_type.lower() not in NORM_CLASS_REGISTRY.keys():
            norm_options = ' | '.join(NORM_CLASS_REGISTRY.keys())
            raise NotImplementedError(f'Requested norm type ({config.norm_type}) is not implemented within this repo (Options: {norm_options}).')
        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]
        self.embedding_fraction = config.embedding_fraction
        self.wte = SharedEmbedding(config.vocab_size, config.d_model, device=config.init_device)
        if self.learned_pos_emb:
            self.wpe = torch.nn.Embedding(config.max_seq_len, config.d_model, device=config.init_device)
        self.emb_drop = nn.Dropout(config.emb_pdrop)
        self.blocks = nn.ModuleList([MPTBlock(device=config.init_device, **config.to_dict()) for _ in range(config.n_layers)])
        self.norm_f = norm_class(config.d_model, device=config.init_device)
        self.rope = config.attn_config['rope']
        self.rope_impl = None
        if self.rope:
            self.rope_impl = config.attn_config['rope_impl']
            self.rotary_embedding = gen_rotary_embedding(rope_head_dim=config.d_model // config.n_heads, rope_impl=self.rope_impl, rope_theta=config.attn_config['rope_theta'], rope_dail_config=config.attn_config['rope_dail_config'], rope_hf_config=config.attn_config['rope_hf_config'], max_seq_len=self.config.max_seq_len)
        if config.init_device != 'meta':
            log.info(f'We recommend using config.init_device="meta" with Composer + FSDP for faster initialization.')
            self.apply(self.param_init_fn)
        self.is_causal = not self.prefix_lm
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attn_bias_shape(self.attn_impl, config.n_heads, config.max_seq_len, self.alibi, prefix_lm=self.prefix_lm, causal=self.is_causal, use_sequence_id=self.attn_uses_sequence_id)
        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                    log.info(f'Removing bias from module={module!r}.')
                    module.register_parameter('bias', None)
                if hasattr(module, 'use_bias'):
                    log.info(f'Setting use_bias=False for module={module!r}.')
                    module.use_bias = False
        log.debug(self)
        log.debug(f"Using {self.config.init_config['name']} initialization.")

    def get_input_embeddings(self) -> Union[SharedEmbedding, nn.Embedding]:
        return self.wte

    def set_input_embeddings(self, value: Union[SharedEmbedding, nn.Embedding]) -> None:
        self.wte = value

    @torch.no_grad()
    def _attn_bias(self, device: torch.device, dtype: torch.dtype, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None) -> Tuple[Optional[torch.Tensor], Optional[torch.ByteTensor]]:
        if not self._attn_bias_initialized:
            if self.attn_bias_shape:
                self.attn_bias = torch.zeros(self.attn_bias_shape, device=device, dtype=dtype)
                self.attn_bias = build_attn_bias(self.attn_impl, self.attn_bias, self.config.n_heads, self.config.max_seq_len, causal=self.is_causal, alibi=self.alibi, alibi_bias_max=self.alibi_bias_max)
            self._attn_bias_initialized = True
        if self.attn_impl == 'flash':
            return (self.attn_bias, attention_mask)
        if self.attn_bias is not None:
            self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)
        attn_bias = self.attn_bias
        if self.prefix_lm:
            assert isinstance(attn_bias, torch.Tensor)
            assert isinstance(prefix_mask, torch.Tensor)
            attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)
        if self.attn_uses_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)
            attn_bias = apply_sequence_id(attn_bias, sequence_id, self.config.max_seq_len)
        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
            else:
                _s_k = max(0, attn_bias.size(-1) - s_k)
                attn_bias = attn_bias[:, :, :, _s_k:]
            if prefix_mask is not None and attention_mask.shape != prefix_mask.shape:
                raise ValueError(f'attention_mask shape={attention_mask.shape} ' + f'and prefix_mask shape={prefix_mask.shape} are not equal.')
            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(~attention_mask.view(-1, 1, 1, s_k), min_val)
        return (attn_bias, attention_mask)

    def _apply_prefix_mask(self, attn_bias: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
        (s_k, s_q) = attn_bias.shape[-2:]
        if s_k != self.config.max_seq_len or s_q != self.config.max_seq_len:
            raise ValueError('attn_bias does not match the expected shape. ' + f'The last two dimensions should both be {self.config.max_length} ' + f'but are {s_k} and {s_q}.')
        seq_len = prefix_mask.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(f'prefix_mask sequence length cannot exceed max_seq_len={self.config.max_seq_len}')
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=prefix_mask.device)).view(1, 1, seq_len, seq_len)
        prefix = prefix_mask.view(-1, 1, 1, seq_len)
        cannot_attend = ~torch.logical_or(causal, prefix.bool())
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        return attn_bias

    def forward(self, 
                input_ids: Optional[torch.LongTensor]=None, 
                past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, 
                attention_mask: Optional[torch.ByteTensor]=None,
                prefix_mask: Optional[torch.ByteTensor]=None, 
                sequence_id: Optional[torch.LongTensor]=None, 
                return_dict: Optional[bool]=None, 
                output_attentions: Optional[bool]=None, 
                output_hidden_states: Optional[bool]=None, 
                use_cache: Optional[bool]=None, 
                inputs_embeds: Optional[torch.Tensor]=None,
                exit_controller = None,
                exit_id = None,
                eval_flop = False,
                eval_time = False,
                ) -> BaseModelOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        if prefix_mask is not None:
            prefix_mask = prefix_mask.bool()
        if not return_dict:
            raise NotImplementedError('return_dict False is not implemented yet for MPT')
        if output_attentions:
            if self.attn_impl != 'torch':
                raise NotImplementedError('output_attentions is not implemented for MPT when using attn_impl `flash` or `triton`.')
        if self.training and attention_mask is not None and (attention_mask[:, 0].sum() != attention_mask.shape[0]):
            raise NotImplementedError('MPT does not support training with left padding.')
        if self.prefix_lm and prefix_mask is None:
            raise ValueError('prefix_mask is a required argument when MPT is configured with prefix_lm=True.')
        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError('sequence_id is a required argument when MPT is configured with attn_uses_sequence_id=True ' + 'and the model is in train mode.')
            elif self.attn_uses_sequence_id is False and sequence_id is not None:
                warnings.warn('MPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. ' + 'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.')
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds.')
        elif input_ids is not None:
            bsz = input_ids.size(0)
            S = input_ids.size(1)
            x = self.wte(input_ids)
            input_device = input_ids.device
        elif inputs_embeds is not None:
            bsz = inputs_embeds.size(0)
            S = inputs_embeds.size(1)
            x = inputs_embeds
            input_device = inputs_embeds.device
        else:
            raise ValueError('You must specify input_ids or inputs_embeds')
        assert S <= self.config.max_seq_len, f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'
        rotary_emb_w_meta_info = None
        past_position = 0
        if past_key_values is not None:
            if len(past_key_values) != self.config.n_layers:
                raise ValueError(f'past_key_values must provide a past_key_value for each attention ' + f'layer in the network (len(past_key_values)={len(past_key_values)!r}; self.config.n_layers={self.config.n_layers!r}).')
            past_position = past_key_values[0][0].size(1)
            if self.attn_impl == 'torch':
                past_position = past_key_values[0][0].size(3)
        if self.learned_pos_emb or self.rope:
            if self.learned_pos_emb and S + past_position > self.config.max_seq_len:
                raise ValueError(f'Cannot forward input with past sequence length {past_position} and current sequence length ' + f'{S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.')
            if self.learned_pos_emb or (self.rope and self.rope_impl == 'hf'):
                pos = torch.arange(past_position, S + past_position, dtype=torch.long, device=input_device).unsqueeze(0)
                if attention_mask is not None:
                    pos = torch.clamp(pos - torch.cumsum((~attention_mask).to(torch.int32), dim=1)[:, past_position:], min=0)
                if self.learned_pos_emb:
                    x = x + self.wpe(pos)
                elif self.rope and self.rope_impl == 'hf':
                    rotary_emb_w_meta_info = {'impl': self.rope_impl, 'rotary_emb': self.rotary_embedding, 'offset_info': pos, 'seq_len': S + past_position}
            elif self.rope and self.rope_impl == 'dail':
                rotary_emb_w_meta_info = {'impl': self.rope_impl, 'rotary_emb': self.rotary_embedding, 'offset_info': past_position, 'seq_len': S + past_position}
        if self.embedding_fraction == 1:
            x = self.emb_drop(x)
        else:
            x_shrunk = x * self.embedding_fraction + x.detach() * (1 - self.embedding_fraction)
            assert isinstance(self.emb_drop, nn.Module)
            x = self.emb_drop(x_shrunk)
        (attn_bias, attention_mask) = self._attn_bias(device=x.device, dtype=torch.float32, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id)
        attention_mask_in_length = gen_attention_mask_in_length(sequence_id=sequence_id, S=S, attn_uses_sequence_id=self.attn_uses_sequence_id, attn_impl=self.attn_impl, attention_mask=attention_mask)
        alibi_slopes = None
        if self.alibi and self.attn_impl == 'flash':
            alibi_slopes = gen_slopes(n_heads=self.config.n_heads, alibi_bias_max=self.alibi_bias_max, device=x.device, return_1d=True)
        presents = () if use_cache else None
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)]
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        flash_attn_padding_info = {}
        if self.attn_impl == 'flash':
            flash_attn_padding_info = gen_flash_attn_padding_info(bsz, S, past_position, x.device, attention_mask_in_length, attention_mask)
        
        
        for (b_idx, block) in enumerate(self.blocks):
            past_key_value = past_key_values[b_idx] if past_key_values is not None else None
            
            
            if eval_flop:
                print('revised')
                from fvcore.nn import FlopCountAnalysis
                from thop import profile
                vis_per_flop = FlopCountAnalysis(block, inputs=(x, past_key_value, attn_bias, rotary_emb_w_meta_info, attention_mask, self.is_causal,
                                                    bool(output_attentions), alibi_slopes, flash_attn_padding_info)).total()
                print(f'fvflop LLM layer {b_idx} flops = {vis_per_flop/1e9:.1f}G, ')
                vis_per_flop = profile(block, inputs=(x, past_key_value, attn_bias, rotary_emb_w_meta_info, attention_mask, self.is_causal,
                                                    bool(output_attentions), alibi_slopes, flash_attn_padding_info))[0]
                print(f'thop LLM layer {b_idx} flops = {vis_per_flop/1e9:.1f}G, ')
            
            if eval_time:
                import time
                torch.cuda.synchronize()
                cur_time = time.time()
            
            (x, attn_weights, present) = block(x, past_key_value=past_key_value, attn_bias=attn_bias, rotary_emb_w_meta_info=rotary_emb_w_meta_info, 
                                               attention_mask=attention_mask, is_causal=self.is_causal, 
                                               output_attentions=bool(output_attentions), alibi_slopes=alibi_slopes,
                                               flash_attn_padding_info=flash_attn_padding_info)
            
            if eval_time:
                torch.cuda.synchronize()
                print(f"LLM {b_idx} layer time: {cur_time-time.time():.4f} seconds")
                cur_time = time.time()
                
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (x,)
            
            
            if presents is not None:
                presents += (present,)
            if output_attentions:
                assert all_self_attns is not None
                all_self_attns = all_self_attns + (attn_weights,)
                
                
            if exit_id is not None:
                if exit_id == b_idx:
                    return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attns,
                                                   exit_layer=b_idx)
                
            if exit_controller is not None:
                if exit_controller(all_hidden_states, b_idx):
                    return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attns,
                                                   exit_layer=b_idx)  
        
        
        x = self.norm_f(x)
        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states = all_hidden_states + (x,)
        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attns,
                                exit_layer=b_idx)

    def param_init_fn(self, module: nn.Module) -> None:
        init_fn_name = self.config.init_config['name']
        MODEL_INIT_REGISTRY[init_fn_name](module=module, n_layers=self.config.n_layers, d_model=self.config.d_model, **self.config.init_config)

    def fsdp_wrap_fn(self, module: nn.Module) -> bool:
        return _fsdp_wrap_fn(self, module)

    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        return isinstance(module, MPTBlock)

class MPTForCausalLM(MPTPreTrainedModel):

    def __init__(self, config: MPTConfig):
        super().__init__(config)
        log.info(f'Instantiating an MPTForCausalLM model from {__file__}')
        self.transformer: MPTModel = MPTModel(config)
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, device=config.init_device)
            self.lm_head._fsdp_wrap = True
        for child in self.transformer.children():
            if isinstance(child, torch.nn.ModuleList):
                continue
            if isinstance(child, torch.nn.Module):
                child._fsdp_wrap = True
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.")
            self.logit_scale = logit_scale

    def get_input_embeddings(self) -> Union[SharedEmbedding, nn.Embedding]:
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value: Union[SharedEmbedding, nn.Embedding]) -> None:
        self.transformer.set_input_embeddings(value)

    def get_output_embeddings(self) -> Union[SharedEmbedding, nn.Embedding, nn.Linear]:
        if self.lm_head is not None:
            return self.lm_head
        return self.transformer.get_input_embeddings()

    def set_output_embeddings(self, new_embeddings: Union[SharedEmbedding, nn.Embedding, nn.Linear]) -> None:
        if self.lm_head is not None:
            self.lm_head = new_embeddings
        else:
            if not isinstance(new_embeddings, (SharedEmbedding, nn.Embedding)):
                raise ValueError('new_embeddings must be an instance of SharedEmbedding ' + f'or nn.Embedding, but got {type(new_embeddings)}.')
            warnings.warn('Using `set_output_embeddings` to set the embedding layer of ' + 'MPTForCausalLM with tied weights. Given weights are tied, ' + 'using `set_input_embeddings` is recommended over using ' + '`set_output_embeddings`.')
            self.transformer.set_input_embeddings(new_embeddings)

    def tie_weights(self) -> None:
        self.lm_head = None

    def set_decoder(self, decoder: MPTModel) -> None:
        self.transformer = decoder

    def get_decoder(self) -> MPTModel:
        return self.transformer

    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None, inputs_embeds: Optional[torch.FloatTensor]=None,
            exit_controller = None,
            exit_id = None,) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache, inputs_embeds=inputs_embeds,
            exit_controller = exit_controller,
            exit_id = exit_id)
        if self.lm_head is not None:
            logits = self.lm_head(outputs.last_hidden_state)
        else:
            out = outputs.last_hidden_state
            out = out.to(self.transformer.wte.weight.device)
            logits = self.transformer.wte(out, True)
        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(f'Multiplying logits by self.logit_scale={self.logit_scale!r}. This will produce uniform (uninformative) outputs.')
            logits *= self.logit_scale
        loss = None
        if labels is not None:
            _labels = torch.roll(labels, shifts=-1)
            _labels[:, -1] = -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), _labels.to(logits.device).view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
                                      exit_layer=outputs.exit_layer)

    def param_init_fn(self, module: nn.Module) -> None:
        init_fn_name = self.config.init_config['name']
        MODEL_INIT_REGISTRY[init_fn_name](module=module, n_layers=self.config.n_layers, d_model=self.config.d_model, **self.config.init_config)

    def fsdp_wrap_fn(self, module: nn.Module) -> bool:
        return _fsdp_wrap_fn(self, module)

    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        act_ckpt_list = getattr(self.config, 'activation_checkpointing_target', None) or ['MPTBlock']
        if isinstance(act_ckpt_list, str):
            act_ckpt_list = [act_ckpt_list]
        elif not isinstance(act_ckpt_list, list):
            raise ValueError(f'activation_checkpointing_target must be either a single string or a list, but got {type(act_ckpt_list)}')
        if 'MPTBlock' in act_ckpt_list or 'mptblock' in act_ckpt_list:
            if len(act_ckpt_list) > 1:
                log.info('Activation checkpointing MPTBlock only (ignoring other sub-block modules specified in activation_checkpointing_target).')
            return isinstance(module, MPTBlock)
        mod_types = ()
        for mod_name in act_ckpt_list:
            if mod_name.lower() == 'mptblock':
                mod_types += (MPTBlock,)
            elif mod_name in ATTN_CLASS_REGISTRY:
                mod_types += (ATTN_CLASS_REGISTRY[mod_name],)
            elif mod_name in FFN_CLASS_REGISTRY:
                mod_types += (FFN_CLASS_REGISTRY[mod_name],)
            elif mod_name in NORM_CLASS_REGISTRY:
                mod_types += (NORM_CLASS_REGISTRY[mod_name],)
            else:
                msg = ', '.join(list(ATTN_CLASS_REGISTRY.keys()) + list(FFN_CLASS_REGISTRY.keys()) + list(NORM_CLASS_REGISTRY.keys()) + ['MPTBlock'])
                raise ValueError(f'{mod_name} (specified in activation_checkpointing_target) is not a recognized option out of available options {msg}.')
        return isinstance(module, mod_types)

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None, inputs_embeds: Optional[torch.Tensor]=None, **kwargs: Any) -> Dict[str, Any]:
        attention_mask = kwargs['attention_mask'].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError('MPT does not support generation with right padding.')
        if self.transformer.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if self.transformer.prefix_lm:
            prefix_mask = torch.ones_like(attention_mask)
            if kwargs.get('use_cache') == False:
                raise NotImplementedError('MPT with prefix_lm=True does not support use_cache=False.')
        else:
            prefix_mask = None
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        model_inputs.update({'attention_mask': attention_mask, 'prefix_mask': prefix_mask, 'sequence_id': sequence_id, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache', True)})
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values: List[Tuple[torch.Tensor, torch.Tensor]], beam_idx: torch.LongTensor) -> List[Tuple[torch.Tensor, ...]]:
        """Used by HuggingFace generate when using beam search with kv-caching.

        See https://github.com/huggingface/transformers/blob/3ec7a47664ebe40c40f4b722f6bb1cd30c3821ec/src/transformers/models/gpt2/modeling_gpt2.py#L1122-L1133
        for an example in transformers.
        """
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past += [tuple((past_state.index_select(0, beam_idx) for past_state in layer_past))]
        return reordered_past