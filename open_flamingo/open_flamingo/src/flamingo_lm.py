import torch.nn as nn
from .helpers import GatedCrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive
import copy

class FlamingoLayer(nn.Module):
    """
    FlamingoLayer is a wrapper around the GatedCrossAttentionBlock and DecoderLayer.
    """

    def __init__(
        self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False, residual=False
    ):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None
        self.residual = residual

        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = (
                gradient_checkpointing
            )
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def clone_parameters(self):
        self.res_layer = copy.deepcopy(self.gated_cross_attn_layer)
        if self.res_layer is not None:
            self.res_layer.requires_grad_(False)

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None and self.media_locations is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        self.media_locations = media_locations

    def condition_use_cached_media(self, use_cached_media):
        self.use_cached_media = use_cached_media

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Cross attention
        if self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")

            if self.media_locations is None:
                raise ValueError(
                    "media_locations must be conditioned before forward pass"
                )

            lang_x = self.gated_cross_attn_layer(
                lang_x,
                self.vis_x,
                media_locations=self.media_locations,
                use_cached_media=self.use_cached_media,
            )
            
            # Residual
            if self.residual and self.res_layer is not None:
                lang_x_res = self.res_layer(
                    lang_x,
                    self.vis_x,
                    media_locations=self.media_locations,
                    attend_previous=self.attend_previous,
                )
                lang_x = (lang_x + lang_x_res) / 2.0

        # Normal decoder layer
        lang_x = self.decoder_layer(
            lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
        )
        return lang_x

    # def forward(
    #     self,
    #     lang_x,
    #     past_key_value=None,
    #     attn_bias=None,
    #     attention_mask=None,
    #     is_causal: bool = True,
    # ):
    #     # Cross attention
    #     if self.gated_cross_attn_layer is not None:
    #         if self.vis_x is None:
    #             raise ValueError("vis_x must be conditioned before forward pass")

    #         if self.media_locations is None:
    #             raise ValueError(
    #                 "media_locations must be conditioned before forward pass"
    #             )

    #         lang_x = self.gated_cross_attn_layer(
    #             lang_x,
    #             self.vis_x,
    #             media_locations=self.media_locations,
    #             use_cached_media=self.use_cached_media,
    #         )
            
    #         # Residual
    #         if self.residual and self.res_layer is not None:
    #             lang_x_res = self.res_layer(
    #                 lang_x,
    #                 self.vis_x,
    #                 media_locations=self.media_locations,
    #                 attend_previous=self.attend_previous,
    #             )
    #             lang_x = (lang_x + lang_x_res) / 2.0

    #     # Normal decoder layer
    #     lang_x = self.decoder_layer(
    #         lang_x, 
    #         past_key_value=past_key_value,
    #         attn_bias=attn_bias,
    #         attention_mask=attention_mask,
    #         is_causal=is_causal
    #     )
    #     return lang_x


class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """
    
    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)
        
    def _delete_decoder_layers(self, indices):
        indices = sorted(indices, reverse=True)
        print(f'deleting layers {indices} in Flamingo...')
        layers = self._get_decoder_layers()
        for i in indices:
            del layers[i]
            del self.gated_cross_attn_layers[i]
            del self.old_decoder_blocks[i] # original language self-attention layers
        self.config.n_layers = len(self._get_decoder_layers())
        print(f'Now the number of layer is {len(self._get_decoder_layers())}')

    def init_flamingo(
        self,
        media_token_id,
        lang_hidden_size,
        vis_hidden_size,
        cross_attn_every_n_layers,
        gradient_checkpointing,
        residual=False,
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        print('-'*100)
        print(self.decoder_layers_attr_name)
        self.old_decoder_blocks = self._get_decoder_layers()
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    dim=lang_hidden_size, dim_visual=vis_hidden_size
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self.init_flamingo_layers(gradient_checkpointing, residual=residual)
        self.media_token_id = media_token_id
        self.initialized_flamingo = True
        self._use_cached_vision_x = False

    def init_flamingo_layers(self, gradient_checkpointing, residual=False):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    FlamingoLayer(
                        gated_cross_attn_layer, decoder_layer, gradient_checkpointing, residual=residual
                    )
                    for gated_cross_attn_layer, decoder_layer in zip(
                        self.gated_cross_attn_layers, self.old_decoder_blocks
                    )
                ]
            )
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        media_locations = input_ids == self.media_token_id

        # if there are media already cached and we're generating and there are no media tokens in the input,
        # we'll assume that ALL input tokens should attend to the last previous media that is cached.
        # this is especially important for HF generate() compatibility, since generate() calls forward()
        # repeatedly one token at a time (with no media tokens).
        # without this check, the model would not attend to any images when generating (after the first token)
        use_cached_media_locations = (
            self._use_cached_vision_x
            and self.is_conditioned()
            and not media_locations.any()
        )

        for layer in self._get_decoder_layers():
            if not use_cached_media_locations:
                layer.condition_media_locations(media_locations)
            layer.condition_use_cached_media(use_cached_media_locations)

        # package arguments for the other parent's forward. since we don't know the order of the arguments,
        # make them all kwargs
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        return super().forward(**kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clone_parameters(self):
        for layer in self._get_decoder_layers():
            layer.clone_parameters()

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_use_cached_media(None)
            
