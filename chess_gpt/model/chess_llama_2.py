import torch
import torch.nn as nn
from transformers import (
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaConfig,
    PretrainedConfig,
    ModernBertConfig,
    ModernBertModel,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

# Weighted loss definitions (remove if you don't use them)
from chess_gpt.custom_liger.transformers.weighted_fused_linear_cross_entropy import (
    WeightedLigerFusedLinearCrossEntropyLoss,
)
from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)

from typing import Optional, Tuple, Union, List, Any, Dict


# Custom annotation combining flash-attn and custom loss kwargs
class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class ChessLlamaConfig(PretrainedConfig):
    model_type = "chess_llama"

    def __init__(self, encoder_config=None, decoder_config=None, **kwargs):
        super().__init__(**kwargs)

        # Initialize with defaults if configs not provided
        encoder_config = encoder_config or {}
        decoder_config = decoder_config or {}

        self.encoder_config = ModernBertConfig(**encoder_config)
        self.decoder_config = LlamaConfig(**decoder_config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)


class ChessLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    config_class = ChessLlamaConfig
    _keys_to_ignore_on_child_class = [
        "fen_input_ids",
        "fen_attention_mask",
    ]
    _tied_weights_keys = ["lm_head.weight"]
    main_input_name = "input_ids"

    def __init__(self, config: ChessLlamaConfig):
        super().__init__(config.decoder_config)
        self.model = LlamaModel(config.decoder_config)
        self.vocab_size = config.decoder_config.vocab_size
        self.lm_head = nn.Linear(
            config.decoder_config.hidden_size,
            config.decoder_config.vocab_size,
            bias=False,
        )

        # FEN encoder
        self.fen_encoder = ModernBertModel(config.encoder_config)
        self.encoder_projection = nn.Linear(
            config.encoder_config.hidden_size, config.decoder_config.hidden_size
        )

        self.post_init()  # from LlamaPreTrainedModel

    # --------------------------------------------------------------------------------
    # Optional freeze/unfreeze helpers
    # --------------------------------------------------------------------------------
    def freeze_encoder(self):
        for param in self.fen_encoder.parameters():
            param.requires_grad = False
        for param in self.encoder_projection.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.fen_encoder.parameters():
            param.requires_grad = True
        for param in self.encoder_projection.parameters():
            param.requires_grad = True

    def freeze_decoder(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.lm_head.parameters():
            param.requires_grad = True

    # --------------------------------------------------------------------------------
    # Standard model methods
    # --------------------------------------------------------------------------------
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # --------------------------------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        fen_input_ids: Optional[torch.LongTensor] = None,
        fen_attention_mask: Optional[torch.LongTensor] = None,
        sample_weights: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Tuple]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # --------------------------------------------------------------------
        # Handle input embeddings
        # --------------------------------------------------------------------
        if inputs_embeds is None:
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            else:
                raise ValueError(
                    "You have to specify either input_ids or inputs_embeds"
                )

        # --------------------------------------------------------------------
        # If we have a FEN prefix on the first pass (or training),
        # incorporate it into the embeddings, attention_mask, etc.
        # --------------------------------------------------------------------
        if fen_input_ids is not None and (self.training or past_key_values is None):
            if labels is not None:
                # Add -100 (ignore index) to align with the new prefix token
                labels = torch.cat(
                    [
                        torch.full(
                            (labels.shape[0], 1),
                            -100,
                            dtype=labels.dtype,
                            device=labels.device,
                        ),
                        labels,
                    ],
                    dim=1,
                )
            inputs_embeds, attention_mask, position_ids, cache_position = (
                self.add_fen_prefix(
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    fen_input_ids,
                    fen_attention_mask,
                    cache_position,
                )
            )

        # Ensure contiguous
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.contiguous()
        if attention_mask is not None:
            attention_mask = attention_mask.contiguous()
        if position_ids is not None:
            position_ids = position_ids.contiguous()

        # --------------------------------------------------------------------
        # Pass through the Llama decoder
        # --------------------------------------------------------------------
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = None
        loss = None

        # --------------------------------------------------------------------
        # Training: compute the shift for labels + hidden states
        # --------------------------------------------------------------------
        if self.training and labels is not None:
            # Shift hidden_states/labels for next-token prediction
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
            shift_labels = shift_labels.view(-1)

            # logits = self.lm_head(shift_hidden_states)

            # loss = self.loss_function(
            #     logits=logits,
            #     labels=shift_labels,  # Use shifted labels
            #     vocab_size=self.config.vocab_size,
            #     **loss_kwargs,
            # )

            # # Handle weighted or unweighted loss
            reduction = "sum" if "num_items_in_batch" in loss_kwargs else "mean"
            if sample_weights is not None:
                loss_fct = WeightedLigerFusedLinearCrossEntropyLoss(reduction=reduction)
                loss = loss_fct(
                    self.lm_head.weight,
                    shift_hidden_states,
                    shift_labels,
                    sample_weights=sample_weights,
                )
            else:
                loss_fct = LigerFusedLinearCrossEntropyLoss(reduction=reduction)
                loss = loss_fct(
                    self.lm_head.weight,
                    shift_hidden_states,
                    shift_labels,
                )
        # --------------------------------------------------------------------
        # Inference: compute logits. Possibly only for the last token
        # --------------------------------------------------------------------
        else:
            if past_key_values is not None:
                # Only compute the last token's logits
                hidden_states = hidden_states[:, -1:, :]
            elif num_logits_to_keep > 0:
                # Keep only the last `num_logits_to_keep` positions
                hidden_states = hidden_states[:, -num_logits_to_keep:, :]

            logits = self.lm_head(hidden_states)

            # If labels are provided in eval mode, compute cross entropy
            if labels is not None:
                loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.vocab_size,
                    **loss_kwargs,
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # --------------------------------------------------------------------
        # Return
        # --------------------------------------------------------------------
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # --------------------------------------------------------------------------------
    # Helper to add FEN prefix embedding and fix attention mask
    # --------------------------------------------------------------------------------
    def add_fen_prefix(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        fen_input_ids: torch.LongTensor,
        fen_attention_mask: Optional[torch.LongTensor],
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor,
        Optional[torch.Tensor],
        Optional[torch.LongTensor],
        Optional[torch.LongTensor],
    ]:
        # Encode the fen_input_ids with ModernBert
        encoder_outputs = self.fen_encoder(
            input_ids=fen_input_ids,
            attention_mask=fen_attention_mask,
            return_dict=True,
        )
        # Use the [CLS] token or first hidden state as the prefix
        cls_token = encoder_outputs.last_hidden_state[:, 0, :]
        cls_token = self.encoder_projection(cls_token).unsqueeze(1)

        # --------------------------------------------------------------------
        # Add prefix to embeddings
        # --------------------------------------------------------------------
        inputs_embeds = torch.cat([cls_token, inputs_embeds], dim=1)

        # --------------------------------------------------------------------
        # Update the attention mask
        # (handle either a 2D or a 4D mask)
        # --------------------------------------------------------------------
        if attention_mask is not None:
            # If it's 2D shape (batch_size, seq_len)
            if attention_mask.dim() == 2:
                # Convert to the model’s dtype (e.g. float16/bfloat16/float32).
                # We'll assume we match inputs_embeds.dtype, which is typical in
                # half-precision training.
                attention_mask = attention_mask.to(inputs_embeds.dtype)

                # You might keep your 1s as keep-tokens or invert them:
                # e.g. 1 = keep, 0 = masked. For flash attention, we typically
                # convert (1 -> 0.0, 0 -> -inf). One approach:
                attention_mask = (1.0 - attention_mask) * torch.finfo(
                    attention_mask.dtype
                ).min

                # Now prepend a prefix mask
                prefix_mask = torch.zeros(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                # Prepend that zero row
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # If it's a 4D shape (used during generation)
            else:
                batch_size = attention_mask.shape[0]
                # We create a single zero column for the prefix
                prefix_mask = torch.zeros(
                    (batch_size, 1, 1, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                prefix_mask = prefix_mask.expand(-1, attention_mask.shape[1], 1, -1)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)

        # --------------------------------------------------------------------
        # Update position_ids
        # --------------------------------------------------------------------
        if position_ids is None:
            position_ids = (
                torch.arange(
                    0,
                    inputs_embeds.size(1),
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                .unsqueeze(0)
                .expand(inputs_embeds.size(0), -1)
            )
        else:
            prefix_position_ids = torch.zeros(
                (position_ids.shape[0], 1),
                dtype=torch.long,
                device=position_ids.device,
            )
            # Shift by +1
            position_ids = position_ids + 1
            position_ids = torch.cat([prefix_position_ids, position_ids], dim=1)

        # --------------------------------------------------------------------
        # Update cache_position if present
        # --------------------------------------------------------------------
        if cache_position is not None:
            cache_position = cache_position + 1

        return inputs_embeds, attention_mask, position_ids, cache_position

    # --------------------------------------------------------------------------------
    # Prepare inputs for generation (beam search, etc.)
    # --------------------------------------------------------------------------------
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        fen_input_ids: Optional[torch.LongTensor] = None,
        fen_attention_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        This method is used during generation to prepare inputs on each iteration.
        """
        first_pass = past_key_values is None or len(past_key_values) == 0
        batch_size = (
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )

        # --------------------------------------------------------------------
        # On the first pass, we may add the FEN prefix
        # --------------------------------------------------------------------
        if first_pass:
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            if fen_input_ids is not None:
                inputs_embeds, attention_mask, position_ids, cache_position = (
                    self.add_fen_prefix(
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        fen_input_ids,
                        fen_attention_mask,
                        cache_position,
                    )
                )

            # ----------------------------------------------------------------
            # Create a 4D causal mask for generation
            # (batch_size, 1, seq_len, seq_len)
            # ----------------------------------------------------------------
            if attention_mask is not None:
                # Make sure to match our model dtype
                attention_mask = attention_mask.to(inputs_embeds.dtype).contiguous()
                seq_length = attention_mask.shape[1]

                # Build an upper-triangular mask in the same dtype
                causal_mask = torch.triu(
                    torch.ones(
                        seq_length,
                        seq_length,
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                    diagonal=1,
                ).bool()

                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask.expand(batch_size, -1, seq_length, -1)
                # Fill with -inf where causal_mask is True
                attention_mask = attention_mask.masked_fill(
                    causal_mask, torch.finfo(attention_mask.dtype).min
                )
                attention_mask = attention_mask.contiguous()

        # --------------------------------------------------------------------
        # Subsequent passes (cached decoding)
        # --------------------------------------------------------------------
        else:
            # We'll only embed the last token if we have new input_ids
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids[:, -1:])

            # Slice out the last position for position_ids
            if position_ids is not None:
                position_ids = position_ids[:, -1:]
            else:
                # If we track a cache_position, increment it
                if cache_position is not None:
                    position_ids = cache_position.unsqueeze(-1)
                else:
                    # Otherwise, fallback to length-based
                    position_ids = torch.LongTensor([input_ids.shape[1] - 1]).to(
                        input_ids.device
                    )
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Make all relevant tensors contiguous
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.contiguous()
        if position_ids is not None:
            position_ids = position_ids.contiguous()

        # Bump cache_position by 1 each step after first pass
        if cache_position is not None and not first_pass:
            cache_position = cache_position + 1

        # Assemble dictionary to feed the next forward call
        model_inputs = {
            "input_ids": input_ids if first_pass else None,
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "use_cache": kwargs.get("use_cache", True),
            "fen_input_ids": fen_input_ids if first_pass else None,
            "fen_attention_mask": fen_attention_mask if first_pass else None,
        }

        return model_inputs

    # --------------------------------------------------------------------------------
    # Reorder cache for beam search
    # --------------------------------------------------------------------------------
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """Reorder the cache for beam search."""
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )
