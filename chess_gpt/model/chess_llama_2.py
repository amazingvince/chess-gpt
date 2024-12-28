import torch
import torch.nn as nn

from typing import Optional, Tuple, Union, List, Any, Dict

# From Hugging Face Transformers
from transformers import (
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaConfig,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.utils import LossKwargs
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

# Weighted loss definitions (remove if you don't use them)
from chess_gpt.custom_liger.transformers.weighted_fused_linear_cross_entropy import (
    WeightedLigerFusedLinearCrossEntropyLoss,
)
from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)


# ---------------------------------------------------------------------
# Custom annotation combining flash-attn and custom loss kwargs
# ---------------------------------------------------------------------
class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    """
    Custom Kwargs class to unify FlashAttention and Loss arguments.
    """

    ...


# ---------------------------------------------------------------------
# ChessLlamaConfig
# ---------------------------------------------------------------------
class ChessLlamaConfig(LlamaConfig):
    """
    Configuration for ChessLlama, which extends LlamaConfig
    with an additional encoder config dict.
    """

    model_type = "chess_llama"

    def __init__(self, encoder_config: Optional[Dict] = None, **kwargs):
        if encoder_config is None:
            # Provide a minimal default
            encoder_config = {
                "hidden_size": 1024,
                "max_position_embeddings": 512,
                "num_attention_heads": 8,
                "intermediate_size": 4096,
                "num_hidden_layers": 4,
            }
        super().__init__(**kwargs)
        self.encoder_config = encoder_config


# ---------------------------------------------------------------------
# BertFenEncoder
# ---------------------------------------------------------------------
class BertFenEncoder(nn.Module):
    """
    A simple Transformer-based encoder for "fen" inputs.
    Returns a tensor of shape [batch_size, seq_len, hidden_size].
    """

    def __init__(self, config):
        super().__init__()
        # self.model_dtype = torch.bfloat16
        # self.layernorm_dtype = torch.float32

        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.num_hidden_layers = config.num_hidden_layers

        self.position_embeddings = nn.Embedding(
            self.max_position_embeddings, self.hidden_size
        )
        self.token_embeddings = nn.Embedding(80, self.hidden_size)  # Adjust as needed

        # Input layer norm
        self.layer_norm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.dropout = nn.Dropout(0.0)

        # Create encoder layer (no final norm inside)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_attention_heads,
            dim_feedforward=self.intermediate_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Create encoder, but remove the final norm
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_hidden_layers,
            norm=None,
        )

        # Separate final norm layer with explicit dtype
        self.final_norm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.residual = True

        # Define projection layers separately for better dtype control
        self.proj_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.proj_activation = nn.GELU()
        self.proj_norm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Move all parameters to model_dtype except LayerNorms

        # self.layer_norm.to(self.layernorm_dtype)
        # self.final_norm.to(self.layernorm_dtype)
        # self.proj_norm.to(self.layernorm_dtype)

    def forward(self, fen_input_ids, fen_attention_mask=None):
        device = fen_input_ids.device  # Get device from input
        seq_length = fen_input_ids.size(1)

        # Ensure all tensors are on the correct device
        position_ids = torch.arange(seq_length, device=device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand_as(fen_input_ids)

        token_embeddings = self.token_embeddings(fen_input_ids).to(device)
        position_embeddings = self.position_embeddings(position_ids).to(device)

        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        if fen_attention_mask is not None:
            attention_mask = fen_attention_mask == 0
        else:
            attention_mask = None

        encoded = self.encoder(embeddings, src_key_padding_mask=attention_mask)

        if self.residual:
            encoded = encoded + embeddings

        encoded = self.final_norm(encoded)
        out = self.proj_linear(encoded)
        out = self.proj_activation(out)
        out = self.proj_norm(out)

        return out


# ---------------------------------------------------------------------
# ChessLlamaForCausalLM
# ---------------------------------------------------------------------
class ChessLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    config_class = ChessLlamaConfig
    _keys_to_ignore_on_child_class = [
        "fen_input_ids",
        "fen_attention_mask",
    ]
    _tied_weights_keys = ["lm_head.weight"]
    main_input_name = "input_ids"

    def __init__(self, config: ChessLlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # FEN encoder
        self.fen_encoder = BertFenEncoder(config)
        self.encoder_projection = nn.Linear(
            config.encoder_config["hidden_size"], config.hidden_size
        )
        self.fen_prefix_added_for_generation = False
        self.post_init()  # from LlamaPreTrainedModel

    # ----------------- Optional freeze/unfreeze helpers ------------------
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

    # --------------------- Standard model methods ------------------------
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

    # ------------------------- Forward pass ------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        **loss_kwargs: KwargsForCausalLM,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if return_dict is None:
            return_dict = self.config.use_return_dict

        device = self.model.device

        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if fen_input_ids is not None:
            fen_input_ids = fen_input_ids.to(device)
        if fen_attention_mask is not None:
            fen_attention_mask = fen_attention_mask.to(device)

        # --------------------------------------------------------------
        # If we have a FEN prefix on the first pass (or training),
        # incorporate it into the embeddings, attention_mask, etc.
        # --------------------------------------------------------------
        if self.training and fen_input_ids is not None and inputs_embeds is None:
            device = input_ids.device
            # Convert input_ids to embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids).to(device)

            # Optionally adjust labels to account for new prefix positions
            if labels is not None:
                # Add -100 (ignore index) for the prefix
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
                ).to(labels.device)
            # Insert FEN prefix
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
        elif inputs_embeds is None and input_ids is not None:
            # On subsequent passes or if fen is not given
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # --------------------------------------------------------------
        # Pass through Llama
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Training: shift hidden states and labels
        # --------------------------------------------------------------
        if self.training and labels is not None:
            # Shift hidden_states/labels for next-token prediction
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten
            shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
            shift_labels = shift_labels.view(-1)

            # Weighted vs unweighted fused cross entropy
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

        # --------------------------------------------------------------
        # Inference: compute logits (optionally for the last token only)
        # --------------------------------------------------------------
        else:
            if past_key_values is not None:
                # Only compute the last token's logits
                hidden_states = hidden_states[:, -1:, :]
            elif num_logits_to_keep > 0:
                # Keep only the last `num_logits_to_keep` positions
                hidden_states = hidden_states[:, -num_logits_to_keep:, :]

            logits = self.lm_head(hidden_states)

            # If labels are provided in eval mode, compute some CE loss
            if labels is not None:
                # Here you could define a simpler CE loss or use HF's built-in
                # If you still want to use your custom fused loss, do:
                # (But you'll need to shift labels if needed.)
                loss = self.simple_eval_loss_function(
                    logits,
                    labels,
                    ignore_index=-100,  # or whatever you prefer
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # Simple evaluation loss (optional helper)
    # ------------------------------------------------------------------
    def simple_eval_loss_function(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        ignore_index: int = -100,
    ) -> torch.FloatTensor:
        """
        A basic cross-entropy for evaluation only (not fused).
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss

    # ------------------------------------------------------------------
    # Helper to add FEN prefix embedding and fix attention mask
    # ------------------------------------------------------------------

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
        """
        Encodes the FEN input via self.fen_encoder, takes the [0th] token
        as a "CLS" prefix, and concatenates it to the main `inputs_embeds`.
        Also updates the attention mask, position_ids, and cache_position.
        """
        # Get target device from inputs_embeds
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # Move all input tensors to the same device
        if fen_input_ids is not None:
            fen_input_ids = fen_input_ids.to(device)
        if fen_attention_mask is not None:
            fen_attention_mask = fen_attention_mask.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if cache_position is not None:
            cache_position = cache_position.to(device)

        # Ensure fen_encoder is on the correct device
        self.fen_encoder = self.fen_encoder.to(device)
        self.encoder_projection = self.encoder_projection.to(device)

        # Get FEN embeddings
        with torch.amp.autocast("cuda", enabled=torch.is_autocast_enabled()):
            fen_outputs = self.fen_encoder(
                fen_input_ids=fen_input_ids,
                fen_attention_mask=fen_attention_mask,
            )

            # Take the [0] token and project it
            cls_token = fen_outputs[:, 0, :]
            cls_token = self.encoder_projection(cls_token).unsqueeze(1)

            # Ensure same dtype as inputs_embeds
            cls_token = cls_token.to(inputs_embeds.device)
            cls_token = cls_token.to(dtype=dtype)

            # Concatenate embeddings
            inputs_embeds = torch.cat([cls_token, inputs_embeds], dim=1)

        # Handle attention mask
        if attention_mask is not None:
            fen_attn_mask = torch.ones(
                (attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device
            )
            attention_mask = torch.cat([fen_attn_mask, attention_mask], dim=1)

        # Handle position IDs
        if position_ids is None:
            position_ids = torch.arange(
                0, inputs_embeds.size(1), dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.size(0), -1)
        else:
            prefix_position_ids = torch.zeros(
                (position_ids.shape[0], 1), dtype=torch.long, device=device
            )
            position_ids = position_ids + 1  # Shift by +1
            position_ids = torch.cat([prefix_position_ids, position_ids], dim=1)

        # Handle cache position
        if cache_position is not None:
            cache_position = cache_position + 1

        return (
            inputs_embeds,
            attention_mask,
            position_ids,
            cache_position,
        )

    def reset_fen_prefix_flag(self):
        """
        Simple helper that you can call before each new generation
        to ensure we always re-inject the FEN prefix on the *first* step
        of each generation call.
        """
        self.fen_prefix_added_for_generation = False

    # ------------------------------------------------------------------
    # Prepare inputs for generation (beam search, etc.)
    # ------------------------------------------------------------------

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        fen_input_ids: Optional[torch.LongTensor] = None,
        fen_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Called automatically by .generate() each decoding step.
        We only want to add the FEN prefix on the *first* step, so we use
        the boolean `self.fen_prefix_added_for_generation`.
        """
        use_cache = kwargs.get("use_cache", True)

        # If no cache yet (past_key_values=None) *and* we haven't already
        # added the FEN prefix for this generation *and* fen_input_ids is provided:
        if (
            (past_key_values is None)
            and not self.fen_prefix_added_for_generation
            and (fen_input_ids is not None)
        ):
            # Convert your input_ids to embeddings
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            # Add the prefix once
            inputs_embeds, attention_mask, _, _ = self.add_fen_prefix(
                inputs_embeds,
                attention_mask,
                position_ids=None,
                fen_input_ids=fen_input_ids,
                fen_attention_mask=fen_attention_mask,
            )

            # Mark that we've done it
            self.fen_prefix_added_for_generation = True

            # In subsequent calls, we won’t pass fen_input_ids again
            fen_input_ids = None
            fen_attention_mask = None

        else:
            # On subsequent steps (or if prefix already added), we only embed the last token
            if (inputs_embeds is None) and (input_ids is not None):
                # If caching is off (use_cache=False), we might get the entire sequence each time,
                # but we do NOT want to re-inject FEN. So we only embed the last token here.
                inputs_embeds = self.get_input_embeddings()(input_ids[:, -1:])

            # If there's an attention_mask, slice it for the new token
            if (attention_mask is not None) and (attention_mask.shape[1] > 1):
                attention_mask = attention_mask[:, -1:]

        return {
            "input_ids": None,  # Because we have inputs_embeds
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "fen_input_ids": None,  # don’t pass these forward again
            "fen_attention_mask": None,
        }

    # ------------------------------------------------------------------
    # Reorder cache for beam search
    # ------------------------------------------------------------------
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
