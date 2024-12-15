import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig, BertModel, BertConfig

class ChessLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, fen_encoder_name_or_path):
        super().__init__(config)
        # Load or initialize the fen encoder (BERT-style)
        if fen_encoder_name_or_path:
            fen_config = BertConfig.from_pretrained(fen_encoder_name_or_path)
            self.fen_encoder = BertModel.from_pretrained(fen_encoder_name_or_path)
        else:
            fen_config = BertConfig(hidden_size=config.hidden_size)
            self.fen_encoder = BertModel(fen_config)

    def forward(
        self,
        input_ids=None,
        # fen_positions=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        # The logic here closely mirrors LlamaForCausalLM but introduces custom positional embeddings.

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.size()
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.size()
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if position_ids is None:
            # If no position_ids are provided, create them as a simple sequence.
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)  # (batch, seq_length)

        if inputs_embeds is None:
            # Get token embeddings
            inputs_embeds = self.model.embed_tokens(input_ids)

        # Add custom positional embeddings
        pos_embeds = self.custom_position_embeddings(position_ids)
        hidden_states = inputs_embeds + pos_embeds

        # Pass modified hidden states to the transformer layers.
        # Note: The LlamaModel inside LlamaForCausalLM expects to handle rope embeddings itself.
        # By feeding in hidden_states here, we effectively bypass rope. 
        # For a proper implementation, you might need to alter how the LlamaModel handles rope embeddings.
        outputs = self.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_ids=None,  # we already applied positional embeddings
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift labels to align them with predictions
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs.past_key_values,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }

# Example usage:
# config = LlamaConfig.from_pretrained("...")
# model = CustomPositionalLlamaForCausalLM(config)
# outputs = model(input_ids=torch.randint(0, config.vocab_size, (1, 10)))
