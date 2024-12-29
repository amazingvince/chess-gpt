from fen_utils import tokenize_fen
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def test_pretrained_model(path):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=True,
    )
    return model

    config = AutoConfig.from_pretrained(
        "/home/vince/code/chess-gpt/chess-llama/config.json",
    )
    config.torch_dtype = torch.bfloat16
    config.use_cache = False

    model = AutoModelForCausalLM.from_config(
        config=config,
    )
    model.train()
    # Add these lines
    model = model.to("cuda:0")
    # model.config._attn_implementation = "flash_attention_2"
    return model


def main():
    path = "/home/vince/code/chess-gpt/tokenizer_building/runtime/autoregressive/chess-llama-decoder-2048"
    model = test_pretrained_model(path)
    move_tokenizer = AutoTokenizer.from_pretrained(path)

    gen_params = {
        "bos_token_id": move_tokenizer.bos_token_id,
        "eos_token_id": move_tokenizer.eos_token_id,
        "pad_token_id": move_tokenizer.pad_token_id,
        "num_beam_groups": 5,
        "diversity_penalty": 1.0,
        "num_return_sequences": 5,
        "max_new_tokens": 100,
        "num_beams": 10,
    }

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    moves = " e2 e4 <|turn|>"

    text = f"<|start|> <|above_2000|> <|standard|> {tokenize_fen(fen)} <|sep|> {moves}"
    move_encodings = move_tokenizer(text, return_tensors="pt").to(model.device)

    model_inputs = {
        "input_ids": move_encodings["input_ids"],
        "attention_mask": move_encodings["attention_mask"],
        **gen_params,
    }

    model.eval()
    # model.forward(**model_inputs)
    outputs = model.generate(**model_inputs)
    # print(f"Output: {move_tokenizer.decode(outputs[0])}")

    for i, output in enumerate(outputs):
        print(
            f"Output {i + 1}: {move_tokenizer.decode(output[model_inputs['input_ids'].shape[-1]:])}"
        )


if __name__ == "__main__":
    # import debugpy

    # debugpy.listen(5678)
    # debugpy.wait_for_client()

    main()
