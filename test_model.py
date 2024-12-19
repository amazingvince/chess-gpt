import os
import json
import time
import random
import logging
import multiprocessing as mp
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set multiprocessing start method to 'spawn'
if __name__ == "__main__":
    mp.set_start_method("spawn")

import chess
import chess.engine
from datasets import load_dataset
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
)
from tokenizer_2 import ChessTokenizer, FENTokenizer
from model.chess_llama import ChessLlamaConfig, ChessLlamaForCausalLM
from tqdm import tqdm
import traceback

# Suppress transformer warnings for cleaner output
transformers.logging.set_verbosity_error()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

AutoConfig.register("chess_llama", ChessLlamaConfig)
AutoModelForCausalLM.register(ChessLlamaConfig, ChessLlamaForCausalLM)


def main():
    model = AutoModelForCausalLM.from_pretrained(
        "amazingvince/chess-llama-mini-v3-2048",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    move_tokenizer = ChessTokenizer()
    fen_tokenizer = FENTokenizer()

    gen_params = {
        "bos_token_id": move_tokenizer.bos_token_id,
        "eos_token_id": move_tokenizer.eos_token_id,
        "pad_token_id": move_tokenizer.eos_token_id,
        "num_beam_groups": 5,
        "diversity_penalty": 1.0,
        "num_return_sequences": 5,
        "max_new_tokens": 10,
        "num_beams": 10,
    }

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    moves = "<|above_2000|><|start|>e2e4<|turn|>e7e5<|turn|>"
    move_encodings = move_tokenizer(moves, return_tensors="pt").to(model.device)
    fen_encodings = fen_tokenizer(fen, return_tensors="pt").to(model.device)

    model_inputs = {
        "input_ids": move_encodings["input_ids"],
        "attention_mask": move_encodings["attention_mask"],
        "fen_input_ids": fen_encodings["input_ids"],
        "fen_attention_mask": fen_encodings["attention_mask"],
        **gen_params,
    }

    outputs = model.generate(**model_inputs)

    for i, output in enumerate(outputs):
        print(f"Output {i + 1}: {move_tokenizer.decode(output)}")


if __name__ == "__main__":
    # import debugpy

    # debugpy.listen(5678)
    # debugpy.wait_for_client()

    main()
