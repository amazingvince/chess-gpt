import os
import json
import time
import random
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import chess
import chess.engine
from datasets import Dataset, DatasetDict, load_dataset
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from fen_utils import tokenize_fen

# Suppress transformer warnings
transformers.logging.set_verbosity_error()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)

from data_loader_fen import create_dataset

config = {
    "lichess_games": 0.9,
    "laion_games": 0.1,
}

# This is a huggingface streaming dataset. I want to use this to get the fens
"""
example: Dataset({
    features: ['text', 'moves', 'fen', 'average_elo', 'weight', 'dataset_source', 'is_valid', 'from_middle', 'eval_score', 'variant'],

})
"""
train_dataset, eval_dataset = create_dataset(config, mid_game_prob=1.0, eval_size=1)


@dataclass
class OpeningBook:
    """A data class that holds chess openings loaded from Lichess dataset."""

    STANDARD_OPENINGS: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        dataset = load_dataset("lichess/chess-openings")
        openings = {}
        for entry in dataset["train"]:
            uci_moves = entry["uci"].split()
            openings[entry["name"]] = uci_moves
        self.STANDARD_OPENINGS = openings

    def get_random_opening(self) -> tuple[str, List[str]]:
        """Returns a random opening name and its moves."""
        name = random.choice(list(self.STANDARD_OPENINGS.keys()))
        return name, self.STANDARD_OPENINGS[name]


class ChessModel:
    def __init__(self, model_path: str, gen_params: Optional[Dict] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=True,
        )

        self.model.eval()
        self.gen_params = gen_params or {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            # "num_beam_groups": 5,
            # "diversity_penalty": 1.0,
            # "num_return_sequences": 5,
            "max_new_tokens": 5,
            # "num_beams": 10,
        }

    @torch.no_grad()
    def get_move(self, board: chess.Board) -> Optional[str]:
        """Generate the next move from the model given the current board state."""
        game_state = self._build_game_state(board)

        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        text = self.build_input_text(start_fen, game_state)

        move_encodings = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        model_inputs = {
            "input_ids": move_encodings["input_ids"],
            "attention_mask": move_encodings["attention_mask"],
            **self.gen_params,
        }

        outputs = self.model.generate(**model_inputs)

        for output in outputs:
            decoded = self.tokenizer.decode(
                output[len(move_encodings["input_ids"][0]) :], skip_special_tokens=False
            )
            move = self._parse_move(decoded)
            if move and self._is_legal_move(move, board):
                return move

        # If no legal move found, return a random legal move
        return random.choice(list(board.legal_moves))

    def build_input_text(self, fen: str, moves: str) -> str:
        return f"<|start|> <|above_2000|> <|standard|> {tokenize_fen(fen)} <|sep|> {moves} {'<|turn|>' if moves else ''}"

    def _build_game_state(self, board: chess.Board) -> str:
        """Build the game state string from the board."""
        moves = []
        for move in board.move_stack:
            uci = move.uci()
            from_square = uci[:2]
            to_square = uci[2:4]
            promotion = f" {uci[4].lower()}" if len(uci) > 4 else ""
            moves.append(f"{from_square} {to_square}{promotion}")

        return " <|turn|> ".join(moves)

    def _parse_move(self, decoded_output: str) -> Optional[str]:
        """Attempt to parse a UCI move from the model's raw output."""
        try:
            parts = decoded_output.split("<|turn|>")[0]
            parts = parts.strip().split(" ")
            return "".join(parts[:2]) + (parts[2].strip() if len(parts) > 2 else "")
        except Exception:
            return None

    def _is_legal_move(self, move: str, board: chess.Board) -> bool:
        """Check if the parsed move is legal in the given board state."""
        try:
            chess_move = chess.Move.from_uci(move)
            return chess_move in board.legal_moves
        except Exception:
            return False


class ChessDatasetGenerator:
    def __init__(
        self,
        model_path: str,
        stockfish_path: str,
        output_dir: str = "chess_dataset",
        stockfish_time: float = 0.1,
    ):
        self.model = ChessModel(model_path)
        self.stockfish_path = stockfish_path
        self.output_dir = output_dir
        self.stockfish_time = stockfish_time
        # self.opening_book = OpeningBook()
        os.makedirs(output_dir, exist_ok=True)

    def setup_stockfish(self, elo: Optional[int] = None) -> chess.engine.SimpleEngine:
        """Setup a Stockfish engine with optional ELO rating."""
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        if elo is None:
            engine.configure(
                {
                    "UCI_LimitStrength": False,
                    "Skill Level": 20,
                    "Hash": 128,
                    "Threads": 4,
                }
            )
        elif elo:
            stockfish_elo = max(1320, min(2800, elo))
            if stockfish_elo <= 1320:
                skill_level = int((stockfish_elo - 800) / (1320 - 800) * 8)
                engine.configure({"Skill Level": max(0, min(8, skill_level))})
            else:
                engine.configure(
                    {
                        "UCI_LimitStrength": True,
                        "UCI_Elo": stockfish_elo,
                        "Skill Level": min(20, (stockfish_elo - 1320) // 75 + 8),
                    }
                )
        return engine

    async def play_parallel_games(
        self, fen: str, opponent_elo: int, max_moves: int = 100
    ) -> Dict[str, List[str]]:
        """Play parallel games where model always moves first."""
        board_model = chess.Board(fen)
        board_stockfish = chess.Board(fen)

        # Determine color from FEN
        model_plays_white = board_model.turn == chess.WHITE

        opponent = self.setup_stockfish(opponent_elo)
        stockfish = self.setup_stockfish(None)  # Full strength

        model_moves = []
        stockfish_moves = []
        move_count = 0

        try:
            while (
                not (board_model.is_game_over() or board_stockfish.is_game_over())
                and move_count < max_moves
            ):
                # Model game - model always moves first in the current position
                if not board_model.is_game_over():
                    # Model's move
                    move = self.model.get_move(board_model)
                    if move:
                        model_moves.append(move)
                        if isinstance(move, str):
                            board_model.push(chess.Move.from_uci(move))
                        else:
                            board_model.push(move)

                    # Opponent's response (if game not over)
                    if not board_model.is_game_over():
                        result = opponent.play(
                            board_model, chess.engine.Limit(time=self.stockfish_time)
                        )
                        board_model.push(result.move)

                # Stockfish game
                if not board_stockfish.is_game_over():
                    # Stockfish's move (playing same position as model)
                    result = stockfish.play(
                        board_stockfish, chess.engine.Limit(time=self.stockfish_time)
                    )
                    stockfish_moves.append(result.move.uci())
                    board_stockfish.push(result.move)

                    # Opponent's response (if game not over)
                    if not board_stockfish.is_game_over():
                        result = opponent.play(
                            board_stockfish,
                            chess.engine.Limit(time=self.stockfish_time),
                        )
                        board_stockfish.push(result.move)

                move_count += 1

        finally:
            opponent.quit()
            stockfish.quit()

        # Determine winners based on results and whose turn it was
        model_winner = None
        stockfish_winner = None

        if board_model.is_game_over():
            result = board_model.result()
            if result == "1-0":
                model_winner = (
                    model_plays_white  # True if model was white and white won
                )
            elif result == "0-1":
                model_winner = (
                    not model_plays_white
                )  # True if model was black and black won
            # Draw results in False for winner

        if board_stockfish.is_game_over():
            result = board_stockfish.result()
            if result == "1-0":
                stockfish_winner = (
                    model_plays_white  # True if stockfish was white and white won
                )
            elif result == "0-1":
                stockfish_winner = (
                    not model_plays_white
                )  # True if stockfish was black and black won
            # Draw results in False for winner

        return {
            "model_moves": model_moves,
            "stockfish_moves": stockfish_moves,
            "model_result": board_model.result(),
            "stockfish_result": board_stockfish.result(),
            "model_winner": model_winner,
            "stockfish_winner": stockfish_winner,
            "model_color": "white" if model_plays_white else "black",
        }

    def generate_dataset(
        self,
        num_positions: int,
        elo_range: Tuple[int, int] = (800, 2800),
        custom_fens: Optional[List[str]] = None,
        max_moves_per_game: int = 100,
    ) -> Dataset:
        """Generate a dataset of parallel games from various positions."""
        data = []

        # If no custom FENs provided, stream from the training dataset
        if not custom_fens:
            # Create an iterator for the training dataset
            dataset_iterator = iter(train_dataset.shuffle().take(num_positions))
            fens = []

            # Stream FENs from the dataset
            for item in dataset_iterator:
                if item["is_valid"] and item["fen"]:  # Ensure we have valid FENs
                    fens.append(item["fen"])
                    if len(fens) >= num_positions:
                        break

            # If we didn't get enough positions, cycle through what we have
            if len(fens) < num_positions:
                logging.warning(
                    f"Only found {len(fens)} valid positions, will cycle through them"
                )
                fens = fens * (num_positions // len(fens) + 1)
                fens = fens[:num_positions]
        else:
            fens = custom_fens

        for i in range(num_positions):
            if i % 10 == 0:
                logging.info(f"Generating game {i + 1}/{num_positions}")

            fen = fens[i]
            opponent_elo = random.randint(*elo_range)

            # Play parallel games
            games = asyncio.run(
                self.play_parallel_games(fen, opponent_elo, max_moves_per_game)
            )

            # Convert moves to UCI format strings if needed
            model_moves_uci = [
                move.uci() if isinstance(move, chess.Move) else move
                for move in games["model_moves"]
            ]

            # Format for the dataset
            entry = {
                "prompt": fen,
                "chosen": " ".join(games["stockfish_moves"]),
                "rejected": " ".join(model_moves_uci),
                "metadata": {
                    "opponent_elo": opponent_elo,
                    "timestamp": datetime.now().isoformat(),
                    "model_result": games["model_result"],
                    "stockfish_result": games["stockfish_result"],
                    "model_color": games["model_color"],
                    "model_winner": games["model_winner"],
                    "stockfish_winner": games["stockfish_winner"],
                    "starting_side": "white"
                    if chess.Board(fen).turn == chess.WHITE
                    else "black",
                },
            }
            data.append(entry)

        # Create HuggingFace dataset
        dataset = Dataset.from_list(data)

        # Save dataset
        dataset.save_to_disk(self.output_dir)

        return dataset


def main():
    config = {
        "model_path": "amazingvince/chess-llama-full-2048",
        "stockfish_path": "/home/vincent/Documents/stockfish-ubuntu-x86-64-vnni512/stockfish/stockfish-ubuntu-x86-64-vnni512",
        "num_positions": 1,
        "elo_range": (800, 2800),
        "output_dir": "chess_training_dataset",
        "max_moves_per_game": 100,
        "stockfish_time": 0.1,  # Time per move in seconds
    }

    generator = ChessDatasetGenerator(
        model_path=config["model_path"],
        stockfish_path=config["stockfish_path"],
        output_dir=config["output_dir"],
        stockfish_time=config["stockfish_time"],
    )

    dataset = generator.generate_dataset(
        num_positions=config["num_positions"],
        elo_range=config["elo_range"],
        max_moves_per_game=config["max_moves_per_game"],
    )

    print(f"Generated dataset with {len(dataset)} examples")
    print(f"Dataset saved to {config['output_dir']}")

    # Print a sample entry
    print("\nSample entry:")
    print(json.dumps(dataset[0], indent=2))


if __name__ == "__main__":
    main()
