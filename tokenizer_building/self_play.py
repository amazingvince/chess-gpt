import os
import json
import time
import random
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import chess
import chess.engine
from datasets import load_dataset
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from fen_utils import tokenize_fen

# Suppress transformer warnings
transformers.logging.set_verbosity_error()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)


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


@dataclass
class GameResult:
    """Stores the final result of a completed chess game."""

    winner: str
    reason: str
    num_moves: int
    duration: float
    failed_moves_local: int
    failed_moves_stockfish: int
    played_as_white: bool
    moves: List[str]


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

        if not torch.cuda.is_available():
            self.model.to(self.device)

        self.model.eval()
        self.gen_params = gen_params or {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "num_beam_groups": 5,
            "diversity_penalty": 1.0,
            "num_return_sequences": 5,
            "max_new_tokens": 10,
            "num_beams": 10,
        }

    @torch.no_grad()
    def get_move(self, board: chess.Board) -> Optional[str]:
        """Generate the next move from the model given the current board state."""
        game_state = self._build_game_state(board)

        # 2 ways starting from current fen or from scratch
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        text = self.build_input_text(start_fen, game_state)

        # fen = board.fen()
        # text = self.build_input_text(fen, "")

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
            # print(f"Decoded: {decoded}")
            # print(f"Move: {move}")
            if move and self._is_legal_move(move, board):
                return move

        return None

    def build_input_text(self, fen: str, moves: str) -> str:
        return f"<|start|> <|above_2000|> <|standard|> {tokenize_fen(fen)} <|sep|> {moves} {'<|turn|>' if moves else ''}"

    def _build_game_state(self, board: chess.Board) -> str:
        """Build the game state string from the board."""
        moves = []
        for move in board.move_stack:
            uci = move.uci()  # Convert chess.Move to UCI string notation
            from_square = uci[:2]
            to_square = uci[2:4]
            promotion = f" {uci[4].lower()}" if len(uci) > 4 else ""
            moves.append(f"{from_square} {to_square} {promotion}")

        return " <|turn|> ".join(moves)

    def _parse_move(self, decoded_output: str) -> Optional[str]:
        """Attempt to parse a UCI move from the model's raw output."""
        try:
            parts = decoded_output.split("<|turn|>")[0]
            parts = parts.strip().split(" ")

            return "".join(parts)
        except Exception:
            return None

    def _is_legal_move(self, move: str, board: chess.Board) -> bool:
        """Check if the parsed move is legal in the given board state."""
        try:
            chess_move = chess.Move.from_uci(move)
            return chess_move in board.legal_moves
        except Exception:
            return False


class ChessGame:
    """Manages a single chess game between the local model and Stockfish."""

    def __init__(
        self,
        model_path: str,
        stockfish_path: str,
        stockfish_time: float = 1.0,
        verbose: bool = True,
    ):
        self.model = ChessModel(model_path)
        self.stockfish_path = stockfish_path
        self.stockfish_time = stockfish_time
        self.verbose = verbose
        self.opening_book = OpeningBook()

    def play_game(self, stockfish_elo: int, local_plays_white: bool) -> GameResult:
        """Play a single game and return the result."""
        if not os.path.exists(self.stockfish_path):
            raise FileNotFoundError("Stockfish binary not found.")

        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

        # Configure Stockfish with ELO rating
        # Minimum ELO is 1320 for UCI_Elo
        stockfish_elo = max(1320, min(2800, stockfish_elo))

        # For ELOs below 1320, we'll only use skill level
        if stockfish_elo <= 1320:
            # Map lower ELOs to skill levels 0-8
            skill_level = int((stockfish_elo - 800) / (1320 - 800) * 8)
            engine.configure({"Skill Level": max(0, min(8, skill_level))})
        else:
            # For higher ELOs, use both UCI_Elo and skill level
            engine.configure(
                {
                    "UCI_LimitStrength": True,
                    "UCI_Elo": stockfish_elo,
                    "Skill Level": min(20, (stockfish_elo - 1320) // 75 + 8),
                }
            )

        try:
            board = chess.Board()
            failed_moves = {"local": 0, "stockfish": 0}
            moves_played = []
            start_time = datetime.now()

            # Play opening moves
            opening_name, opening_moves = self.opening_book.get_random_opening()
            for move_uci in opening_moves:
                move = chess.Move.from_uci(move_uci)
                moves_played.append(move_uci)
                board.push(move)
                if self.verbose:
                    print(f"\nPlaying opening {opening_name}")
                    print(board)

            # Main game loop
            while not board.is_game_over() and not self._has_excessive_failures(
                failed_moves
            ):
                is_local_turn = (len(moves_played) % 2 == 0) == local_plays_white
                move = None

                if is_local_turn:
                    move_uci = self.model.get_move(board)
                    if move_uci:
                        move = chess.Move.from_uci(move_uci)
                    else:
                        failed_moves["local"] += 1
                else:
                    result = engine.play(
                        board, chess.engine.Limit(time=self.stockfish_time)
                    )
                    move = result.move
                    if not move:
                        failed_moves["stockfish"] += 1

                if move:
                    moves_played.append(move.uci())
                    board.push(move)
                    if self.verbose:
                        print(f"\nMove: {move.uci()}")
                        print(board)
                        if not is_local_turn:
                            print(
                                f"Stockfish (ELO {stockfish_elo}) played: {move.uci()}"
                            )

            return self._create_game_result(
                board, moves_played, failed_moves, start_time, local_plays_white
            )

        finally:
            engine.quit()

    def _has_excessive_failures(self, failed_moves: Dict[str, int]) -> bool:
        """Check if either side has failed too many times."""
        return any(fails >= 3 for fails in failed_moves.values())

    def _create_game_result(
        self,
        board: chess.Board,
        moves_played: List[str],
        failed_moves: Dict[str, int],
        start_time: datetime,
        local_plays_white: bool,
    ) -> GameResult:
        """Create a GameResult based on the final state of the board."""
        duration = (datetime.now() - start_time).total_seconds()
        winner, reason = self._determine_winner(
            board, failed_moves, local_plays_white, moves_played
        )

        return GameResult(
            winner=winner,
            reason=reason,
            num_moves=len(moves_played),
            duration=duration,
            failed_moves_local=failed_moves["local"],
            failed_moves_stockfish=failed_moves["stockfish"],
            played_as_white=local_plays_white,
            moves=moves_played,  # Include the full move list
        )

    def _determine_winner(
        self,
        board: chess.Board,
        failed_moves: Dict[str, int],
        local_plays_white: bool,
        moves_played: List[str],
    ) -> Tuple[str, str]:
        """Determine the outcome of the game."""
        if failed_moves["local"] >= 3:
            return "stockfish", "local_model_failed"
        if failed_moves["stockfish"] >= 3:
            return "local", "stockfish_failed"

        if board.is_checkmate():
            is_white_win = len(moves_played) % 2 == 1
            winner = "local" if is_white_win == local_plays_white else "stockfish"
            return winner, "checkmate"

        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_fifty_moves()
        ):
            return "draw", "draw"

        return "draw", "unknown"


def main():
    config = {
        "model_path": "amazingvince/chess-llama-full-2048",
        "stockfish_path": "/home/vincent/Documents/stockfish-ubuntu-x86-64-vnni512/stockfish/stockfish-ubuntu-x86-64-vnni512",
        # Adjusted ELO ratings considering the 1320 minimum
        "stockfish_elos": [800, 1000, 1320, 1600, 2000, 2400, 2800],
        "games_per_elo": 5,
        "stockfish_time": 1.0,
        "verbose": False,
    }

    results = {}
    for elo in config["stockfish_elos"]:
        elo_results = []
        print(f"\nPlaying games against Stockfish ELO {elo}")

        game = ChessGame(
            config["model_path"],
            config["stockfish_path"],
            config["stockfish_time"],
            config["verbose"],
        )

        for i in range(config["games_per_elo"]):
            print(f"\nGame {i+1} of {config['games_per_elo']}")
            local_plays_white = random.choice([True, False])
            result = game.play_game(elo, local_plays_white)
            elo_results.append(
                {
                    "winner": result.winner,
                    "reason": result.reason,
                    "num_moves": result.num_moves,
                    "duration": result.duration,
                    "failed_moves_local": result.failed_moves_local,
                    "failed_moves_stockfish": result.failed_moves_stockfish,
                    "played_as_white": result.played_as_white,
                    "moves": result.moves,
                }
            )

        results[elo] = elo_results

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/chess_results_elo.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
