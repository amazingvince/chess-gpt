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

import chess
import chess.engine
from datasets import load_dataset
import torch
import transformers
from transformers import AutoModelForCausalLM
from tokenizer import ChessTokenizer
from tqdm import tqdm

# Suppress transformer warnings for cleaner output
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
        # Load the dataset from Hugging Face
        dataset = load_dataset("lichess/chess-openings")

        # Convert the UCI moves strings to lists and create the dictionary
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
class MoveResult:
    """Holds the result of a move attempt by the local model."""

    move: Optional[str] = None
    is_resignation: bool = False
    failed_to_move: bool = False
    first_move_invalid: bool = (
        False  # True if the first candidate move was invalid but a subsequent was chosen.
    )


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


@dataclass
class GameStats:
    """Collects statistics across multiple games."""

    results: List[GameResult] = field(default_factory=list)
    invalid_first_moves: int = 0
    incomplete_games: int = 0
    start_time: datetime = field(default_factory=datetime.now)

    def add_result(self, result: GameResult):
        self.results.append(result)
        if result.reason == "local_model_failed":
            self.incomplete_games += 1

    def get_summary(self, stockfish_level: Optional[int]) -> Dict:
        """Generate a summary of all recorded results."""
        total_games = len(self.results)
        if total_games == 0:
            return {}

        local_wins = sum(1 for r in self.results if r.winner == "local")
        stockfish_wins = sum(1 for r in self.results if r.winner == "stockfish")
        draws = sum(1 for r in self.results if r.winner == "draw")

        local_wins_as_white = sum(
            1 for r in self.results if r.winner == "local" and r.played_as_white
        )
        local_wins_as_black = sum(
            1 for r in self.results if r.winner == "local" and not r.played_as_white
        )

        return {
            "stockfish_level": stockfish_level,
            "total_games": total_games,
            "local_wins": local_wins,
            "local_win_rate": round(local_wins / total_games * 100, 2),
            "local_wins_as_white": local_wins_as_white,
            "local_wins_as_black": local_wins_as_black,
            "stockfish_wins": stockfish_wins,
            "stockfish_win_rate": round(stockfish_wins / total_games * 100, 2),
            "draws": draws,
            "draw_rate": round(draws / total_games * 100, 2),
            "failed_moves_total": sum(r.failed_moves_local for r in self.results),
            "avg_moves_per_game": round(
                sum(r.num_moves for r in self.results) / total_games, 2
            ),
            "avg_game_duration": round(
                sum(r.duration for r in self.results) / total_games, 2
            ),
            "invalid_first_moves": self.invalid_first_moves,
            "incomplete_games": self.incomplete_games,
        }

    def save_results(self, filename: str):
        """Save the results and summary to a JSON file."""
        results_data = {
            "summary": self.get_summary(stockfish_level=None),
            "individual_games": [
                {
                    "winner": r.winner,
                    "reason": r.reason,
                    "num_moves": r.num_moves,
                    "duration": r.duration,
                    "failed_moves_local": r.failed_moves_local,
                    "failed_moves_stockfish": r.failed_moves_stockfish,
                    "played_as_white": r.played_as_white,
                }
                for r in self.results
            ],
            "invalid_first_moves": self.invalid_first_moves,
            "incomplete_games": self.incomplete_games,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filename, "w") as f:
            json.dump(results_data, f, indent=2)


class ChessModel:
    """A wrapper class around a chess-oriented language model."""

    def __init__(self, model_path: str, gen_params: Optional[Dict] = None):
        self.tokenizer = ChessTokenizer()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16, use_cache=True
        )
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
    def get_move(
        self, board: chess.Board, game_state: Optional[str] = None
    ) -> MoveResult:
        """Generate the next move from the model given the current board state."""
        if game_state is None:
            game_state = self._build_game_state(board)

        inputs = self.tokenizer(
            game_state, return_tensors="pt", return_token_type_ids=False
        ).to(self.model.device)
        outputs = self.model.generate(**self.gen_params, **inputs)

        chosen_move = None
        first_move_invalid = False
        for i, output in enumerate(outputs):
            decoded = self.tokenizer.decode(
                output[len(inputs["input_ids"][0]) :], skip_special_tokens=False
            )
            move = self._parse_move(decoded)
            if move and self._is_legal_move(move, board):
                chosen_move = move
                # If i > 0, that means the first candidate was invalid and we had to choose a later one
                if i > 0:
                    first_move_invalid = True
                break

        if chosen_move:
            return MoveResult(move=chosen_move, first_move_invalid=first_move_invalid)

        return MoveResult(failed_to_move=True)

    def _build_game_state(self, board: chess.Board) -> str:
        """Build the game state string from the board's move stack."""
        moves = ["<|start|>"]
        for move in board.move_stack:
            moves.append(move.uci())
            moves.append("<|turn|>")
        return "".join(moves)

    def _parse_move(self, decoded_output: str) -> Optional[str]:
        """Attempt to parse a UCI move from the model's raw output."""
        try:
            parts = decoded_output.split("<|turn|>")[0]
            move = parts.replace("<|start|>", "").replace("<|end|>", "").strip()
            return "".join(move.split()) if len(move) >= 4 else None
        except Exception:
            return None

    def _is_legal_move(self, move: str, board: chess.Board) -> bool:
        """Check if the parsed move is legal in the given board state."""
        try:
            chess_move = chess.Move.from_uci(move)
            return chess_move in board.legal_moves
        except Exception:
            return False


class ChessGameManager:
    """Manages the process of playing a single chess game between the local model and Stockfish."""

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

    def play_game(
        self, stockfish_level: int, local_plays_white: bool, stats: GameStats
    ) -> GameResult:
        """Play a single game and return the result."""
        if not os.path.exists(self.stockfish_path):
            logging.error("Stockfish binary not found at specified path.")
            raise FileNotFoundError("Stockfish binary not found.")

        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        engine.configure({"Skill Level": stockfish_level})

        board = chess.Board()
        game_state = "<|start|>"
        failed_moves = {"local": 0, "stockfish": 0}
        moves_played: List[str] = []
        start_time = datetime.now()

        # Choose a random opening and play its moves
        opening_name, opening_moves = self.opening_book.get_random_opening()
        try:
            # Play opening moves
            for i, move_uci in enumerate(opening_moves):
                move = chess.Move.from_uci(move_uci)
                moves_played.append(move_uci)
                board.push(move)
                game_state += move_uci + "<|turn|>"

                if self.verbose:
                    self._print_game_state(
                        board, move, f"Playing opening {opening_name}: Move {i+1}"
                    )

            # Continue with the regular gameplay
            while not board.is_game_over() and not self._has_excessive_failures(
                failed_moves
            ):
                is_local_turn = (len(moves_played) % 2 == 0) == local_plays_white
                move = self._get_next_move(
                    board, game_state, engine, is_local_turn, failed_moves, stats
                )

                if move:
                    moves_played.append(move.uci())
                    board.push(move)
                    game_state += move.uci() + "<|turn|>"

                    if self.verbose:
                        self._print_game_state(board, move)

            # Return the final result
            return self._create_game_result(
                board, moves_played, failed_moves, start_time, local_plays_white
            )
        finally:
            engine.quit()

    def _get_next_move(
        self,
        board: chess.Board,
        game_state: str,
        engine: chess.engine.SimpleEngine,
        is_local_turn: bool,
        failed_moves: Dict[str, int],
        stats: GameStats,
    ) -> Optional[chess.Move]:
        """Determine the next move from either the local model or Stockfish."""
        if is_local_turn:
            result = self.model.get_move(board, game_state)
            if result.failed_to_move:
                failed_moves["local"] += 1
                return None
            # If the first generated candidate was invalid, increment stats
            if result.first_move_invalid:
                stats.invalid_first_moves += 1
            return chess.Move.from_uci(result.move)
        else:
            # Stockfish's move
            result = engine.play(board, chess.engine.Limit(time=self.stockfish_time))
            if not result.move:
                failed_moves["stockfish"] += 1
                return None
            return result.move

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
            # Checkmate: If an odd number of moves have been played, the last move was by Black.
            is_white_win = len(moves_played) % 2 == 1
            winner = "local" if is_white_win == local_plays_white else "stockfish"
            return winner, "checkmate"

        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_fifty_moves()
        ):
            return "draw", "draw"

        # Fallback outcome:
        return "draw", "unknown"

    def _has_excessive_failures(self, failed_moves: Dict[str, int]) -> bool:
        """Check if either side has failed too many times."""
        return any(fails >= 3 for fails in failed_moves.values())

    def _print_game_state(
        self, board: chess.Board, move: chess.Move, message: str = ""
    ):
        """Print the current game state if verbose is True."""
        if self.verbose:
            print(f"\n{message}")
            print("Current position:")
            print(board)
            print(f"Move played: {move.uci()}")
            print("-" * 40)


@dataclass
class ParallelGameConfig:
    """Configuration dataclass for running games in parallel."""

    model_path: str
    stockfish_path: str
    games_per_level: int
    stockfish_levels: List[int]
    stockfish_time: float = 1.0
    verbose: bool = False
    color_strategy: str = "random"  # "random", "white", or "black"
    num_workers: int = mp.cpu_count()


def play_single_game(args):
    """Function to run a single game in parallel."""
    level, config, game_num = args

    # Initialize fresh instances for this process
    game_manager = ChessGameManager(
        config.model_path, config.stockfish_path, config.stockfish_time, config.verbose
    )

    stats = GameStats()

    color_choice = {
        "random": lambda: random.choice([True, False]),
        "white": lambda: True,
        "black": lambda: False,
    }[config.color_strategy]()

    result = game_manager.play_game(level, color_choice, stats)
    return {"level": level, "game_num": game_num, "result": result, "stats": stats}


def run_parallel_games(config: ParallelGameConfig) -> Dict[int, Dict]:
    """Run multiple games in parallel and aggregate results, with a progress bar."""
    os.makedirs("results", exist_ok=True)
    all_results = defaultdict(GameStats)
    tasks = [
        (level, config, game_num)
        for level in config.stockfish_levels
        for game_num in range(config.games_per_level)
    ]

    logging.info(f"Starting parallel execution with {config.num_workers} workers.")
    start_time = time.time()

    # Setup the progress bar
    total_tasks = len(tasks)
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures = [executor.submit(play_single_game, task) for task in tasks]
        with tqdm(total=total_tasks, desc="Running Games", unit="game") as pbar:
            for future in as_completed(futures):
                try:
                    outcome = future.result()
                    level = outcome["level"]
                    game_result = outcome["result"]
                    all_results[level].add_result(game_result)
                    pbar.update(1)  # Update progress bar for each completed game

                    # Update the progress bar description to show elapsed time
                    elapsed = time.time() - start_time
                    pbar.set_postfix_str(f"Elapsed: {elapsed:.2f}s")
                except Exception as e:
                    logging.error(f"Error in game execution: {str(e)}")
                    pbar.update(1)  # Still increment to keep progress accurate

    final_results = {}
    for level in config.stockfish_levels:
        stats = all_results[level]
        summary = stats.get_summary(level)
        final_results[level] = summary
        stats.save_results(f"results/chess_results_level_{level}.json")

    combined_results = {
        "config": {
            "games_per_level": config.games_per_level,
            "stockfish_levels": config.stockfish_levels,
            "stockfish_time": config.stockfish_time,
            "num_workers": config.num_workers,
            "color_strategy": config.color_strategy,
        },
        "results": final_results,
        "total_duration": time.time() - start_time,
    }

    with open("results/combined_results.json", "w") as f:
        json.dump(combined_results, f, indent=2)

    return final_results


def main(config_dict: Dict):
    """Main entry point for running multiple games against different Stockfish levels."""
    config = ParallelGameConfig(
        model_path=config_dict["model_path"],
        stockfish_path=config_dict["stockfish_path"],
        games_per_level=config_dict["games_per_level"],
        stockfish_levels=config_dict["stockfish_levels"],
        stockfish_time=config_dict["stockfish_time"],
        verbose=config_dict["verbose"],
        color_strategy=config_dict["color_strategy"],
        num_workers=config_dict.get("num_workers", mp.cpu_count()),
    )

    results = run_parallel_games(config)
    print("\nFinal Results Summary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    CONFIG = {
        "model_path": "/home/vince/code/chess-gpt/runtime/autoregressive/chess-llama-mini-v3-2048/checkpoint-15000",
        "stockfish_path": "/home/vince/code/chess-gpt/stockfish/stockfish/stockfish-ubuntu-x86-64-avx512",
        "games_per_level": 500,
        "stockfish_levels": [0, 5, 10, 15, 20],
        "stockfish_time": 1.0,
        "verbose": False,
        "color_strategy": "random",
        "num_workers": 16,
    }
    main(CONFIG)
