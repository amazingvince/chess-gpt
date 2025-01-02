import os
import json
import time
import random
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import itertools

import chess
import chess.engine
from datasets import Dataset
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


@dataclass
class GameState:
    board: chess.Board
    board_stockfish: chess.Board
    model_moves: List[str]
    stockfish_moves: List[str]
    model_plays_white: bool
    opponent_elo: int

    @classmethod
    def from_fen(cls, fen: str, opponent_elo: int) -> "GameState":
        """Create a GameState instance from a FEN string."""
        board = chess.Board(fen=fen)
        board_stockfish = chess.Board(fen=fen)
        return cls(
            board=board,
            board_stockfish=board_stockfish,
            model_moves=[],
            stockfish_moves=[],
            model_plays_white=board.turn == chess.WHITE,
            opponent_elo=opponent_elo,
        )


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
            "max_new_tokens": 5,
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

        return random.choice([move.uci() for move in board.legal_moves])

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


class BatchedChessModel(ChessModel):
    @torch.no_grad()
    def get_moves_batch(
        self, boards: List[chess.Board], batch_size: int = 32
    ) -> List[Optional[str]]:
        """Generate moves for multiple boards in parallel."""
        all_moves = []

        for i in range(0, len(boards), batch_size):
            batch_boards = boards[i : i + batch_size]
            game_states = [self._build_game_state(board) for board in batch_boards]
            start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            texts = [self.build_input_text(start_fen, state) for state in game_states]

            move_encodings = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            outputs = self.model.generate(
                input_ids=move_encodings["input_ids"],
                attention_mask=move_encodings["attention_mask"],
                **self.gen_params,
            )

            for output, board in zip(outputs, batch_boards):
                decoded = self.tokenizer.decode(
                    output[len(move_encodings["input_ids"][0]) :],
                    skip_special_tokens=False,
                )
                move = self._parse_move(decoded)
                if move and self._is_legal_move(move, board):
                    all_moves.append(move)
                else:
                    all_moves.append(
                        random.choice([move.uci() for move in board.legal_moves])
                    )

        return all_moves


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

    async def play_parallel_games_batch(
        self,
        fens: List[str],
        opponent_elos: List[int],
        max_moves: int = 100,
        batch_size: int = 32,
    ) -> List[Dict]:
        """Play multiple games in parallel with batched model inference."""
        games: List[GameState] = []
        for fen, elo in zip(fens, opponent_elos):
            # Create two separate board instances from the same FEN
            # This ensures both games start from identical positions
            board = chess.Board(fen=fen)
            board_stockfish = chess.Board(fen=fen)

            # Determine if model plays white based on the starting position
            model_plays_white = board.turn == chess.WHITE

            games.append(
                GameState(
                    board=board,
                    board_stockfish=board_stockfish,
                    model_moves=[],
                    stockfish_moves=[],
                    model_plays_white=model_plays_white,
                    opponent_elo=elo,
                )
            )

        engines = {
            "opponent": [self.setup_stockfish(elo) for elo in opponent_elos],
            "stockfish": [self.setup_stockfish(None) for _ in range(len(fens))],
        }

        try:
            move_count = 0
            while move_count < max_moves:
                # Track active games separately for model and Stockfish
                active_model_games = [g for g in games if not g.board.is_game_over()]
                active_stockfish_games = [
                    g for g in games if not g.board_stockfish.is_game_over()
                ]

                if not (active_model_games or active_stockfish_games):
                    break

                # Handle model games
                if active_model_games:
                    active_boards = [g.board for g in active_model_games]
                    model_moves = self.model.get_moves_batch(active_boards, batch_size)

                    for game, move in zip(active_model_games, model_moves):
                        game.model_moves.append(move)
                        game.board.push(chess.Move.from_uci(move))

                        if not game.board.is_game_over():
                            engine_idx = games.index(game)
                            result = await asyncio.get_event_loop().run_in_executor(
                                self.executor,
                                engines["opponent"][engine_idx].play,
                                game.board,
                                chess.engine.Limit(time=self.stockfish_time),
                            )
                            game.board.push(result.move)

                # Handle Stockfish games
                stockfish_tasks = []
                for game in active_stockfish_games:
                    engine_idx = games.index(game)
                    task = asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        engines["stockfish"][engine_idx].play,
                        game.board_stockfish,
                        chess.engine.Limit(time=1.0),
                    )
                    stockfish_tasks.append((game, task))

                for game, task in stockfish_tasks:
                    result = await task
                    game.stockfish_moves.append(result.move.uci())
                    game.board_stockfish.push(result.move)

                    if not game.board_stockfish.is_game_over():
                        engine_idx = games.index(game)
                        result = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            engines["opponent"][engine_idx].play,
                            game.board_stockfish,
                            chess.engine.Limit(time=self.stockfish_time),
                        )
                        game.board_stockfish.push(result.move)

                move_count += 1

        finally:
            for engine_list in engines.values():
                for engine in engine_list:
                    engine.quit()

        # Format results with separate outcomes for model and Stockfish games
        results = []
        for game in games:
            model_winner = None
            stockfish_winner = None

            if game.board.is_game_over():
                result = game.board.result()
                if result == "1-0":
                    model_winner = game.model_plays_white
                elif result == "0-1":
                    model_winner = not game.model_plays_white

            if game.board_stockfish.is_game_over():
                result = game.board_stockfish.result()
                if result == "1-0":
                    stockfish_winner = game.model_plays_white
                elif result == "0-1":
                    stockfish_winner = not game.model_plays_white

            results.append(
                {
                    "model_moves": game.model_moves,
                    "stockfish_moves": game.stockfish_moves,
                    "model_result": game.board.result()
                    if game.board.is_game_over()
                    else "unfinished",
                    "stockfish_result": game.board_stockfish.result()
                    if game.board_stockfish.is_game_over()
                    else "unfinished",
                    "model_winner": model_winner,
                    "stockfish_winner": stockfish_winner,
                    "model_color": "white" if game.model_plays_white else "black",
                }
            )

        return results


class OptimizedChessDatasetGenerator(ChessDatasetGenerator):
    def __init__(
        self,
        model_path: str,
        stockfish_path: str,
        output_dir: str = "chess_dataset",
        stockfish_time: float = 0.1,
        num_parallel_games: int = 8,
    ):
        # Initialize with BatchedChessModel instead of ChessModel
        self.model = BatchedChessModel(model_path)
        self.stockfish_path = stockfish_path
        self.output_dir = output_dir
        self.stockfish_time = stockfish_time
        self.num_parallel_games = num_parallel_games
        self.executor = ThreadPoolExecutor(max_workers=num_parallel_games * 2)

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, data: List[Dict], batch_num: int):
        """Save a checkpoint of the current batch."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"batch_{batch_num}.json")
        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved checkpoint to {checkpoint_path}")

    def load_existing_checkpoints(self) -> Tuple[List[Dict], int]:
        """Load all existing checkpoints and return combined data and number of processed positions."""
        all_data = []
        processed_positions = 0

        checkpoint_files = sorted(
            [
                f
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith("batch_") and f.endswith(".json")
            ],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )

        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_file)
            with open(checkpoint_path, "r") as f:
                batch_data = json.load(f)
                all_data.extend(batch_data)
                processed_positions += len(batch_data)

        logging.info(
            f"Loaded {len(checkpoint_files)} checkpoints with {processed_positions} total positions"
        )
        return all_data, processed_positions

    async def play_parallel_games_batch(
        self,
        fens: List[str],
        opponent_elos: List[int],
        max_moves: int = 100,
        batch_size: int = 32,
    ) -> List[Dict]:
        """Play multiple games in parallel with batched model inference."""
        games: List[GameState] = []
        for fen, elo in zip(fens, opponent_elos):
            games.append(GameState.from_fen(fen, elo))

        engines = {
            "opponent": [self.setup_stockfish(elo) for elo in opponent_elos],
            "stockfish": [self.setup_stockfish(None) for _ in range(len(fens))],
        }

        try:
            move_count = 0
            while move_count < max_moves:
                # Track active games separately for model and Stockfish
                active_model_games = [g for g in games if not g.board.is_game_over()]
                active_stockfish_games = [
                    g for g in games if not g.board_stockfish.is_game_over()
                ]

                if not (active_model_games or active_stockfish_games):
                    break

                # Handle model games
                if active_model_games:
                    active_boards = [g.board for g in active_model_games]
                    model_moves = self.model.get_moves_batch(active_boards, batch_size)

                    for game, move in zip(active_model_games, model_moves):
                        game.model_moves.append(move)
                        game.board.push(chess.Move.from_uci(move))

                        if not game.board.is_game_over():
                            engine_idx = games.index(game)
                            result = await asyncio.get_event_loop().run_in_executor(
                                self.executor,
                                engines["opponent"][engine_idx].play,
                                game.board,
                                chess.engine.Limit(time=self.stockfish_time),
                            )
                            game.board.push(result.move)

                # Handle Stockfish games
                stockfish_tasks = []
                for game in active_stockfish_games:
                    engine_idx = games.index(game)
                    task = asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        engines["stockfish"][engine_idx].play,
                        game.board_stockfish,
                        chess.engine.Limit(time=self.stockfish_time),
                    )
                    stockfish_tasks.append((game, task))

                for game, task in stockfish_tasks:
                    result = await task
                    game.stockfish_moves.append(result.move.uci())
                    game.board_stockfish.push(result.move)

                    if not game.board_stockfish.is_game_over():
                        engine_idx = games.index(game)
                        result = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            engines["opponent"][engine_idx].play,
                            game.board_stockfish,
                            chess.engine.Limit(time=self.stockfish_time),
                        )
                        game.board_stockfish.push(result.move)

                move_count += 1

        finally:
            for engine_list in engines.values():
                for engine in engine_list:
                    engine.quit()

        # Format results with separate outcomes for model and Stockfish games
        results = []
        for game in games:
            model_winner = None
            stockfish_winner = None

            if game.board.is_game_over():
                result = game.board.result()
                if result == "1-0":
                    model_winner = game.model_plays_white
                elif result == "0-1":
                    model_winner = not game.model_plays_white

            if game.board_stockfish.is_game_over():
                result = game.board_stockfish.result()
                if result == "1-0":
                    stockfish_winner = game.model_plays_white
                elif result == "0-1":
                    stockfish_winner = not game.model_plays_white

            results.append(
                {
                    "model_moves": game.model_moves,
                    "stockfish_moves": game.stockfish_moves,
                    "model_result": game.board.result()
                    if game.board.is_game_over()
                    else "unfinished",
                    "stockfish_result": game.board_stockfish.result()
                    if game.board_stockfish.is_game_over()
                    else "unfinished",
                    "model_winner": model_winner,
                    "stockfish_winner": stockfish_winner,
                    "model_color": "white" if game.model_plays_white else "black",
                }
            )

        return results

    async def generate_dataset_parallel(
        self,
        streaming_dataset: Iterator,
        num_positions: int,
        elo_range: Tuple[int, int] = (800, 2800),
        max_moves_per_game: int = 100,
        batch_size: int = 32,
    ) -> Dataset:
        # Load any existing checkpoints
        data, processed_positions = self.load_existing_checkpoints()
        batch_num = len(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith("batch_")]
        )

        # Skip already processed positions in the streaming dataset
        for _ in range(processed_positions):
            next(streaming_dataset, None)

        # Process positions in parallel batches
        while processed_positions < num_positions:
            batch_size = min(
                self.num_parallel_games, num_positions - processed_positions
            )

            # Get batch of positions from streaming dataset
            batch_items = list(itertools.islice(streaming_dataset, batch_size))
            batch_fens = []
            batch_metadata = []

            for item in batch_items:
                if item["is_valid"] and item["fen"] and item["variant"] == "standard":
                    batch_fens.append(item["fen"])
                    batch_metadata.append(
                        {
                            "dataset_source": item.get("dataset_source"),
                            "eval_score": item.get("eval_score"),
                        }
                    )

            if not batch_fens:
                continue

            logging.info(
                f"Processing games {processed_positions + 1}-{processed_positions + len(batch_fens)}/{num_positions}"
            )

            # Generate random ELO ratings for all games
            opponent_elos = [random.randint(*elo_range) for _ in range(len(batch_fens))]

            # Play games in parallel
            games_results = await self.play_parallel_games_batch(
                batch_fens, opponent_elos, max_moves_per_game, batch_size
            )

            # Format entries
            batch_data = []
            for fen, elo, game_result, meta in zip(
                batch_fens, opponent_elos, games_results, batch_metadata
            ):
                entry = {
                    "prompt": fen,
                    "chosen": " ".join(game_result["stockfish_moves"]),
                    "rejected": " ".join(game_result["model_moves"]),
                    "metadata": {
                        "opponent_elo": elo,
                        "dataset_source": meta["dataset_source"],
                        "eval_score": meta["eval_score"],
                        "timestamp": datetime.now().isoformat(),
                        "model_result": game_result["model_result"],
                        "stockfish_result": game_result["stockfish_result"],
                        "model_color": game_result["model_color"],
                        "model_winner": game_result["model_winner"],
                        "stockfish_winner": game_result["stockfish_winner"],
                        "starting_side": "white"
                        if chess.Board(fen).turn == chess.WHITE
                        else "black",
                    },
                }
                batch_data.append(entry)

            # Save checkpoint for this batch
            self.save_checkpoint(batch_data, batch_num)
            batch_num += 1

            # Add batch data to main data list
            data.extend(batch_data)
            processed_positions += len(batch_fens)

        # Create and save final dataset
        dataset = Dataset.from_list(data)
        dataset.save_to_disk(self.output_dir)
        return dataset


def main():
    import itertools
    from data_loader_fen import create_dataset

    ds_config = {
        "lichess_games": 0.9,
        "laion_games": 0.1,
    }

    # Create streaming dataset
    train_dataset, eval_dataset = create_dataset(
        ds_config, mid_game_prob=1.0, eval_size=1, seed=6969
    )

    # Generator configuration
    config = {
        "model_path": "amazingvince/chess-llama-full-2048",
        "stockfish_path": "/home/vincent/Documents/stockfish-ubuntu-x86-64-vnni512/stockfish/stockfish-ubuntu-x86-64-vnni512",  # Update this path
        "num_positions": 32000,
        "elo_range": (800, 2800),
        "output_dir": "chess_training_dataset",
        "max_moves_per_game": 100,
        "stockfish_time": 0.1,
        "num_parallel_games": 16,
        "batch_size": 32,
    }

    # Initialize generator
    generator = OptimizedChessDatasetGenerator(
        model_path=config["model_path"],
        stockfish_path=config["stockfish_path"],
        output_dir=config["output_dir"],
        stockfish_time=config["stockfish_time"],
        num_parallel_games=config["num_parallel_games"],
    )

    # Generate dataset
    dataset = asyncio.run(
        generator.generate_dataset_parallel(
            streaming_dataset=iter(train_dataset.shuffle()),
            num_positions=config["num_positions"],
            elo_range=config["elo_range"],
            max_moves_per_game=config["max_moves_per_game"],
            batch_size=config["batch_size"],
        )
    )

    print(f"Generated dataset with {len(dataset)} examples")
    print(f"Dataset saved to {config['output_dir']}")
    print("\nSample entry:")
    print(json.dumps(dataset[0], indent=2))


if __name__ == "__main__":
    main()
