import io
from dataclasses import dataclass
from typing import Dict, List, Optional

import chess.pgn
from datasets import Dataset, interleave_datasets, load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChessExample:
    """Represents a single chess game example with its metadata."""

    text: str
    moves: List[str]
    average_elo: float
    weight: float = 1.0
    dataset_source: str = "lichess_games"
    is_valid: bool = True  # New field to track validity

    def to_dict(self) -> Dict:
        """Convert the example to a dictionary format."""
        return {
            key: (
                float(value)
                if key in ["average_elo", "weight"] and value is not None
                else value
            )
            for key, value in self.__dict__.items()
        }


class ChessProcessor:
    """Processes chess game data from various sources."""

    def __init__(self, max_moves: int = 500):
        self.max_moves = max_moves

    def process_game(self, example: Dict, source: str = "lichess_games") -> Dict:
        """Process a single game from any source."""
        moves = self._get_moves(example, source)
        if not moves:
            # Return an invalid example instead of None
            return ChessExample(
                text="",
                moves=[],
                average_elo=0.0,
                weight=0.0,
                dataset_source=source,
                is_valid=False,
            ).to_dict()

        moves = moves[: self.max_moves]
        avg_elo = self._calculate_elo(example, source)

        return ChessExample(
            text=self._format_moves(
                moves, self._get_elo_token(avg_elo, source), source
            ),
            moves=moves,
            average_elo=avg_elo,
            weight=self._calculate_weight(avg_elo, source),
            dataset_source=source,
            is_valid=True,
        ).to_dict()

    def _get_moves(self, example: Dict, source: str) -> List[str]:
        """Extract moves from different source formats."""
        try:
            if source == "lichess_games":
                movetext = example.get("movetext", "")
                if not movetext:
                    return []

                game = chess.pgn.read_game(io.StringIO(movetext))
                if not game:
                    return []

                moves = list(game.mainline_moves())
                if not moves:
                    return []

                return [move.uci() for move in moves]

            elif source == "laion_games":
                return example.get("Moves", [])

            return []

        except Exception as e:
            logger.warning(f"Error processing moves from {source}: {str(e)}")
            return []

    def _calculate_elo(self, example: Dict, source: str) -> float:
        """Calculate average ELO rating."""
        if source == "laion_games":
            return 2000.0
        return (
            int(example.get("WhiteElo", 1000)) + int(example.get("BlackElo", 1000))
        ) / 2

    def _calculate_weight(self, elo: float, source: str) -> float:
        """Calculate game weight based on ELO and source."""
        if source == "laion_games":
            return 2.0
        if elo < 1000:
            return 0.2
        if elo > 2000:
            return 1.5
        return 0.2 + (elo - 1000) * 0.0013

    @staticmethod
    def _get_elo_token(elo: float, source: str) -> str:
        """Get appropriate ELO token."""
        if source == "laion_games":
            return "<|engine|>"
        if elo < 1000:
            return "<|below_1000|>"
        if elo < 2000:
            return "<|1000_2000|>"
        return "<|above_2000|>"

    @staticmethod
    def add_eos_token(moves: List[str], source: str) -> Optional[str]:
        # Create a board from the starting FEN

        if source == "laion_games":
            return "<|end|>"

        board = chess.Board()

        # Try to play through all moves
        try:
            for move in moves:
                board.push_uci(move)

            # If game is finished (checkmate, stalemate, etc.), add EOS token
            if board.is_game_over():
                return "<|end|>"

        except ValueError:
            # Handle invalid moves by not adding EOS token
            return None

    @staticmethod
    def _format_moves(moves: List[str], elo_token: str, source: str) -> str:
        """Format moves into the required string format."""
        formatted = []
        for move in moves:
            from_square = move[:2]
            to_square = move[2:4]
            promotion = f" {move[4].lower()}" if len(move) > 4 else ""
            formatted.append(f"{from_square} {to_square}{promotion}")
        return f"{elo_token} <|start|> {' <|turn|> '.join(formatted)} {ChessProcessor.add_eos_token(moves, source) or '<|turn|>'}"


def create_dataset(
    config: Dict[str, float], eval_size: int = 2000
) -> tuple[Dataset, Dataset]:
    """Create training and evaluation datasets."""
    processor = ChessProcessor()

    datasets = {
        "lichess_games": {
            "path": "Lichess/standard-chess-games",
            "process": lambda x: processor.process_game(x, "lichess_games"),
            "remove_cols": [
                "Event",
                "Site",
                "White",
                "Black",
                "Result",
                "WhiteTitle",
                "BlackTitle",
                "WhiteElo",
                "BlackElo",
                "WhiteRatingDiff",
                "BlackRatingDiff",
                "UTCDate",
                "UTCTime",
                "ECO",
                "Opening",
                "Termination",
                "TimeControl",
                "movetext",
            ],
        },
        "laion_games": {
            "path": "laion/strategic_game_chess",
            "process": lambda x: processor.process_game(x, "laion_games"),
            "remove_cols": ["Moves", "Termination", "Result"],
        },
    }

    processed_datasets = []
    probabilities = []
    total_prob = sum(config.values())

    for name, prob in config.items():
        if name not in datasets:
            raise ValueError(f"Unknown dataset: {name}")

        dataset_info = datasets[name]
        dataset = load_dataset(dataset_info["path"], streaming=True)["train"]

        # Process and filter based on is_valid flag
        processed = dataset.map(
            dataset_info["process"], remove_columns=dataset_info["remove_cols"]
        ).filter(lambda x: x["is_valid"])

        processed_datasets.append(processed)
        probabilities.append(prob / total_prob)

    combined = interleave_datasets(processed_datasets, probabilities=probabilities)
    combined = combined.shuffle(seed=42069)

    # Create evaluation dataset
    eval_dataset = combined.take(eval_size)
    eval_dataset = Dataset.from_list(list(eval_dataset))
    train_dataset = combined.skip(eval_size)

    return train_dataset, eval_dataset


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Example configuration
    config = {"lichess_games": 0.7, "laion_games": 0.3}

    # Create datasets
    train_dataset, eval_dataset = create_dataset(config)

    # Print some statistics
    print("Train dataset:", train_dataset)
    print("Eval dataset:", eval_dataset)

    # Example of processing one item
    sample = next(iter(train_dataset))
    print("\nSample item:", sample)

    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained(
        "/home/vince/code/chess-gpt/tokenizer_building/tokenizer-chess"
    )
    out = next(iter(train_dataset))
    print(out)
    print(len(t(out["text"])["input_ids"]))
    print(t.decode(t(out["text"])["input_ids"], skip_special_tokens=True))
