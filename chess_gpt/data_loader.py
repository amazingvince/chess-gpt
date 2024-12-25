import io
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn
from datasets import load_dataset, Dataset, interleave_datasets


@dataclass
class ChessExample:
    fen: str
    moves: Optional[List[str]] = None
    average_elo: Optional[float] = None
    weight: float = 1.0
    dataset_source: Optional[str] = None
    from_middle: bool = False

    def to_dict(self):
        result = {}
        for k, v in asdict(self).items():
            # Convert certain numeric fields to float if needed
            if k in ["average_elo", "eval_score", "weight"] and v is not None:
                v = float(v)
            result[k] = v
        return result


@dataclass
class DatasetConfig:
    name: str
    probability: float
    enabled: bool = True

    def __post_init__(self):
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1")


class ChessDataProcessor:
    def __init__(
        self,
        mid_game_prob: float = 0.0,
        max_moves: int = 500,
    ):
        self.mid_game_prob = mid_game_prob
        self.max_moves = max_moves
        self.initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def _parse_pgn(self, movetext: str) -> List[str]:
        game = chess.pgn.read_game(io.StringIO(movetext))
        if not game:
            return []
        return [move.uci() for move in game.mainline_moves()]

    def _normalize_eval_score(self, score: float, is_mate: bool = False) -> float:
        if is_mate:
            return 20000.0 if score > 0 else -20000.0
        return max(min(score, 20000.0), -20000.0)

    def _calculate_game_weight(self, avg_elo: Optional[float]) -> float:
        if avg_elo is None:
            return 1.0
        if avg_elo < 1000:
            return 0.2
        if avg_elo > 2000:
            return 1.5
        # Linear interpolation between 1000 and 2000 Elo
        # At 1000 Elo: 0.2
        # At 2000 Elo: 1.5
        # Range of 1.3 (1.5 - 0.2) over 1000 Elo points = 0.0013 per point
        return 0.2 + (avg_elo - 1000) * 0.0013

    def _get_random_position_from_moves(
        self, moves: List[str]
    ) -> Tuple[str, List[str]]:
        board = chess.Board()
        valid_moves = []
        for move in moves:
            try:
                board.push_uci(move)
                valid_moves.append(move)
            except:
                break

        if len(valid_moves) < 2:
            return self.initial_fen, valid_moves

        position_idx = random.randint(0, len(valid_moves) - 1)
        board = chess.Board()
        for move in valid_moves[:position_idx]:
            board.push_uci(move)
        return board.fen(), valid_moves[position_idx:]

    def process_lichess_game(self, example: Dict) -> Optional[Dict]:
        moves = self._parse_pgn(example.get("movetext", None))

        avg_elo = (
            int(example.get("WhiteElo", 1000)) + int(example.get("BlackElo", 1000))
        ) / 2
        weight = self._calculate_game_weight(avg_elo)

        from_middle = random.random() < self.mid_game_prob
        if from_middle:
            fen, moves = self._get_random_position_from_moves(moves)
        else:
            fen = self.initial_fen

        moves = moves[: self.max_moves]

        return ChessExample(
            fen=fen,
            moves=moves,
            average_elo=avg_elo,
            weight=weight,
            dataset_source="lichess_games",
            from_middle=from_middle,
        ).to_dict()

    def process_lichess_eval(self, example: Dict) -> Optional[Dict]:
        fen = example.get("fen")
        line = example.get("line")

        # Drop sample if fen or line is missing
        if fen is None or line is None or not line.strip():
            return None

        eval_score = example.get("score", 0)
        is_mate = example.get("is_mate", False)
        depth = example.get("depth", 1)

        # For evals, no mid-game probability selection (already a single FEN)
        from_middle = False

        return ChessExample(
            fen=fen,
            moves=line.split(),
            eval_score=self._normalize_eval_score(eval_score, is_mate),
            weight=min(depth / 30.0, 1.0),
            dataset_source="lichess_evals",
            from_middle=from_middle,
        ).to_dict()

    def process_puzzle(self, example: Dict) -> Optional[Dict]:
        fen = example.get("FEN", None)
        moves = example.get("Moves", None)

        puzzle_rating = float(example.get("Rating", 1500))
        weight = 1.0 - min(abs(puzzle_rating - 1400) / 1000.0, 0.5)

        # Puzzles are typically positions, so from_middle is False
        from_middle = False

        return ChessExample(
            fen=fen,
            moves=moves.split() if moves else None,
            weight=weight,
            dataset_source="puzzles",
            from_middle=from_middle,
        ).to_dict()

    def process_laion_game(self, example: Dict) -> Optional[Dict]:
        moves = example.get("Moves", None)

        from_middle = random.random() < self.mid_game_prob
        if from_middle and moves:
            fen, moves = self._get_random_position_from_moves(moves)
        else:
            fen = self.initial_fen

        moves = moves[: self.max_moves]

        return ChessExample(
            fen=fen,
            moves=moves,
            average_elo=2000,
            weight=2.0,
            dataset_source="laion_games",
            from_middle=from_middle,
        ).to_dict()


class ChessDatasetManager:
    """Manages the configuration and loading of chess datasets"""

    AVAILABLE_DATASETS = {
        "lichess_games": {
            "path": "Lichess/standard-chess-games",
            "columns_to_remove": [
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
        "lichess_evals": {
            "path": "Lichess/chess-position-evaluations",
            "columns_to_remove": ["fen", "line", "depth", "knodes", "cp", "mate"],
        },
        "puzzles": {
            "path": "Lichess/chess-puzzles",
            "columns_to_remove": [
                "PuzzleId",
                "FEN",
                "Moves",
                "Rating",
                "RatingDeviation",
                "Popularity",
                "NbPlays",
                "Themes",
                "GameUrl",
                "OpeningTags",
            ],
        },
        "laion_games": {
            "path": "laion/strategic_game_chess",
            "columns_to_remove": ["Moves", "Termination", "Result"],
        },
    }

    def __init__(self, processor: ChessDataProcessor):
        self.processor = processor
        self.dataset_configs = {}

        # Initialize with default configurations (all disabled)
        for dataset_name in self.AVAILABLE_DATASETS:
            self.dataset_configs[dataset_name] = DatasetConfig(
                name=dataset_name, probability=0.0, enabled=False
            )

    def add_dataset(self, name: str, probability: float, enabled: bool = True):
        """Add or update a dataset configuration"""
        if name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        self.dataset_configs[name] = DatasetConfig(
            name=name, probability=probability, enabled=enabled
        )

    def _normalize_probabilities(self) -> Dict[str, float]:
        """Normalize probabilities of enabled datasets to sum to 1"""
        enabled_configs = {
            name: config
            for name, config in self.dataset_configs.items()
            if config.enabled
        }

        if not enabled_configs:
            raise ValueError("No datasets are enabled")

        total_prob = sum(config.probability for config in enabled_configs.values())

        if total_prob == 0:
            # If all probabilities are 0, set equal probabilities
            equal_prob = 1.0 / len(enabled_configs)
            return {name: equal_prob for name in enabled_configs}

        # Normalize probabilities
        return {
            name: (config.probability / total_prob)
            for name, config in enabled_configs.items()
        }

    def make_dataset(self) -> Dataset:
        """Create combined dataset based on current configuration"""
        # Get normalized probabilities for enabled datasets
        normalized_probs = self._normalize_probabilities()

        # Initialize processor methods mapping
        processor_methods = {
            "lichess_games": self.processor.process_lichess_game,
            "lichess_evals": self.processor.process_lichess_eval,
            "puzzles": self.processor.process_puzzle,
            "laion_games": self.processor.process_laion_game,
        }

        # Load and process enabled datasets
        processed_datasets = []
        probabilities = []

        for dataset_name, probability in normalized_probs.items():
            # Load dataset
            dataset_info = self.AVAILABLE_DATASETS[dataset_name]
            raw_dataset = load_dataset(dataset_info["path"], streaming=True)["train"]

            # Process dataset
            processed_dataset = raw_dataset.map(
                processor_methods[dataset_name],
                remove_columns=dataset_info["columns_to_remove"],
            )

            processed_datasets.append(processed_dataset)
            probabilities.append(probability)

        # Combine datasets
        return interleave_datasets(processed_datasets, probabilities=probabilities)


def make_train_dataset(
    datasets_config: Dict[str, float], mid_game_prob: float = 0.0
) -> Dataset:
    """
    Create the training dataset with configurable dataset selection

    Args:
        datasets_config: Dictionary mapping dataset names to their probabilities
                        e.g., {"lichess_games": 0.35, "lichess_evals": 0.25}
    """
    processor = ChessDataProcessor(mid_game_prob=mid_game_prob)
    manager = ChessDatasetManager(processor)

    # Configure datasets based on input
    for dataset_name, probability in datasets_config.items():
        manager.add_dataset(dataset_name, probability)

    return manager.make_dataset()


def get_eval_dataset(
    dataset: Dataset,
    size: int = 10000,
    make_eval_dataset: bool = False,
    eval_path: str = "amazingvince/chess_eval_set",
) -> Tuple[Dataset, Dataset]:
    """
    Create or load an evaluation dataset.

    Args:
        dataset: The training dataset.
        size: Number of samples to use for evaluation.
        make_eval_dataset: Whether to create a new evaluation dataset from the training set.
        eval_path: Hub path for the evaluation dataset.

    Returns:
        A tuple of (training_dataset, evaluation_dataset).
    """
    # Filter out examples without moves or FEN
    dataset = dataset.shuffle(seed=42069).filter(
        lambda x: (x["moves"] is not None or x["fen"] is not None)
    )

    if make_eval_dataset:
        # Create new evaluation dataset from training data
        eval_dataset = dataset.take(size)
        eval_dataset = Dataset.from_list(list(eval_dataset))
        # Uncomment and configure if you want to push the dataset to the hub:
        # eval_dataset.push_to_hub(eval_path)

        # Remove evaluation samples from training dataset
        dataset = dataset.skip(size)
    else:
        # Load existing evaluation dataset
        eval_dataset = load_dataset(eval_path)["train"]

    return dataset, eval_dataset


def make_training_and_eval_datasets(
    datasets_config: Dict[str, float],
    mid_game_prob: float = 0.0,
    eval_size: int = 2000,
    make_eval_dataset: bool = True,
    eval_path: str = "amazingvince/chess_eval_set",
) -> Tuple[Dataset, Dataset]:
    """
    Set up the training and evaluation datasets.

    Args:
        datasets_config: Configuration for dataset selection and probabilities.
        eval_size: Number of samples to use for evaluation.
        make_eval_dataset: Whether to create a new evaluation dataset.
        eval_path: Hub path for the evaluation dataset.

    Returns:
        A tuple (train_dataset, eval_dataset).
    """
    # Create initial training dataset with specified configuration
    train_dataset = make_train_dataset(datasets_config, mid_game_prob=mid_game_prob)

    # Split into training and evaluation datasets
    train_dataset, eval_dataset = get_eval_dataset(
        train_dataset,
        size=eval_size,
        make_eval_dataset=make_eval_dataset,
        eval_path=eval_path,
    )

    return train_dataset, eval_dataset


if __name__ == "__main__":
    train_dataset, eval_dataset = make_training_and_eval_datasets()
    # print(list(train_dataset.take(10)))
    # print(list(eval_dataset.take(10)))
