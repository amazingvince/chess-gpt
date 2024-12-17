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


class ChessDataProcessor:
    def __init__(
        self,
        mid_game_prob: float = 0.7,
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
            return 0.5
        if avg_elo > 2000:
            return 2.0
        return 0.5 + (avg_elo - 1000) * 0.001

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
        # if not moves:
        #     return None

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

    # def process_lichess_eval(self, example: Dict) -> Optional[Dict]:
    #     fen = example.get("fen")
    #     line = example.get("line")

    #     # Drop sample if fen or line is missing
    #     if fen is None or line is None or not line.strip():
    #         return None

    #     eval_score = example.get("score", 0)
    #     is_mate = example.get("is_mate", False)
    #     depth = example.get("depth", 1)

    #     # For evals, no mid-game probability selection (already a single FEN)
    #     from_middle = False

    #     return ChessExample(
    #         fen=fen,
    #         moves=line.split(),
    #         eval_score=self._normalize_eval_score(eval_score, is_mate),
    #         weight=min(depth / 30.0, 1.0),
    #         dataset_source="lichess_evals",
    #         from_middle=from_middle,
    #     ).to_dict()

    def process_puzzle(self, example: Dict) -> Optional[Dict]:
        fen = example.get("FEN", None)
        moves = example.get("Moves", None)

        # # Drop if FEN or Moves is missing
        # if fen is None or moves is None or not moves.strip():
        #     return None

        puzzle_rating = float(example.get("Rating", 1500))
        weight = 1.0 - min(abs(puzzle_rating - 1400) / 1000.0, 0.5)

        # Puzzles are typically positions, so from_middle is False
        from_middle = False

        return ChessExample(
            fen=fen,
            moves=moves.split(),
            weight=weight,
            dataset_source="puzzles",
            from_middle=from_middle,
        ).to_dict()

    def process_laion_game(self, example: Dict) -> Optional[Dict]:
        moves = example.get("Moves", None)
        # if not moves:
        #     return None

        from_middle = random.random() < self.mid_game_prob
        if from_middle:
            fen, moves = self._get_random_position_from_moves(moves)
        else:
            fen = self.initial_fen

        moves = moves[: self.max_moves]

        return ChessExample(
            fen=fen,
            moves=moves,
            average_elo=2000,
            weight=1.5,
            dataset_source="laion_games",
            from_middle=from_middle,
        ).to_dict()


def make_train_dataset() -> Dataset:
    """
    Create the training dataset by loading and processing multiple datasets.
    Interleave them with specified probabilities.
    """

    processor = ChessDataProcessor()

    # Load datasets in streaming mode
    datasets = {
        "lichess_games": load_dataset("Lichess/standard-chess-games", streaming=True)[
            "train"
        ],
        # "lichess_evals": load_dataset(
        #     "Lichess/chess-position-evaluations", streaming=True
        # )["train"],
        "laion_games": load_dataset("laion/strategic_game_chess", streaming=True)[
            "train"
        ],
        "puzzles": load_dataset("Lichess/chess-puzzles", streaming=True)["train"],
    }

    # Columns to remove after mapping
    datasets_columns = {
        "lichess_games": [
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
        "lichess_evals": ["fen", "line", "depth", "knodes", "cp", "mate"],
        "laion_games": ["Moves", "Termination", "Result"],
        "puzzles": [
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
    }

    # Process each dataset
    processed_datasets = {
        "lichess_games": datasets["lichess_games"].map(
            processor.process_lichess_game,
            remove_columns=datasets_columns["lichess_games"],
        ),
        # "lichess_evals": datasets["lichess_evals"].map(
        #     processor.process_lichess_eval,
        #     remove_columns=datasets_columns["lichess_evals"],
        # ),
        "puzzles": datasets["puzzles"].map(
            processor.process_puzzle, remove_columns=datasets_columns["puzzles"]
        ),
        "laion_games": datasets["laion_games"].map(
            processor.process_laion_game, remove_columns=datasets_columns["laion_games"]
        ),
    }

    # Interleave datasets with updated probabilities
    combined_dataset = interleave_datasets(
        [
            processed_datasets["lichess_games"],
            # processed_datasets["lichess_evals"],
            processed_datasets["puzzles"],
            processed_datasets["laion_games"],
        ],
        # probabilities=[0.35, 0.25, 0.20, 0.20],
        probabilities=[0.45, 0.20, 0.35],
    )

    return combined_dataset


def get_eval_dataset(
    dataset: Dataset, size: int = 10000, make_eval_dataset: bool = False
) -> Tuple[Dataset, Dataset]:
    """
    Create or load an evaluation dataset.

    Args:
        dataset: The training dataset.
        size: Number of samples to use for evaluation.
        make_eval_dataset: Whether to create a new evaluation dataset from the training set.

    Returns:
        A tuple of (training_dataset, evaluation_dataset).
    """
    path = "amazingvince/chess_eval_set"
    dataset = dataset.shuffle(seed=42).filter(
        lambda x: (x["moves"] is not None or x["fen"] is not None)
    )

    if make_eval_dataset:
        eval_dataset = dataset.take(size)
        eval_dataset = Dataset.from_list(list(eval_dataset))
        # Uncomment and configure if you want to push the dataset to the hub:
        eval_dataset.push_to_hub(path)
    else:
        dataset = dataset.skip(size)
        eval_dataset = load_dataset(path)

    return dataset, eval_dataset


# run this function to get the training and evaluation datasets in clm script
def set_up_data() -> Tuple[Dataset, Dataset]:
    """
    Set up the training and evaluation datasets.

    Returns:
        A tuple (train_dataset, eval_dataset).
    """
    train_dataset = make_train_dataset()
    train_dataset, eval_dataset = get_eval_dataset(train_dataset)
    return train_dataset, eval_dataset


if __name__ == "__main__":
    train_dataset, eval_dataset = set_up_data()
    # print(list(train_dataset.take(10)))
    # print(list(eval_dataset.take(10)))
