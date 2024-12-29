import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn
import chess.variant
import logging
import random
from datasets import Dataset, interleave_datasets, load_dataset


from fen_utils import tokenize_fen

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# # Remove default handlers
# for handler in logger.handlers[:]:
#     logger.removeHandler(handler)

# # Create a file handler
# file_handler = logging.FileHandler("chess_example.log")
# file_handler.setLevel(logging.DEBUG)

# # Create a logging format
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# file_handler.setFormatter(formatter)

# # Add the file handler to the logger
# logger.addHandler(file_handler)


@dataclass
class ChessExample:
    """Represents a single chess game example with its metadata."""

    text: str
    moves: List[str]
    fen: str
    average_elo: Optional[float] = None
    weight: float = 1.0
    dataset_source: str = "lichess_games"
    is_valid: bool = True
    from_middle: bool = False
    eval_score: Optional[float] = None
    variant: str = "standard"

    def to_dict(self) -> Dict:
        """Convert the example to a dictionary format."""
        return {
            key: (
                float(value)
                if key in ["average_elo", "weight", "eval_score"] and value is not None
                else value
            )
            for key, value in self.__dict__.items()
        }


class ChessProcessor:
    """Processes chess game data from various sources."""

    DATASET_CONFIGS = {
        "lichess_games": {
            "path": "Lichess/standard-chess-games",
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
        "lichess_960": {
            "path": "Lichess/chess960-chess-games",
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
                "Termination",
                "TimeControl",
                "movetext",
                "FEN",
            ],
            "variant": "chess960",
        },
        "lichess_antichess": {
            "path": "Lichess/antichess-chess-games",
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
                "Termination",
                "TimeControl",
                "movetext",
            ],
            "variant": "antichess",
        },
        "lichess_atomic": {
            "path": "Lichess/atomic-chess-games",
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
                "Termination",
                "TimeControl",
                "movetext",
            ],
            "variant": "atomic",
        },
        "lichess_evals": {
            "path": "Lichess/chess-position-evaluations",
            "remove_cols": ["fen", "line", "depth", "knodes", "cp", "mate"],
        },
        "puzzles": {
            "path": "Lichess/chess-puzzles",
            "remove_cols": [
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
            "remove_cols": ["Moves", "Termination", "Result"],
        },
    }

    def __init__(self, max_moves: int = 500, mid_game_prob: float = 0.7):
        self.max_moves = max_moves
        self.mid_game_prob = mid_game_prob
        self.variant_boards = {
            "standard": chess.Board,
            "chess960": chess.Board.from_chess960_pos,
            # Use the correct chess.variant module
            "antichess": chess.variant.AntichessBoard,
            "atomic": chess.variant.AtomicBoard,
        }

    def _create_board(self, variant: str, fen: Optional[str] = None) -> chess.Board:
        """Create a board of the specified variant."""
        try:
            if variant == "chess960" and fen:
                # For Chess960, we need to parse the FEN to get the position number
                board = chess.Board(fen, chess960=True)
                scharnagl = board.chess960_pos()
                if scharnagl is None:
                    logger.warning(
                        "Invalid Chess960 position, falling back to standard chess"
                    )
                    return chess.Board(fen)
                return chess.Board.from_chess960_pos(scharnagl)
            elif variant in self.variant_boards:
                board_constructor = self.variant_boards[variant]
                if variant in ["antichess", "atomic"]:
                    # For variant boards, we need to create the board first then set FEN if provided
                    board = board_constructor()
                    if fen:
                        board.set_fen(fen)
                    return board
                else:
                    # For standard and chess960, we can pass FEN directly
                    return board_constructor(fen) if fen else board_constructor()
            else:
                logger.warning(
                    f"Unknown variant '{variant}', falling back to standard chess"
                )
                return chess.Board(fen) if fen else chess.Board()
        except Exception as e:
            logger.error(f"Error creating board for variant {variant}: {str(e)}")
            return chess.Board(fen) if fen else chess.Board()

    def _get_random_position_from_moves(
        self,
        moves: List[str],
        variant: str = "standard",
        initial_fen: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        """Get a random position from a sequence of moves.

        For Chess960, initial_fen must be provided to maintain correct castling rights.
        """
        # Create initial board with proper variant and starting position
        board = self._create_board(variant, initial_fen)
        valid_moves = []

        # Track if this is Chess960
        is_960 = variant == "chess960"
        if is_960 and initial_fen is None:
            logger.warning(
                "Chess960 position requested without initial FEN; castling rights may be incorrect."
            )

        # Validate and collect moves
        for move in moves:
            move = self._translate_castling_move(board, move)
            try:
                # For Chess960, we may need to ensure castling moves are properly translated
                board.push_uci(move)
                valid_moves.append(move)
            except Exception as e:
                logger.warning(f"Invalid move '{move}' for variant {variant}: {str(e)}")
                break

        if len(valid_moves) < 2:
            return board.fen(), valid_moves

        # Get random position index
        position_idx = random.randint(0, len(valid_moves) - 1)

        # Replay moves up to that random position
        board = self._create_board(variant, initial_fen)
        for move in valid_moves[:position_idx]:
            move = self._translate_castling_move(board, move)
            try:
                board.push_uci(move)
            except Exception as e:
                logger.error(f"Error replaying move '{move}': {str(e)}")
                break

        return board.fen(), valid_moves[position_idx:]

    def process_game(self, example: Dict, source: str = "lichess_games") -> Dict:
        """Process examples from any source."""
        variant = self.DATASET_CONFIGS.get(source, {}).get("variant", "standard")

        if source == "lichess_960":
            # Extract starting FEN if available
            starting_fen = example.get("FEN", None)
            if not starting_fen:
                logger.warning(
                    "Chess960 game without initial FEN; falling back to standard."
                )
                variant = "standard"
            if starting_fen:
                try:
                    board = self._create_board("chess960", starting_fen)
                    starting_fen = board.fen()
                except Exception as e:
                    logger.error(f"Error processing Chess960 FEN: {str(e)}")
                    logger.error(f"Raw FEN: {starting_fen}")
                    return self._create_invalid_example(source)

        # Special handling by source
        if source == "lichess_evals":
            return self._process_lichess_eval(example)
        elif source == "puzzles":
            return self._process_puzzle(example)
        elif source in [
            "lichess_games",
            "lichess_960",
            "lichess_antichess",
            "lichess_atomic",
        ]:
            return self._process_lichess_game(example, source, variant)
        elif source == "laion_games":
            return self._process_laion_game(example)
        else:
            logger.warning(f"Unknown source: {source}")
            return self._create_invalid_example(source)

    def _process_lichess_game(self, example: Dict, source: str, variant: str) -> Dict:
        """Process a LiChess game (standard, 960, antichess, atomic)."""
        initial_fen = example.get("FEN", None)
        if variant == "chess960" and not initial_fen:
            logger.warning(
                "Chess960 game without initial FEN; falling back to standard."
            )
            variant = "standard"

        # For Chess960, try to get the initial position from the game

        moves = self._get_moves(example, source, initial_fen)
        if not moves:
            return self._create_invalid_example(source)

        from_middle = random.random() < self.mid_game_prob
        if from_middle:
            fen, moves = self._get_random_position_from_moves(
                moves, variant, initial_fen
            )
        else:
            board = self._create_board(variant, initial_fen)
            fen = board.fen()

        moves = moves[: self.max_moves]
        avg_elo = self._calculate_elo(example)

        # Format final text
        text = self._format_moves(
            fen=fen,
            moves=moves,
            elo_token=self._get_elo_token(avg_elo),
            source=source,
            variant=variant,
            from_middle=from_middle,
        )

        return ChessExample(
            text=text,
            moves=moves,
            fen=fen,
            variant=variant,
            average_elo=avg_elo,
            weight=self._calculate_weight(avg_elo),
            dataset_source=source,
            is_valid=True,
            from_middle=from_middle,
        ).to_dict()

    def _process_laion_game(self, example: Dict) -> Dict:
        """Process a LAION game (placeholder logic—adapt as needed)."""
        moves = example.get("Moves")
        if not moves:
            return self._create_invalid_example("laion_games")

        moves = moves[: self.max_moves]
        board = self._create_board("standard")
        from_middle = random.random() < self.mid_game_prob
        if from_middle:
            fen, moves = self._get_random_position_from_moves(
                moves, "standard", board.fen()
            )
        else:
            board = self._create_board("standard", board.fen())
            fen = board.fen()

        text = self._format_moves(
            fen=fen,
            moves=moves,
            elo_token="<|above_2000|>",
            source="laion_games",
            variant="standard",
        )

        return ChessExample(
            text=text,
            moves=moves,
            fen=fen,
            variant="standard",
            dataset_source="laion_games",
            from_middle=False,
        ).to_dict()

    # TODO: Fix this if ever needed NOT WORKING ATM!
    def _process_lichess_eval(self, example: Dict) -> Dict:
        """Process a lichess evaluation position."""
        fen = example.get("fen")
        line = example.get("line")

        if not fen or not line:
            return self._create_invalid_example("lichess_evals")

        eval_score = example.get("score", 0)
        is_mate = example.get("is_mate", False)
        depth = example.get("depth", 1)

        moves = line.split()
        moves = moves[: self.max_moves]

        text = self._format_moves(
            fen=fen,
            moves=moves,
            elo_token="<|engine|>",
            source="lichess_evals",
            variant="standard",
            from_middle=True,
        )

        return ChessExample(
            text=text,
            moves=moves,
            fen=fen,
            eval_score=self._normalize_eval_score(eval_score, is_mate),
            weight=min(depth / 30.0, 1.0),
            dataset_source="lichess_evals",
            from_middle=True,
            variant="standard",
        ).to_dict()

    def _process_puzzle(self, example: Dict) -> Dict:
        """Process a chess puzzle."""
        fen = example.get("FEN")
        moves_str = example.get("Moves")

        if not fen or not moves_str:
            return self._create_invalid_example("puzzles")

        moves = moves_str.split()
        moves = moves[: self.max_moves]

        puzzle_rating = float(example.get("Rating", 1500))
        weight = 1.0 - min(abs(puzzle_rating - 1400) / 1000.0, 0.5)

        text = self._format_moves(
            fen=fen,
            moves=moves,
            elo_token="<|above_2000|>",
            source="puzzles",
            variant="standard",
            from_middle=True,
        )

        return ChessExample(
            text=text,
            moves=moves,
            fen=fen,
            weight=weight,
            dataset_source="puzzles",
            from_middle=True,
            variant="standard",
        ).to_dict()

    def _get_moves(
        self, example: Dict, source: str, initial_fen: Optional[str] = None
    ) -> List[str]:
        """Extract moves from PGN-based or text-based game formats with improved error handling."""
        try:
            if source in [
                "lichess_960",
                "lichess_games",
                "lichess_antichess",
                "lichess_atomic",
            ]:
                # Map source to variant
                variant_map = {
                    "lichess_960": "Chess960",
                    "lichess_games": "Standard",
                    "lichess_antichess": "Antichess",
                    "lichess_atomic": "Atomic",
                }
                variant = variant_map[source]

                # Log the initial state
                logger.debug(
                    f"Processing {variant} game with initial FEN: {initial_fen or 'standard starting position'}"
                )

                # Create a proper PGN string with headers
                pgn_str = f"""[Variant "{variant}"]
[FEN "{initial_fen if initial_fen else chess.STARTING_FEN}"]
[Event "Dummy"]
[Site "{variant}"]
[Date "2024.01.01"]
{example.get("movetext", "")}\n"""

                # Log the full PGN for debugging
                logger.debug(f"Generated PGN:\n{pgn_str}")

                # Read game and extract moves
                game = chess.pgn.read_game(io.StringIO(pgn_str))
                if not game:
                    logger.error(f"Failed to parse game from PGN in {source}")
                    logger.debug(f"Raw movetext: {example.get('movetext', '')}")
                    return []

                try:
                    # Create board for move validation
                    board = self._create_board(variant.lower(), initial_fen)
                    moves = []

                    # Validate each move
                    for move in game.mainline_moves():
                        move_uci = move.uci()
                        legal_moves = [m.uci() for m in board.legal_moves]

                        if move_uci not in legal_moves:
                            logger.warning(
                                f"Move {move_uci} not in legal moves for position.\n"
                                f"FEN: {board.fen()}\n"
                                f"Legal moves: {legal_moves}"
                            )

                        moves.append(move_uci)
                        try:
                            board.push_uci(move_uci)
                        except ValueError as e:
                            logger.error(
                                f"Invalid move {move_uci} in {variant}:\n"
                                f"Position: {board.fen()}\n"
                                f"Error: {str(e)}"
                            )
                            break

                    return moves

                except Exception as e:
                    logger.error(
                        f"Error validating moves for {variant} game:\n"
                        f"Initial FEN: {initial_fen}\n"
                        f"Error: {str(e)}"
                    )
                    return []

            elif source == "laion_games":
                moves = example.get("Moves", "")
                if not moves:
                    return []

                # Validate moves against a board
                board = chess.Board()
                moves = []

                logger.debug(f"Processing LAION game moves: {moves}")

                for move in moves:
                    try:
                        legal_moves = [m.uci() for m in board.legal_moves]
                        if move not in legal_moves:
                            logger.warning(
                                f"Move {move} not in legal moves.\n"
                                f"Position: {board.fen()}\n"
                                f"Legal moves: {legal_moves}"
                            )
                        moves.append(move)
                        board.push_uci(move)
                    except Exception as e:
                        logger.error(
                            f"Invalid move {move} in LAION game:\n"
                            f"Position: {board.fen()}\n"
                            f"Error: {str(e)}"
                        )
                        break
                return moves

            return []

        except Exception as e:
            logger.error(
                f"Unexpected error processing moves from {source}:\n"
                f"Error: {str(e)}\n"
                f"Example data: {example}"
            )
            return []

    def _translate_castling_move(self, board: chess.Board, move_uci: str) -> str:
        """Translate castling moves with improved logging."""
        # Quick check: if already legal, return immediately
        legal_uci_moves = [m.uci() for m in board.legal_moves]
        if move_uci in legal_uci_moves:
            return move_uci

        # Log the attempted castling move
        if move_uci in ("e1h1", "e1a1", "e8h8", "e8a8"):
            logger.debug(
                f"Attempting to translate castling move {move_uci}\n"
                f"Position: {board.fen()}\n"
                f"Legal moves: {legal_uci_moves}"
            )

            king_origin = move_uci[:2]
            for lm in board.legal_moves:
                if board.is_castling(lm) and lm.uci().startswith(king_origin):
                    logger.debug(f"Translated castling move {move_uci} to {lm.uci()}")
                    return lm.uci()

            logger.warning(
                f"Failed to translate castling move {move_uci}\n"
                f"Position: {board.fen()}\n"
                f"No matching legal castling move found"
            )

        return move_uci

    def _calculate_elo(self, example: Dict) -> float:
        """Calculate average ELO rating from WhiteElo and BlackElo.

        Args:
            example: Dictionary containing game data with possible WhiteElo and BlackElo keys

        Returns:
            float: Average ELO rating, defaults to 1000.0 if ratings are invalid
        """
        try:
            white_elo = example.get("WhiteElo")
            black_elo = example.get("BlackElo")

            # Convert to int if values exist, otherwise use 1000
            white_elo = int(white_elo) if white_elo is not None else 1000
            black_elo = int(black_elo) if black_elo is not None else 1000

            # Validate reasonable ELO range (e.g., between 100 and 4000)
            if not (100 <= white_elo <= 4000 and 100 <= black_elo <= 4000):
                return 1000.0

            return (white_elo + black_elo) / 2

        except (ValueError, TypeError):
            return 1000.0

    def _calculate_weight(self, elo: float) -> float:
        """Calculate game weight based on ELO."""
        if elo < 1000:
            return 0.2
        if elo > 2000:
            return 1.5
        # Linear scale between 1000-2000
        return 0.2 + (elo - 1000) * 0.0013

    @staticmethod
    def _get_elo_token(elo: float) -> str:
        """Get a rough ELO category token."""
        if elo < 1000:
            return "<|below_1000|>"
        if elo < 2000:
            return "<|1000_2000|>"
        return "<|above_2000|>"

    @staticmethod
    def _normalize_eval_score(score: float, is_mate: bool) -> float:
        """Normalize evaluation score for engine lines. Example approach:
        - If mate, clamp the score to a large +/-
        - Else keep cp as is
        """
        if is_mate:
            return max(min(score, 9999), -9999)
        return score

    @classmethod
    def add_eos_token(
        cls,
        moves: List[str],
        source: str,
        variant: str = "standard",
        initial_fen: Optional[str] = None,
        from_middle: bool = False,
    ) -> str:
        """Add appropriate end-of-sequence or turn-separator token."""
        # Special cases for evaluation positions and puzzles
        if source in ["laion_games"]:
            return "<|end|>"

        try:
            # If no initial FEN is provided, use the standard starting position
            if not isinstance(initial_fen, str):
                print("initial_fen is not a string")
                print(moves, source, variant, initial_fen, from_middle)
            starting_fen = (
                initial_fen if isinstance(initial_fen, str) else chess.STARTING_FEN
            )

            # Create appropriate board based on variant
            if variant == "chess960" and not from_middle:
                if starting_fen:
                    board = chess.Board(starting_fen, chess960=True)
                    scharnagl = board.chess960_pos()
                    if scharnagl is not None:
                        board = chess.Board.from_chess960_pos(scharnagl)
                    else:
                        # Fallback to standard chess if invalid Chess960 position
                        board = chess.Board(starting_fen)
                else:
                    # Without FEN, use standard chess board
                    board = chess.Board()
            elif variant == "antichess":
                board = chess.variant.AntichessBoard(fen=starting_fen)
            elif variant == "atomic":
                board = chess.variant.AtomicBoard(fen=starting_fen)
            else:
                board = chess.Board(fen=starting_fen)

            # Apply all moves using the static translate_castling_move method
            for move in moves:
                move = cls._translate_castling_move_static(board, move)
                board.push_uci(move)

            # Check game ending conditions specific to each variant
            if variant == "antichess":
                # In Antichess, winning means losing all pieces or having no legal moves
                return "<|end|>" if len(board.legal_moves) == 0 else "<|turn|>"
            elif variant == "atomic":
                # In Atomic chess, game ends when king is blown up or standard checkmate
                return "<|end|>" if board.is_game_over(claim_draw=False) else "<|turn|>"
            else:
                # Standard chess and Chess960 use regular game-over conditions
                return "<|end|>" if board.is_game_over(claim_draw=True) else "<|turn|>"

        except (ValueError, AssertionError) as e:
            logger.warning(f"Error checking game-over status: {str(e)}")
            # If there's an illegal move or other error, fall back to turn token
            return "<|turn|>"

    @staticmethod
    def variant_token_map(variant: str) -> str:
        """Map a variant to a token."""
        return {
            "standard": "<|standard|>",
            "chess960": "<|chess_960|>",
            "antichess": "<|anti_chess|>",
            "atomic": "<|atomic_chess|>",
        }.get(variant, "<|standard|>")

    @staticmethod
    def _format_moves(
        fen: str,
        moves: List[str],
        elo_token: str,
        source: str,
        variant: str,
        from_middle: bool = False,
    ) -> str:
        """
        Format moves into the required string format.
        Currently, this uses the same approach as the original code:
          - ELO token
          - <|start|> <move1> <|turn|> <move2> ...
          - <|end|> or <|turn|> depending on game-over
        If you'd like to incorporate the fen/variant textually, you can do so.
        """
        formatted_moves = []
        for move in moves:
            from_square = move[:2]
            to_square = move[2:4]
            promotion = f" {move[4].lower()}" if len(move) > 4 else ""
            formatted_moves.append(f"{from_square} {to_square} {promotion}")

        move_sequence = " <|turn|> ".join(formatted_moves)
        # eos_token = ChessProcessor.add_eos_token(
        #     moves, source, variant, fen, from_middle
        # )
        eos_token = "<|turn|>"

        # Example with placeholders for fen/variant if desired in the prompt text
        # (you can expand this if you want to embed them):
        text = f"<|start|> {elo_token} {ChessProcessor.variant_token_map(variant)} {tokenize_fen(fen)} <|sep|> {move_sequence} {eos_token}"
        return text

    def _create_invalid_example(self, source: str) -> Dict:
        """Create a dict for an invalid example."""
        return ChessExample(
            text="",
            moves=[],
            fen="",
            dataset_source=source,
            is_valid=False,
        ).to_dict()


def create_dataset(
    config: Dict[str, float], eval_size: int = 2048, mid_game_prob: float = 0.0
) -> Tuple[Dataset, Dataset]:
    """
    Create training and evaluation datasets by interleaving from multiple sources.
    """
    processor = ChessProcessor(mid_game_prob=mid_game_prob)
    datasets = []
    probabilities = []
    total_prob = sum(config.values())

    for name, prob in config.items():
        if name not in ChessProcessor.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {name}")

        dataset_info = ChessProcessor.DATASET_CONFIGS[name]

        # Load the dataset with its specific configuration
        dataset = load_dataset(
            dataset_info["path"],
            streaming=True,
        )["train"]

        # Create a closure to ensure the source name is properly captured
        def process_with_source(example, src_name=name):
            return processor.process_game(example, src_name)

        # Process the dataset with the correct source
        processed = dataset.map(
            process_with_source,
            remove_columns=dataset_info["remove_cols"],
        ).filter(lambda x: x["is_valid"])

        datasets.append(processed)
        probabilities.append(prob / total_prob)

    # Interleave datasets with their respective probabilities
    combined = interleave_datasets(
        datasets,
        probabilities=probabilities,
    ).shuffle(seed=42)

    # Split into eval and train
    eval_dataset = combined.take(eval_size)
    eval_dataset = Dataset.from_list(list(eval_dataset))
    train_dataset = combined.skip(eval_size)

    return train_dataset, eval_dataset


if __name__ == "__main__":
    # Example configuration
    config = {
        "lichess_games": 0.35,
        "lichess_960": 0.1,
        "lichess_antichess": 0.05,
        "lichess_atomic": 0.05,
        "puzzles": 0.2,
        "laion_games": 0.35,
    }

    # Create datasets with 30% chance of mid-game starts
    train_dataset, eval_dataset = create_dataset(
        config, mid_game_prob=0.7, eval_size=2000
    )

    # Print some statistics
    print("Train dataset:", train_dataset)
    print("Eval dataset:", eval_dataset)

    # Example of processing one item
    sample = next(iter(train_dataset))
    print("\nSample item:", sample)

    sample = next(iter(train_dataset))
    print("\nSample item:", sample)

    sample = next(iter(train_dataset))
    print("\nSample item:", sample)
