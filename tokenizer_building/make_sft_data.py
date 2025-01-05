import chess
import chess.engine
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class EvalTokens:
    """Single evaluation token with optional magnitude."""

    quality: str  # <mate good>//<mate bad>/<good>/<bad>/<equal>
    magnitude: Optional[int] = None  # None for mate/equal, int for good/bad


class Evaluator:
    """
    Maps engine evaluations to tokens from the perspective
    of the player to move.
    """

    # Centipawn thresholds for magnitude
    MAGNITUDE_LEVELS = [
        30,  # tiny edge
        60,  # small advantage
        100,  # clear advantage
        175,  # big advantage
        250,  # winning
        400,  # very winning
        600,  # completely winning
        900,  # totally winning
    ]

    @staticmethod
    def get_eval_tokens(
        score: chess.engine.PovScore, player_turn: chess.Color
    ) -> EvalTokens:
        """
        Convert an engine score to evaluation tokens from the perspective of `player_turn`.

        Args:
            score (chess.engine.PovScore): The engine score.
            player_turn (chess.Color): True for White, False for Black.
        """
        # Handle mate scores first
        if score.is_mate():
            mate_moves = score.relative.moves
            if mate_moves is not None:
                # If positive, side to move can force mate; if negative, side to move is getting mated
                quality = "<mate good>" if mate_moves > 0 else "<mate bad>"
                return EvalTokens(quality)

        # Otherwise, it's a centipawn score
        cp_score = score.relative.score(mate_score=100000)

        # Near 0 is equal
        if -15 <= cp_score <= 15:
            return EvalTokens("<equal>")

        # Determine magnitude
        magnitude = 0
        for level in Evaluator.MAGNITUDE_LEVELS:
            if abs(cp_score) > level:
                magnitude = level
            else:
                break

        # If we didn't exceed the first threshold, but it's outside [-15, 15]
        if magnitude == 0:
            magnitude = Evaluator.MAGNITUDE_LEVELS[0]

        # Positive is good for side to move, negative is bad
        quality = "<good>" if cp_score > 0 else "<bad>"
        return EvalTokens(quality, magnitude)


class ChessReasoningGenerator:
    """
    Generates 'chain of thought' style reasoning for a chess position
    using a UCI chess engine and retrieving principal variations.
    """

    def __init__(self, engine_path: str):
        self.engine_path = engine_path

    def _format_line(
        self, moves: List[chess.Move], score: chess.engine.PovScore, board: chess.Board
    ) -> str:
        """
        Format a single variation line in UCI notation along with evaluation tokens.
        """
        move_strs = [m.uci() for m in moves]
        eval_tokens = Evaluator.get_eval_tokens(score, board.turn)

        # Format based on token type
        if eval_tokens.magnitude is None:
            return f"{' '.join(move_strs)} {eval_tokens.quality}"
        else:
            return (
                f"{' '.join(move_strs)} {eval_tokens.quality} <{eval_tokens.magnitude}>"
            )

    def generate_reasoning(
        self, fen: str, num_lines: int = 3, depth: int = 25, moves_to_show: int = 4
    ) -> str:
        """
        Generate chain-of-thought reasoning for a given FEN position.
        """
        board = chess.Board(fen)
        player_turn = board.turn
        legal_moves = list(board.legal_moves)

        # Early exit if no legal moves
        if not legal_moves:
            return None

        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            try:
                # Ask for multiple lines at the chosen depth
                analysis = engine.analyse(
                    board, limit=chess.engine.Limit(depth=depth), multipv=num_lines
                )

                # Top line analysis for detecting forced mate
                top_info = analysis[0] if analysis else {}
                top_score = top_info.get("score") if top_info else None

                # If there's a mate in top line, do a deeper search
                if top_score and top_score.is_mate():
                    deeper_depth = max(depth, 25)
                    mate_analysis = engine.analyse(
                        board, limit=chess.engine.Limit(depth=deeper_depth), multipv=1
                    )
                    if mate_analysis:
                        analysis[0] = mate_analysis[0]
            except Exception:
                analysis = []

        # Build the output
        reasoning_parts = []

        # Position evaluation from the top line
        if analysis and "score" in analysis[0]:
            pos_tokens = Evaluator.get_eval_tokens(analysis[0]["score"], player_turn)
            if pos_tokens.magnitude is None:
                reasoning_parts.append(f"<pos>{pos_tokens.quality}</pos>")
            else:
                reasoning_parts.append(
                    f"<pos>{pos_tokens.quality} <{pos_tokens.magnitude}></pos>"
                )

        # List lines
        reasoning_parts.append("<consider>")

        lines_added = 0
        for info in analysis:
            pv = info.get("pv", [])
            score = info.get("score", None)
            if not pv or not score:
                continue

            # Show full line for mate variations, otherwise limit to moves_to_show
            moves = pv if score.is_mate() else pv[:moves_to_show]

            if moves:
                line_str = self._format_line(moves, score, board)
                reasoning_parts.append(line_str)
                lines_added += 1

            if lines_added >= num_lines:
                break

        # If no lines were added but we have legal moves, show at least one
        if lines_added == 0 and legal_moves:
            move = legal_moves[0]
            reasoning_parts.append(f"{move.uci()} (fallback move)")

        reasoning_parts.append("</consider>")

        # Best Move Selection
        best_move = analysis[0].get("pv", [None])[0] if analysis else None
        if not best_move and legal_moves:
            best_move = legal_moves[0]

        move_str = best_move.uci() if best_move else "no legal moves"
        reasoning_parts.append(f"<choose>{move_str}</choose>")

        return "\n".join(reasoning_parts)


def main():
    test_positions = [
        # 1) Equal position from starting position (White to move)
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # 2) Small advantage for White, Black to move
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 2",
        # 3) Potential advantage for White
        "rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        # 4) Mate in 2 for White, Black to move
        "r1bq2r1/b4pk1/p1pp1p2/1p2pP2/1P2P1PB/3P4/1PPQ2P1/R3K2R w",
    ]

    generator = ChessReasoningGenerator(
        engine_path="/home/vincent/Documents/stockfish-ubuntu-x86-64-vnni512/stockfish/stockfish-ubuntu-x86-64-vnni512"
    )

    for fen in test_positions:
        print("\nPosition:", fen)
        print("-" * 50)
        reasoning = generator.generate_reasoning(
            fen, num_lines=3, depth=12, moves_to_show=4
        )
        print(reasoning)
        print("-" * 50)


if __name__ == "__main__":
    main()
