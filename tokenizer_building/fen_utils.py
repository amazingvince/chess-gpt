import chess
from typing import List, Set, Dict, Tuple

# Minimized token vocabulary with clear separation of token types
PIECE_TOKENS = {
    "P": "<|P|>",
    "N": "<|N|>",
    "B": "<|B|>",
    "R": "<|R|>",
    "Q": "<|Q|>",
    "K": "<|K|>",
    "p": "<|p|>",
    "n": "<|n|>",
    "b": "<|b|>",
    "r": "<|r|>",
    "q": "<|q|>",
    "k": "<|k|>",
    "empty": "<|_|>",
}

SPECIAL_TOKENS = {
    "turn_w": "<|w|>",
    "turn_b": "<|b|>",
    "castle_K": "<|K+|>",
    "castle_Q": "<|Q+|>",
    "castle_k": "<|k+|>",
    "castle_q": "<|q+|>",
    "no_castle": "<|-|>",
    "no_ep": "<|ep-|>",
}

# Create reverse mappings
REVERSE_PIECE_TOKENS = {v: k for k, v in PIECE_TOKENS.items()}
REVERSE_SPECIAL_TOKENS = {v: k for k, v in SPECIAL_TOKENS.items()}


def get_all_tokens() -> Set[str]:
    """Return the set of all possible tokens."""
    tokens = set(PIECE_TOKENS.values()) | set(SPECIAL_TOKENS.values())

    # Add en passant square tokens
    for file in "abcdefgh":
        for rank in "36":  # Only ranks 3 and 6 are possible
            tokens.add(f"<|{file}{rank}|>")

    return tokens


def tokenize_fen(fen_string: str) -> str:
    """
    Convert a FEN string to space-separated tokens.
    Works with both standard chess and Chess960 positions.

    Args:
        fen_string: A valid FEN string
    Returns:
        Space-separated string of tokens
    """
    try:
        board = chess.Board(fen_string, chess960=True)
    except ValueError as e:
        raise ValueError(f"Invalid FEN string: {e}")

    tokens = []

    # Tokenize piece placement by processing FEN board part directly
    board_part = fen_string.split()[0]
    ranks = board_part.split("/")

    for rank in ranks:
        current_pos = 0
        for char in rank:
            if char.isdigit():
                # Add empty squares
                num_empty = int(char)
                tokens.extend([PIECE_TOKENS["empty"]] * num_empty)
            else:
                # Add piece token
                tokens.append(PIECE_TOKENS[char])

    # Tokenize turn
    tokens.append(SPECIAL_TOKENS[f'turn_{board.turn and "w" or "b"}'])

    # Tokenize castling rights (works for both standard and Chess960)
    if board.castling_rights:
        if board.has_kingside_castling_rights(chess.WHITE):
            tokens.append(SPECIAL_TOKENS["castle_K"])
        if board.has_queenside_castling_rights(chess.WHITE):
            tokens.append(SPECIAL_TOKENS["castle_Q"])
        if board.has_kingside_castling_rights(chess.BLACK):
            tokens.append(SPECIAL_TOKENS["castle_k"])
        if board.has_queenside_castling_rights(chess.BLACK):
            tokens.append(SPECIAL_TOKENS["castle_q"])
    else:
        tokens.append(SPECIAL_TOKENS["no_castle"])

    # Tokenize en passant
    if board.ep_square is not None:
        ep_square = chess.square_name(board.ep_square)
        tokens.append(f"<|{ep_square}|>")
    else:
        tokens.append(SPECIAL_TOKENS["no_ep"])

    # Move numbers
    tokens.append(str(board.halfmove_clock))
    tokens.append(str(board.fullmove_number))

    return " ".join(tokens)


def detokenize_fen(tokenized_string: str) -> str:
    """
    Convert a tokenized string back to a valid FEN string.
    Works with both standard chess and Chess960 positions.

    Args:
        tokenized_string: Space-separated string of tokens
    Returns:
        Valid FEN string
    """
    tokens = tokenized_string.split()

    # First 64 tokens are the board position
    board_tokens = tokens[:64]
    ranks = []

    # Process each rank
    for rank in range(8):
        rank_tokens = board_tokens[rank * 8 : (rank + 1) * 8]
        empty_count = 0
        rank_str = ""

        for token in rank_tokens:
            if token == PIECE_TOKENS["empty"]:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += REVERSE_PIECE_TOKENS[token]

        if empty_count > 0:
            rank_str += str(empty_count)
        ranks.append(rank_str)

    position = "/".join(ranks)

    # Token after board position (index 64) is the turn
    turn = "w" if tokens[64] == SPECIAL_TOKENS["turn_w"] else "b"

    # Process castling rights, starting from index 65
    idx = 65
    castling_tokens = []

    while idx < len(tokens) and tokens[idx] in {
        SPECIAL_TOKENS["castle_K"],
        SPECIAL_TOKENS["castle_Q"],
        SPECIAL_TOKENS["castle_k"],
        SPECIAL_TOKENS["castle_q"],
        SPECIAL_TOKENS["no_castle"],
    }:
        if tokens[idx] == SPECIAL_TOKENS["no_castle"]:
            castling_tokens = ["-"]
            idx += 1
            break
        if tokens[idx] == SPECIAL_TOKENS["castle_K"]:
            castling_tokens.append("K")
        if tokens[idx] == SPECIAL_TOKENS["castle_Q"]:
            castling_tokens.append("Q")
        if tokens[idx] == SPECIAL_TOKENS["castle_k"]:
            castling_tokens.append("k")
        if tokens[idx] == SPECIAL_TOKENS["castle_q"]:
            castling_tokens.append("q")
        idx += 1

    castling = "".join(castling_tokens) if castling_tokens else "-"

    # Process en passant
    ep_token = tokens[idx]
    ep_square = "-" if ep_token == SPECIAL_TOKENS["no_ep"] else ep_token[2:-2]

    # Last two tokens are move numbers
    halfmove = tokens[idx + 1]
    fullmove = tokens[idx + 2]

    # Combine all parts
    fen = f"{position} {turn} {castling} {ep_square} {halfmove} {fullmove}"

    # Validate the resulting FEN
    try:
        chess.Board(fen, chess960=True)
        return fen
    except ValueError as e:
        raise ValueError(f"Generated invalid FEN string: {e}")


def print_vocabulary():
    """Print all possible tokens."""
    tokens = get_all_tokens()
    print(f"Total vocabulary size: {len(tokens)}")
    print("\nTokens:")
    for token in sorted(tokens):
        print(token)


# Example usage and testing
if __name__ == "__main__":
    # Test cases including Chess960 positions
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Standard start
        "nnrbqkbr/pppppppp/8/8/8/8/PPPPPPPP/NNRBQKBR w KQkq - 0 1",  # Chess960 position
        "3R4/8/K7/pB2b3/1p6/1P2k3/3p4/8 w - - 4 58",  # Midgame position
        "2kr3r/ppp1nppp/2n1p3/2b5/4P3/2N2N2/PPP2PPP/R1B2RK1 w - - 0 12",  # Another position
    ]

    print("Testing FEN tokenization and detokenization:")
    print("-" * 50)

    for fen in test_fens:
        print(f"\nOriginal FEN: {fen}")
        try:
            tokenized = tokenize_fen(fen)
            print(f"Tokenized: {tokenized}")
            detokenized = detokenize_fen(tokenized)
            print(f"Detokenized: {detokenized}")
            print(f"Match: {fen == detokenized}")

            # Verify both are valid positions
            chess.Board(fen, chess960=True)
            chess.Board(detokenized, chess960=True)

        except ValueError as e:
            print(f"Error: {e}")
        except IndexError as e:
            print(f"Index Error: {e}")
