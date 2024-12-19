from transformers import PreTrainedTokenizer
from typing import List, Optional, Dict, Any, Union, Tuple
import re
import chess
import json
import os
import torch

from transformers.tokenization_utils import BatchEncoding


from transformers import PreTrainedTokenizer
from typing import List, Optional, Dict, Any
import re


class ChessTokenizer(PreTrainedTokenizer):
    def __init__(
        self, model_max_length: int = 2048, padding_side: str = "right", **kwargs
    ):
        self.special_tokens = {
            "start_token": "<|start|>",
            "turn_token": "<|turn|>",
            "end_token": "<|end|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
        }

        # Generate chess square tokens (a1-h8)
        files = "abcdefgh"[::-1]
        ranks = "12345678"
        squares = [f"{f}{r}" for f in files for r in ranks]
        promotions = ["q", "r", "b", "n"]
        elo = ["<|below_1000|>", "<|1000_2000|>", "<|above_2000|>"]

        # Build vocabulary
        vocab = {}
        current_id = 0

        # Add special tokens
        for token in self.special_tokens.values():
            vocab[token] = current_id
            current_id += 1

        # Add squares
        for square in squares:
            vocab[square] = current_id
            current_id += 1

        # Add promotion pieces
        for piece in promotions:
            vocab[piece] = current_id
            current_id += 1

        # Add elo
        for elo in elo:
            vocab[elo] = current_id
            current_id += 1

        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}

        super().__init__(
            model_max_length=model_max_length, padding_side=padding_side, **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()

    def _tokenize(self, text: str) -> List[str]:
        # Split on special tokens and moves
        pattern = r"(<\|[^|]+\|>|[a-h][1-8][a-h][1-8][qrnb]?)"
        tokens = []

        for match in re.finditer(pattern, text):
            token = match.group()
            if len(token) >= 4 and not token.startswith("<|"):  # It's a move
                tokens.append(token[:2])  # from square
                tokens.append(token[2:4])  # to square
                if len(token) > 4:  # promotion piece
                    tokens.append(token[4])
            else:  # Special token
                tokens.append(token)

        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.special_tokens["unk_token"]])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.special_tokens["unk_token"])

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return token_ids_0

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        mask = [
            1 if self.ids_to_tokens[id] in self.special_tokens.values() else 0
            for id in token_ids_0
        ]
        return mask

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return [0] * len(token_ids_0)

    @property
    def pad_token(self) -> str:
        return self.special_tokens["pad_token"]

    @property
    def unk_token(self) -> str:
        return self.special_tokens["unk_token"]

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> tuple:
        import os
        import json

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        return (vocab_file,)


class FENTokenizer(PreTrainedTokenizer):
    """
    A more robust tokenizer for FEN strings, compatible with BERT-style tokenization.
    This tokenizer:
    - Converts a FEN into a sequence of known tokens.
    - Handles special tokens for side-to-move ([WHITE], [BLACK]) and standard FEN fields.
    - Properly encodes and decodes all fields of a FEN string.
    """

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    padding_side = "right"

    def __init__(
        self,
        vocab_file=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
    ):
        # Start with special tokens
        # We'll use a dedicated symbol for space to make reconstruction easier.
        # Let's pick a symbol that doesn't appear in standard FEN strings, like "▁".
        # Alternatively, you can use a normal space " " if you prefer.
        special_tokens = [
            pad_token,  # [PAD]
            unk_token,  # [UNK]
            cls_token,  # [CLS]
            sep_token,  # [SEP]
            mask_token,  # [MASK]
            "[BLACK]",
            "[WHITE]",
            "▁",  # space symbol
        ]

        base_tokens = [
            # Pieces
            "p",
            "n",
            "b",
            "r",
            "q",
            "k",
            "P",
            "N",
            "B",
            "R",
            "Q",
            "K",
            # Digits for empty squares and move counters
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            # Files
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            # Slash for ranks
            "/",
            # Hyphen for no castling or no en passant
            "-",
            # Castling rights (treated as individual tokens)
            # Note: K,Q,k,q appear also as pieces, but that's fine as long as we
            # tokenize fields correctly.
            # En passant squares (a3, ..., h3, a6, ..., h6)
            "a3",
            "b3",
            "c3",
            "d3",
            "e3",
            "f3",
            "g3",
            "h3",
            "a6",
            "b6",
            "c6",
            "d6",
            "e6",
            "f6",
            "g6",
            "h6",
        ]

        # Load vocab if provided
        if vocab_file is not None and os.path.isfile(vocab_file):
            with open(vocab_file, "r") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {}
            for token in special_tokens:
                self.vocab.setdefault(token, len(self.vocab))
            for token in base_tokens:
                self.vocab.setdefault(token, len(self.vocab))

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    def _tokenize(self, fen: str) -> List[str]:
        """
        Tokenize a FEN string into known tokens.
        Assume the input FEN string does NOT include [CLS] and [SEP].
        We handle them automatically when encoding if needed.
        """
        fen = fen.strip()
        # Split by space-like fields:
        parts = fen.split(" ")
        # Standard FEN has at least 4 fields, often 6:
        # board, side_to_move, castling, en_passant, halfmove_clock, fullmove_number
        # If some fields are missing, handle gracefully.

        board = parts[0] if len(parts) > 0 else ""
        side = parts[1] if len(parts) > 1 else "w"
        castling = parts[2] if len(parts) > 2 else "-"
        en_passant = parts[3] if len(parts) > 3 else "-"
        halfmove = parts[4] if len(parts) > 4 else "0"
        fullmove = parts[5] if len(parts) > 5 else "1"

        tokens = []
        # Board tokenization: one character or slash per token
        for char in board:
            # Each piece or digit or slash is already a token
            if char in self.vocab:
                tokens.append(char)
            else:
                # Unexpected char => [UNK]
                tokens.append(self.unk_token)

        # Space (field separator)
        tokens.append("▁")

        # Side to move
        if side == "w":
            tokens.append("[WHITE]")
        elif side == "b":
            tokens.append("[BLACK]")
        else:
            tokens.append(self.unk_token)

        tokens.append("▁")

        # Castling rights: could be "-", or some combination of KQkq
        if castling == "-":
            tokens.append("-")
        else:
            for c in castling:
                if c in self.vocab:
                    tokens.append(c)
                else:
                    tokens.append(self.unk_token)

        tokens.append("▁")

        # En passant: either "-" or one of a3...h3 or a6...h6
        if en_passant == "-":
            tokens.append("-")
        else:
            if en_passant in self.vocab:
                tokens.append(en_passant)
            else:
                # Unexpected en passant square
                tokens.extend(
                    [c if c in self.vocab else self.unk_token for c in en_passant]
                )

        tokens.append("▁")

        # Halfmove: possibly a multi-digit number
        for digit in halfmove:
            tokens.append(digit if digit in self.vocab else self.unk_token)

        tokens.append("▁")

        # Fullmove: multi-digit number
        for digit in fullmove:
            tokens.append(digit if digit in self.vocab else self.unk_token)

        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert tokens back into a FEN string.
        We know the structure:
        board + space + side + space + castling + space + en_passant + space + halfmove + space + fullmove
        We'll split by "▁" since we used it as a field separator.
        """
        # Remove special tokens that might appear
        tokens = [
            t
            for t in tokens
            if t
            not in [self.pad_token, self.cls_token, self.sep_token, self.mask_token]
        ]

        # Split by the field separator
        # Fields: [board_tokens, side_tokens, castling_tokens, en_passant_tokens, halfmove_tokens, fullmove_tokens]
        if "▁" not in tokens:
            # If we don't find any field separator, just join as is.
            return "".join(tokens).strip()

        fields = []
        current_field = []
        for t in tokens:
            if t == "▁":
                if current_field:
                    fields.append(current_field)
                    current_field = []
                else:
                    # Empty field (should not happen, but let's handle gracefully)
                    fields.append([])
            else:
                current_field.append(t)
        if current_field:
            fields.append(current_field)

        # fields should now be: board, side, castling, en_passant, halfmove, fullmove
        # Some FENs might have fewer fields if user provided incomplete FEN
        # Let's handle them gracefully.
        board_tokens = fields[0] if len(fields) > 0 else []
        side_tokens = fields[1] if len(fields) > 1 else ["[WHITE]"]
        castling_tokens = fields[2] if len(fields) > 2 else ["-"]
        en_passant_tokens = fields[3] if len(fields) > 3 else ["-"]
        halfmove_tokens = fields[4] if len(fields) > 4 else ["0"]
        fullmove_tokens = fields[5] if len(fields) > 5 else ["1"]

        # Reconstruct board
        board = "".join(board_tokens)

        # Reconstruct side
        # Expect either [WHITE] or [BLACK]
        side = "w"
        if side_tokens == ["[BLACK]"]:
            side = "b"
        elif side_tokens == ["[WHITE]"]:
            side = "w"
        else:
            # Unknown side, try to decode best effort
            side = "".join(side_tokens)
            side = side.replace("[WHITE]", "w").replace("[BLACK]", "b")

        # Castling
        if castling_tokens == ["-"]:
            castling = "-"
        else:
            castling = "".join(castling_tokens)

        # En passant
        if en_passant_tokens == ["-"]:
            en_passant = "-"
        else:
            en_passant = "".join(en_passant_tokens)

        # Halfmove & fullmove
        halfmove = "".join(halfmove_tokens)
        fullmove = "".join(fullmove_tokens)

        # Construct final FEN
        fen_str = f"{board} {side} {castling} {en_passant} {halfmove} {fullmove}"
        return fen_str.strip()

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        vocab_path = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
        return (vocab_path,)


# Example usage:
if __name__ == "__main__":
    tokenizer = FENTokenizer()

    # Example FEN:
    fen = "3q1rk1/1Q1b1ppp/p4b2/8/5B2/4P3/PPr1BPPP/3R1RK1 w - - 1 17"
    # Encode with special tokens:
    encoded = tokenizer(fen, return_tensors="pt", add_special_tokens=True)
    print("Encoded:", encoded)

    # Decode back
    decoded = tokenizer.decode(encoded["input_ids"][0])
    print("Decoded:", decoded)

    # Another example with different FENs in batch:
    fens = [
        "r3k2r/pp1q1ppp/3b1n2/2p1N3/3P4/4P3/PP3PPP/R1BQ1RK1 b kq - 3 12",
        "7k/R7/6K1/8/8/8/8/8 b - - 44 113",
    ]
    batch_encoded = tokenizer(
        fens, padding=True, return_tensors="pt", add_special_tokens=True
    )
    print("Batch Encoded IDs:", batch_encoded["input_ids"])

    # Decode batch
    for i in range(len(fens)):
        d = tokenizer.decode(batch_encoded["input_ids"][i])
        print(f"Decoded FEN {i}:", d)

    #
    print("\nChessTokenizer")
    tokenizer = ChessTokenizer()

    # Single string
    game = "<|below_1000|><|start|>e2e4<|turn|>e7e5<|turn|>d7d8q<|end|>"
    encoded = tokenizer(game)
    print("Single encode:", encoded)

    # Batch encoding
    games = [
        "<|above_2000|><|start|>e2e4<|turn|>e7e5<|turn|>d7d8q<|end|>",
        "<|1000_2000|><|start|>d2d4<|turn|>d7d5<|end|>",
    ]
    batch_encoded = tokenizer(games, padding=True, return_tensors="pt")
    print("\nBatch encode:", batch_encoded["input_ids"])

    # Decode
    decoded = tokenizer.decode(encoded["input_ids"])
    print("\nDecoded:", decoded)

    print(tokenizer.decode(batch_encoded["input_ids"][0]))
    print(tokenizer.decode(batch_encoded["input_ids"][1]))
    print(tokenizer.vocab_size)
