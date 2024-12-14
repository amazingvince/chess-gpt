from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from typing import List, Tuple, Union


class ChessTokenizer:
    """A tokenizer for chess moves and game states."""

    # Special tokens used for the model
    SPECIAL_TOKENS = {
        "bos_token": "<|start|>",
        "eos_token": "<|end|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "turn_token": "<|turn|>",
    }

    # Piece symbols for promotions
    PROMOTION_TOKENS = ["q", "r", "b", "n"]

    @staticmethod
    def generate_square_tokens():
        """Generate all possible square coordinates on a chess board.
        revese the files so it maps to the correct square on the board
        """
        files = "abcdefgh"[::-1]
        ranks = "12345678"
        board = [f"{f}{r}" for f in files for r in ranks]

        # print board as nice 8x8 grid
        # for i in range(8):
        #     print(board[i * 8 : i * 8 + 8])
        return board

    def __init__(self):
        """Initialize the chess tokenizer with all necessary tokens."""
        # Combine all token types
        self.vocabulary = (
            list(self.SPECIAL_TOKENS.values())
            + self.PROMOTION_TOKENS
            + self.generate_square_tokens()
        )

        # Create vocabulary dictionary with indices
        vocab_dict = {token: i for i, token in enumerate(self.vocabulary)}

        # Initialize the base tokenizer with WordLevel model
        self.tokenizer = Tokenizer(
            WordLevel(vocab=vocab_dict, unk_token=self.SPECIAL_TOKENS["unk_token"])
        )

        # Set up pre-tokenizer to split on whitespace
        self.tokenizer.pre_tokenizer = WhitespaceSplit()

        # Set up post-processing for special tokens
        self.tokenizer.post_processor = TemplateProcessing(
            single="$A",  # Changed to just use the content without adding extra tokens
            special_tokens=[],  # No special tokens in post-processing
        )

        # Create the fast tokenizer
        self.fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token=self.SPECIAL_TOKENS["unk_token"],
            bos_token=self.SPECIAL_TOKENS["bos_token"],
            eos_token=self.SPECIAL_TOKENS["eos_token"],
            pad_token=self.SPECIAL_TOKENS["pad_token"],
            return_token_type_ids=False,
        )

    def save(self, path="tokenizer-chess"):
        """Save the tokenizer to the specified path."""
        self.fast_tokenizer.save_pretrained(path)

    def encode(self, text):
        """Encode text to token IDs."""
        # Pre-process the text to properly split moves and preserve special tokens
        tokens = []
        current_token = ""
        i = 0

        while i < len(text):
            if text[i] == "<":
                # Handle special tokens
                end_idx = text.find(">", i)
                if end_idx != -1:
                    special_token = text[i : end_idx + 1]
                    if special_token in self.SPECIAL_TOKENS.values():
                        if current_token:
                            tokens.extend(self._split_move(current_token))
                            current_token = ""
                        tokens.append(special_token)
                        i = end_idx + 1
                        continue

            if text[i] not in ["<", "|", ">"]:
                current_token += text[i]
                if len(current_token) == 4:  # Regular move
                    tokens.extend(self._split_move(current_token))
                    current_token = ""
                elif len(current_token) == 5:  # Promotion move
                    tokens.extend(self._split_promotion_move(current_token))
                    current_token = ""
            i += 1

        if current_token:
            tokens.extend(self._split_move(current_token))

        # Join tokens with spaces and encode
        return self.fast_tokenizer.encode(" ".join(tokens))

    def _split_move(self, move):
        """Split a chess move into its components."""
        if len(move) == 4:
            return [move[:2], move[2:]]
        elif len(move) == 5:
            return self._split_promotion_move(move)
        return [move]

    def _split_promotion_move(self, move):
        """Split a promotion move into its components."""
        return [move[:2], move[2:4], move[4]]

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        # Get the raw decoded text
        decoded = self.fast_tokenizer.decode(token_ids)

        # Post-process to remove extra spaces and combine moves
        result = ""
        tokens = decoded.split()
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in self.SPECIAL_TOKENS.values():
                result += token
            else:
                # Check if it's a move (two squares possibly followed by promotion piece)
                if i + 1 < len(tokens) and len(token) == 2 and len(tokens[i + 1]) == 2:
                    if (
                        i + 2 < len(tokens)
                        and len(tokens[i + 2]) == 1
                        and tokens[i + 2] in self.PROMOTION_TOKENS
                    ):
                        # Promotion move
                        result += token + tokens[i + 1] + tokens[i + 2]
                        i += 2
                    else:
                        # Regular move
                        result += token + tokens[i + 1]
                        i += 1
            i += 1

        return result


# Usage example
if __name__ == "__main__":
    chess_tokenizer = ChessTokenizer()

    # Test the tokenizer with a sample move sequence
    test_sequence = "<|start|>e2e4<|turn|>e7e5<|turn|>d7d8q<|end|>"
    encoded = chess_tokenizer.encode(test_sequence)
    decoded = chess_tokenizer.decode(encoded)

    print("Original:", test_sequence)
    print("Encoded:", encoded)
    print("Decoded:", decoded)

    # Print vocabulary for debugging
    print("\nVocabulary:")
    for i, token in enumerate(chess_tokenizer.vocabulary):
        print(f"{i}: {token}")

    chess_tokenizer.save()
