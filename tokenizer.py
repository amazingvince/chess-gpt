import re
from typing import Dict, List, Optional, Tuple

import chess
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding


class FENTokenizer(PreTrainedTokenizer):
    def __init__(
        self, model_max_length: int = 2048, padding_side: str = "right", **kwargs
    ):
        self.special_tokens = {
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
        }

        # Build vocabulary for FEN tokens
        vocab = {}
        current_id = 0

        # Add special tokens
        for token in self.special_tokens.values():
            vocab[token] = current_id
            current_id += 1

        # Add piece tokens
        pieces = "pnbrqkPNBRQK"
        for piece in pieces:
            vocab[piece] = current_id
            current_id += 1

        # Add numbers 1-8 for empty squares
        for i in range(1, 9):
            vocab[str(i)] = current_id
            current_id += 1

        # Add FEN separators
        vocab["/"] = current_id
        current_id += 1

        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}

        super().__init__(
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs
        )

    def _tokenize(self, text: str) -> List[str]:
        """
        Converts a string into a sequence of tokens for FEN notation.
        
        Args:
            text (str): The FEN string to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        # Split FEN string into individual characters/tokens
        tokens = []
        for char in text:
            tokens.append(char)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        """
        Converts a token to its corresponding vocabulary ID.
        
        Args:
            token (str): Token to convert
            
        Returns:
            int: The ID of the token in the vocabulary
        """
        return self.vocab.get(token, self.vocab[self.special_tokens["unk_token"]])

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts a vocabulary ID back to its corresponding token.
        
        Args:
            index (int): ID to convert
            
        Returns:
            str: The token corresponding to the ID
        """
        return self.ids_to_tokens.get(index, self.special_tokens["unk_token"])

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.
        
        Returns:
            Dict[str, int]: The vocabulary
        """
        return self.vocab.copy()

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.
        
        Returns:
            int: Vocabulary size
        """
        return len(self.vocab)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the tokenizer vocabulary to a file.
        
        Args:
            save_directory (str): Directory where to save the vocabulary
            filename_prefix (Optional[str]): Optional prefix for the vocabulary filename
            
        Returns:
            Tuple[str]: Paths to the saved vocabulary files
        """
        import json
        import os

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        return (vocab_file,)


class ChessTokenizer(PreTrainedTokenizer):
    def __init__(
        self, model_max_length: int = 2048, padding_side: str = "right", **kwargs
    ):
        # Fen tokenizer. Uses a separate vocab to ensure LLM gives valid output.
        self.fen_tokenizer = FENTokenizer(model_max_length=model_max_length, padding_side=padding_side)
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
    
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        **kwargs,
    ) -> BatchEncoding:
        del kwargs  # Unused
        input_ids = []
        fed_ids = []
        for text in batch_text_or_text_pairs:
            ids = self.convert_tokens_to_ids(tokenized_text["text"])
            input_ids.append((ids, None))
            fed_tokens = self.fen_tokenizer._tokenize(tokenized_text["fen_positions"])
            fed_ids.append(self.fen_tokenizer.convert_tokens_to_ids(fed_tokens))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
        )
        batch_outputs["fen_input_ids"] = fed_ids
        return BatchEncoding(batch_outputs)

    def _tokenize(self, text: str) -> List[str]:
        def get_board_fen_index(fen_to_inputs, board):
            fen = board.fen()
            if fen not in fen_to_inputs:
                fen_tokens = self.fen_tokenizer._tokenize(fen)
                fed_inputs = self.fen_tokenizer.convert_tokens_to_ids(fen_tokens)
                fen_to_inputs[fen] = (len(fen_to_inputs), fed_inputs)
            return fen_to_inputs[fen][0]
            
        # Split on special tokens and moves
        pattern = r"(<\|[^|]+\|>|[a-h][1-8][a-h][1-8][qrnb]?)"
        tokens = []
        fen_token_ids = []
        board = chess.Board()
        fen_to_inputs: dict[str, dict[int, list[str]]] = {}
        for match in re.finditer(pattern, text):
            token = match.group()
            
            board_pos_idx = get_board_fen_index(fen_to_inputs, board)
            if len(token) >= 4 and not token.startswith("<|"):  # It's a move
                from_square = token[:2]
                to_square = token[2:4]  # to square
                tokens.extend([from_square, to_square])
                # Add fen position of previous move
                fen_token_ids.extend(2*[board_pos_idx])
                promotion = ''
                if len(token) > 4:  # promotion piece
                    promotion = token[4]
                    tokens.append(promotion)
                    fen_token_ids.append(board_pos_idx)
                # Update fen position
                move = chess.Move.from_uci(from_square + to_square + promotion)
                board_pos_idx = get_board_fen_index(fen_to_inputs, board)
                board.push(move)
            else:  # Special token
                tokens.append(token)
                fen_token_ids.append(board_pos_idx)

        return {
            "text": tokens, 
            "fen_token_ids":fen_token_ids, 
            "fen_token_map": fen_to_inputs
        }

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
        import json
        import os

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        return (vocab_file,)


# Example usage
if __name__ == "__main__":
    tokenizer = ChessTokenizer()

    # Single string
    game = "<|start|>e2e4<|turn|>e7e5<|turn|>d7d8q<|end|>"
    encoded = tokenizer(game)
    print("Single encode:", encoded)

    # Batch encoding
    games = [
        "<|start|>e2e4<|turn|>e7e5<|turn|>d7d8q<|end|>",
        "<|start|>d2d4<|turn|>d7d5<|end|>",
    ]
    batch_encoded = tokenizer(games, padding=True, return_tensors="pt")
    print("\nBatch encode:", batch_encoded["input_ids"])

    # Decode
    decoded = tokenizer.decode(encoded["input_ids"])
    print("\nDecoded:", decoded)

    print(tokenizer.decode(batch_encoded["input_ids"][0]))
    print(tokenizer.decode(batch_encoded["input_ids"][1]))
