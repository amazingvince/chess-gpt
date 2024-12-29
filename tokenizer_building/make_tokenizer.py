from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
from fen_utils import get_all_tokens

#

tokens = [
    "<|start|>",
    "<|end|>",
    "<|turn|>",
    "<|unk|>",
    "<|pad|>",
    "<|sep|>",
    "<|below_1000|>",
    "<|1000_2000|>",
    "<|above_2000|>",
    "<|engine|>",
    "<|standard|>",
    "<|anti_chess|>",
    "<|atomic_chess|>",
    "<|chess_960|>",
]
files = "abcdefgh"[::-1]
ranks = "12345678"
squares = [f"{f}{r}" for f in files for r in ranks]

tokens.extend(squares)

tokens.extend(["q", "r", "b", "n"])

tokens.extend(get_all_tokens())


tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = WhitespaceSplit()

tokenizer.add_tokens(tokens)


t = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<|unk|>",
    bos_token="<|start|>",
    eos_token="<|end|>",
    pad_token="<|pad|>",
    special_tokens=[
        "<|start|>",
        "<|end|>",
        "<|unk|>",
        "<|pad|>",
        "<|below_1000|>",
        "<|1000_2000|>",
        "<|above_2000|>",
        "<|engine|>",
        "<|standard|>",
        "<|anti_chess|>",
        "<|atomic_chess|>",
        "<|chess_960|>",
    ],
    return_token_type_ids=False,
)


t.save_pretrained("tokenizer-chess")
