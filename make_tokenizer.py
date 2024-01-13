from transformers import PreTrainedTokenizerFast

# fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from transformers import PreTrainedTokenizerFast


# 

files = ['<|start|>', '<|end|>', '<|turn|>', '<|unk|>', '<|pad|>',
         '1/2-1/2', '1-0', '0-1',
         'FIVEFOLD_REPETITION', 'INSUFFICIENT_MATERIAL', 'STALEMATE', 'CHECKMATE', 'SEVENTYFIVE_MOVES', 
         'Q', 'R', 'B', 'N', 
         'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 
         'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 
         'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 
         'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 
         'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 
         'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 
         'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8',
         'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
         ]


tokenizer = Tokenizer(BPE())
# tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)

tokenizer.pre_tokenizer = WhitespaceSplit()

tokenizer.add_tokens(files)

t = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<|unk|>",
    bos_token = "<|start|>",
    eos_token = "<|end|>",
    pad_token="<|pad|>",
    special_tokens=[ '<|start|>', '<|end|>', '<|unk|>', '<|pad|>',  'FIVEFOLD_REPETITION', 'INSUFFICIENT_MATERIAL', 'STALEMATE', 'CHECKMATE', 'SEVENTYFIVE_MOVES', '1/2-1/2', '1-0', '0-1'],
    return_token_type_ids=False,
)



t.save_pretrained("tokenizer-chess-2")


# t.save("tokenizer-chess.json")

from transformers import AutoTokenizer

# t = AutoTokenizer.from_pretrained("BEE-spoke-data/verysmol_llama-v11-KIx2")

# print(t("<|start|> P d2 d4 <|turn|> N g8 f6 <|turn|> P c2 c4 <|turn|> P e7 e6 <|turn|> N g1 f3 <|turn|> P b7 b6 <|turn|> P g2 g3 <|turn|> B c8 b7 <|turn|> B f1 g2 <|turn|> B f8 e7 <|turn|> N b1 c3 <|turn|> P d7 d5 <|turn|> B c1 g5 <|turn|> N b8 d7 <|turn|> P c4 d5 <|turn|> N f6 d5 <|turn|> B g5 e7 <|turn|> N d5 e7 <|turn|> P e2 e4 <|turn|> P h7 h6 <|turn|> Q d1 e2 <|turn|> P a7 a5 <|turn|> Q e2 e3 <|turn|> B b7 a6 <|turn|> B g2 f1 <|turn|> B a6 b7 <|turn|> B f1 e2 <|turn|> P c7 c5 <|turn|> B e2 b5 <|turn|> B b7 c6 <|turn|> R a1 d1 <|turn|> B c6 b5 <|turn|> N c3 b5 <|turn|> K e8 g8 <|turn|> K e1 g1 <|turn|> Q d8 c8 <|turn|> Q e3 e2 <|turn|> Q c8 b7 <|turn|> R d1 d2 <|turn|> R f8 d8 <|turn|> R d2 c2 <|turn|> N d7 f6 <|turn|> R f1 e1 <|turn|> P c5 d4 <|turn|> R c2 c7 <|turn|> P d4 d3 <|turn|> R c7 b7 <|turn|> P d3 e2 <|turn|> R b7 e7 <|turn|> R d8 d1 <|turn|> K g1 g2 <|turn|> K g8 f8 <|turn|> R e7 b7 <|turn|> R a8 d8 <|turn|> R e1 e2 <|turn|> N f6 d7 <|turn|> R e2 e3 <|turn|> K f8 g8 <|turn|> P e4 e5 <|turn|> R d1 d5 <|turn|> N b5 c3 <|turn|> R d5 c5 <|turn|> R e3 d3 <|turn|> P f7 f6 <|turn|> P e5 f6 <|turn|> R c5 d5 <|turn|> N c3 d5 <|turn|> P e6 d5 <|turn|> R d3 d5 <|turn|> P b6 b5 <|turn|> R d5 d7 <|turn|> R d8 d7 <|turn|> R b7 d7 <|turn|> P g7 g6 <|turn|> N f3 e5 <|turn|> P g6 g5 <|turn|> N e5 g6 <|turn|> P h6 h5 <|turn|> R d7 g7 <|turn|>", return_token_type_ids=False))





"""
from transformers import PreTrainedTokenizerFast

# fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from transformers import PreTrainedTokenizerFast


# 

files = {k:i for i, k in enumerate([
         '<|turn|>', '<|unk|>', '<|pad|>', '<|start|>', '<|end|>'
         'K', 'Q', 'R', 'B', 'N', 'P', 
         'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 
         'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 
         'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 
         'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 
         'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 
         'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 
         'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8',
         'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8'])}


tokenizer = Tokenizer(BPE(files, [], unk_token="<|unk|>"))
# tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)

tokenizer.pre_tokenizer = WhitespaceSplit()

t = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<|unk|>",
    bos_token = "<|start|>",
    eos_token = "<|end|>",
    pad_token="<|pad|>",
    special_tokens=['<|turn|>', '<|unk|>', '<|pad|>', '<|start|>', '<|end|>'],
    return_token_type_ids=False
)

# tokenizer.add_tokens(files)

# t.save_pretrained("tokenizer-chess")


# t.save("tokenizer-chess.json")

from transformers import AutoTokenizer

# t = AutoTokenizer.from_pretrained("BEE-spoke-data/verysmol_llama-v11-KIx2")

print(t.tokenize("P d2 d4 <|turn|> N g8 f6 <|turn|> P c2 c4 <|turn|> P e7 e6 <|turn|> N g1 f3 <|turn|> P b7 b6 <|turn|> P g2 g3 <|turn|> B c8 b7 <|turn|> B f1 g2 <|turn|> B f8 e7 <|turn|> N b1 c3 <|turn|> P d7 d5 <|turn|> B c1 g5 <|turn|> N b8 d7 <|turn|> P c4 d5 <|turn|> N f6 d5 <|turn|> B g5 e7 <|turn|> N d5 e7 <|turn|> P e2 e4 <|turn|> P h7 h6 <|turn|> Q d1 e2 <|turn|> P a7 a5 <|turn|> Q e2 e3 <|turn|> B b7 a6 <|turn|> B g2 f1 <|turn|> B a6 b7 <|turn|> B f1 e2 <|turn|> P c7 c5 <|turn|> B e2 b5 <|turn|> B b7 c6 <|turn|> R a1 d1 <|turn|> B c6 b5 <|turn|> N c3 b5 <|turn|> K e8 g8 <|turn|> K e1 g1 <|turn|> Q d8 c8 <|turn|> Q e3 e2 <|turn|> Q c8 b7 <|turn|> R d1 d2 <|turn|> R f8 d8 <|turn|> R d2 c2 <|turn|> N d7 f6 <|turn|> R f1 e1 <|turn|> P c5 d4 <|turn|> R c2 c7 <|turn|> P d4 d3 <|turn|> R c7 b7 <|turn|> P d3 e2 <|turn|> R b7 e7 <|turn|> R d8 d1 <|turn|> K g1 g2 <|turn|> K g8 f8 <|turn|> R e7 b7 <|turn|> R a8 d8 <|turn|> R e1 e2 <|turn|> N f6 d7 <|turn|> R e2 e3 <|turn|> K f8 g8 <|turn|> P e4 e5 <|turn|> R d1 d5 <|turn|> N b5 c3 <|turn|> R d5 c5 <|turn|> R e3 d3 <|turn|> P f7 f6 <|turn|> P e5 f6 <|turn|> R c5 d5 <|turn|> N c3 d5 <|turn|> P e6 d5 <|turn|> R d3 d5 <|turn|> P b6 b5 <|turn|> R d5 d7 <|turn|> R d8 d7 <|turn|> R b7 d7 <|turn|> P g7 g6 <|turn|> N f3 e5 <|turn|> P g6 g5 <|turn|> N e5 g6 <|turn|> P h6 h5 <|turn|> R d7 g7 <|turn|>", return_token_type_ids=False))


"""