from chess_gpt.utils import ChessDataCollator
from chess_gpt.tokenizer import ChessTokenizer, FENTokenizer
from chess_gpt.data_loader import make_training_and_eval_datasets
from torch.utils.data import DataLoader


train_dataset, eval_dataset = make_training_and_eval_datasets()

data_collator = ChessDataCollator(
    move_tokenizer=ChessTokenizer(),
    fen_tokenizer=FENTokenizer(),
    mlm=False,
    max_length=2048,
    pad_to_multiple_of=8,
    return_tensors="pt",
)

# Test the collator
# Convert the dataset slice to a list
#
#
train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    collate_fn=data_collator,
    num_workers=2,
    pin_memory=True,
)
batch = next(iter(train_dataloader))

# batch = {
#     "input_ids": move_encodings["input_ids"],
#     "attention_mask": move_encodings["attention_mask"],
#     "fen_input_ids": fen_encodings["input_ids"],
#     "fen_attention_mask": fen_encodings["attention_mask"],
#     "labels": labels,
#     "sample_weights": weight_tensor,
# }

# Print the first batch
print("Batch keys:", batch.keys())

for FEN, MOVES in zip(batch["fen_input_ids"], batch["input_ids"]):
    print("FEN NOT DECODE ", FEN)
    print("FEN:", FENTokenizer().decode(FEN.tolist()))
    print("#" * 80)
    print("MOVE NOT DECODE ", MOVES)
    print("MOVES:", ChessTokenizer().decode(MOVES.tolist()))
