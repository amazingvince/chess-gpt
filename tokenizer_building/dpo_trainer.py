import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import ORPOTrainer, ORPOConfig, DPOConfig, DPOTrainer
from datasets import load_dataset

from fen_utils import tokenize_fen

from liger_kernel.transformers.trainer import LigerORPOTrainer  # noqa: F401

# Basic logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def build_input_text(fen: str) -> str:
    return f"<|start|> <|above_2000|> <|standard|> {tokenize_fen(fen)} <|sep|> "


def process_moves(moves: list) -> str:
    formatted_moves = []
    for move in moves:
        from_square = move[:2]
        to_square = move[2:4]
        promotion = f" {move[4].lower()}" if len(move) > 4 else ""
        formatted_moves.append(f"{from_square} {to_square} {promotion}")

    return " <|turn|> ".join(formatted_moves) + " <|turn|>"


def process_row(example: dict) -> dict:
    """Process the moves in a chess game example."""
    # Extract the moves from the example
    example["prompt"] = build_input_text(example["prompt"])
    example["chosen"] = process_moves(example["chosen"].split(" "))
    example["rejected"] = process_moves(example["rejected"].split(" "))

    return example


def main(model_name, dataset_name, output_dir):
    # Set seed for reproducibility
    set_seed(42)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    # Load dataset
    raw_datasets = load_dataset(dataset_name)

    # Process datasets with chat template
    processed_datasets = raw_datasets.map(
        process_row,
        remove_columns=["metadata"],
        desc="Applying chat template formatting",
    )

    # # Rename columns to match ORPO trainer expectations
    # for split in processed_datasets.keys():
    #     processed_datasets[split] = processed_datasets[split].rename_columns({
    #         "text_prompt": "prompt",
    #         "text_chosen": "chosen",
    #         "text_rejected": "rejected",
    #     })

    if "test" not in processed_datasets.keys():
        processed_datasets = processed_datasets["train"].train_test_split(
            test_size=0.05
        )

    # Setup training arguments
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     num_train_epochs=3,
    #     per_device_train_batch_size=1,
    #     gradient_accumulation_steps=16,
    #     learning_rate=2e-5,
    #     bf16=True,
    #     tf32=True,
    #     logging_steps=10,
    #     save_steps=100,
    #     eval_steps=100
    # )
    # orpo_args = ORPOConfig(
    #     # Small learning rate to prevent catastrophic forgetting
    #     learning_rate=1e-4,
    #     # Linear learning rate decay over training
    #     lr_scheduler_type="linear",
    #     # Maximum combined length of prompt + completion
    #     max_length=1024,
    #     # Maximum length for input prompts
    #     max_prompt_length=512,
    #     # Controls weight of the odds ratio loss (λ in paper)
    #     # beta=0.1,
    #     # Batch size for training
    #     per_device_train_batch_size=1,
    #     per_device_eval_batch_size=1,
    #     # Helps with training stability by accumulating gradients before updating
    #     gradient_accumulation_steps=32,
    #     # Memory-efficient optimizer for CUDA, falls back to adamw_torch for CPU/MPS
    #     optim="paged_adamw_8bit",
    #     # Number of training epochs
    #     num_train_epochs=3,
    #     # When to run evaluation
    #     evaluation_strategy="steps",
    #     # Evaluate every 20% of training
    #     eval_steps=0.2,
    #     # Log metrics every step
    #     logging_steps=10,
    #     # Gradual learning rate warmup
    #     warmup_steps=5,
    #     # Disable external logging
    #     report_to="wandb",
    #     # Where to save model/checkpoints
    #     output_dir="./results/",
    #     bf16=True,
    #     tf32=True,
    #     remove_unused_columns=False,
    #     use_liger_kernel=True,
    # )

    # # Initialize ORPO trainer
    # trainer = ORPOTrainer(
    #     model=model,
    #     args=orpo_args,
    #     train_dataset=processed_datasets["train"],
    #     eval_dataset=processed_datasets["test"],
    #     tokenizer=tokenizer,
    # )

    DPO_Config = DPOConfig(
        # Small learning rate to prevent catastrophic forgetting
        learning_rate=1e-6,
        beta=0.1,
        # Linear learning rate decay over training
        lr_scheduler_type="cosine",
        # Maximum combined length of prompt + completion
        max_length=1024,
        # Maximum length for input prompts
        max_prompt_length=512,
        # Controls weight of the odds ratio loss (λ in paper)
        # beta=0.1,
        # Batch size for training
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        # Helps with training stability by accumulating gradients before updating
        gradient_accumulation_steps=4,
        # Memory-efficient optimizer for CUDA, falls back to adamw_torch for CPU/MPS
        optim="adamw_torch",
        # Number of training epochs
        num_train_epochs=1,
        save_steps=1000,
        save_total_limit=1,
        # When to run evaluation
        evaluation_strategy="steps",
        # Evaluate every 20% of training
        eval_steps=0.1,
        # Log metrics every step
        logging_steps=10,
        # Gradual learning rate warmup
        warmup_steps=5,
        # Disable external logging
        report_to="wandb",
        # Where to save model/checkpoints
        output_dir="./results_v2/",
        bf16=True,
        tf32=True,
        remove_unused_columns=False,
        use_liger_kernel=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=DPO_Config,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["test"],
        tokenizer=tokenizer,
    )
    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)

    # Save tokenizer and configuration
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)


if __name__ == "__main__":
    # Set basic parameters
    model_name = "amazingvince/chess-llama-pretrain"
    dataset_name = "amazingvince/chess_dpo"
    output_dir = "output_v2/"
    main(model_name=model_name, dataset_name=dataset_name, output_dir=output_dir)
