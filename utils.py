from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import logging
from transformers import DataCollatorForLanguageModeling, Trainer
from transformers.utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from tqdm import tqdm


from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase
import torch
import logging

logger = logging.getLogger(__name__)
from tokenizer_2 import ChessTokenizer, FENTokenizer


@dataclass
class ChessDataCollator:
    """
    Data collator for chess training that handles both FEN and move sequences.
    Inherits from DataCollatorBase for HuggingFace compatibility.
    """

    move_tokenizer: ChessTokenizer
    fen_tokenizer: FENTokenizer
    mlm: bool = False
    pad_to_multiple_of: int = 8
    return_tensors: str = "pt"
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        logger.debug(f"Collating batch of size {len(features)}")

        # Extract FEN strings and pre-tokenize moves
        fen_texts = self.fen_pre_tokenize(features)
        move_texts = self.pre_tokenize(features)
        weights = [f.get("weight", 1.0) for f in features]

        # Tokenize FEN sequences
        fen_encodings = self.fen_tokenizer(
            fen_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Tokenize move sequences
        move_encodings = self.move_tokenizer(
            move_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Convert weights to tensor
        weight_tensor = torch.tensor(weights, dtype=torch.float)

        # Prepare labels for moves (shift input_ids)
        labels = move_encodings["input_ids"].clone()
        labels = torch.roll(labels, -1, dims=1)
        labels[:, -1] = -100  # Mask last token

        # Create final batch
        batch = {
            "input_ids": move_encodings["input_ids"],
            "attention_mask": move_encodings["attention_mask"],
            "fen_input_ids": fen_encodings["input_ids"],
            "fen_attention_mask": fen_encodings["attention_mask"],
            "labels": labels,
            "sample_weights": weight_tensor,
        }

        logger.debug(f"Batch shapes: {[(k, v.shape) for k, v in batch.items()]}")
        return batch

    @staticmethod
    def add_elo_token(features: Dict[str, List]) -> List[str]:
        elo = features.get("average_elo", 999)
        dataset_source = features.get("dataset_source", None)

        if dataset_source == "puzzles" or dataset_source == "laion_games":
            return "<|engine|>"
        if elo < 1000:
            return "<|below_1000|>"
        elif elo < 1999:
            return "<|1000_2000|>"
        else:
            "<|above_2000|>"

        return

    @staticmethod
    def pre_tokenize(features: Dict[str, List]) -> List[str]:
        """
        Pre-tokenize move sequences by adding special tokens.

        Args:
            features: Dictionary containing "Moves" key with list of move sequences

        Returns:
            List of pre-tokenized move sequences with special tokens
        """

        return [
            f"{ChessDataCollator.add_elo_token(f)}<|start|>{('<|turn|>'.join(f['moves']))}"
            for f in features
        ]

    @staticmethod
    def fen_pre_tokenize(features: Dict[str, List]) -> List[str]:
        """
        Pre-tokenize move sequences by adding special tokens.

        Args:
            features: Dictionary containing "Moves" key with list of move sequences

        Returns:
            List of pre-tokenized move sequences with special tokens
        """

        return [f"[CLS]{f['fen']}[SEP]" for f in features]


from transformers import Trainer
from typing import Optional, Dict, Union, Any, Tuple, List
from torch.utils.data import DataLoader, Dataset
import torch
from transformers.trainer_utils import seed_worker
import logging

logger = logging.getLogger(__name__)


class ChessModelTrainer(Trainer):

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # We use the same batch_size as for eval.
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))
