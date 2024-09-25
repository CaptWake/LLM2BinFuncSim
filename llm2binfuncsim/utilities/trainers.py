from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer

from utilities.losses import SupConLoss

if TYPE_CHECKING:
    import networkx as nx
    from datasets import Dataset


def collate_padding_train_valid_labels(data, G, data_collator):
    batch_size: int = len(data)
    labels: torch.Tensor = torch.eye(batch_size)
    for i in range(batch_size):
        for j in range(batch_size):
            if G.has_edge(data[i]["asm_hash"], data[j]["asm_hash"]):
                labels[i][j] = labels[j][i] = 1
    batch = data_collator(
        [
            {k: v for k, v in b.items() if k in ["input_ids", "attention_mask"]}
            for b in data
        ]
    )
    batch["labels"] = labels
    return batch


class MLMTrainer(Trainer):
    def __init__(
        self,
        eval_data_collator,
        **kwargs,
    ):
        self.eval_data_collator = eval_data_collator
        super().__init__(**kwargs)

    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        return DataLoader(
            eval_dataset,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.eval_data_collator,
        )


class SupConLossTrainer(Trainer):
    def __init__(
        self,
        gs: Optional[list["nx.Graph"]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gs = gs

    def compute_loss(
        self,
        model: "nn.Module",
        inputs: dict[str, "torch.Tensor" | Any],
        return_outputs=False,
    ):
        criterion: SupConLoss = SupConLoss()
        labels: "torch.Tensor" = inputs.pop("labels")
        cls: "torch.Tensor" = model(**inputs).last_hidden_state[:, 0]
        loss = criterion(features=cls, mask=labels)
        return (loss, cls) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self.gs is None:
            raise ValueError(
                "Trainer: training requires a list of graphs (gs parameter)."
            )
        return DataLoader(
            self.train_dataset,
            num_workers=self.args.dataloader_num_workers,
            batch_size=self.args.per_device_train_batch_size,
            pin_memory=self.args.dataloader_pin_memory,
            collate_fn=lambda x: collate_padding_train_valid_labels(
                x, self.gs[0], self.data_collator
            ),
        )

    def get_test_dataloader(self, test_dataset: "Dataset") -> DataLoader:
        return DataLoader(
            test_dataset,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
        )
