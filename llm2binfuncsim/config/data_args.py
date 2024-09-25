import json
import os
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel


class DatasetAttr(BaseModel):
    training_nodes_file_name: str
    training_edges_file_name: str
    validation_nodes_file_name: str
    validation_edges_file_name: str
    test_nodes_file_name: str
    test_edges_file_name: str
    input_feature: str


@dataclass
class DataArguments:
    r"""
    Arguments specifying the data we are going to input our model for training, evaluation and test.
    """

    dataset_dir: Optional[str] = field(
        default="data",
        metadata={"help": "The name of the folder containing datasets."},
    )
    cutoff_len: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum length of the model inputs after tokenization."},
    )
    subsampling_probs: Optional[str] = field(
        default=None,
        metadata={
            "help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."
        },
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save or load the preprocessed datasets."},
    )

    def init_for_training(self, seed: int):  # support mixing multiple datasets
        self.seed: int = seed
        try:
            with open(os.path.join(self.dataset_dir, "dataset_info.json"), "r") as f:
                dataset_info = json.load(f)
        except Exception:
            raise ValueError("Cannot find dataset_info.json in `dataset_dir`.")

        if self.subsampling_probs is not None:
            self.subsampling_probs: list[float] = [
                float(prob.strip()) for prob in self.subsampling_probs.split(",")
            ]

        self.dataset_attr: DatasetAttr = DatasetAttr.model_validate(dataset_info)
