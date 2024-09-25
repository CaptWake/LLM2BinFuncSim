from itertools import chain
from typing import TYPE_CHECKING

import networkx as nx
import polars as pl
from datasets import Dataset
from numpy.typing import ArrayLike
from polars import DataFrame

from llm2binfuncsim.samplers.pair_sampler import *
from llm2binfuncsim.utilities.constants import POOL_SIZE
from llm2binfuncsim.utilities.loggers import SimpleLogger, get_logger

if TYPE_CHECKING:
    from datasets import DatasetDict
    from transformers import DataCollatorForLanguageModeling, TrainingArguments
    from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

    from llm2binfuncsim.config.data_args import DataArguments


logger: SimpleLogger = get_logger()


def insert_random_mask(
    batch: dict, data_collator: "DataCollatorForLanguageModeling"
) -> dict:
    """As reported in https://huggingface.co/learn/nlp-course/chapter7/3#fine-tuning-distilbert-with-accelerate,
    ````
    We saw that DataCollatorForLanguageModeling also applies random masking with each evaluation, so we’ll see some fluctuations in our perplexity scores with each training run. One way to eliminate this source of randomness is to apply the masking once on the whole test set, and then use the default data collator.
    ```
    Args:
        batch (dict): The input batch of data.
        data_collator (DataCollatorForLanguageModeling): The data collator object.

    Returns:
        dict: A dictionary containing masked inputs.
    """
    batch = {k: batch[k] for k in ["input_ids", "attention_mask"] if k in batch}
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {k: v.numpy() for k, v in masked_inputs.items()}


def simple_tokenizing_function(
    examples: dict,
    tokenizer: "PreTrainedTokenizer",
    input_feature: str,
    cutoff_len: int,
) -> "BatchEncoding":
    """Simply tokenizing the input feature of the given dataset with a truncation strategy
    Args:
        examples (dict): A dictionary containing example inputs.
        tokenizer: The tokenizer used to encode the text
        cutoff_len: The maximum number of tokens
        input_feature (str): The name of the feature containing the input text.
    Returns:
        BatchEncoding: Tokenized features for the examples.
    """
    features: "BatchEncoding" = tokenizer(
        examples[input_feature],
        max_length=cutoff_len,
        truncation=True,
    )
    features["rid"] = examples["rid"]  # Keep track of
    features["asm_hash"] = examples["asm_hash"]
    return features


def tokenize_node_dataset(
    nodes_ds: Dataset,
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
) -> Dataset:

    with training_args.main_process_first(desc="Tokenizing data"):
        tokenized_ds = nodes_ds.map(
            lambda x: simple_tokenizing_function(
                examples=x,
                tokenizer=tokenizer,
                input_feature="asm_code",
                cutoff_len=data_args.cutoff_len,  # type: ignore
            ),
            batched=True,
        )
    return tokenized_ds


def preprocess_da_datasets(
    nodes_ds: dict[str, Dataset],
    tokenizer: "PreTrainedTokenizer",
    mlm_data_collator: "DataCollatorForLanguageModeling",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
) -> dict[str, Dataset | dict[str, ArrayLike]]:
    assert data_args.cutoff_len is not None

    tokenized_ds = {}
    with training_args.main_process_first(desc="Tokenizing data"):
        for split_name, ds in nodes_ds.items():
            if split_name == "train":
                tokenized_ds[split_name] = ds.map(
                    lambda x: simple_tokenizing_function(
                        examples=x,
                        tokenizer=tokenizer,
                        input_feature="asm_code",
                        cutoff_len=data_args.cutoff_len,  # type: ignore
                    ),
                    batched=True,
                    remove_columns=ds.column_names,
                )
            else:
                tokenized_ds[split_name] = ds.map(
                    lambda x: simple_tokenizing_function(
                        examples=x,
                        tokenizer=tokenizer,
                        input_feature="asm_code",
                        cutoff_len=data_args.cutoff_len,  # type: ignore
                    ),
                    batched=True,
                ).map(
                    lambda x: insert_random_mask(
                        batch=x, data_collator=mlm_data_collator
                    ),
                    batched=True,
                    remove_columns=ds.column_names,
                )
    return tokenized_ds


def preprocess_sct_datasets(
    nodes_ds: "DatasetDict",
    edges_ds: "DatasetDict",
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
) -> tuple[list[nx.Graph], dict[str, Dataset]]:

    ds: dict[str, Dataset] = {}
    gs: list[nx.Graph] = []
    assert data_args.cutoff_len is not None

    with training_args.main_process_first(desc="Tokenizing data"):
        for split_name in nodes_ds:
            edges_df: DataFrame = pl.from_pandas(edges_ds[split_name].to_pandas())

            # tokenize the nodes_df
            tokenized_ds: Dataset = nodes_ds[split_name].map(
                lambda x: simple_tokenizing_function(
                    examples=x,
                    tokenizer=tokenizer,
                    input_feature="asm_code",
                    cutoff_len=data_args.cutoff_len,  # type: ignore
                ),
                batched=True,
            )

            node_to_rid: dict[str, int] = {}

            def update_row_id_map(x):
                # TODO: What if there are multiple chunks? In this case we retrieve only the last one
                node_to_rid[x[1]] = x[0]
                return 0  # necessary

            pl.from_pandas(tokenized_ds.to_pandas()).with_row_index().select(
                pl.col("index"), pl.col("asm_hash")
            ).map_rows(lambda x: update_row_id_map(x))

            G: nx.Graph = nx.from_edgelist(edges_df[["from", "to"]].to_numpy())
            gs.append(G)
            edge_list = list(G.edges)

            if split_name == "train":
                batch_sampler = SoftBatchPairSampler(
                    edge_list, node_to_rid, training_args.per_device_train_batch_size
                )
                split_name = "train_dataset"

            elif split_name == "validation":
                batch_sampler = SoftBatchPairSampler(
                    edge_list,
                    node_to_rid,
                    training_args.per_device_train_batch_size,
                    static=True,
                )
                split_name = "eval_dataset"
            else:
                G.remove_edges_from(nx.selfloop_edges(G))
                batch_sampler = StrongBatchPairSampler(G, node_to_rid, POOL_SIZE)
                split_name = "test_dataset"

            # generate a new dataset stacking batches on bottom the other, for distributed setup (avoid sharding issue)
            sample_idx: list[int] = [batch_idx for batch_idx in chain(*batch_sampler)]
            ds[split_name] = tokenized_ds.select(sample_idx)

    return gs, ds
