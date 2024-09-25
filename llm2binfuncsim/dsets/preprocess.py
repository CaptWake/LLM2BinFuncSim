from itertools import chain
from typing import TYPE_CHECKING

import networkx as nx
import polars as pl
from datasets import Dataset
from polars import DataFrame
from samplers.pair_sampler import *
from utilities import POOL_SIZE, SimpleLogger, get_logger

if TYPE_CHECKING:
    import numpy as np
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


def preprocess_da_datasets(
    nodes_ds: dict[str, Dataset],
    tokenizer: "PreTrainedTokenizer",
    mlm_data_collator: "DataCollatorForLanguageModeling",
    data_args: "DataArguments",
) -> dict[str, Dataset | dict[str, "np.array"]]:
    tokenized_ds = {}
    for split_name, ds in nodes_ds.items():
        if split_name == "train":
            tokenized_ds[split_name] = ds.map(
                lambda x: simple_tokenizing_function(
                    examples=x,
                    tokenizer=tokenizer,
                    input_feature="asm_code",
                    cutoff_len=data_args.cutoff_len,
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
                    cutoff_len=data_args.cutoff_len,
                ),
                batched=True,
            ).map(
                lambda x: insert_random_mask(batch=x, data_collator=mlm_data_collator),
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

    with training_args.main_process_first(desc="Tokenizing data"):
        for split_name in nodes_ds:
            edges_df: DataFrame = pl.from_pandas(edges_ds[split_name].to_pandas())

            # tokenize the nodes_df
            tokenized_ds: Dataset = nodes_ds[split_name].map(
                lambda x: simple_tokenizing_function(
                    examples=x,
                    tokenizer=tokenizer,
                    input_feature="asm_code",
                    cutoff_len=data_args.cutoff_len,
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

    def tokenize_data(self, dataset_obj, logger):
        """Tokenize the data according to the specified task and chunking strategy.
        Args:
            dataset_obj: The dataset object containing the data to be tokenized.
            logger (Logger): The logger object for logging (cache management).
        Returns:
            Dataset: The tokenized dataset (with setter).
        """
        # Extract useful info from Dataset object
        input_feature, ds = dataset_obj.input_feature, dataset_obj.nodes_ds
        # Cache name depends on name of the dataset and chunking strategy
        cache_path = (
            logger.tokenized_data_dir
        )  # os.path.join(logger.cache_dir, f"tokenized_corpus")
        create_dir(cache_path)
        # If training, a validation dataset will be also available (handling DatasetDict and not Dataset)
        cache_reference = create_cache_folder(
            ds, cache_path=cache_path, cache_filename="tokenized.arrow"
        )
        # For now, no more options at this level
        tokenizing_function = self.simple_tokenizing_function
        tokenized_datasets = self.map_function(
            ds,
            tokenizing_function,
            cache_reference,
            arguments={"input_feature": input_feature},
            load_from_cache_file=self.load_from_cache,
        )
        # Always perform chunking (worst case: we set chunk_id == log id) > chunks of 1 element
        # We'll need another cache path
        cache_reference = create_cache_folder(
            tokenized_datasets,
            cache_path=cache_path,
            cache_filename="chunked_tokenized.arrow",
        )
        tokenized_ds = self.map_function(
            tokenized_datasets,
            self.chunk_assemblies,
            cache_name=cache_reference if self.load_from_cache else None,
            load_from_cache_file=self.load_from_cache,
        )
        dataset_obj.set_tokenized_corpus(tokenized_ds)
