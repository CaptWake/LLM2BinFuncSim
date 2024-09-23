from itertools import chain
from typing import (
    TYPE_CHECKING,
    Literal,
)

import polars as pl
import networkx as nx
from llm2binfuncsim.utilities.logger import get_logger
from llm2binfuncsim.samplers.pair_sampler import *

if TYPE_CHECKING:
    from datasets import DatasetDict, Dataset
    from llm2binfuncsim.utilities.logger import SimpleLogger
    from llm2binfuncsim.config.data_args import DataArguments
    from transformers import TrainingArguments
    from polars import DataFrame
    from transformers.tokenization_utils import PreTrainedTokenizer, BatchEncoding


logger: SimpleLogger = get_logger()


def preprocess_dataset(
    nodes_ds: DatasetDict,
    edges_ds: DatasetDict,
    data_args: DataArguments,
    training_args: TrainingArguments,
    stage: Literal["da", "cl"],
) -> dict[str, Dataset]:

    def simple_tokenizing_function(
        tokenizer: PreTrainedTokenizer, examples: dict, input_feature: str
    ) -> dict:
        """Simply tokenizing the input feature of the given dataset.
        Also implements a truncation vs simple_chunking strategty (decides whether to truncate or not the input text).
        Args:
            examples (dict): A dictionary containing example inputs.
            input_feature (str): The name of the feature containing the input text.
        Returns:
            dict: Tokenized features for the examples.
        """
        features: BatchEncoding = tokenizer(
            examples[input_feature],
            max_length=512,  # model_max_length,
            truncation=True,
        )
        features["rid"] = examples["rid"]  # Keep track of
        features["asm_hash"] = examples["asm_hash"]
        return features

    ds: dict[str, Dataset] = {}
    # For now, no more options at this level
    with training_args.main_process_first(desc="Tokenizing data"):
        for split_name in nodes_ds:
            nodes_df: DataFrame = pl.from_pandas(
                nodes_ds[split_name].set_format(type="pandas")
            ).with_row_index()
            edges_df: DataFrame = pl.from_pandas(
                edges_ds[split_name].set_format(type="pandas")
            )
            node_to_rid: dict[str, int] = {}

            def update_row_id_map(x):
                # TODO: What if there are multiple chunks? In this case we retrieve only the last one
                node_to_rid[x[1]] = x[0]
                return 0  # necessary

            nodes_df.select(pl.col("index"), pl.col("asm_hash")).map_rows(
                lambda x: update_row_id_map(x)
            )

            G: nx.Graph = nx.from_edgelist(edges_df[["from", "to"]].to_numpy())
            edge_list = list(G.edges)

            if split_name == "train":
                batch_sampler: SoftBatchPairSampler = SoftBatchPairSampler(
                    edge_list, node_to_rid, training_args.per_device_train_batch_size
                )
                split_name = "train_dataset"

            elif split_name == "validation":
                batch_sampler: SoftBatchPairSampler = SoftBatchPairSampler(
                    edge_list,
                    node_to_rid,
                    training_args.per_device_eval_batch_size,
                    static=True,
                )
                split_name = "eval_dataset"

            # generate a new dataset stacking batches on bottom the other, for distributed setup (avoid sharding issue)
            sample_idx: list[int] = [batch_idx for batch_idx in chain(*batch_sampler)]

            # tokenize the nodes_df
            tokenized_ds: Dataset = nodes_df.select(sample_idx).map(
                lambda x: simple_tokenizing_function, input_feature="code"
            )

            ds[split_name] = tokenized_ds

            # batch_size=args.per_device_train_batch_size,
            # collate_fn=lambda x: collate_padding_train_valid_labels(
            #     x, G, data_collator
            # ),
            # )

    return ds

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

    def chunk_assemblies(self, sample):
        """Chunk the assemblies into smaller parts.
        Args:
            sample (dict): A dictionary containing assembly inputs.
        Returns:
            dict: A dictionary containing chunked assemblies and related information.
        """
        chunks, attentions, rids, asm_hash = [], [], [], [], []
        for sample_id, assembly in enumerate(sample["input_ids"]):
            chunks += [assembly]
            attentions += [[1 for el in assembly]]
            rids += [sample["rid"][sample_id]]
            asm_hash += [sample["asm_hash"][sample_id]]
        return {
            "rid": rids,
            "asm_hash": asm_hash,
            "input_ids": chunks,
            "attention_mask": attentions,
        }
