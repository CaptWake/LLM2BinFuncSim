from typing import Optional, cast

import polars as pl
# from classes.datasets.base_dataset import LogDataset
from datasets import Dataset, DatasetDict
from polars import DataFrame, Series
from transformers import TrainingArguments

from llm2binfuncsim.config.data_args import DataArguments, DatasetAttr
from llm2binfuncsim.utilities.loggers import SimpleLogger, get_logger


def _read_dataset_format(ds_path: str) -> DataFrame:
    if ds_path.endswith(".csv"):
        return pl.read_csv(ds_path)
    elif ds_path.endswith((".parquet", ".parquet.gzip")):
        return pl.read_parquet(ds_path)
    else:
        raise ValueError(f"Unsupported file format: {ds_path}")


def get_dataset(
    dataset_args: DataArguments, training_args: TrainingArguments
) -> Dataset | tuple[DatasetDict, DatasetDict]:

    def _build_hf_dataset_from_path(
        ds_nodes_path: str, ds_edge_list_path: Optional[str] = None, prob: float = 1.0
    ) -> Dataset | tuple[Dataset, Dataset]:

        df_nodes: DataFrame = (
            _read_dataset_format(ds_nodes_path)
            .with_row_index(name="rid")
            .sample(
                fraction=prob if ds_edge_list_path is None else 1,
                seed=dataset_args.seed,
            )
        )
        if ds_edge_list_path is None:
            return Dataset(df_nodes.to_arrow())

        # we want to sample only a fraction of the data, to be specific, 1% for training and 10% for validation
        # the validation is smaller than the training set, so we can afford to sample more data

        df_edge_list: DataFrame = _read_dataset_format(ds_edge_list_path).sample(
            fraction=prob, seed=dataset_args.seed
        )

        # Get unique asm_hash values from 'from' and 'to' columns of df_edge_list
        unique_asm_hashes: Series = (
            df_edge_list.select(
                pl.concat_list([pl.col("from"), pl.col("to")]).alias("combined")
            )
            .select(pl.col("combined").flatten().unique())
            .to_series()
        )

        # Filter df_nodes based on asm_hash presence in df_edge_list
        df_nodes: DataFrame = df_nodes.filter(
            pl.col("asm_hash").is_in(unique_asm_hashes)
        )

        return Dataset(df_nodes.to_arrow()), Dataset(df_edge_list.to_arrow())

    logger: SimpleLogger = get_logger()
    dataset_attr: DatasetAttr = dataset_args.dataset_attr

    if training_args.do_predict:
        assert dataset_attr.pred_nodes_file_name is not None
        ds = _build_hf_dataset_from_path(
            dataset_attr.pred_nodes_file_name,
            prob=(
                dataset_args.subsampling_probs[0]
                if len(dataset_args.subsampling_probs) > 0
                else 1.0
            ),
        )
        assert isinstance(ds, Dataset)
        return ds

    # Load training set first
    train_nodes_ds: Dataset
    train_edges_ds: Dataset

    subsampling_probs: list[float] = dataset_args.subsampling_probs

    nodes_ds: DatasetDict = DatasetDict()
    edge_list_ds: DatasetDict = DatasetDict()

    if training_args.do_train:
        assert (
            dataset_attr.training_nodes_file_name is not None
            and dataset_attr.training_edges_file_name is not None
        )
        train_nodes_ds, train_edges_ds = cast(
            tuple[Dataset, Dataset],
            _build_hf_dataset_from_path(
                dataset_attr.training_nodes_file_name,
                dataset_attr.training_edges_file_name,
                subsampling_probs[0],
            ),
        )

        logger.debug(f"\tLoaded rows for TRAINING...")
        nodes_ds.update({"train": train_nodes_ds})
        edge_list_ds.update({"train": train_edges_ds})

        # Now, validation set!
        validation_nodes_ds: Dataset
        validation_edges_ds: Dataset

        assert (
            dataset_attr.validation_nodes_file_name is not None
            and dataset_attr.validation_edges_file_name is not None
        )

        validation_nodes_ds, validation_edges_ds = cast(
            tuple[Dataset, Dataset],
            _build_hf_dataset_from_path(
                dataset_attr.validation_nodes_file_name,
                dataset_attr.validation_edges_file_name,
                subsampling_probs[1],
            ),
        )

        logger.debug(f"\tLoaded rows for VALIDATION...")
        nodes_ds.update({"validation": validation_nodes_ds})
        edge_list_ds.update({"validation": validation_edges_ds})

    if training_args.do_eval:
        test_nodes_ds: Dataset
        test_edges_ds: Dataset

        assert (
            dataset_attr.test_nodes_file_name is not None
            and dataset_attr.test_edges_file_name is not None
        )

        test_nodes_ds, test_edges_ds = cast(tuple[Dataset, Dataset], _build_hf_dataset_from_path(
            dataset_attr.test_nodes_file_name,
            dataset_attr.test_edges_file_name,
            subsampling_probs[2],
        ))

        logger.debug(f"\tLoaded rows for TESTING...")
        nodes_ds.update({"test": test_nodes_ds})
        edge_list_ds.update({"test": test_edges_ds})

    return nodes_ds, edge_list_ds
