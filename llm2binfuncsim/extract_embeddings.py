# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/language-modeling/run_clm.py

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from datasets import Dataset

from llm2binfuncsim.dsets.loader import get_dataset
from llm2binfuncsim.dsets.preprocess import tokenize_node_dataset
from llm2binfuncsim.tuner.core.loader import load_model_and_tokenizer
from llm2binfuncsim.utilities.loggers import SimpleLogger, get_logger
from llm2binfuncsim.utilities.misc import extract_cls_from_predictions
from llm2binfuncsim.utilities.trainers import EmbeddingsExtractionTrainer

if TYPE_CHECKING:
    from transformers import TrainingArguments
    from llm2binfuncsim.config.data_args import DataArguments
    from llm2binfuncsim.config.model_args import ModelArguments

from transformers import DataCollatorWithPadding


def run_emb(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
):
    logger: SimpleLogger = get_logger()
    logger.setLevel(logging.DEBUG)

    logger.debug("Loading dataset...")

    nodes_ds = get_dataset(data_args, training_args)
    assert isinstance(nodes_ds, Dataset)

    logger.debug(nodes_ds.__str__())

    logger.debug("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_args)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logger.debug("Tokenize the dataset...")
    tokenized_ds = tokenize_node_dataset(nodes_ds, tokenizer, data_args, training_args)

    trainer = EmbeddingsExtractionTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
    )

    outputs = trainer.predict(
        tokenized_ds.remove_columns(
            column_names=[
                "rid",
                "graph_id",
                "node_id",
                "asm_code",
                "asm_hash",
                "id",
            ]
        )
    )
    embeddings = extract_cls_from_predictions(outputs)

    data = {
        "asm_hash": nodes_ds["asm_hash"],
        "embeddings": np.reshape(embeddings, (len(tokenized_ds), -1)),
    }
    node_embeddings_df = pl.DataFrame(data)

    df = pl.from_pandas(nodes_ds.to_pandas())

    embeddings_path = os.path.join(
        training_args.output_dir,
        "extracted_embeddings",
    )

    os.makedirs(embeddings_path, exist_ok=True)

    df.join(node_embeddings_df, on="asm_hash", how="left").drop_nulls().write_parquet(
        os.path.join(embeddings_path, "node_embeddings.parquet")
    )
