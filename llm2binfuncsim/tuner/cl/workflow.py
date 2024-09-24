# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/language-modeling/run_clm.py

import math
from typing import TYPE_CHECKING
import logging

from llm2binfuncsim.dsets import (
    get_dataset,
    preprocess_cl_datasets,
)

# from llmtuner.extras.ploting import plot_loss
from tuner.core.loader import load_model_and_tokenizer
from transformers import (
    DataCollatorWithPadding,
    Trainer,
)
from llm2binfuncsim.utilities import (
    SimpleLogger,
    get_logger,
    SupConLossTrainer,
    compute_top_k,
)

if TYPE_CHECKING:
    from llm2binfuncsim.config import DataArguments, ModelArguments
    from datasets import DatasetDict
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        TrainingArguments,
    )


def run_cl(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    # finetuning_args: FinetuningArguments,
):
    logger: SimpleLogger = get_logger()
    logger.setLevel(logging.DEBUG)

    nodes_ds: DatasetDict
    edge_list_ds: DatasetDict

    logger.debug("Loading dataset...")

    nodes_ds, edge_list_ds = get_dataset(data_args)

    logger.debug(nodes_ds)
    logger.debug(edge_list_ds)

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    logger.debug("Loading model and tokenizer...")

    model, tokenizer = load_model_and_tokenizer(
        model_args,
        # finetuning_args,
        # training_args.do_train,
        stage="cl",
    )

    test_nodes_ds = nodes_ds.pop("test")
    test_edge_list_ds = edge_list_ds.pop("test")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training
    if training_args.do_train:
        logger.debug("Preprocessing train eval dataset...")
        gs, ds = preprocess_cl_datasets(
            nodes_ds, edge_list_ds, tokenizer, data_args, training_args
        )
        # Initialize our Trainer
        trainer = SupConLossTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            gs=gs,
            **ds
        )
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()

    # Evaluation
    if training_args.do_eval:
        logger.debug("Preprocessing test dataset...")
        # we want to evaluate top_1@100
        _, test_ds = preprocess_cl_datasets(
            test_nodes_ds, test_edge_list_ds, tokenizer, data_args, training_args
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        predictions = trainer.predict(
            test_dataset=test_ds["test_dataset"].remove_columns(
                ["rid", "graph_id", "node_id", "asm_code", "asm_hash", "id"]
            )
        )
        compute_top_k(predictions[0][0][:, 0, :], pool_size=100, k=1)
