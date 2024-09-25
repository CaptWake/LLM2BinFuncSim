# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/language-modeling/run_clm.py

import logging
from typing import TYPE_CHECKING

from transformers import DataCollatorWithPadding
from tuner.core.loader import load_model_and_tokenizer

from dsets import get_dataset, preprocess_sct_datasets
from utilities import (
    POOL_SIZE,
    K,
    SimpleLogger,
    SupConLossTrainer,
    compute_top_k,
    get_logger,
)

if TYPE_CHECKING:
    from datasets import DatasetDict
    from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments

    from llm2binfuncsim.config import DataArguments, ModelArguments


def run_sct(
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

    nodes_ds, edge_list_ds = get_dataset(data_args, training_args)

    logger.debug(nodes_ds.__str__())
    logger.debug(edge_list_ds.__str__())

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    logger.debug("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_args,
        # finetuning_args,
        # training_args.do_train,
        stage="sct",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logger.debug("Preprocessing datasets...")
    gs, ds = preprocess_sct_datasets(
        nodes_ds, edge_list_ds, tokenizer, data_args, training_args
    )

    # Initialize our Trainer
    trainer = SupConLossTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **({"gs": gs} if training_args.do_train else {}),
        **(
            {k: v for k, v in ds.items() if k in ["train_dataset"]}
            if training_args.do_train
            else {}
        ),
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()

    # Evaluation
    if training_args.do_eval:
        trainer.add_callback()
        # we want to evaluate top_1@100
        predictions = trainer.predict(
            test_dataset=ds["test_dataset"].remove_columns(
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
        metric: dict[str, float] = {
            f"top_{K}@{POOL_SIZE}": compute_top_k(
                predictions[0][0][:, 0, :], pool_size=POOL_SIZE, k=K
            )
        }
        trainer.log_metrics("eval", metric)
