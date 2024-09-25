# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/language-modeling/run_clm.py

import logging
import math
from typing import TYPE_CHECKING

from llm2binfuncsim.dsets.loader import get_dataset
from llm2binfuncsim.dsets.preprocess import preprocess_da_datasets
from llm2binfuncsim.tuner.core.loader import load_model_and_tokenizer
from llm2binfuncsim.utilities.loggers import SimpleLogger, get_logger
from llm2binfuncsim.utilities.trainers import MLMTrainer

if TYPE_CHECKING:
    from transformers import TrainingArguments

    from llm2binfuncsim.config.data_args import DataArguments
    from llm2binfuncsim.config.model_args import ModelArguments

from transformers import DataCollatorForLanguageModeling, default_data_collator


def run_da(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
):
    logger: SimpleLogger = get_logger()
    logger.setLevel(logging.DEBUG)

    logger.debug("Loading dataset...")

    nodes_ds, _ = get_dataset(data_args, training_args)

    logger.debug(nodes_ds.__str__())

    logger.debug("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_args, stage="da")

    mlm_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    logger.debug("Preprocessing datasets...")
    tokenized_ds = preprocess_da_datasets(
        nodes_ds, tokenizer, mlm_data_collator, data_args, training_args
    )

    # Initialize our Trainer
    trainer = MLMTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=mlm_data_collator,
        eval_data_collator=default_data_collator,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
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
        metrics = trainer.evaluate(
            eval_dataset=tokenized_ds["test"], metric_key_prefix="eval"
        )
        try:
            # metrics["eval_loss"] already contains the mean of the batch losses
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
