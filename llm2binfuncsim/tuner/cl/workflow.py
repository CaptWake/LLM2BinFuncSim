# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/language-modeling/run_clm.py

import math
from typing import TYPE_CHECKING, Optional
import logging

from llm2binfuncsim.dsets import get_dataset  # , preprocess_dataset

# from llmtuner.extras.ploting import plot_loss
from tuner.core.loader import load_model_and_tokenizer
from transformers import DataCollatorForLanguageModeling, Trainer
from utilities.logger import SimpleLogger, get_logger

if TYPE_CHECKING:
    from llm2binfuncsim.config import DataArguments, ModelArguments, DatasetAttr
    from datasets import DatasetDict
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        TrainingArguments,
        TrainerCallback,
    )


def run_cl(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    # finetuning_args: FinetuningArguments,
    # callbacks: Optional[list[TrainerCallback]] = None,
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

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    """
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **preprocess_dataset(
            nodes_ds, edge_list_ds, tokenizer, data_args, training_args, stage="cl"
        )
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
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics: Dict[str, float] = trainer.evaluate(metric_key_prefix="eval")
        try:
            perplexity: float = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    """
