# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/language-modeling/run_clm.py

import logging
from typing import TYPE_CHECKING

from transformers import DataCollatorWithPadding

from llm2binfuncsim.dsets import get_dataset, preprocess_sct_datasets
from llm2binfuncsim.tuner.core import load_model_and_tokenizer
from llm2binfuncsim.utilities.constants import POOL_SIZE, K
from llm2binfuncsim.utilities.loggers import SimpleLogger, get_logger
from llm2binfuncsim.utilities.metrics import compute_top_k
from llm2binfuncsim.utilities.misc import extract_cls_from_predictions
from llm2binfuncsim.utilities.trainers import SupConLossTrainer

if TYPE_CHECKING:
    from transformers import TrainingArguments

    from llm2binfuncsim.config import DataArguments, ModelArguments


def run_sct(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
):
    logger: SimpleLogger = get_logger()
    logger.setLevel(logging.DEBUG)

    logger.debug("Loading dataset...")

    nodes_ds, edge_list_ds = get_dataset(data_args, training_args)

    logger.debug(nodes_ds.__str__())
    logger.debug(edge_list_ds.__str__())

    logger.debug("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_args,
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
                extract_cls_from_predictions(predictions), pool_size=POOL_SIZE, k=K
            )
        }
        trainer.log_metrics("eval", metric)
