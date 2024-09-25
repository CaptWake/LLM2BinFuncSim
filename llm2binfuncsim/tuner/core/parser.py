import os
import sys
from typing import Any, Dict, Optional, Tuple

import datasets
import transformers
from transformers import HfArgumentParser, TrainingArguments

from llm2binfuncsim.config.data_args import DataArguments
from llm2binfuncsim.config.model_args import ModelArguments


def _parse_args(
    parser: HfArgumentParser, args: Optional[Dict[str, Any]] = None
) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()


def parse_train_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
        )
    )
    return _parse_args(parser, args)


def parse_infer_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[ModelArguments, DataArguments]:
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
        )
    )
    return _parse_args(parser, args)


def get_train_args(args: Optional[Dict[str, Any]] = None) -> Tuple[
    ModelArguments,
    DataArguments,
    TrainingArguments,
]:
    (
        model_args,
        data_args,
        training_args,
    ) = parse_train_args(args)

    # Setup logging

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.logging.set_verbosity(log_level)
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()

    # Check arguments
    data_args.init_for_training(training_args.seed)
    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    return (model_args, data_args, training_args)
