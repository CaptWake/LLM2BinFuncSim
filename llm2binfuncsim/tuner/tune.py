from typing import Any, Optional

from tuner.core import get_train_args
from tuner.da import run_da

from tuner.sct import run_sct


def run_exp(
    args: Optional[dict[str, Any]] = None,
):
    model_args, data_args, training_args = get_train_args(args)
    if model_args.stage == "sct":
        run_sct(model_args, data_args, training_args)
    elif model_args.stage == "da":
        run_da(model_args, data_args, training_args)
    else:
        raise ValueError("Unknown task.")
