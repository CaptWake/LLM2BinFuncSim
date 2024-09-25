from typing import TYPE_CHECKING, Any, Optional

from llm2binfuncsim.tuner.core import get_train_args

# from llmtuner.tuner.core.utils import is_first_node
from llm2binfuncsim.tuner.sct import run_sct

# import wandb


if TYPE_CHECKING:
    from transformers import TrainerCallback


def run_exp(
    args: Optional[dict[str, Any]] = None,
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    (model_args, data_args, training_args) = get_train_args(args)
    # if is_first_node():
    #    wandb.login(key=finetuning_args.wandb_token)
    #    wandb.init(
    #        project=finetuning_args.wandb_project,
    #        tags=[*finetuning_args.wandb_tags] if finetuning_args.wandb_tags else None,
    #    )

    # if finetuning_args.stage == "sct":
    run_sct(model_args, data_args, training_args)
    # else:
    #    raise ValueError("Unknown task.")
    """
    elif finetuning_args.stage == "da":
        run_sft(
            model_args,
            data_args,
            training_args,
            finetuning_args,
            generating_args,
            callbacks,
        )
    """
