from typing import TYPE_CHECKING, Any, Optional

# import wandb

# from llmtuner.extras.logging import get_logger
from llm2binfuncsim.tuner.core import (
    # get_infer_args,
    get_train_args,
    # load_model_and_tokenizer,
)

# from llmtuner.tuner.core.utils import is_first_node
from llm2binfuncsim.tuner.cl import run_cl

if TYPE_CHECKING:
    from transformers import TrainerCallback


# logger = get_logger(__name__)


def run_exp(
    args: Optional[dict[str, Any]] = None,
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    (
        model_args,
        data_args,
        training_args,
        #  finetuning_args,
        #        generating_args,
    ) = get_train_args(args)
    # if is_first_node():
    #    wandb.login(key=finetuning_args.wandb_token)
    #    wandb.init(
    #        project=finetuning_args.wandb_project,
    #        tags=[*finetuning_args.wandb_tags] if finetuning_args.wandb_tags else None,
    #    )

    # if finetuning_args.stage == "cl":
    run_cl(model_args, data_args, training_args)
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
