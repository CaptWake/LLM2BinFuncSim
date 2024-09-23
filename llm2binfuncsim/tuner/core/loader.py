from typing import TYPE_CHECKING, Optional, Literal
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM


if TYPE_CHECKING:
    from llm2binfuncsim.config import ModelArguments
    from transformers import PreTrainedModel, PreTrainedTokenizer


def load_model_and_tokenizer(
    model_args: "ModelArguments",
    # finetuning_args: FinetuningArguments,
    stage: Optional[Literal["da", "cl"]] = "cl",
) -> tuple["PreTrainedModel", "PreTrainedTokenizer"]:

    if stage == "da":
        model: "PreTrainedModel" = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path
        )
    else:
        model: "PreTrainedModel" = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path
        )

    tokenizer: "PreTrainedTokenizer" = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        add_prefix_space=True,
        return_special_tokens_mask=(True if stage == "da" else False),
    )

    return model, tokenizer
