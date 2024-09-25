from dataclasses import dataclass, field
from typing import Literal, Optional

from torch import bfloat16, float16, float32


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models."
        }
    )
    stage: Optional[Literal["sct", "da", "emb"]] = field(
        default="sct",
        metadata={"help": "Which stage will be performed in training."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co."
        },
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    split_special_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not the special tokens should be split during the tokenization process."
        },
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `huggingface-cli login`."
        },
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory(s) containing the delta model checkpoints as well as the configurations."
        },
    )
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."},
    )
    hf_auth_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    use_custom_callback: Optional[bool] = field(
        default=False, metadata={"help": "Whether enable custom callbacks."}
    )
    call_back_save_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of epochs before each call back save."},
    )

    def __post_init__(self):
        self.model_max_length = None

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError(
                "`split_special_tokens` is only supported for slow tokenizers."
            )

        if self.checkpoint_dir is not None:  # support merging multiple lora weights
            self.checkpoint_dir = [cd.strip() for cd in self.checkpoint_dir.split(",")]

        if self.use_auth_token == True and self.hf_auth_token is not None:
            from huggingface_hub.hf_api import HfFolder  # lazy load

            HfFolder.save_token(self.hf_auth_token)
