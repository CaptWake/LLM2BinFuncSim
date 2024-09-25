from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from transformers.trainer_utils import PredictionOutput


def extract_cls_from_predictions(predictions: "PredictionOutput") -> ArrayLike:
    return predictions[0][0][:, 0, :]
