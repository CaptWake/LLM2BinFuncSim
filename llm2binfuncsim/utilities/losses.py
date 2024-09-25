from typing import Optional

import torch
import torch.nn as nn


class Similarity(nn.Module):
    """
    Cosine similarity
    """

    def __init__(self, temperature: float):
        super().__init__()
        self.temp: float = temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.cos(x, y) / self.temp


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    SimCSE: https://arxiv.org/pdf/2104.08821.pdf
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: Optional[float] = None,
    ):
        """Initialization module.
        Arguments:
            temperature (float, optional) -- Similarities will be weighted by a temperature factor: temperatures between [0,1] will act as an amplification factor. Defaults to 0.07.
            contrast_mode (str, optional) -- In case of images, how many filters consider for the similarity. On our case, "all" is equivalent to "one" > text has only 1 filter. Defaults to 'all'.
            base_temperature (float, optional) -- Another factor which weight the logits. If None, logits are not weighted (see below). Defaults to None.
        """
        super(SupConLoss, self).__init__()
        self.temperature: float = temperature
        self.contrast_mode: str = contrast_mode
        self.base_temperature: float = (
            base_temperature if base_temperature != None else self.temperature
        )
        self.criterion: Similarity = Similarity(self.temperature)

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Compute loss for model.
        Args:
            features: hidanchor_dot_contrastden vector of shape [bsz, hidden_dim].
            labels: ground truth of shape [bsz]
        Returns:
            A loss scalar.
        """
        # 1. Get device
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")
        # 2. Extract the batch size
        batch_size: int = features.shape[0]
        ## 3c. The mask is [bsz, bsz]
        # [[1, 1, 0, 0, 0, 0]
        # [1, 1, 0, 0, 0, 0],
        # [0, 0, 1, 1, 0, 0],
        # [0, 0, 1, 1, 0, 0],
        # [0, 0, 0, 0, 1, 1],
        # [0, 0, 0, 0, 1, 1]]
        # 4. Extracr anchor and contrast feature (same vector on our case)
        contrast_feature = features
        anchor_feature = features
        # 5. Compute similarity
        anchor_dot_contrast = self.criterion(
            anchor_feature.unsqueeze(1), contrast_feature.unsqueeze(0)
        )
        # 6. Since we want value <0, normalize rows picking max logit
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # 7. Normalize
        logits = (
            anchor_dot_contrast - logits_max.detach()
        )  # act as a scaling (all values are now < 0)
        # 8. mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        # 9. Exponentiate: since values are <0, exp_logits will be at most 1
        exp_logits = torch.exp(logits) * logits_mask
        # 10. This is where we take into account the negative sample (having different labels)
        #   Let's say that all the positive sample with respect to the anchor had the maximum logit (only one with '0', after row 76):
        #   On line 89, we'd obtain a loss of 0, if that was simply enough.
        #   However, we also want to penalize the model if all the negative samples are not far enough (if they are far, exp_logits.sum(1) is small).
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # 11. compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # 12. Scale wrt temperature parameters (if defined)
        loss = (
            (-(self.temperature / self.base_temperature) * mean_log_prob_pos)
            .view(1, batch_size)
            .mean()
        )
        return loss
