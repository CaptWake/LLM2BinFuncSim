from typing import TYPE_CHECKING

import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    import torch


def compute_top_k(node_embeddings, pool_size, k) -> float:
    top_k = 0
    n_eval = 0
    # loop retrieving a batch of 100 elements from node_embeddings
    pool_size += 1  # we count also in the pool the target node

    for i in range(
        0, len(node_embeddings) - (len(node_embeddings) % pool_size), pool_size
    ):
        pool = node_embeddings[i : i + pool_size].squeeze()
        ranks = cosine_similarity(pool)[0]
        # Exlude self-distance
        ranks = ranks[1:]
        # the element in position 0 is the positive sample by construction (random walk)
        y = ranks[0]
        rank = np.sort(ranks)[::-1]
        # +1 because the positive sample is in position 0
        position = rank.tolist().index(y) + 1
        top_k += 0 if position > k else 1
        n_eval += 1

    return top_k / n_eval


def compute_mrr_k(batch_embeddings: "torch.Tensor", label: int, k=1):
    batch_size: int = batch_embeddings.size(dim=0)
    ranks: "torch.Tensor" = F.cosine_similarity(
        batch_embeddings[:1, :].repeat(batch_size - 1, 1),
        batch_embeddings[1:, :],
        dim=-1,
    )
    y = ranks[label]
    ranks: "torch.Tensor" = ranks.sort(descending=True)[0]
    position: int = ranks.tolist().index(y) + 1
    return 0 if position > k else 1 / position
