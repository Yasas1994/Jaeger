import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """ArcFace additive angular margin loss."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = 0.5,
        scale: float = 30.0,
        onehot: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        self.onehot = onehot
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, labels: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, D), labels: (B, C) one-hot or (B,) long
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cos_t = torch.matmul(embeddings_norm, weight_norm.t())

        if self.onehot:
            target = labels.argmax(dim=1)
        else:
            target = labels

        # Additive angular margin
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        sin_t = torch.sqrt(1.0 - cos_t.pow(2) + 1e-6)
        cos_t_plus_m = cos_t * cos_m - sin_t * sin_m
        one_hot = F.one_hot(target, num_classes=self.num_classes).to(cos_t.dtype)
        logits = one_hot * cos_t_plus_m + (1.0 - one_hot) * cos_t
        logits = logits * self.scale
        return F.cross_entropy(logits, target)
