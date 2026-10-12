from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CapsuleClassificationLoss(nn.Module):
    def __init__(
        self,
        margin_positive: float = 0.9,
        margin_negative: float = 0.1,
        negative_weight: float = 0.5,
        ce_weight: float = 1.0,
        margin_weight: float = 0.25,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.margin_positive = margin_positive
        self.margin_negative = margin_negative
        self.negative_weight = negative_weight
        self.ce_weight = ce_weight
        self.margin_weight = margin_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        capsule_lengths: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

        target_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        positive_cost = torch.relu(self.margin_positive - capsule_lengths).pow(2)
        negative_cost = torch.relu(capsule_lengths - self.margin_negative).pow(2)
        margin_loss = (
            target_one_hot * positive_cost + self.negative_weight * (1.0 - target_one_hot) * negative_cost
        ).sum(dim=1).mean()

        total_loss = self.ce_weight * ce_loss + self.margin_weight * margin_loss
        stats = {
            "total_loss": float(total_loss.detach().item()),
            "ce_loss": float(ce_loss.detach().item()),
            "margin_loss": float(margin_loss.detach().item()),
        }
        return total_loss, stats