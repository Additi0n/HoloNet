from __future__ import annotations

import torch


@torch.no_grad()
def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> torch.Tensor:
    num_classes = confusion_matrix.size(0)
    indices = targets * num_classes + predictions
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion_matrix += counts.view(num_classes, num_classes)
    return confusion_matrix


@torch.no_grad()
def compute_metrics(confusion_matrix: torch.Tensor) -> dict[str, float]:
    confusion = confusion_matrix.float()
    total = confusion.sum().clamp_min(1.0)
    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)

    accuracy = true_positive.sum() / total
    precision = true_positive / predicted.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)

    return {
        "accuracy": float(accuracy.item()),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())