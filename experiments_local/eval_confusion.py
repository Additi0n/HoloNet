"""Per-class metrics + confusion matrices from the trained best.pt checkpoints.

Fills the missing per-class / confusion-matrix error analysis. Reuses the exact
test split via holonet.build_dataloaders. Outputs JSON + a confusion-matrix PNG
per dataset (row-normalised, publication style).

    python experiments_local/eval_confusion.py
"""
from __future__ import annotations

import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from _common import RUNS, DATA_ROOT, DATASETS, OUT, safe_load, build_model_from_ckpt
from holonet import build_dataloaders


def _font():
    for n in ("Segoe UI", "Arial", "DejaVu Sans"):
        try:
            font_manager.findfont(n, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [n, "DejaVu Sans"]
            return
        except Exception:
            continue


@torch.no_grad()
def confusion(model, loader, device, k):
    cm = torch.zeros(k, k, dtype=torch.long)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        pred = model(images)["logits"].argmax(1).cpu()
        idx = targets * k + pred
        cm += torch.bincount(idx, minlength=k * k).view(k, k)
    return cm.numpy()


def per_class(cm):
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    predicted = cm.sum(0)
    prec = tp / np.maximum(predicted, 1)
    rec = tp / np.maximum(support, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-8)
    return prec, rec, f1, support


def plot_cm(cm, names, ds, path):
    row = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(4.5, n * 0.55), max(4.0, n * 0.5)))
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{ds} — row-normalised confusion", fontsize=10, fontweight="bold")
    thr = 0.5
    for i in range(n):
        for j in range(n):
            v = row[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > thr else "#333")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    _font()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    out = {}
    for run_name, ds in DATASETS.items():
        ckpt = safe_load(RUNS / run_name / "best.pt", device)
        model, di = build_model_from_ckpt(ckpt, device)
        _, _, test_loader, info = build_dataloaders(
            data_root=DATA_ROOT, dataset_name=ds, image_size=96,
            batch_size=128, num_workers=0, val_ratio=0.1, seed=42,
            balanced_sampling=False,
        )
        names = list(info.class_names)
        k = info.num_classes
        cm = confusion(model, test_loader, device, k)
        prec, rec, f1, sup = per_class(cm)
        acc = float(np.diag(cm).sum() / cm.sum())
        out[ds] = {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(prec.mean()), 4),
            "macro_recall": round(float(rec.mean()), 4),
            "macro_f1": round(float(f1.mean()), 4),
            "per_class": {
                names[i]: {"precision": round(float(prec[i]), 4),
                           "recall": round(float(rec[i]), 4),
                           "f1": round(float(f1[i]), 4),
                           "support": int(sup[i])}
                for i in range(k)
            },
            "confusion_matrix": cm.tolist(),
            "class_names": names,
        }
        plot_cm(cm, names, ds, OUT / f"confusion_{ds}.png")
        worst = sorted(range(k), key=lambda i: f1[i])[:3]
        print(f"[{ds}] acc={acc:.4f} macroF1={f1.mean():.4f} | "
              f"weakest: {', '.join(f'{names[i]}(F1={f1[i]:.3f})' for i in worst)}")
    (OUT / "confusion_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT / 'confusion_metrics.json'} + confusion_*.png")


if __name__ == "__main__":
    main()
