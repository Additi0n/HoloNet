"""Cross-platform transfer probe — addresses the #1 review issue (is it really
"cross-platform"?).

For every (source -> target) pair we FREEZE the HoloNet backbone trained on the
source platform, extract a class-agnostic embedding (global-average-pooled
context_refine feature, 96-D) on the target's train/test splits, then fit a
linear classifier on the target's labels and evaluate on the target test set.

  diagonal (source == target)  = in-domain linear-probe (upper reference)
  off-diagonal (source != tgt) = does a backbone trained on platform A produce
                                 features that linearly separate platform B?

If off-diagonal macro-F1 is well above the random baseline, the Bin2Img+HoloNet
representation carries platform-transferable structure -> evidence for the
"cross-platform" claim that single-platform training alone cannot provide.

    python experiments_local/transfer_probe.py
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from _common import RUNS, DATA_ROOT, DATASETS, OUT, safe_load, build_model_from_ckpt
from holonet.data import _scan_split, MalwareImageDataset, _build_transforms
from holonet.metrics import compute_metrics, update_confusion_matrix


def _font():
    for n in ("Segoe UI", "Arial", "DejaVu Sans"):
        try:
            font_manager.findfont(n, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [n, "DejaVu Sans"]
            return
        except Exception:
            continue


@torch.no_grad()
def extract_features(model, hook, split_dir, device, image_size=96):
    """Frozen forward over a split -> (N x 96) GAP features + labels."""
    class_names, samples = _scan_split(split_dir)
    ds = MalwareImageDataset(samples, _build_transforms(image_size, train=False))
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    feats, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        hook["feat"] = None
        model(images)
        f = hook["feat"].mean(dim=(2, 3))  # GAP over 12x12 -> (B, 96)
        feats.append(f.cpu())
        labels.append(targets)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy(), class_names


def linear_probe(xtr, ytr, xte, yte, k, device, epochs=300):
    """Standardise features, fit a single linear layer, return test metrics."""
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    xtr = torch.tensor((xtr - mu) / sd, dtype=torch.float32, device=device)
    xte = torch.tensor((xte - mu) / sd, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    clf = nn.Linear(xtr.shape[1], k).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    clf.train()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(clf(xtr), ytr_t).backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(xte).argmax(1).cpu()
    cm = torch.zeros(k, k, dtype=torch.long)
    update_confusion_matrix(cm, torch.tensor(yte), pred)
    return compute_metrics(cm)


def random_baseline(ytr, yte, k):
    """Macro-F1 of a class-prior random classifier (reference floor)."""
    counts = np.bincount(ytr, minlength=k).astype(float)
    prior = counts / counts.sum()
    rng = np.random.default_rng(0)
    pred = rng.choice(k, size=len(yte), p=prior)
    cm = torch.zeros(k, k, dtype=torch.long)
    update_confusion_matrix(cm, torch.tensor(yte), torch.tensor(pred))
    return compute_metrics(cm)["macro_f1"]


def main():
    _font()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    names = list(DATASETS.values())  # [BODMAS, CICMalDroid2020, ELF2020]

    # cache target features once (independent of source we need per-source backbone,
    # so we recompute per source; targets' label sets are fixed).
    f1 = np.zeros((len(names), len(names)))
    acc = np.zeros((len(names), len(names)))
    rand = {}
    detail = {}

    for si, (src_run, src_ds) in enumerate(DATASETS.items()):
        ckpt = safe_load(RUNS / src_run / "best.pt", device)
        model, _ = build_model_from_ckpt(ckpt, device)
        hook = {"feat": None}
        model.context_refine.register_forward_hook(
            lambda m, i, o: hook.__setitem__("feat", o.detach()))
        for ti, (tgt_run, tgt_ds) in enumerate(DATASETS.items()):
            root = DATA_ROOT / tgt_ds
            xtr, ytr, cls = extract_features(model, hook, root / "train", device)
            xte, yte, _ = extract_features(model, hook, root / "test", device)
            k = len(cls)
            m = linear_probe(xtr, ytr, xte, yte, k, device)
            f1[si, ti] = m["macro_f1"]
            acc[si, ti] = m["accuracy"]
            if tgt_ds not in rand:
                rand[tgt_ds] = random_baseline(ytr, yte, k)
            detail[f"{src_ds}->{tgt_ds}"] = {"macro_f1": round(m["macro_f1"], 4),
                                             "accuracy": round(m["accuracy"], 4)}
            print(f"  {src_ds:>16} -> {tgt_ds:<16}  probe macroF1={m['macro_f1']:.4f}")

    out = {
        "sources": names, "targets": names,
        "macro_f1_matrix": np.round(f1, 4).tolist(),
        "accuracy_matrix": np.round(acc, 4).tolist(),
        "random_baseline_macro_f1": {k: round(v, 4) for k, v in rand.items()},
        "detail": detail,
        "note": "rows=source backbone (frozen), cols=target dataset; diagonal=in-domain probe",
    }
    (OUT / "transfer_probe.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # heatmap
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(f1, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Target dataset (linear probe + test)")
    ax.set_ylabel("Source backbone (frozen)")
    ax.set_title("Cross-platform transfer — probe Macro-F1", fontsize=10, fontweight="bold")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{f1[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if f1[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "transfer_probe.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\n=== random-baseline macro-F1 (floor) ===")
    for ds, r in rand.items():
        print(f"  {ds}: {r:.4f}")
    print(f"\nwrote {OUT/'transfer_probe.json'} + transfer_probe.png")


if __name__ == "__main__":
    main()
