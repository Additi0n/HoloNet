"""Confidence and capsule-length statistics over the whole test set (R3-15, R3-16).

Replaces the two-samples-per-class confidence table. For each dataset and seed:
mean top-1 softmax confidence (all / correct / incorrect), the label-smoothing
optimum 1 - eps + eps/K that bounds what training rewards, expected calibration
error (15 bins), and how well confidence and class-capsule length separate correct
from incorrect predictions (AUROC), plus their Spearman correlation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from runs import run_dir

HERE = Path(__file__).resolve().parent
EPS = 0.05
RUNS = {ds: [run_dir("full", ds, s) for s in (42, 1, 2)] for ds in ("BODMAS", "CICMalDroid2020", "ELF2020")}


def ranks(x: np.ndarray) -> np.ndarray:
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(len(x))
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return sums[inv] / counts[inv]


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    r = ranks(score)
    n1, n0 = positive.sum(), (~positive).sum()
    return float((r[positive].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(conf[idx == b].mean() - correct[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def one(npz: Path) -> dict:
    d = np.load(npz)
    y, prob, lengths = d["y"], d["prob"], d["capsule_lengths"]
    k = prob.shape[1]
    pred = prob.argmax(1)
    conf = prob.max(1)
    length = lengths[np.arange(len(pred)), pred]
    correct = pred == y
    optimum = 1 - EPS + EPS / k
    return {
        "n": int(len(y)), "error_rate": round(float((~correct).mean()), 4),
        "label_smoothing_optimum": round(optimum, 4),
        "conf_mean_all": round(float(conf.mean()), 4),
        "conf_mean_correct": round(float(conf[correct].mean()), 4),
        "conf_mean_incorrect": round(float(conf[~correct].mean()), 4),
        "frac_conf_above_optimum": round(float((conf > optimum).mean()), 4),
        "ece": round(ece(conf, correct), 4),
        "auroc_conf_correct_vs_incorrect": round(auroc(conf, correct), 4),
        "auroc_length_correct_vs_incorrect": round(auroc(length, correct), 4),
        "length_mean_correct": round(float(length[correct].mean()), 4),
        "length_mean_incorrect": round(float(length[~correct].mean()), 4),
        "spearman_length_vs_conf": round(spearman(length, conf), 4),
    }


def main() -> None:
    out = {}
    for ds, runs in RUNS.items():
        per_seed = [one(r / "test_predictions.npz") for r in runs]
        keys = [k for k in per_seed[0] if k != "n"]
        out[ds] = {"per_seed": per_seed,
                   "mean": {k: round(float(np.mean([s[k] for s in per_seed])), 4) for k in keys},
                   "std": {k: round(float(np.std([s[k] for s in per_seed])), 4) for k in keys}}
        m, s = out[ds]["mean"], out[ds]["std"]
        print(f"== {ds}  (3 seeds)")
        for k in keys:
            print(f"   {k:36s} {m[k]:.4f} +- {s[k]:.4f}")
    (HERE / "results" / "confidence_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
