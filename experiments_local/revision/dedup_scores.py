"""Re-score test predictions with leaked samples removed (reviewer R3-6)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"


def macro_f1(y: np.ndarray, p: np.ndarray, k: int) -> float:
    f1 = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def scores(npz: Path, dedup: dict) -> dict:
    d = np.load(npz)
    names, y, p = d["names"], d["y"], d["pred"]
    k = d["prob"].shape[1]
    exact = set(dedup["leaked_test_files_exact"]); near = set(dedup["leaked_test_files_near"])
    out = {}
    for label, drop in (("full", set()), ("minus_exact", exact), ("minus_near", near)):
        m = np.array([n not in drop for n in names])
        out[label] = {"n": int(m.sum()), "accuracy": round(float((y[m] == p[m]).mean()), 4),
                      "macro_f1": round(macro_f1(y[m], p[m], k), 4)}
    return out


if __name__ == "__main__":
    dedup = json.loads((RES / "dedup.json").read_text(encoding="utf-8"))
    for run in sys.argv[1:]:
        run = Path(run)
        h = json.loads((run / "history.json").read_text(encoding="utf-8"))
        s = scores(run / "test_predictions.npz", dedup[h["dataset"]])
        print(f"{run.name:46s} " + "  ".join(f"{k}: n={v['n']} acc={v['accuracy']:.4f} F1={v['macro_f1']:.4f}" for k, v in s.items()))
