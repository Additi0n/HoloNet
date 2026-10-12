"""Supplementary Table S3: test Accuracy and Macro-F1 of every training run.

Writes article/revision/supplementary/TableS3_per_seed_results.csv with one row per
(split, model, dataset, seed), read from each run's history.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from aggregate_results import DATASETS, MODELS, REV, ROOT, SEEDS, run_dir

OUT = ROOT / "article" / "revision" / "supplementary" / "TableS3_per_seed_results.csv"


def metrics(d: Path) -> dict | None:
    """Scores from test_predictions.npz when present (as in aggregate.json), else from history.json."""
    f = d / "history.json"
    if not f.is_file():
        return None
    h = json.loads(f.read_text(encoding="utf-8"))
    out = {"best_epoch": h["best_epoch"], **{k: round(v, 4) for k, v in h["test_metrics"].items() if k != "loss"}}
    if (d / "test_predictions.npz").is_file():
        z = np.load(d / "test_predictions.npz")
        y, p, k = z["y"], z["pred"], z["prob"].shape[1]
        prec, rec, f1 = [], [], []
        for c in range(k):
            tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
            prec.append(tp / (tp + fp) if tp + fp else 0.0)
            rec.append(tp / (tp + fn) if tp + fn else 0.0)
            f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        out.update(accuracy=round(float((y == p).mean()), 4), macro_precision=round(float(np.mean(prec)), 4),
                   macro_recall=round(float(np.mean(rec)), 4), macro_f1=round(float(np.mean(f1)), 4))
    return out


def main() -> None:
    rows = []
    for ds in DATASETS:
        for m in MODELS:
            for s in SEEDS:
                r = metrics(run_dir(m, ds, s))
                if r:
                    rows.append({"split": "original", "model": m, "dataset": ds, "seed": s, **r})
        d = ds.lower()
        extra = [("grouped", m, f"grouped_{p}_{d}_s{{}}") for m, p in
                 [("full", "holonet_full"), ("res2net50", "base_res2net50"), ("convnext_tiny", "base_convnext_tiny")]]
        extra += [("original", f"channels_{c}", f"holonet_ch_{c}_{d}_s{{}}") for c in ("rgb", "gray")]
        if ds == "BODMAS":
            extra.append(("grouped_no_near_uniform", "full", "grouped_noflat_holonet_full_bodmas_s{}"))
        for split, m, pattern in extra:
            for s in (42, 1, 2):
                r = metrics(REV / pattern.format(s))
                if r:
                    rows.append({"split": split, "model": m, "dataset": ds, "seed": s, **r})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
