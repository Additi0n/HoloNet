"""Collect all runs, report mean +- std over seeds and test each model against
HoloNet (R3-9).

Seeds 42, 1, 2 of the full model come from all_results/runs and runs/variance,
seed-42 ablations from all_results/runs, everything else from runs/revision.
Missing test_predictions.npz files are generated first.

Two tests per comparison: Welch's t-test over seed-level Macro-F1, and McNemar's
exact test on per-sample correctness for each shared seed (reported as the number
of seeds with p < 0.05). Welch p-values are also Holm-corrected over all
comparisons against the full model on each dataset. Scores are given on the original test set and on the
test set without exact pixel duplicates of training images.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REV = ROOT / "runs" / "revision"
DATASETS = ("BODMAS", "CICMalDroid2020", "ELF2020")
SEEDS = (42, 1, 2, 3, 4)
MODELS = ("full", "no_ckan", "no_transformer", "no_margin", "dynamic_routing", "no_capsule",
          "resnext50", "resnext_cbam", "res2net50", "convnext_tiny", "vit_s8")


def run_dir(model: str, ds: str, seed: int) -> Path:
    d = ds.lower()
    legacy = {
        ("full", 42): ROOT / "all_results/runs" / f"holonet_{d}",
        ("full", 1): ROOT / "runs/variance" / f"var_{d}_s1",
        ("full", 2): ROOT / "runs/variance" / f"var_{d}_s2",
        ("no_ckan", 42): ROOT / "all_results/runs" / f"ablation_no_ckan_{d}_no_ckan",
        ("no_transformer", 42): ROOT / "all_results/runs" / f"ablation_no_transformer_{d}_no_transformer",
        ("no_margin", 42): ROOT / "all_results/runs" / f"ablation_no_margin_{d}_no_margin",
    }
    if (model, seed) in legacy:
        return legacy[(model, seed)]
    prefix = "holonet_" if model in MODELS[:6] else "base_"
    return REV / f"{prefix}{model}_{d}_s{seed}"


def macro_f1(y, p, k):
    out = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))


def mcnemar_exact(a_correct, b_correct) -> float:
    b = int(np.sum(a_correct & ~b_correct)); c = int(np.sum(~a_correct & b_correct))
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    dedup = json.loads((HERE / "results" / "dedup.json").read_text(encoding="utf-8"))
    runs = {(m, ds, s): run_dir(m, ds, s) for m in MODELS for ds in DATASETS for s in SEEDS}
    done = {k: v for k, v in runs.items() if (v / "history.json").is_file()}
    need = [str(v) for v in done.values() if not (v / "test_predictions.npz").is_file()]
    if need:
        subprocess.check_call([sys.executable, str(HERE / "dump_predictions.py"), *need], cwd=ROOT)

    preds, metrics = {}, defaultdict(dict)
    for (m, ds, s), d in done.items():
        z = np.load(d / "test_predictions.npz")
        names, y, p = z["names"], z["y"], z["pred"]
        k = z["prob"].shape[1]
        keep = np.array([n not in set(dedup[ds]["leaked_test_files_exact"]) for n in names])
        preds[(m, ds, s)] = (y, p)
        metrics[(m, ds)][s] = {"acc": float((y == p).mean()), "f1": macro_f1(y, p, k),
                               "acc_dedup": float((y[keep] == p[keep]).mean()), "f1_dedup": macro_f1(y[keep], p[keep], k)}

    table = {}
    for ds in DATASETS:
        ref = metrics.get(("full", ds), {})
        for m in MODELS:
            per_seed = metrics.get((m, ds), {})
            if not per_seed:
                continue
            row = {"n_seeds": len(per_seed), "seeds": sorted(per_seed)}
            for key in ("acc", "f1", "acc_dedup", "f1_dedup"):
                v = [per_seed[s][key] for s in sorted(per_seed)]
                row[key] = {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 4)}
            if m != "full" and len(per_seed) > 1 and len(ref) > 1:
                a = [ref[s]["f1"] for s in ref]; b = [per_seed[s]["f1"] for s in per_seed]
                row["welch_p_f1_vs_full"] = round(float(stats.ttest_ind(a, b, equal_var=False).pvalue), 4)
                shared = sorted(set(ref) & set(per_seed))
                ps = [mcnemar_exact(*(lambda r, o: (r[0] == r[1], o[0] == o[1]))(preds[("full", ds, s)], preds[(m, ds, s)]))
                      for s in shared]
                row["mcnemar_p_per_seed"] = [round(x, 4) for x in ps]
                row["mcnemar_seeds_significant"] = f"{sum(x < 0.05 for x in ps)}/{len(ps)}"
            table.setdefault(ds, {})[m] = row
        # Holm-Bonferroni over all comparisons against the full model on this dataset
        tested = sorted((r["welch_p_f1_vs_full"], m) for m, r in table.get(ds, {}).items() if "welch_p_f1_vs_full" in r)
        running = 0.0
        for i, (p, m) in enumerate(tested):
            running = max(running, min(1.0, (len(tested) - i) * p))
            table[ds][m]["welch_p_holm"] = round(running, 4)

    (HERE / "results" / "aggregate.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    for ds, rows in table.items():
        print(f"\n== {ds}")
        print(f"   {'model':16s} n  Acc            Macro-F1        F1 (dedup)      Welch p  Holm p   McNemar sig")
        for m, r in rows.items():
            print(f"   {m:16s} {r['n_seeds']}  {r['acc']['mean']:.4f}+-{r['acc']['std']:.4f}  "
                  f"{r['f1']['mean']:.4f}+-{r['f1']['std']:.4f}  {r['f1_dedup']['mean']:.4f}+-{r['f1_dedup']['std']:.4f}  "
                  f"{r.get('welch_p_f1_vs_full', ''):>7}  {r.get('welch_p_holm', ''):>7}  "
                  f"{r.get('mcnemar_seeds_significant', '')}")


if __name__ == "__main__":
    main()
