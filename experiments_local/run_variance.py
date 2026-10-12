"""Multi-seed retraining to add variance (std) + significance to the main results.

Re-runs the EXACT reported protocol (200 epochs, same hyper-params) with extra
random seeds, then aggregates mean +/- std per dataset/metric. The originally
reported run (seed 42, in runs/launcher) is folded in automatically.

Runs smallest dataset first so partial results are useful early. Writes
incrementally to results/variance.json after every finished run.

    python experiments_local/run_variance.py --seeds 1 2 --epochs 200
    python experiments_local/run_variance.py --seeds 1 2 --datasets ELF2020
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from _common import PROJECT_ROOT, OUT

# ordered smallest -> largest train set (fastest feedback first)
ORDER = ["ELF2020", "CICMalDroid2020", "BODMAS"]
LAUNCHER = {
    "ELF2020": PROJECT_ROOT / "runs" / "launcher" / "holonet_elf2020" / "history.json",
    "CICMalDroid2020": PROJECT_ROOT / "runs" / "launcher" / "holonet_cicmaldroid2020" / "history.json",
    "BODMAS": PROJECT_ROOT / "runs" / "launcher" / "holonet_bodmas" / "history.json",
}
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


def read_test_metrics(history_json: Path):
    if not history_json.is_file():
        return None
    data = json.loads(history_json.read_text(encoding="utf-8"))
    return data.get("test_metrics")


def run_one(ds, seed, epochs):
    out_dir = PROJECT_ROOT / "runs" / "variance"
    run_name = f"var_{ds.lower()}_s{seed}"
    hist = out_dir / run_name / "history.json"
    if read_test_metrics(hist) is not None:
        print(f"[skip] {run_name} already done")
        return read_test_metrics(hist)
    cmd = [sys.executable, str(PROJECT_ROOT / "train.py"),
           "--dataset", ds, "--epochs", str(epochs), "--seed", str(seed),
           "--cache-images", "--output-dir", str(out_dir),
           "--run-name", run_name, "--experiment-name", "var"]
    # If a partial run exists (checkpoint but no history.json), resume from it so a
    # reaped/interrupted run continues instead of restarting from epoch 1.
    last_pt = out_dir / run_name / "last.pt"
    if last_pt.is_file():
        cmd += ["--resume", str(last_pt)]
        print(f"[resume] {run_name} from {last_pt}")
    print(f"[run] {run_name}: {' '.join(cmd[2:])}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    return read_test_metrics(hist)


def aggregate(collected):
    agg = {}
    for ds, runs in collected.items():
        agg[ds] = {"n_seeds": len(runs), "seeds": sorted(r["seed"] for r in runs)}
        for m in METRICS:
            vals = [r["metrics"][m] for r in runs if r["metrics"]]
            if vals:
                agg[ds][m] = {
                    "mean": round(statistics.mean(vals), 4),
                    "std": round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 4),
                    "values": [round(v, 4) for v in vals],
                }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--datasets", type=str, nargs="+", default=ORDER)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "variance.json"
    collected = {ds: [] for ds in args.datasets}

    for ds in args.datasets:
        # fold in the originally reported seed-42 run
        base = read_test_metrics(LAUNCHER[ds])
        if base:
            collected[ds].append({"seed": 42, "metrics": base})
        for seed in args.seeds:
            m = run_one(ds, seed, args.epochs)
            if m:
                collected[ds].append({"seed": seed, "metrics": m})
            out_json.write_text(json.dumps(
                {"raw": collected, "summary": aggregate(collected)}, indent=2), encoding="utf-8")
            print(f"[saved] {out_json}")

    print("\n=== variance summary ===")
    for ds, s in aggregate(collected).items():
        if "macro_f1" in s:
            f = s["macro_f1"]
            print(f"{ds}: Macro-F1 {f['mean']} ± {f['std']}  (seeds {s['seeds']})")


if __name__ == "__main__":
    main()
