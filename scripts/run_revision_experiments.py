"""Revision experiments for applsci-4589086 (reviewer requests R3-6, R3-9, R3-12, R3-13, R2-7).

Runs every job through train.py under the original protocol (200 epochs, AdamW
3e-4, cosine, batch 32, label smoothing 0.05, best-val-Macro-F1 checkpoint) and
skips jobs whose history.json already exists, so the script can be restarted.

Runs trained for the original submission (seeds 42, 1, 2 of the full model and the
seed-42 ablations, in all_results/runs and runs/variance) are reused when present;
on a fresh checkout, or with --from-scratch, they are trained here as well.

The queue is rebuilt from runs/revision/plan.json each time a worker picks a job,
so seeds and the baselines retrained on the grouped split can be changed without
restarting. Data loading follows the time of day: images cached in RAM with
in-process augmentation by day (low CPU use), multi-process loading at night
(faster, high CPU use). Running jobs are stopped at the switch and resumed from
last.pt in the new mode; both paths give the same images and augmentation.

    python scripts/run_revision_experiments.py [--workers 2] [--dry-run] [--from-scratch]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_local" / "revision"))
from runs import legacy_dir  # noqa: E402

OUT = ROOT / "runs" / "revision"
LOGS = OUT / "logs"
PLAN = OUT / "plan.json"
PYTHON = sys.executable

DATASETS = ("BODMAS", "CICMalDroid2020", "ELF2020")
COMMON = ["--epochs", "200", "--batch-size", "32"]

# HoloNet variants: name -> extra train.py flags
VARIANTS = {
    "full": [],
    "no_ckan": ["--disable-use-ckan"],
    "no_transformer": ["--router", "pooling"],
    "no_margin": ["--margin-weight", "0"],
    "dynamic_routing": ["--router", "dynamic"],
    "no_capsule": ["--router", "token"],
}

DEFAULT_PLAN = {
    "holonet_seeds": [42, 1, 2, 3, 4],
    "baselines": ["resnext50", "convnext_tiny", "vit_s8", "res2net50", "resnext_cbam"],
    "baseline_seeds": [42, 1, 2],
    "grouped_seeds": [42, 1, 2],
    "grouped_baselines": [],  # filled in once the baselines on the original split are known
    "channel_seeds": [42, 1, 2],
    "channel_modes": ["rgb", "gray"],
    "night_hours": [2, 8],     # [start, end): multi-process loading in this window
    "night_loader_workers": 4,
}

lock = threading.Lock()
running: set[str] = set()


def load_plan() -> dict:
    if not PLAN.is_file():
        PLAN.write_text(json.dumps(DEFAULT_PLAN, indent=1), encoding="utf-8")
    return {**DEFAULT_PLAN, **json.loads(PLAN.read_text(encoding="utf-8-sig"))}


def all_jobs(plan: dict) -> list[tuple[str, list[str]]]:
    """Queue order: HoloNet variants; HoloNet on the grouped split; HoloNet on the grouped
    BODMAS split without near-uniform images; baselines on the grouped split; baselines
    on the original split."""
    jobs = []
    for seed in plan["holonet_seeds"]:  # seed-major, so complete seed sets finish first
        for variant, flags in VARIANTS.items():
            for ds in DATASETS:
                if legacy_dir(variant, ds, seed) is not None:
                    continue  # trained for the original submission
                jobs.append((f"holonet_{variant}_{ds.lower()}_s{seed}", ["--dataset", ds, "--seed", str(seed), *flags]))
    grouped = ["--data-root", "RGBA_grouped"]
    for seed in plan["grouped_seeds"]:
        for ds in DATASETS:
            jobs.append((f"grouped_holonet_full_{ds.lower()}_s{seed}", ["--dataset", ds, "--seed", str(seed), *grouped]))
    # the split script writes this file last, so the image folders are complete by then
    if (ROOT / "experiments_local" / "revision" / "results" / "grouped_split_noflat.json").is_file():
        for seed in plan["grouped_seeds"]:
            jobs.append((f"grouped_noflat_holonet_full_bodmas_s{seed}",
                         ["--dataset", "BODMAS", "--seed", str(seed), "--data-root", "RGBA_grouped_noflat"]))
    # input-channel ablation on the stored images (R3-3); RGBA is the full model above
    for seed in plan["channel_seeds"]:
        for mode in plan["channel_modes"]:
            for ds in DATASETS:
                jobs.append((f"holonet_ch_{mode}_{ds.lower()}_s{seed}",
                             ["--dataset", ds, "--seed", str(seed), "--channels", mode]))
    for seed in plan["grouped_seeds"]:
        for arch in plan["grouped_baselines"]:
            for ds in DATASETS:
                jobs.append((f"grouped_base_{arch}_{ds.lower()}_s{seed}",
                             ["--dataset", ds, "--seed", str(seed), "--arch", arch, *grouped]))
    for seed in plan["baseline_seeds"]:
        for arch in plan["baselines"]:
            for ds in DATASETS:
                jobs.append((f"base_{arch}_{ds.lower()}_s{seed}", ["--dataset", ds, "--seed", str(seed), "--arch", arch]))
    return jobs


def is_done(name: str) -> bool:
    return (OUT / name / "history.json").is_file()


def is_night(plan: dict) -> bool:
    """Multi-process loading inside night_hours, or until a one-off "fast_until" time."""
    until = plan.get("fast_until")  # e.g. "2026-10-06 14:05"
    if until and time.strftime("%Y-%m-%d %H:%M") < until:
        return True
    start, end = plan["night_hours"]
    return start <= time.localtime().tm_hour < end


def loader_flags(night: bool, plan: dict) -> list[str]:
    if night:
        return ["--disable-cache-images", "--num-workers", str(plan["night_loader_workers"])]
    return ["--cache-images"]


def log(msg: str) -> None:
    with lock:
        line = f"{time.strftime('%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(OUT / "progress.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def next_job():
    plan = load_plan()
    with lock:
        for name, flags in all_jobs(plan):
            if name not in running and not is_done(name):
                running.add(name)
                return name, flags
    return None


def run(name: str, flags: list[str]) -> None:
    run_dir = OUT / name
    last = run_dir / "last.pt"
    start = time.time()
    while True:
        plan = load_plan()
        night = is_night(plan)
        cmd = [PYTHON, "train.py", *COMMON, *loader_flags(night, plan), *flags, "--run-name", name,
               "--output-dir", str(OUT), "--experiment-name", "revision"]
        if last.is_file():
            cmd += ["--resume", str(last)]
        env = dict(os.environ, TQDM_DISABLE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        log(f"start {name}  [{'night: multi-process loading' if night else 'day: cached'}]")
        with open(LOGS / f"{name}.log", "a", encoding="utf-8") as f:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
            switched = False
            while proc.poll() is None:
                time.sleep(30)
                if proc.poll() is None and is_night(load_plan()) != night:
                    proc.terminate()
                    proc.wait()
                    switched = True
        if switched:
            log(f"switch      {name}  -> {'day' if night else 'night'} mode, resuming")
            continue
        break
    ok = proc.returncode == 0 and is_done(name)
    if ok and last.is_file():
        last.unlink()  # best.pt is kept; last.pt is only needed for resuming
    status = "done" if ok else f"FAILED ({proc.returncode})"
    log(f"{status:12s} {name}  {(time.time() - start) / 60:.1f} min")


def worker() -> None:
    while True:
        job = next_job()
        if job is None:
            return
        try:
            run(*job)
        finally:
            with lock:
                running.discard(job[0])
        if not is_done(job[0]):
            return  # a failed job is not retried in a loop; fix it and restart the script


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-scratch", action="store_true",
                        help="ignore the pre-revision runs and train every job here")
    args = parser.parse_args()
    if args.from_scratch:
        os.environ["HOLONET_FROM_SCRATCH"] = "1"  # read by runs.legacy_dir, also in the analysis scripts
    LOGS.mkdir(parents=True, exist_ok=True)
    plan = load_plan()
    jobs = all_jobs(plan)
    pending = [j for j in jobs if not is_done(j[0])]
    print(f"{len(jobs)} jobs, {len(pending)} pending, {args.workers} workers, night={is_night(plan)}")
    if args.dry_run:
        for name, flags in pending:
            print(" ", name, " ".join(flags))
        return
    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("worker pool finished")


if __name__ == "__main__":
    main()
