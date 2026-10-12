"""Where each training run lives.

Runs trained for the original submission (seeds 42, 1, 2 of the full model and the
seed-42 ablations) are in all_results/runs and runs/variance. Every other run, and
every run on a fresh checkout, is in runs/revision, named as
scripts/run_revision_experiments.py names it. The legacy folder is used only if it
holds a finished run (history.json); setting HOLONET_FROM_SCRATCH=1 ignores it.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "runs" / "revision"
HOLONET_VARIANTS = ("full", "no_ckan", "no_transformer", "no_margin", "dynamic_routing", "no_capsule")


def from_scratch() -> bool:
    return os.environ.get("HOLONET_FROM_SCRATCH", "") not in ("", "0")


def legacy_dir(model: str, ds: str, seed: int) -> Path | None:
    """The pre-revision folder of this run, or None if there is none."""
    d = ds.lower()
    legacy = {
        ("full", 42): ROOT / "all_results/runs" / f"holonet_{d}",
        ("full", 1): ROOT / "runs/variance" / f"var_{d}_s1",
        ("full", 2): ROOT / "runs/variance" / f"var_{d}_s2",
        ("no_ckan", 42): ROOT / "all_results/runs" / f"ablation_no_ckan_{d}_no_ckan",
        ("no_transformer", 42): ROOT / "all_results/runs" / f"ablation_no_transformer_{d}_no_transformer",
        ("no_margin", 42): ROOT / "all_results/runs" / f"ablation_no_margin_{d}_no_margin",
    }
    path = legacy.get((model, seed))
    if path is None or from_scratch() or not (path / "history.json").is_file():
        return None
    return path


def revision_name(model: str, ds: str, seed: int) -> str:
    prefix = "holonet_" if model in HOLONET_VARIANTS else "base_"
    return f"{prefix}{model}_{ds.lower()}_s{seed}"


def run_dir(model: str, ds: str, seed: int) -> Path:
    """Folder of a HoloNet variant or baseline run on the original split."""
    return legacy_dir(model, ds, seed) or REV / revision_name(model, ds, seed)
