"""Shared helpers for the local supplementary experiments."""
from __future__ import annotations

import pathlib
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNS = PROJECT_ROOT / "runs" / "launcher"
DATA_ROOT = PROJECT_ROOT / "RGBA"
OUT = PROJECT_ROOT / "experiments_local" / "results"

# run dir -> dataset name (as used by build_dataloaders / RGBA/<name>)
DATASETS = {
    "holonet_bodmas": "BODMAS",
    "holonet_cicmaldroid2020": "CICMalDroid2020",
    "holonet_elf2020": "ELF2020",
}


def safe_load(path, device):
    """torch.load tolerant of Linux PosixPath objects pickled into the ckpt."""
    saved = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        return torch.load(path, map_location=device, weights_only=False)
    finally:
        pathlib.PosixPath = saved


def build_model_from_ckpt(ckpt, device):
    from holonet.model import HoloNet
    a = dict(ckpt.get("args", {}))
    di = dict(ckpt.get("dataset_info", {}))
    model = HoloNet(
        num_classes=int(di["num_classes"]),
        image_size=int(a.get("image_size", 96)),
        stem_channels=tuple(a.get("stem_channels", [32, 64, 96])),
        primary_capsule_dim=int(a.get("primary_capsule_dim", 16)),
        primary_capsule_types=int(a.get("primary_capsule_types", 12)),
        transformer_dim=int(a.get("transformer_dim", 128)),
        transformer_heads=int(a.get("transformer_heads", 4)),
        transformer_layers=int(a.get("transformer_layers", 2)),
        ckan_grid_size=int(a.get("ckan_grid_size", 8)),
        dropout=float(a.get("dropout", 0.1)),
        use_ckan=bool(a.get("use_ckan", True)),
        use_transformer_router=bool(a.get("use_transformer_router", True)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, di
