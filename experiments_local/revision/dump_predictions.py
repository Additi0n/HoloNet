"""Per-sample test predictions for trained runs (feeds de-duplicated scoring,
McNemar tests and confidence statistics).

Writes <run_dir>/test_predictions.npz with file names (class/file, the same keys
as results/dedup.json), labels, predicted labels, softmax probabilities and, for
capsule models, class-capsule lengths. Runs that already have the file are skipped.

    python experiments_local/revision/dump_predictions.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import train as train_module  # noqa: E402
from holonet.data import _post_transform, _pre_transform, _scan_split  # noqa: E402
from torchvision import transforms  # noqa: E402
from PIL import Image  # noqa: E402


def _parser_defaults() -> dict:
    argv, sys.argv = sys.argv, ["train.py"]
    try:
        return vars(train_module.parse_args())
    finally:
        sys.argv = argv


def load_state(path: Path) -> dict:
    # checkpoints trained on Linux pickle PosixPath objects
    import pathlib
    posix = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = posix


def model_from_checkpoint(state: dict, num_classes: int) -> torch.nn.Module:
    defaults = _parser_defaults()
    saved = dict(state["args"])
    saved.setdefault("arch", "holonet")
    saved.setdefault("router", None)
    saved.setdefault("routing_iterations", 3)
    args = argparse.Namespace(**{**defaults, **saved})
    model = train_module.build_model(num_classes, args)
    model.load_state_dict(state["model_state"])
    return model


@torch.no_grad()
def dump(run_dir: Path, device: str = "cuda") -> None:
    out = run_dir / "test_predictions.npz"
    if out.is_file():
        return
    state = load_state(run_dir / "best.pt")
    info = state["dataset_info"]
    model = model_from_checkpoint(state, info["num_classes"]).to(device).eval()

    # grouped-split runs train on RGBA_grouped; older Linux checkpoints store absolute paths
    data_root = ROOT / Path(str(state["args"].get("data_root", "RGBA")).replace("\\", "/")).name
    test_dir = data_root / info["dataset_name"] / "test"
    _, samples = _scan_split(test_dir, class_names=info["class_names"])
    channels = state["args"].get("channels", "rgba")
    tf = transforms.Compose([*_pre_transform(info["image_size"]).transforms,
                             *_post_transform(info["image_size"], train=False, channels=channels).transforms])

    names, labels, probs, lengths = [], [], [], []
    for start in range(0, len(samples), 256):
        batch = samples[start:start + 256]
        x = torch.stack([tf(Image.open(p).convert("RGBA")) for p, _ in batch]).to(device)
        with torch.autocast("cuda", enabled=device == "cuda"):
            o = model(x)
        probs.append(torch.softmax(o["logits"].float(), 1).cpu())
        lengths.append(o["capsule_lengths"].float().cpu())
        names += [f"{p.parent.name}/{p.name}" for p, _ in batch]
        labels += [y for _, y in batch]
    probs = torch.cat(probs).numpy()
    np.savez_compressed(
        out,
        names=np.array(names),
        y=np.array(labels),
        pred=probs.argmax(1),
        prob=probs.astype(np.float32),
        capsule_lengths=torch.cat(lengths).numpy().astype(np.float32),
        is_capsule=np.array(state["args"].get("arch", "holonet") == "holonet"
                            and state["args"].get("router") != "token"),
    )
    acc = (probs.argmax(1) == np.array(labels)).mean()
    print(f"{run_dir.name:48s} n={len(labels):5d} acc={acc:.4f}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        dump(Path(d))
