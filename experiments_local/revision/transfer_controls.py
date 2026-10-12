"""Controls for the cross-platform linear probe (reviewer R3-8, R3 minor-10).

Same probe protocol as experiments_local/transfer_probe.py (frozen features,
standardised, one linear layer, 300 full-batch Adam steps), applied to:

  trained      HoloNet backbones trained on each source platform, seeds 42/1/2
  random_init  untrained HoloNet backbones, 5 random initialisations
  pixels       raw image average-pooled to 5 x 5 x 4 (100-D), no network at all
  histogram    24-bin intensity histogram per channel (96-D), no spatial layout

plus the expected Macro-F1 of three guessing rules (prior sampling, uniform,
majority class). If random_init or the model-free features score close to the
trained backbones, the probe measures generic image statistics, not learned
malware fingerprints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_local"))

from _common import safe_load, build_model_from_ckpt  # noqa: E402
from transfer_probe import extract_features, linear_probe  # noqa: E402
from holonet.data import _scan_split, MalwareImageDataset, _build_transforms  # noqa: E402
from holonet.model import HoloNet  # noqa: E402

DATA = ROOT / "RGBA"
DATASETS = ("BODMAS", "CICMalDroid2020", "ELF2020")
TRAINED = {
    42: {d: ROOT / "all_results" / "runs" / f"holonet_{d.lower()}" for d in DATASETS},
    1: {d: ROOT / "runs" / "variance" / f"var_{d.lower()}_s1" for d in DATASETS},
    2: {d: ROOT / "runs" / "variance" / f"var_{d.lower()}_s2" for d in DATASETS},
}
RANDOM_INITS = (0, 1, 2, 3, 4)
DEVICE = torch.device("cuda")
# the fused nn.TransformerEncoder inference path crashed (illegal memory access)
# while sharing the GPU with training jobs; the standard path gives the same outputs
torch.backends.mha.set_fastpath_enabled(False)


def hooked(model):
    hook = {"feat": None}
    model.context_refine.register_forward_hook(lambda m, i, o: hook.__setitem__("feat", o.detach()))
    return hook


_cache: dict = {}


def images(ds: str, split: str):
    """Normalised test-time tensors of a split (cached)."""
    key = (ds, split)
    if key not in _cache:
        _, samples = _scan_split(DATA / ds / split)
        d = MalwareImageDataset(samples, _build_transforms(96, train=False))
        loader = torch.utils.data.DataLoader(d, batch_size=512, shuffle=False, num_workers=0)
        xs, ys = zip(*loader)
        _cache[key] = (torch.cat(xs), torch.cat(ys).numpy())
    return _cache[key]


def model_free(ds: str, split: str, kind: str):
    x, y = images(ds, split)
    if kind == "pixels":
        f = F.adaptive_avg_pool2d(x, 5).flatten(1)
    else:
        v = ((x * 0.5 + 0.5).clamp(0, 1) * 23.999).long()  # back to [0, 1], 24 bins
        f = torch.stack([F.one_hot(v[:, c].flatten(1), 24).float().mean(1) for c in range(4)], 1).flatten(1)
    return f.numpy(), y


def probe_backbone(model, tgt: str) -> float:
    hook = hooked(model)
    xtr, ytr, cls = extract_features(model, hook, DATA / tgt / "train", DEVICE)
    xte, yte, _ = extract_features(model, hook, DATA / tgt / "test", DEVICE)
    return linear_probe(xtr, ytr, xte, yte, len(cls), DEVICE)["macro_f1"]


def guessing(ytr: np.ndarray, yte: np.ndarray, k: int) -> dict:
    p = np.bincount(ytr, minlength=k) / len(ytr)   # training prior
    q = np.bincount(yte, minlength=k) / len(yte)   # test class frequencies
    # expected per-class F1 of independent guessing with distribution g: 2 q g / (q + g)
    exp_f1 = lambda g: float(np.mean(2 * q * g / (q + g)))
    maj = np.zeros(k); maj[p.argmax()] = 1
    return {"prior_sampling": round(exp_f1(p), 4), "uniform": round(exp_f1(np.full(k, 1 / k)), 4),
            "majority_class": round(exp_f1(maj), 4)}


def main() -> None:
    res = {"trained": {}, "random_init": {}, "pixels": {}, "histogram": {}, "guessing": {}}
    for tgt in DATASETS:
        _, ytr = images(tgt, "train"); _, yte = images(tgt, "test")
        k = int(ytr.max()) + 1
        res["guessing"][tgt] = guessing(ytr, yte, k)
        for kind in ("pixels", "histogram"):
            xtr, ytr_ = model_free(tgt, "train", kind)
            xte, yte_ = model_free(tgt, "test", kind)
            res[kind][tgt] = round(linear_probe(xtr, ytr_, xte, yte_, k, DEVICE)["macro_f1"], 4)
        print(f"{tgt}: pixels {res['pixels'][tgt]}  histogram {res['histogram'][tgt]}  guessing {res['guessing'][tgt]}")

    for src in DATASETS:
        for tgt in DATASETS:
            key = f"{src}->{tgt}"
            vals = []
            for seed, runs in TRAINED.items():
                ckpt = safe_load(runs[src] / "best.pt", DEVICE)
                model, _ = build_model_from_ckpt(ckpt, DEVICE)
                vals.append(probe_backbone(model, tgt))
            res["trained"][key] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                                   "per_seed": [round(v, 4) for v in vals]}
            print(f"  trained     {key:34s} {np.mean(vals):.4f} +- {np.std(vals):.4f}")

    # an untrained backbone has no source; the class count of its head is irrelevant to the features
    for tgt in DATASETS:
        vals = []
        for s in RANDOM_INITS:
            torch.manual_seed(s)
            model = HoloNet(num_classes=12).to(DEVICE).eval()
            vals.append(probe_backbone(model, tgt))
        res["random_init"][tgt] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                                   "per_seed": [round(v, 4) for v in vals]}
        print(f"  random_init -> {tgt:16s} {np.mean(vals):.4f} +- {np.std(vals):.4f}")

    out = HERE / "results" / "transfer_controls.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
