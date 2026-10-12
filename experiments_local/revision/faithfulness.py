"""Deletion and insertion curves for Grad-CAM and input-gradient saliency over the
full test sets (Petsiuk et al., 2018; reviewer R3-17).

Pixels are ranked by the explanation map (upsampled to 96 x 96). Deletion sets the
top-ranked pixels to zero bytes (all four channels, raw value 0) in 5% steps and
tracks the softmax probability of the originally predicted class; insertion starts
from an all-zero image and restores pixels in the same order. Lower deletion AUC
and higher insertion AUC mean a more faithful map. A random ranking is the control.

Grad-CAM target layer: HoloNet.context_refine (96 channels, 12 x 12), the last
convolutional feature map before the primary capsules.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments_local"))
from _common import safe_load, build_model_from_ckpt  # noqa: E402
from holonet.data import _scan_split, MalwareImageDataset, _build_transforms  # noqa: E402
from runs import run_dir  # noqa: E402

DEVICE = torch.device("cuda")
# the fused nn.TransformerEncoder inference path crashed (illegal memory access)
# while sharing the GPU with training jobs; the standard path gives the same outputs
torch.backends.mha.set_fastpath_enabled(False)
STEPS = 20
RUNS = {d: run_dir("full", d, 42) for d in ("BODMAS", "CICMalDroid2020", "ELF2020")}
ZERO = -1.0  # raw byte 0 after Normalize(mean=0.5, std=0.5)
# replacement for removed pixels: "zero" bytes, or a Gaussian-blurred copy of the image
# (RISE's choice), which is not a no-op on regions that already hold zero bytes
REPLACE = sys.argv[1] if len(sys.argv) > 1 else "zero"


def explanations(model, x):
    """Grad-CAM (context_refine) and saliency maps for the predicted class, batched."""
    store = {}
    h1 = model.context_refine.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    x = x.clone().requires_grad_(True)
    logits = model(x)["logits"]
    pred = logits.argmax(1)
    a = store["a"]; a.retain_grad()
    logits.gather(1, pred[:, None]).sum().backward()
    h1.remove()
    weights = a.grad.mean((2, 3), keepdim=True)
    cam = F.relu((weights * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    sal = x.grad.abs().amax(1)
    return pred.detach(), cam.detach(), sal.detach()


@torch.no_grad()
def curve(model, x, pred, order, mode):
    """Probability of the predicted class as pixels are deleted / inserted in `order`."""
    b, c, h, w = x.shape
    n = h * w
    base = torch.full_like(x, ZERO) if REPLACE == "zero" else gaussian_blur(x, kernel_size=11, sigma=5.0)
    probs = []
    for s in range(STEPS + 1):
        k = int(round(n * s / STEPS))
        mask = torch.zeros(b, n, device=x.device)
        if k:
            mask.scatter_(1, order[:, :k], 1.0)
        mask = mask.view(b, 1, h, w)
        if mode == "deletion":
            xs = x * (1 - mask) + base * mask
        else:
            xs = x * mask + base * (1 - mask)
        p = torch.softmax(model(xs)["logits"].float(), 1).gather(1, pred[:, None])[:, 0]
        probs.append(p)
    return torch.stack(probs, 1)  # b x (STEPS+1)


def auc(c: torch.Tensor) -> torch.Tensor:
    return torch.trapezoid(c, dx=1.0 / STEPS, dim=1)


def main() -> None:
    res = {}
    for ds, run in RUNS.items():
        model, _ = build_model_from_ckpt(safe_load(run / "best.pt", DEVICE), DEVICE)
        model.eval()
        _, samples = _scan_split(ROOT / "RGBA" / ds / "test")
        loader = torch.utils.data.DataLoader(MalwareImageDataset(samples, _build_transforms(96, train=False)),
                                             batch_size=128, shuffle=False, num_workers=0)
        g = torch.Generator(device="cpu").manual_seed(0)
        curves = {f"{m}_{mode}": [] for m in ("gradcam", "saliency", "random") for mode in ("deletion", "insertion")}
        zero_share = {m: [] for m in ("gradcam", "saliency", "random")}
        for x, _ in loader:
            x = x.to(DEVICE)
            pred, cam, sal = explanations(model, x)
            b = x.shape[0]
            orders = {
                "gradcam": cam.flatten(1).argsort(1, descending=True),
                "saliency": sal.flatten(1).argsort(1, descending=True),
                "random": torch.rand(b, 96 * 96, generator=g).argsort(1).to(DEVICE),
            }
            is_zero = (x <= ZERO + 1e-6).all(1).flatten(1).float()  # all four bytes are 0
            top = int(0.1 * 96 * 96)
            for m, order in orders.items():
                zero_share[m].append(is_zero.gather(1, order[:, :top]).mean(1).cpu())
            for m, order in orders.items():
                for mode in ("deletion", "insertion"):
                    curves[f"{m}_{mode}"].append(curve(model, x, pred, order, mode).cpu())
        res[ds] = {}
        for key, parts in curves.items():
            c = torch.cat(parts)
            res[ds][key] = {"auc_mean": round(float(auc(c).mean()), 4), "auc_std": round(float(auc(c).std()), 4),
                            "mean_curve": [round(float(v), 4) for v in c.mean(0)]}
        res[ds]["zero_byte_share_top10pct"] = {m: round(float(torch.cat(v).mean()), 4) for m, v in zero_share.items()}
        r = res[ds]
        print(f"{ds:16s} zero-byte share in top-10% pixels: {r['zero_byte_share_top10pct']}")
        print(f"{ds:16s} n={len(samples)}  deletion AUC (lower=better): " +
              "  ".join(f"{m} {r[m + '_deletion']['auc_mean']:.3f}" for m in ("gradcam", "saliency", "random")) +
              "  | insertion AUC (higher=better): " +
              "  ".join(f"{m} {r[m + '_insertion']['auc_mean']:.3f}" for m in ("gradcam", "saliency", "random")))
    res["_protocol"] = {"steps": STEPS, "replacement": REPLACE,
                        "gradcam_layer": "context_refine (96 x 12 x 12)", "checkpoint": "seed 42 best-val"}
    (HERE / "results" / f"faithfulness_{REPLACE}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
