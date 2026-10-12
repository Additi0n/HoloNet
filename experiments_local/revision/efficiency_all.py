"""Parameters, FLOPs and latency for HoloNet, its routing variants and the image
baselines (R1-1, R2-7, R3-18).

Parameters count only weights that receive a gradient (an ablated block that is
built but never called is excluded). FLOPs come from torch.utils.flop_counter for
one 4 x 96 x 96 sample (one multiply-add = 2 FLOPs; matmul/conv only, counted
outside the fused attention fast path). Latency (--latency) needs an
otherwise idle GPU/CPU: median of 200 timed forward passes after 30 warm-up passes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from holonet import HoloNet  # noqa: E402
from holonet.baselines import BaselineClassifier  # noqa: E402

K = 12  # BODMAS head; other heads differ by a few hundred weights
MODELS = {
    "HoloNet (Transformer routing)": lambda: HoloNet(K),
    "HoloNet, dynamic routing (3 it.)": lambda: HoloNet(K, router="dynamic"),
    "HoloNet, pooling router": lambda: HoloNet(K, router="pooling"),
    "HoloNet, no capsules (token head)": lambda: HoloNet(K, router="token"),
    "HoloNet, no Conv-KAN": lambda: HoloNet(K, use_ckan=False),
    "Res2Net-50": lambda: BaselineClassifier("res2net50", K),
    "ResNeXt-50": lambda: BaselineClassifier("resnext50", K),
    "ResNeXt-50 + CBAM (ResNeXt+)": lambda: BaselineClassifier("resnext_cbam", K),
    "ConvNeXt-T": lambda: BaselineClassifier("convnext_tiny", K),
    "ViT-S/8": lambda: BaselineClassifier("vit_s8", K),
}


def active_params(model: torch.nn.Module) -> int:
    model.train()
    model.zero_grad(set_to_none=True)
    model(torch.randn(2, 4, 96, 96))["logits"].sum().backward()
    n = sum(p.numel() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return n


def flops(model: torch.nn.Module) -> int:
    # train mode: in eval mode nn.TransformerEncoder/MultiheadAttention take a fused
    # fast path that the FLOP counter does not see, which undercounts attention
    model.train()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(torch.randn(1, 4, 96, 96))
    return fc.get_total_flops()


@torch.no_grad()
def latency_ms(model: torch.nn.Module, device: str, batch: int) -> float:
    model = model.to(device).eval()
    x = torch.randn(batch, 4, 96, 96, device=device)
    times = []
    for i in range(230):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= 30:
            times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()
    out_path = HERE / "results" / "efficiency_all.json"
    res = json.loads(out_path.read_text()) if out_path.is_file() else {}
    torch.backends.cudnn.benchmark = True
    for name, make in MODELS.items():
        torch.manual_seed(0)
        model = make()
        r = res.setdefault(name, {})
        r["params_M"] = round(active_params(model) / 1e6, 3)
        r["gflops"] = round(flops(model) / 1e9, 3)
        if args.latency:
            torch.set_num_threads(8)
            r["cpu_ms_bs1"] = round(latency_ms(model, "cpu", 1), 2)
            r["gpu_ms_bs1"] = round(latency_ms(model, "cuda", 1), 2)
            r["gpu_ms_bs32"] = round(latency_ms(model, "cuda", 32), 2)
            model.cpu()
        print(f"{name:36s} " + "  ".join(f"{k}={v}" for k, v in r.items()))
    out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
