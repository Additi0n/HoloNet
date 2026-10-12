"""Efficiency benchmark for HoloNet — fills the missing efficiency experiment.

Reports, per dataset checkpoint: parameter count, FLOPs (1 sample), and measured
latency / throughput on both CPU and (if available) GPU at batch 1 and batch 32.

    python experiments_local/bench_efficiency.py
"""
from __future__ import annotations

import json
import time

import torch

from _common import RUNS, DATASETS, OUT, safe_load, build_model_from_ckpt


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, device):
    """FLOPs for a single 4x96x96 forward, via torch's built-in FLOP counter."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    x = torch.randn(1, 4, 96, 96, device=device)
    fc = FlopCounterMode(display=False)
    try:
        with fc:
            model(x)
        return fc.get_total_flops()
    except Exception:
        return None


@torch.no_grad()
def latency(model, device, batch, iters=50, warmup=10):
    x = torch.randn(batch, 4, 96, 96, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt * 1000.0, batch / dt  # ms/batch, samples/s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.insert(0, torch.device("cuda"))

    results = {}
    # parameters/FLOPs are config-dependent; report per dataset (num_classes differs)
    for run_name, ds in DATASETS.items():
        ckpt = safe_load(RUNS / run_name / "best.pt", torch.device("cpu"))
        rec = {}
        for device in devices:
            model, di = build_model_from_ckpt(ckpt, device)
            if device.type == "cpu":
                rec["params"] = count_params(model)
                rec["params_M"] = round(count_params(model) / 1e6, 3)
                flops = count_flops(model, device)
                rec["flops_1sample"] = flops
                rec["gflops_1sample"] = round(flops / 1e9, 3) if flops else None
            ms1, tp1 = latency(model, device, 1)
            ms32, tp32 = latency(model, device, 32)
            rec[device.type] = {
                "latency_bs1_ms": round(ms1, 3),
                "throughput_bs1_sps": round(tp1, 1),
                "latency_bs32_ms": round(ms32, 3),
                "throughput_bs32_sps": round(tp32, 1),
            }
        results[ds] = rec
        p = rec["params_M"]
        g = rec["gflops_1sample"]
        line = f"[{ds}] params={p}M  FLOPs={g}G/sample"
        for d in ("cuda", "cpu"):
            if d in rec:
                line += f"  | {d}: bs1={rec[d]['latency_bs1_ms']}ms ({rec[d]['throughput_bs1_sps']}/s), " \
                         f"bs32={rec[d]['throughput_bs32_sps']}/s"
        print(line)

    (OUT / "efficiency.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'efficiency.json'}")


if __name__ == "__main__":
    main()
