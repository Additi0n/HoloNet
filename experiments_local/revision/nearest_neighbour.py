"""Nearest training image for every test image in raw pixel space (reviewer R3-6).

RMSE over the 96 x 96 x 4 RGBA array scaled to [0, 1]. RMSE 0 is an exact copy;
small values are near copies (a few bytes changed). Adds the per-file distances and
the near-duplicate sets to results/dedup.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "RGBA"
RES = Path(__file__).resolve().parent / "results"
THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.05)
NEAR = 0.01  # threshold used for the de-duplicated test set


def load(split_dir: Path):
    names, labels, arrs = [], [], []
    for cls in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for f in sorted(cls.iterdir()):
            names.append(f"{cls.name}/{f.name}")
            labels.append(cls.name)
            arrs.append(np.asarray(Image.open(f).convert("RGBA"), dtype=np.uint8).reshape(-1))
    return names, np.array(labels), torch.from_numpy(np.stack(arrs))


def main() -> None:
    report = json.loads((RES / "dedup.json").read_text(encoding="utf-8"))
    dev = "cuda"
    for ds in ("BODMAS", "CICMalDroid2020", "ELF2020"):
        _, tr_lab, tr = load(DATA / ds / "train")
        te_names, te_lab, te = load(DATA / ds / "test")
        trf = tr.to(dev).half() / 255
        tr_sq = (trf.float() ** 2).sum(1)
        trg = tr.to(dev)
        best_d, best_i = [], []
        for chunk in te.split(256):
            q = chunk.to(dev)
            qf = q.half() / 255
            d2 = (qf.float() ** 2).sum(1, keepdim=True) + tr_sq[None] - 2 * (qf @ trf.T).float()
            cand = d2.topk(64, dim=1, largest=False).indices  # fp16 shortlist
            # exact fp32 RMSE against the shortlisted training images
            diff = (q[:, None].float() - trg[cand].float()) / 255
            exact_rmse = diff.pow(2).mean(-1).sqrt()
            v, j = exact_rmse.min(1)
            best_d.append(v); best_i.append(cand.gather(1, j[:, None])[:, 0])
        rmse = torch.cat(best_d).cpu().numpy()
        idx = torch.cat(best_i).cpu().numpy()
        # the pixel hash is authoritative for exact copies the shortlist may miss
        exact = set(report[ds]["leaked_test_files_exact"])
        rmse[[n in exact for n in te_names]] = 0.0
        same = tr_lab[idx] == te_lab
        stats = {}
        for t in THRESHOLDS:
            m = rmse <= t
            stats[str(t)] = {"count": int(m.sum()), "fraction": round(float(m.mean()), 4),
                             "same_label_fraction": round(float(same[m].mean()), 4) if m.any() else None}
        near = [n for n, r in zip(te_names, rmse) if r <= NEAR]
        report[ds]["nn_rmse_thresholds"] = stats
        report[ds]["nn_rmse_quantiles"] = {q: round(float(np.quantile(rmse, q)), 4) for q in (0.05, 0.25, 0.5)}
        report[ds]["near_rmse_threshold"] = NEAR
        report[ds]["leaked_test_files_near"] = near
        report[ds]["test_near_duplicates_of_train"] = len(near)
        report[ds].pop("near_threshold_bits", None)
        line = "  ".join(f"<= {t}: {s['count']} ({s['fraction']:.1%}, same label {s['same_label_fraction']})" for t, s in stats.items())
        print(f"{ds:16s} {line}\n{'':16s} median NN RMSE {np.median(rmse):.3f}")
        del trf, tr_sq, trg
        torch.cuda.empty_cache()
    (RES / "dedup.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
