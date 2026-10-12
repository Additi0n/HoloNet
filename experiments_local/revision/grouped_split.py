"""Leakage-free re-split: near-duplicate clusters are kept on one side (R3-6).

Pools train + test, links every pair of images with pixel RMSE <= NEAR (exact
fp32 check on an fp16 shortlist), takes connected components as groups, then
assigns whole groups to test, class by class (group label = majority label), until
each class holds ~30% of its samples in test, the original ratio. Writes
RGBA_grouped/<dataset>/{train,test}/<class>/ as hard links and
article/revision/release/<dataset>_grouped_split.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC, DST = ROOT / "RGBA", ROOT / "RGBA_grouped"
REL = ROOT / "article" / "revision" / "release"
NEAR, TEST_FRAC, SEED, SHORTLIST = 0.01, 0.30, 0, 128


def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i


FLAT_STD = 0.02  # pixel std below which an image is a near-uniform grey (see the flat-image audit)


def is_flat(path: Path) -> bool:
    return float(np.asarray(Image.open(path).convert("RGBA"), np.float32).std() / 255) < FLAT_STD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["BODMAS", "CICMalDroid2020", "ELF2020"])
    ap.add_argument("--exclude-flat", action="store_true",
                    help="drop near-uniform images from both sides before grouping")
    ap.add_argument("--out", default="RGBA_grouped")
    args = ap.parse_args()
    dst = ROOT / args.out
    tag = "_noflat" if args.exclude_flat else ""
    stats = {}
    for ds in args.datasets:
        files = []
        for split in ("train", "test"):
            for cls in sorted(p for p in (SRC / ds / split).iterdir() if p.is_dir()):
                files += [(f, cls.name, split) for f in sorted(cls.iterdir())]
        if args.exclude_flat:
            before = len(files)
            files = [x for x in files if not is_flat(x[0])]
            print(f"{ds}: excluded {before - len(files)} near-uniform images")
        X = torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGBA"), np.uint8).reshape(-1) for f, _, _ in files])).cuda()
        Xh = X.half() / 255
        sq = (Xh.float() ** 2).sum(1)
        n = len(files)
        parent = list(range(n))
        for s in range(0, n, 256):
            q = Xh[s:s + 256]
            d2 = (q.float() ** 2).sum(1, keepdim=True) + sq[None] - 2 * (q @ Xh.T).float()
            cand = d2.topk(SHORTLIST, 1, largest=False).indices
            rmse = ((X[s:s + 256, None].float() - X[cand].float()) / 255).pow(2).mean(-1).sqrt()
            for a, b in (rmse <= NEAR).nonzero().tolist():
                i, j = s + a, int(cand[a, b])
                if i != j:
                    ri, rj = find(parent, i), find(parent, j)
                    if ri != rj:
                        parent[ri] = rj
        del X, Xh; torch.cuda.empty_cache()

        groups = defaultdict(list)
        for i in range(n):
            groups[find(parent, i)].append(i)
        by_class = defaultdict(list)
        mixed = 0
        for g in groups.values():
            labels = Counter(files[i][1] for i in g)
            mixed += len(labels) > 1
            by_class[labels.most_common(1)[0][0]].append(g)
        rng = random.Random(SEED)
        new_split = {}
        for cls, gs in by_class.items():
            rng.shuffle(gs)
            total = sum(len(g) for g in gs)
            in_test = 0
            for g in gs:
                side = "test" if in_test < TEST_FRAC * total and in_test + len(g) <= TEST_FRAC * total * 1.5 else "train"
                in_test += len(g) if side == "test" else 0
                for i in g:
                    new_split[i] = side

        out_rows = []
        for i, (f, cls, old) in enumerate(files):
            target = dst / ds / new_split[i] / cls / f.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(f, target)
            out_rows.append((f.stem.replace("VirusShare_", ""), cls, old, new_split[i]))
        with open(REL / f"{ds}_grouped{tag}_split.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["hash", "label", "original_split", "grouped_split"]); w.writerows(out_rows)

        sizes = sorted((len(g) for g in groups.values()), reverse=True)
        per_class = {}
        for cls in sorted({c for _, c, _ in files}):
            idx = [i for i, (_, c, _) in enumerate(files) if c == cls]
            per_class[cls] = {"train": sum(new_split[i] == "train" for i in idx), "test": sum(new_split[i] == "test" for i in idx)}
        stats[ds] = {"samples": n, "groups": len(groups), "singletons": sum(s == 1 for s in sizes),
                     "largest_groups": sizes[:5], "mixed_label_groups": mixed, "per_class": per_class}
        print(ds, json.dumps({k: v for k, v in stats[ds].items() if k != "per_class"}))
        for cls, c in per_class.items():
            print(f"   {cls:10s} train {c['train']:5d} test {c['test']:5d}  ({c['test'] / (c['train'] + c['test']):.0%})")
    (ROOT / "experiments_local" / "revision" / "results" / f"grouped_split{tag}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
