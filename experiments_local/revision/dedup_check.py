"""Duplicate and train/test leakage audit of the RGBA images (reviewer R3-6).

Exact duplicates: identical decoded RGBA pixels. Near duplicates: 64-bit
difference hash (dHash) of the luminance, Hamming distance <= NEAR_BITS. Results
go to results/dedup.json together with the file lists, so the test set can be
re-scored with the leaked samples removed.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "RGBA"
OUT = Path(__file__).resolve().parent / "results"
NEAR_BITS = 4  # of 64


def dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def scan(split_dir: Path):
    rows = []
    for cls in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for f in sorted(cls.iterdir()):
            img = Image.open(f).convert("RGBA")
            rows.append((f"{cls.name}/{f.name}", cls.name, hashlib.md5(img.tobytes()).hexdigest(), dhash(img)))
    return rows


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.uint8)
    for shift in range(0, 64, 8):
        c += np.unpackbits(((x >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint8)[..., None], axis=-1).sum(-1).astype(np.uint8)
    return c


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for ds in ("BODMAS", "CICMalDroid2020", "ELF2020"):
        train, test = scan(DATA / ds / "train"), scan(DATA / ds / "test")
        # exact duplicates
        by_md5 = defaultdict(list)
        for split, rows in (("train", train), ("test", test)):
            for name, cls, md5, _ in rows:
                by_md5[md5].append((split, name, cls))
        groups = [g for g in by_md5.values() if len(g) > 1]
        within_train = sum(len([x for x in g if x[0] == "train"]) - 1 for g in groups if sum(x[0] == "train" for x in g) > 1)
        within_test = sum(len([x for x in g if x[0] == "test"]) - 1 for g in groups if sum(x[0] == "test" for x in g) > 1)
        train_md5 = {r[2] for r in train}
        exact_leak = [r[0] for r in test if r[2] in train_md5]
        label_conflicts = sum(1 for g in groups if len({x[2] for x in g}) > 1)

        # near duplicates of test images in train (dHash)
        th = np.array([r[3] for r in train], dtype=np.uint64)
        near_leak = []
        for name, cls, md5, h in test:
            if md5 in train_md5:
                continue
            d = popcount64(th ^ np.uint64(h))
            if d.min() <= NEAR_BITS:
                near_leak.append(name)

        report[ds] = {
            "train": len(train),
            "test": len(test),
            "duplicate_groups": len(groups),
            "redundant_copies_within_train": within_train,
            "redundant_copies_within_test": within_test,
            "duplicate_groups_with_conflicting_labels": label_conflicts,
            "test_exact_duplicates_of_train": len(exact_leak),
            "test_near_duplicates_of_train": len(near_leak),
            "near_threshold_bits": NEAR_BITS,
            "leaked_test_files_exact": exact_leak,
            "leaked_test_files_near": near_leak,
        }
        r = report[ds]
        print(f"{ds:16s} train {r['train']:6d} test {r['test']:5d} | dup groups {r['duplicate_groups']:5d} "
              f"(label conflicts {label_conflicts}) | test exact-in-train {len(exact_leak):5d} "
              f"({len(exact_leak) / len(test):.1%}) near {len(near_leak):5d} ({len(near_leak) / len(test):.1%})")
    (OUT / "dedup.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
