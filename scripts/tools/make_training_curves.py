# -*- coding: utf-8 -*-
"""Render the training-curve figure from the .dat tables shipped with the paper.

The LaTeX build draws this figure with pgfplots, but the docx pipeline has to
substitute a raster image. article/images/training_curves_acc_f1.png carries
Chinese labels, so the English manuscript needs its own copy; both are produced
here from the same data the pgfplots block reads, so the two builds cannot drift.

Usage:  python make_training_curves.py <data_dir> <out.png> [--lang zh|en]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

DATA_DIR = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LANG = "en"
if "--lang" in sys.argv:
    LANG = sys.argv[sys.argv.index("--lang") + 1]

# dataset -> (.dat stem, best epoch, ymin)  -- best epochs match the pgfplots block
PANELS = [
    ("CICMalDroid2020", "cicmaldroid2020", 123, 0.5),
    ("BODMAS", "bodmas", 160, 0.5),
    ("ELF2020", "elf2020", 113, 0.35),
]

TEXT = {
    "en": dict(sup="Training curves on the three datasets", acc="Accuracy",
               f1="Macro-F1", ep="Epoch", tr="Training set", va="Validation set"),
    "zh": dict(sup="三个数据集的训练过程曲线", acc="准确率",
               f1="Macro-F1", ep="Epoch", tr="训练集", va="验证集"),
}[LANG]

if LANG == "zh":
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False

TRAIN_C, VAL_C = "#4C78A8", "#E45756"


def read(stem):
    rows = (DATA_DIR / f"training_curves_{stem}.dat").read_text().strip().splitlines()
    head = rows[0].split()
    cols = {k: [] for k in head}
    for line in rows[1:]:
        for k, v in zip(head, line.split()):
            cols[k].append(float(v))
    return cols


fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4))

for col, (pretty, stem, best_ep, ymin) in enumerate(PANELS):
    d = read(stem)
    ep = d["epoch"]
    for row, (tr_key, va_key) in enumerate([("train_acc", "val_acc"),
                                            ("train_f1", "val_f1")]):
        ax = axes[row, col]
        ax.plot(ep, d[tr_key], color=TRAIN_C, ls="--", lw=1.4,
                label=TEXT["tr"] if (row == 0 and col == 0) else None)
        ax.plot(ep, d[va_key], color=VAL_C, lw=1.4,
                label=TEXT["va"] if (row == 0 and col == 0) else None)
        ax.axvline(best_ep, color="black", ls=":", lw=1.2)
        ax.plot([best_ep], [d[va_key][best_ep - 1]], "o", color="black", ms=5)
        ax.set_xlim(1, 200)
        ax.set_ylim(ymin, 1.0)
        ax.grid(True, color="0.88", lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if row == 0:
            ax.set_title(pretty, fontsize=12)
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(TEXT["ep"], fontsize=11)
        if col == 0:
            ax.set_ylabel(TEXT["acc"] if row == 0 else TEXT["f1"], fontsize=11)

fig.suptitle(TEXT["sup"], fontsize=15, y=0.99)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
           ncol=2, frameon=False, fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
print(f"wrote {OUT}")
