# -*- coding: utf-8 -*-
"""Repair two figures for the Applied Sciences submission.

1. dataset_samples_montage.png ships with its title clipped at both canvas edges
   ("sentative Bin2Img Samples Across Three Pla"). The pixels are gone, so the
   title band is repainted and the text redrawn at a size that fits the canvas.
   The sample images themselves are untouched.

2. transfer_probe.png carries an em dash in its title. It is redrawn from
   experiments_local/results/transfer_probe.json using the same plotting code as
   experiments_local/transfer_probe.py, with the dash replaced. No model or
   dataset is needed, only the recorded results.

Usage:  python fix_paper_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "article" / "images"
RESULTS = ROOT / "experiments_local" / "results"

# ---------------------------------------------------------------- montage ----
TITLE = "Representative Bin2Img Samples Across Three Platforms"
BAND_BOTTOM = 78          # everything above this row is the clipped title
TITLE_COLOUR = (31, 59, 92)

src = IMAGES / "dataset_samples_montage.png"
im = Image.open(src).convert("RGB")
w, h = im.size

draw = ImageDraw.Draw(im)
draw.rectangle([0, 0, w, BAND_BOTTOM], fill="white")

font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
size = 34
while size > 12:
    font = ImageFont.truetype(str(font_path), size)
    box = font.getbbox(TITLE)
    if box[2] - box[0] <= w - 40:      # keep a 20px margin each side
        break
    size -= 1

box = font.getbbox(TITLE)
tw, th = box[2] - box[0], box[3] - box[1]
draw.text(((w - tw) // 2 - box[0], (BAND_BOTTOM - th) // 2 - box[1] + 4),
          TITLE, font=font, fill=TITLE_COLOUR)

out = IMAGES / "dataset_samples_montage_fixed.png"
im.save(out)
print(f"montage title redrawn at {size}px -> {out.name}")

# ---------------------------------------------------------- transfer probe ---
data = json.loads((RESULTS / "transfer_probe.json").read_text(encoding="utf-8"))
names = data["sources"]
f1 = np.array(data["macro_f1_matrix"], dtype=float)

fig, ax = plt.subplots(figsize=(5.2, 4.4))
img = ax.imshow(f1, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(names)))
ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Target dataset (linear probe + test)")
ax.set_ylabel("Source backbone (frozen)")
ax.set_title("Cross-platform transfer: probe Macro-F1", fontsize=10, fontweight="bold")
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{f1[i, j]:.3f}", ha="center", va="center", fontsize=8,
                color="white" if f1[i, j] < 0.6 else "black")
fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
out2 = IMAGES / "transfer_probe_fixed.png"
fig.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"transfer probe replotted from {RESULTS.name}/transfer_probe.json -> {out2.name}")
print("  matrix:", np.round(f1, 4).tolist())
