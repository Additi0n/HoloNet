# -*- coding: utf-8 -*-
"""Produce an English-labelled copy of the Grad-CAM figure.

make_paper_heatmaps.py renders the row labels and the colorbar caption in
Chinese. Re-running it needs torch plus the checkpoints and the RGBA dataset,
so for the English manuscript we relabel the rendered PNG in place: the text
sits in two margins that contain nothing else (the left strip and the band
below the colorbar tick labels), so painting over them loses no plot content.

Usage:  python localise_gradcam_en.py <in.png> <out.png>
"""
import sys
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])

LEFT_MARGIN = 88          # x < this holds only the rotated row labels
LEFT_STOP = 300           # ...but stop this far above the bottom, so the
                          # colorbar's leftmost tick label is not clipped
BOTTOM_MARGIN = 88        # y > h - this holds only the colorbar caption
ROW_LABELS = ["Best model", "Last-epoch model"]
ROW_CENTRES = (0.25, 0.64)  # fraction of image height
CBAR_LABEL = "Grad-CAM normalised response (low → high)"
FONT_SIZE = 34

font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
font = ImageFont.truetype(str(font_path), FONT_SIZE)

im = Image.open(SRC).convert("RGB")
w, h = im.size
draw = ImageDraw.Draw(im)

# clear the two text margins
draw.rectangle([0, 0, LEFT_MARGIN, h - LEFT_STOP], fill="white")
draw.rectangle([0, h - BOTTOM_MARGIN, w, h], fill="white")

# rotated row labels
for label, frac in zip(ROW_LABELS, ROW_CENTRES):
    box = font.getbbox(label)
    tw, th = box[2] - box[0], box[3] - box[1]
    tile = Image.new("RGB", (tw + 8, th + 12), "white")
    ImageDraw.Draw(tile).text((4 - box[0], 6 - box[1]), label, font=font, fill="black")
    tile = tile.rotate(90, expand=True)
    im.paste(tile, ((LEFT_MARGIN - tile.width) // 2,
                    int(h * frac) - tile.height // 2))

# colorbar caption
box = font.getbbox(CBAR_LABEL)
draw.text(((w - (box[2] - box[0])) // 2, h - BOTTOM_MARGIN + 18),
          CBAR_LABEL, font=font, fill="black")

im.save(DST)
print(f"wrote {DST}  ({im.size[0]}x{im.size[1]})")
