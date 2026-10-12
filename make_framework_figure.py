"""HoloNet framework figure (Figure 1) -- icon/glyph driven, minimal text.

Each operation is depicted by a small pictogram rather than a text box:
  Bin2Img        -> byte strip morphing into an RGBA pixel grid
  Conv (stride 2)-> kernel-grid icon + shrinking 3D tensor cuboids
  Conv-KAN       -> learnable spline/RBF activation curve
  Primary caps   -> a field of oriented capsule (pose) vectors
  Self-attention -> attention-matrix heatmap with Q/K/V bars (x2 stacked)
  Squash         -> long-vector -> short-vector compression glyph
  Prediction     -> class-probability bar chart with the winner highlighted

Faithful to holonet/model.py. Outputs vector PDF (+ PNG preview).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon

OUT_DIR = Path(__file__).resolve().parent / "article" / "images"

BAND = {
    "A": dict(fill="#EEF2F7", edge="#D5DEEA"),
    "B": dict(fill="#FBF4E6", edge="#ECDBB6"),
    "C": dict(fill="#F1ECFA", edge="#D9CFEE"),
}
C_KAN = dict(fc="#FBE3CB", ec="#E08A3C", tc="#7A4410")
C_OP = dict(fc="#DCEFD9", ec="#5BA85A", tc="#234e22")
C_OUT = dict(fc="#E7DCF6", ec="#8A5CC4", tc="#432a6b")
INK = "#33414F"


def _setup_font():
    for name in ("Segoe UI", "Arial", "DejaVu Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
        except Exception:
            continue
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.unicode_minus"] = False


_setup_font()


def _shade(hex_color, f):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, min(255, int(c * f))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# ----------------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------------
def band(ax, x, y, w, h, key, title):
    s = BAND[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
                 linewidth=1.3, facecolor=s["fill"], edgecolor=s["edge"], zorder=0.5))
    ax.text(x + w / 2, y - 20, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1d2730", zorder=4)


def arrow(ax, p1, p2, color=INK, lw=1.7, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                 lw=lw, color=color, zorder=2.6, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def tag(ax, cx, cy, text, fs=8, color=INK, weight="normal", style="normal"):
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight=weight, style=style, zorder=5)


def cuboid(ax, cx, cy, w, h, d=10, base="#9aa6b2", label=None, label_pos="below"):
    x, y = cx - w / 2, cy - h / 2
    front, top, side = base, _shade(base, 1.12), _shade(base, 0.82)
    ec = _shade(base, 0.6)
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)],
                 closed=True, facecolor=side, edgecolor=ec, lw=1.0, zorder=3))
    ax.add_patch(Polygon([(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d), (x + w, y + h)],
                 closed=True, facecolor=top, edgecolor=ec, lw=1.0, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                 linewidth=1.1, facecolor=front, edgecolor=ec, zorder=3.1))
    if label:
        if label_pos == "below":
            ax.text(cx + d / 2, y - 7, label, ha="center", va="top", fontsize=7.6,
                    color=INK, fontweight="bold")
        elif label_pos == "above":
            ax.text(cx + d / 2, y + h + d + 7, label, ha="center", va="bottom",
                    fontsize=7.6, color=INK, fontweight="bold")
        elif label_pos == "left":
            ax.text(x - 9, cy, label, ha="right", va="center", fontsize=7.6,
                    color=INK, fontweight="bold")
        elif label_pos == "right":
            ax.text(x + w + d + 9, cy, label, ha="left", va="center", fontsize=7.6,
                    color=INK, fontweight="bold")
    return (cx + d / 2, cy + d / 2, w, h)


def edge(c, s):
    cx, cy, w, h = c
    return {"t": (cx, cy + h / 2), "b": (cx, cy - h / 2),
            "l": (cx - w / 2, cy), "r": (cx + w / 2, cy)}[s]


def oplus(ax, cx, cy, r=8.5, sym="+"):
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=INK, lw=1.4, zorder=4))
    ax.text(cx, cy, sym, ha="center", va="center", fontsize=11, color=INK, zorder=5)


def file_icon(ax, x, y, w, h, tab_text, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2",
                 linewidth=1.2, facecolor="white", edgecolor="#7c8794", zorder=3))
    ax.add_patch(Polygon([(x + w, y + h), (x + w - 8, y + h), (x + w, y + h - 8)],
                 closed=True, facecolor="#dfe4ea", edgecolor="#7c8794", lw=0.8, zorder=3))
    ax.add_patch(FancyBboxPatch((x + 3, y + 3), w - 6, 9, boxstyle="round,pad=0,rounding_size=1",
                 linewidth=0, facecolor=col, zorder=3))
    ax.text(x + (w - 3) / 2, y + 7.5, tab_text, ha="center", va="center", fontsize=6.4,
            color="white", fontweight="bold", zorder=4)


# ---- operation glyphs ----
def byte_strip(ax, cx, cy, cols=4, rows=2, cell=11):
    rng = np.random.default_rng(7)
    w = cols * cell
    h = rows * cell
    x0, y0 = cx - w / 2, cy - h / 2
    for r in range(rows):
        for c in range(cols):
            b = rng.integers(0, 2)
            ax.add_patch(FancyBboxPatch((x0 + c * cell, y0 + (rows - 1 - r) * cell), cell - 1.5, cell - 1.5,
                         boxstyle="round,pad=0,rounding_size=0.8", linewidth=0.6,
                         facecolor="#f4f4f1", edgecolor="#c4c4bc", zorder=3))
            ax.text(x0 + c * cell + cell / 2 - 0.7, y0 + (rows - 1 - r) * cell + cell / 2 - 0.7,
                    str(b), ha="center", va="center", fontsize=6.4, color="#555", zorder=4)


def pixel_grid(ax, cx, cy, n=5, cell=10):
    rng = np.random.default_rng(3)
    palette = ["#d98c8c", "#d9b96a", "#9ac49a", "#7fb0d8", "#b79ad0", "#e0a35c", "#82c7bd"]
    w = n * cell
    x0, y0 = cx - w / 2, cy - w / 2
    for r in range(n):
        for c in range(n):
            col = palette[rng.integers(0, len(palette))]
            ax.add_patch(FancyBboxPatch((x0 + c * cell, y0 + (n - 1 - r) * cell), cell - 1.2, cell - 1.2,
                         boxstyle="round,pad=0,rounding_size=0.8", linewidth=0.5,
                         facecolor=col, edgecolor=_shade(col, 0.7), zorder=3))


def kernel_glyph(ax, cx, cy, n=3, cell=6.5, base="#cdddef"):
    w = n * cell
    x0, y0 = cx - w / 2, cy - w / 2
    for r in range(n):
        for c in range(n):
            hot = (r == 1 and c == 1)
            ax.add_patch(FancyBboxPatch((x0 + c * cell, y0 + r * cell), cell - 1, cell - 1,
                         boxstyle="round,pad=0,rounding_size=0.6", linewidth=0.6,
                         facecolor="#7fb0e6" if hot else base, edgecolor="#5B8FCF", zorder=4))


def spline_glyph(ax, cx, cy, w=40, h=26):
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=2",
                 linewidth=1.0, facecolor="white", edgecolor=C_KAN["ec"], zorder=4))
    xs = np.linspace(x0 + 4, x0 + w - 4, 60)
    t = np.linspace(-2.4, 2.4, 60)
    # sum of a few RBF bumps -> wiggly learnable curve
    ys = (np.exp(-((t + 1.4) ** 2) / 0.5) * 0.9 - np.exp(-((t - 0.2) ** 2) / 0.4) * 0.7
          + np.exp(-((t - 1.6) ** 2) / 0.5) * 1.0)
    ys = y0 + h / 2 + ys * (h * 0.26)
    ax.plot(xs, ys, color=C_KAN["ec"], lw=1.6, zorder=5)
    for mu in (-1.4, 0.2, 1.6):
        bx = x0 + 4 + (mu + 2.4) / 4.8 * (w - 8)
        ax.plot([bx, bx], [y0 + 3, y0 + h - 3], color="#e9c19a", lw=0.7, ls=(0, (2, 2)), zorder=4.5)


def capsule_field(ax, cx, cy, rows=6, cols=6, step=14):
    """Grid of oriented arrows = capsule pose vectors."""
    rng = np.random.default_rng(11)
    palette = plt.get_cmap("twilight")
    w = cols * step
    h = rows * step
    x0, y0 = cx - w / 2, cy - h / 2
    # backing card
    ax.add_patch(FancyBboxPatch((x0 - 6, y0 - 6), w + 12, h + 12,
                 boxstyle="round,pad=0,rounding_size=5", linewidth=1.2,
                 facecolor="#fffdf6", edgecolor=C_KAN["ec"], zorder=2.8))
    for r in range(rows):
        for c in range(cols):
            ang = rng.uniform(0, 2 * np.pi)
            ln = rng.uniform(3.5, 6.0)
            px = x0 + c * step + step / 2
            py = y0 + r * step + step / 2
            dx, dy = ln * np.cos(ang), ln * np.sin(ang)
            col = palette((ang / (2 * np.pi)) * 0.8 + 0.1)
            ax.add_patch(FancyArrowPatch((px - dx, py - dy), (px + dx, py + dy),
                         arrowstyle="-|>", mutation_scale=5, lw=1.1, color=col, zorder=4))
    return (cx, cy, w, h)


def attn_heat(ax, cx, cy, n=6, cell=10):
    rng = np.random.default_rng(5)
    m = np.eye(n) * 0.9 + rng.uniform(0, 0.35, (n, n))
    m += np.exp(-((np.arange(n)[:, None] - np.arange(n)[None, :]) ** 2) / 3.0) * 0.5
    m = m / m.max()
    cmap = plt.get_cmap("YlOrRd")
    w = n * cell
    x0, y0 = cx - w / 2, cy - w / 2
    for r in range(n):
        for c in range(n):
            ax.add_patch(FancyBboxPatch((x0 + c * cell, y0 + (n - 1 - r) * cell), cell - 0.8, cell - 0.8,
                         boxstyle="round,pad=0,rounding_size=0.5", linewidth=0.4,
                         facecolor=cmap(m[r, c]), edgecolor="#e8c9a0", zorder=4))
    return (cx, cy, w, w)


def squash_glyph(ax, cx, cy, w=74, h=52, label="Squash", label_pos="above"):
    """Pictogram for the squash nonlinearity: a long input vector compressed to a
    short one (direction kept), over the saturating length curve s=|u|^2/(1+|u|^2)
    that caps the output length at 1."""
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=4",
                 linewidth=1.4, facecolor="white", edgecolor=C_OP["ec"], zorder=3.5))
    # --- plot area ---
    px0, py0 = x0 + 11, y0 + 9
    pw, ph = w - 20, h - 18
    ax.plot([px0, px0], [py0, py0 + ph], color="#b7b7b7", lw=0.8, zorder=4)        # y axis
    ax.plot([px0, px0 + pw], [py0, py0], color="#b7b7b7", lw=0.8, zorder=4)        # x axis
    ax.plot([px0, px0 + pw], [py0 + ph, py0 + ph], color="#cc9a6a", lw=0.9,        # asymptote = 1
            ls=(0, (3, 2)), zorder=4)
    ax.text(px0 + pw + 3, py0 + ph, "1", ha="left", va="center", fontsize=6.6,
            color="#b07a3a", zorder=5)
    t = np.linspace(0, 3.4, 80)
    s = t ** 2 / (1 + t ** 2)
    ax.plot(px0 + t / 3.4 * pw, py0 + s * ph, color=C_OP["ec"], lw=2.3, zorder=5)
    # --- input(long) -> output(short) vector cue, top-left of the curve ---
    ax.annotate("", xy=(px0 + pw * 0.30, py0 + ph * 0.86), xytext=(px0 + pw * 0.30, py0 + ph * 0.40),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#9bbf9b"), zorder=4)
    if label_pos == "above":
        ax.text(cx, y0 + h + 9, label, ha="center", va="bottom", fontsize=8.6,
                color=C_OP["tc"], fontweight="bold")
    elif label_pos == "below":
        ax.text(cx, y0 - 9, label, ha="center", va="top", fontsize=8.6,
                color=C_OP["tc"], fontweight="bold")
    elif label_pos == "right":
        ax.text(x0 + w + 9, cy, label, ha="left", va="center", fontsize=8.6,
                color=C_OP["tc"], fontweight="bold")
    return (cx, cy, w, h)


def length_bars(ax, cx, cy_top, n=5, w=66, bar_h=9, gap=2.4):
    cols = ["#f4c7b6", "#f6d9a8", "#f6ecb0", "#d8ebbf", "#c2d8ee"]
    lens = [0.9, 0.55, 0.7, 0.4, 0.6]
    for i in range(n):
        y = cy_top - i * (bar_h + gap)
        ax.add_patch(FancyBboxPatch((cx - w / 2, y - bar_h), w * lens[i], bar_h,
                     boxstyle="round,pad=0,rounding_size=1.5", linewidth=0.7,
                     facecolor=cols[i], edgecolor=_shade(cols[i], 0.7), zorder=3))
        ax.add_patch(FancyBboxPatch((cx - w / 2, y - bar_h), w, bar_h,
                     boxstyle="round,pad=0,rounding_size=1.5", linewidth=0.6,
                     facecolor="none", edgecolor="#cfcfcf", zorder=3.1))
    return (cx, cy_top, w, n * (bar_h + gap))


def prob_bars(ax, cx, cy, heights, hi):
    w = len(heights) * 12
    x0 = cx - w / 2
    base_y = cy - 18
    for i, hgt in enumerate(heights):
        col = C_OUT["ec"] if i == hi else "#cdb6e8"
        ax.add_patch(FancyBboxPatch((x0 + i * 12, base_y), 9, hgt, boxstyle="round,pad=0,rounding_size=1",
                     linewidth=0.6, facecolor=col, edgecolor=_shade(C_OUT["ec"], 0.8), zorder=4))
    ax.plot([x0 - 2, x0 + w], [base_y, base_y], color="#9a9a9a", lw=0.9, zorder=4)


def class_tokens(ax, cx, cy, k=4, s=14, gap=5):
    cols = ["#f0c27a", "#f3c6b8", "#bfd9ef", "#c5e0c3"]
    w = k * (s + gap)
    x0 = cx - w / 2
    for i in range(k):
        ax.add_patch(FancyBboxPatch((x0 + i * (s + gap), cy - s / 2), s, s,
                     boxstyle="round,pad=0,rounding_size=2", linewidth=1.0,
                     facecolor=cols[i % len(cols)], edgecolor="#9a8a5a", zorder=4))
    ax.text(x0 + s / 2, cy, "C", ha="center", va="center", fontsize=7.5, fontweight="bold",
            color="#5a4a1a", zorder=5)
    return (cx, cy, w, s)


# ----------------------------------------------------------------------------
def main() -> None:
    fig, ax = plt.subplots(figsize=(15.0, 5.9))
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 590)
    ax.axis("off")

    band(ax, 8, 50, 590, 510, "A", "(a) Feature extraction")
    band(ax, 614, 50, 300, 510, "B", "(b) Primary capsules")
    band(ax, 930, 50, 562, 510, "C", "(c) Class-token Transformer routing")

    # =================== Panel A ===================
    # --- intake row (top), strictly horizontal ---
    file_icon(ax, 40, 492, 32, 38, "EXE", "#6b9bd1")
    file_icon(ax, 52, 480, 32, 38, "ELF", "#7bbf7b")
    file_icon(ax, 64, 468, 32, 38, "APK", "#e0a35c")
    byte_strip(ax, 195, 497, cols=4, rows=2)
    arrow(ax, (104, 492), (167, 497))
    pixel_grid(ax, 312, 497, n=5, cell=10)
    arrow(ax, (227, 497), (282, 497))
    tag(ax, 254, 518, "Bin2Img", fs=7.6, weight="bold", style="italic")
    incube = cuboid(ax, 425, 497, 42, 42, d=12, base="#9aa6b2", label="4 × 96 × 96", label_pos="above")
    arrow(ax, (344, 497), edge(incube, "l"))

    # --- conv chain (vertical), dim labels on the LEFT, op glyphs on the RIGHT ---
    cx = 425
    s1 = cuboid(ax, cx, 412, 40, 40, d=11, base="#9cc0ea", label="32 × 48 × 48", label_pos="left")
    s2 = cuboid(ax, cx, 344, 30, 30, d=10, base="#7fb0e6", label="64 × 24 × 24", label_pos="left")
    s3 = cuboid(ax, cx, 286, 22, 22, d=9, base="#5f97da", label="96 × 12 × 12", label_pos="left")
    arrow(ax, edge(incube, "b"), edge(s1, "t"))
    arrow(ax, edge(s1, "b"), edge(s2, "t"))
    arrow(ax, edge(s2, "b"), edge(s3, "t"))
    gx = 512
    kernel_glyph(ax, gx, 456); tag(ax, gx, 474, "conv ↓2", fs=6.6, color="#5B8FCF")
    kernel_glyph(ax, gx, 388); tag(ax, gx, 406, "conv ↓2", fs=6.6, color="#5B8FCF")
    spline_glyph(ax, gx, 338, w=44, h=24); tag(ax, gx, 357, "Conv-KAN", fs=6.6, color=C_KAN["ec"])

    # --- Conv-KAN detail inset (bottom band); tied to the step above by colour + title ---
    ix, iy, iw, ih = 50, 95, 510, 150
    ax.add_patch(FancyBboxPatch((ix, iy), iw, ih, boxstyle="round,pad=0,rounding_size=8",
                 linewidth=1.2, facecolor="white", edgecolor=C_KAN["ec"],
                 linestyle=(0, (5, 3)), zorder=2.2, alpha=0.96))
    tag(ax, ix + iw / 2, iy + ih - 13, "Conv-KAN block (detail of the orange step)", fs=8.2,
        color=C_KAN["tc"], weight="bold")
    inslab = cuboid(ax, ix + 70, iy + 70, 18, 50, d=8, base="#f0b9a0")
    dwg = kernel_glyph(ax, ix + 195, iy + 80, base="#f6d9c0"); tag(ax, ix + 195, iy + 100, "DWConv", fs=6.8, color=C_KAN["ec"])
    spline_glyph(ax, ix + 320, iy + 80, w=46, h=28); tag(ax, ix + 320, iy + 102, "spline φ (RBF)", fs=6.8, color=C_KAN["ec"])
    oplus(ax, ix + 257, iy + 40)
    outslab = cuboid(ax, ix + 410, iy + 64, 16, 44, d=8, base="#cdd7e6")
    arrow(ax, (ix + 88, iy + 74), (ix + 178, iy + 80), lw=1.2)
    arrow(ax, (ix + 88, iy + 64), (ix + 300, iy + 78), lw=1.1, rad=-0.22)
    arrow(ax, (ix + 195, iy + 67), (ix + 250, iy + 47), lw=1.2, rad=-0.12)
    arrow(ax, (ix + 320, iy + 65), (ix + 266, iy + 47), lw=1.2, rad=0.16)
    arrow(ax, (ix + 267, iy + 40), edge(outslab, "l"), lw=1.2)
    oplus(ax, ix + 14, iy - 16, r=6)
    tag(ax, ix + 66, iy - 16, "Concatenate", fs=7.2)

    # =================== Panel B ===================
    bcube = cuboid(ax, 668, 360, 24, 64, d=10, base="#cdd7e6", label="96 × 12 × 12", label_pos="below")
    arrow(ax, edge(s3, "r"), edge(bcube, "l"), rad=-0.04)  # A -> B data flow (low, clears op glyphs)
    field = capsule_field(ax, 800, 360, rows=6, cols=6, step=15)
    arrow(ax, edge(bcube, "r"), edge(field, "l"), rad=0.0)
    kernel_glyph(ax, 726, 405, base="#f6d9c0"); tag(ax, 726, 423, "caps conv", fs=6.6, color=C_KAN["ec"])
    tag(ax, 812, 296, "432 capsule vectors", fs=8, weight="bold")
    tag(ax, 812, 283, "(6 × 6 × 12 types, 16-D)", fs=7)
    squash_glyph(ax, 800, 460)
    arrow(ax, (800, 448), edge(field, "t"))
    # capsule type maps (12 type channels) -- decomposition aside, fills lower area.
    # Down-link hugs the field's far-left so it never crosses the right-shifted captions.
    arrow(ax, (744, 308), (700, 200), lw=1.1, rad=-0.16)
    cuboid(ax, 700, 182, 96, 12, d=6, base="#f3c6b8")
    cuboid(ax, 700, 164, 96, 12, d=6, base="#f6e6b0")
    cuboid(ax, 700, 146, 96, 12, d=6, base="#bfe0c2")
    tag(ax, 748, 128, "12 capsule-type maps", fs=7.2)

    # =================== Panel C ===================
    ctok = class_tokens(ax, 1000, 505, k=4)
    tag(ax, 1000, 484, "K class tokens", fs=7.4)
    # attention block (stacked x2)
    ax_cx, ax_cy = 1175, 360
    for off in (8, 4):
        ax.add_patch(FancyBboxPatch((ax_cx - 92 + off, ax_cy - 78 + off), 184, 156,
                     boxstyle="round,pad=0,rounding_size=8", linewidth=1.0,
                     facecolor="#fff", edgecolor="#c79a6a", alpha=0.7, zorder=2.2))
    ax.add_patch(FancyBboxPatch((ax_cx - 92, ax_cy - 78), 184, 156, boxstyle="round,pad=0,rounding_size=8",
                 linewidth=1.3, facecolor="white", edgecolor="#c79a6a", zorder=2.5))
    tag(ax, ax_cx, ax_cy + 66, "Self-Attention × 2", fs=8.4, color="#8a5a2a", weight="bold")
    heat = attn_heat(ax, ax_cx, ax_cy + 4, n=6, cell=11)
    # Q/K/V mini bars
    for i, (lab, col) in enumerate(zip("QKV", ["#9B72CE", "#c79a6a", "#5BA85A"])):
        bx = ax_cx - 52 + i * 52
        ax.add_patch(FancyBboxPatch((bx - 12, ax_cy - 70), 24, 10, boxstyle="round,pad=0,rounding_size=2",
                     linewidth=0.8, facecolor=_shade(col, 1.25), edgecolor=col, zorder=4))
        tag(ax, bx, ax_cy - 65, lab, fs=7, color=col, weight="bold")
        arrow(ax, (bx, ax_cy - 59), (ax_cx - 22 + i * 22, ax_cy - 20), lw=0.9)

    # inputs into attention: tokens(view) + class tokens + PE
    arrow(ax, edge(field, "r"), (ax_cx - 92, ax_cy + 30), rad=-0.12)
    tag(ax, 940, 350, "view", fs=7.2, style="italic")
    ax.add_patch(Circle((948, 300), 11, facecolor="#fde9b8", edgecolor="#D9B23C", lw=1.2, zorder=4))
    ax.plot([942, 945, 948, 951, 954], [300, 305, 300, 295, 300], color="#b9892a", lw=1.1, zorder=5)
    tag(ax, 948, 282, "Pos. Enc.", fs=6.8)
    oplus(ax, 980, 330, r=8)
    arrow(ax, (959, 300), (974, 326), lw=1.1)
    arrow(ax, (988, 330), (ax_cx - 92, ax_cy + 6), lw=1.2, rad=0.05)
    arrow(ax, edge(ctok, "b"), (ax_cx - 30, ax_cy + 78), rad=0.0)

    # output: take cls -> squash -> lengths -> prediction
    arrow(ax, (ax_cx + 92, ax_cy + 30), (1320, 470), rad=-0.16)
    tag(ax, 1340, 488, "take cls", fs=7.2, style="italic")
    sqg = squash_glyph(ax, 1360, 430)
    arrow(ax, (1340, 455), (1352, 442), lw=1.1)
    lb = length_bars(ax, 1370, 400, n=5)
    tag(ax, 1370, 400 - 5 * 11.4 - 8, "capsule lengths", fs=7.2)
    arrow(ax, (1378, 420), (1370, 402), lw=1.1)
    # prediction bar chart
    pred_cx, pred_cy = 1200, 165
    ax.add_patch(FancyBboxPatch((pred_cx - 70, pred_cy - 40), 140, 88, boxstyle="round,pad=0,rounding_size=6",
                 linewidth=1.3, facecolor="#f6f1fb", edgecolor=C_OUT["ec"], zorder=3))
    prob_bars(ax, pred_cx, pred_cy + 6, [14, 40, 22, 10, 30, 16], hi=1)
    tag(ax, pred_cx, pred_cy - 30, "Prediction", fs=9.5, color=C_OUT["tc"], weight="bold")
    arrow(ax, (1340, 360), edge((pred_cx, pred_cy, 140, 88), "r"), rad=0.25)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.06)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "framework_pipeline_v2.pdf"
    png = OUT_DIR / "framework_pipeline_v2.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
