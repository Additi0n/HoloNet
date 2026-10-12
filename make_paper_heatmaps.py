"""Render publication-quality Grad-CAM heatmaps for the paper.

Reuses the trained checkpoints and the Grad-CAM machinery from
``run_explainability.py`` but, instead of saving low-resolution alpha-blended
overlays, it extracts the raw CAM float fields and composes a clean matplotlib
figure: per dataset a (original | Grad-CAM | overlay) triptych, with a best vs
last comparison and a single shared colorbar.

Run from the project root:

    python make_paper_heatmaps.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import run_explainability as rex


def _gaussian_blur(cam: torch.Tensor, sigma: float = 1.1, ksize: int = 7) -> torch.Tensor:
    """Separable Gaussian blur on a 2D tensor, no scipy dependency."""
    coords = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, ksize, ksize)
    padded = cam.view(1, 1, *cam.shape)
    padded = F.pad(padded, (ksize // 2,) * 4, mode="reflect")
    return F.conv2d(padded, kernel_2d)[0, 0]

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs" / "launcher"
DATA_ROOT = PROJECT_ROOT / "RGBA"
OUT_DIR = PROJECT_ROOT / "article" / "images"

# One representative family per dataset, kept consistent with the explainability
# panels already shown in the paper (Wabot / Adware / Tsunami).
DATASETS = [
    ("holonet_bodmas", "BODMAS", "Wabot"),
    ("holonet_cicmaldroid2020", "CICMalDroid2020", "Adware"),
    ("holonet_elf2020", "ELF2020", "Tsunami"),
]

CMAP = "turbo"  # blue (low) -> red (high), matches the caption's colour wording


def setup_cjk_font() -> None:
    """Pick a CJK-capable sans font so Chinese labels render correctly."""
    for name in ("Microsoft YaHei", "SimHei", "Microsoft JhengHei", "SimSun"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def _safe_torch_load(path, device):
    """Load a checkpoint that may embed Linux PosixPath objects on Windows."""
    import pathlib
    saved = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath  # PosixPath can't instantiate on Windows
    try:
        return torch.load(path, map_location=device, weights_only=False)
    finally:
        pathlib.PosixPath = saved


def load_model(run_name: str, device: torch.device):
    ckpt_path = RUNS_ROOT / run_name / "best.pt"
    last_path = RUNS_ROOT / run_name / "last.pt"
    best_ckpt = _safe_torch_load(ckpt_path, device)
    last_ckpt = _safe_torch_load(last_path, device)
    best_model, args_dict, dataset_info = rex.build_model_from_checkpoint(best_ckpt, device)
    last_model, _, _ = rex.build_model_from_checkpoint(last_ckpt, device)
    image_size = int(args_dict.get("image_size", 96))
    return best_model, last_model, image_size, dataset_info


def pick_family_sample(model, samples, class_names, family, image_size, device):
    """Return the highest-confidence correctly classified sample of ``family``."""
    family_idx = class_names.index(family)
    family_samples = [(p, y) for (p, y) in samples if y == family_idx]
    best_record = None
    best_conf = -1.0
    model.eval()
    with torch.no_grad():
        for path, label in family_samples:
            image = Image.open(path).convert("RGBA")
            image = TF.resize(image, [image_size, image_size])
            tensor = rex.normalize_tensor(TF.to_tensor(image)).unsqueeze(0).to(device)
            out = model(tensor)
            prob = torch.softmax(out["logits"], dim=1)[0]
            conf, pred = prob.max(dim=0)
            if int(pred.item()) == label and float(conf.item()) > best_conf:
                best_conf = float(conf.item())
                best_record = (path, label)
    if best_record is None and family_samples:  # fall back to any sample
        best_record = family_samples[0]
    return best_record, family_idx


def compute_cam(model, image_path, image_size, target_index, device):
    """Return (rgb uint8 HxWx3, cam float HxW in [0,1]) for one image."""
    image = Image.open(image_path).convert("RGBA")
    image = TF.resize(image, [image_size, image_size])
    input_tensor = rex.normalize_tensor(TF.to_tensor(image)).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    hook = rex.GradCAMHook(model.context_refine)
    model.zero_grad(set_to_none=True)
    outputs = model(input_tensor)
    outputs["logits"][0, target_index].backward()

    if hook.activations is None or hook.gradients is None:
        hook.close()
        raise RuntimeError("Grad-CAM hooks captured nothing")

    weights = hook.gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * hook.activations).sum(dim=1, keepdim=True).relu()
    cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = rex.normalize_map(cam)
    hook.close()

    # Light smoothing for a cleaner, less pixelated heatmap.
    cam = _gaussian_blur(cam.detach(), sigma=1.1).numpy()
    cam = cam - cam.min()
    cam = cam / max(cam.max(), 1e-8)

    rgb = rex.denormalize_tensor(input_tensor[0].detach().cpu())[:3]
    rgb = (rgb.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return rgb, cam


def draw_overlay(ax, rgb, cam):
    """Dimmed original + turbo heatmap with response-driven alpha."""
    base = (0.45 * rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    ax.imshow(base)
    alpha = np.clip(cam, 0, 1) ** 0.8 * 0.9
    ax.imshow(cam, cmap=CMAP, vmin=0.0, vmax=1.0, alpha=alpha, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    setup_cjk_font()
    device = torch.device("cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: best-vs-last overlay comparison (2 rows x 3 datasets) ----
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.0))
    row_labels = ["最佳模型 (best)", "末轮模型 (last)"]

    collected = {}
    for col, (run_name, pretty, family) in enumerate(DATASETS):
        print(f"[heatmap] {pretty}: loading checkpoints ...")
        best_model, last_model, image_size, dataset_info = load_model(run_name, device)
        dataset_name = str(dataset_info["dataset_name"])
        class_names, samples = rex.scan_test_samples(DATA_ROOT, dataset_name)
        record, family_idx = pick_family_sample(best_model, samples, class_names, family, image_size, device)
        path = record[0]
        print(f"[heatmap] {pretty}: sample={Path(path).name} family={family}")

        rgb_b, cam_b = compute_cam(best_model, path, image_size, family_idx, device)
        rgb_l, cam_l = compute_cam(last_model, path, image_size, family_idx, device)
        collected[pretty] = (family, rgb_b, cam_b)

        draw_overlay(axes[0, col], rgb_b, cam_b)
        draw_overlay(axes[1, col], rgb_l, cam_l)
        axes[0, col].set_title(f"{pretty}\n{family}", fontsize=11, pad=6)

    for row in range(2):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=11)
        axes[row, 0].tick_params(left=False, labelleft=False)

    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.05, pad=0.06, aspect=40)
    cbar.set_label("Grad-CAM 归一化响应（低 → 高）", fontsize=10)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    out1 = OUT_DIR / "gradcam_heatmaps_v2.png"
    fig.savefig(out1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[heatmap] wrote {out1}")

    # ---- Figure 2: per-dataset triptych (original | heatmap | overlay) ----
    fig2, axes2 = plt.subplots(3, 3, figsize=(7.6, 7.8))
    col_titles = ["原始图像", "Grad-CAM 热力图", "叠加结果"]
    for r, (pretty, (family, rgb, cam)) in enumerate(collected.items()):
        axes2[r, 0].imshow(rgb)
        im = axes2[r, 1].imshow(cam, cmap=CMAP, vmin=0.0, vmax=1.0, interpolation="bilinear")
        draw_overlay(axes2[r, 2], rgb, cam)
        for c in range(3):
            axes2[r, c].set_xticks([])
            axes2[r, c].set_yticks([])
        axes2[r, 0].set_ylabel(f"{pretty}\n{family}", fontsize=10)
        if r == 0:
            for c in range(3):
                axes2[r, c].set_title(col_titles[c], fontsize=11, pad=6)

    sm2 = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=CMAP)
    sm2.set_array([])
    cbar2 = fig2.colorbar(sm2, ax=axes2, orientation="horizontal",
                          fraction=0.045, pad=0.05, aspect=40)
    cbar2.set_label("Grad-CAM 归一化响应（低 → 高）", fontsize=10)
    cbar2.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    out2 = OUT_DIR / "gradcam_triptych_v2.png"
    fig2.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"[heatmap] wrote {out2}")


if __name__ == "__main__":
    main()
