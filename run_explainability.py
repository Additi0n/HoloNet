from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageOps
import torch
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from holonet.model import HoloNet


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "RGBA"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "explainability"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
NORMALIZE_MEAN = (0.5, 0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5, 0.5)


@dataclass(slots=True)
class SampleRecord:
    image_path: str
    true_index: int
    predicted_index: int
    confidence: float


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest_name = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest_name, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--disable-{name}", dest=dest_name, action="store_false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explainability artifacts for trained HoloNet checkpoints")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dirs", type=Path, nargs="*", default=None)
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=None)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--datasets", type=str, nargs="*", default=None)
    parser.add_argument("--include-runs", type=str, nargs="*", default=None)
    parser.add_argument("--exclude-runs", type=str, nargs="*", default=None)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-scan-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-mode", type=str, choices=["predicted", "true"], default="predicted")
    add_bool_flag(parser, "correct-only", True, "Use only correctly predicted samples for explanations.")
    add_bool_flag(parser, "full-only", True, "Explain only full models and skip ablation checkpoints by default.")
    add_bool_flag(parser, "amp", False, "Enable automatic mixed precision on CUDA during sample selection.")
    return parser.parse_args()


def resolve_path(path: Path, base_dir: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, SampleRecord):
        return asdict(value)
    return value


def is_full_experiment(run_name: str, args_dict: dict[str, Any]) -> bool:
    experiment_name = str(args_dict.get("experiment_name", run_name))
    return not experiment_name.startswith("ablation_")


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        return [resolve_path(path) for path in args.checkpoints]

    if args.run_dirs:
        checkpoints: list[Path] = []
        for run_dir in args.run_dirs:
            resolved_run_dir = resolve_path(run_dir)
            if args.full_only and resolved_run_dir.name.startswith("ablation_"):
                continue
            checkpoint_path = resolved_run_dir / args.checkpoint_name
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
            checkpoints.append(checkpoint_path)
        if not checkpoints:
            raise FileNotFoundError("no checkpoints selected for explainability")
        return checkpoints

    runs_root = resolve_path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    include_runs = set(args.include_runs or [])
    exclude_runs = set(args.exclude_runs or [])
    checkpoints: list[Path] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if run_dir.name in {"launcher", "robustness", "explainability"}:
            continue
        if args.full_only and run_dir.name.startswith("ablation_"):
            continue
        if include_runs and run_dir.name not in include_runs:
            continue
        if run_dir.name in exclude_runs:
            continue
        checkpoint_path = run_dir / args.checkpoint_name
        if checkpoint_path.is_file():
            checkpoints.append(checkpoint_path)

    if not checkpoints:
        raise FileNotFoundError("no checkpoints found for explainability")
    return checkpoints


def normalize_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    return TF.normalize(image_tensor, mean=NORMALIZE_MEAN, std=NORMALIZE_STD)


def denormalize_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORMALIZE_MEAN, device=image_tensor.device).view(-1, 1, 1)
    std = torch.tensor(NORMALIZE_STD, device=image_tensor.device).view(-1, 1, 1)
    return (image_tensor * std + mean).clamp(0.0, 1.0)


class InferenceDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, samples: list[tuple[Path, int]], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGBA")
        image = TF.resize(image, [self.image_size, self.image_size])
        tensor = normalize_tensor(TF.to_tensor(image))
        return tensor, label, str(image_path)


def scan_test_samples(data_root: Path, dataset_name: str, max_samples: int | None = None) -> tuple[list[str], list[tuple[Path, int]]]:
    split_dir = data_root / dataset_name / "test"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"test split not found: {split_dir}")

    class_names = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples: list[tuple[Path, int]] = []

    for class_name in class_names:
        class_dir = split_dir / class_name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, class_to_idx[class_name]))
                if max_samples is not None and len(samples) >= max_samples:
                    return class_names, samples
    return class_names, samples


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    args_dict = dict(checkpoint.get("args", {}))
    dataset_info = dict(checkpoint.get("dataset_info", {}))
    num_classes = int(dataset_info["num_classes"])
    stem_channels = tuple(args_dict.get("stem_channels", [32, 64, 96]))

    model = HoloNet(
        num_classes=num_classes,
        image_size=int(args_dict.get("image_size", 96)),
        stem_channels=stem_channels,
        primary_capsule_dim=int(args_dict.get("primary_capsule_dim", 16)),
        primary_capsule_types=int(args_dict.get("primary_capsule_types", 12)),
        transformer_dim=int(args_dict.get("transformer_dim", 128)),
        transformer_heads=int(args_dict.get("transformer_heads", 4)),
        transformer_layers=int(args_dict.get("transformer_layers", 2)),
        ckan_grid_size=int(args_dict.get("ckan_grid_size", 8)),
        dropout=float(args_dict.get("dropout", 0.1)),
        use_ckan=bool(args_dict.get("use_ckan", True)),
        use_transformer_router=bool(args_dict.get("use_transformer_router", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, args_dict, dataset_info


class GradCAMHook:
    def __init__(self, module: nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = module.register_forward_hook(self._forward_hook)
        self.backward_handle = module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        self.gradients = grad_output[0].detach() if grad_output[0] is not None else None

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def normalize_map(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = heatmap.detach().float()
    heatmap = heatmap - heatmap.min()
    maximum = heatmap.max().clamp_min(1e-8)
    return heatmap / maximum


def heatmap_to_overlay(base_image: Image.Image, heatmap: torch.Tensor, alpha: float = 0.45) -> tuple[Image.Image, Image.Image]:
    heatmap = normalize_map(heatmap.cpu())
    gray = TF.to_pil_image(heatmap.unsqueeze(0))
    colorized = ImageOps.colorize(gray.convert("L"), black="#001d3d", mid="#ffdd00", white="#d00000")
    base_rgb = base_image.convert("RGB")
    overlay = Image.blend(base_rgb, colorized, alpha=alpha)
    return colorized, overlay


def compose_panel(original: Image.Image, saliency_overlay: Image.Image, gradcam_overlay: Image.Image) -> Image.Image:
    images = [original.convert("RGB"), saliency_overlay.convert("RGB"), gradcam_overlay.convert("RGB")]
    width, height = images[0].size
    panel = Image.new("RGB", (width * len(images), height), color="white")
    for index, image in enumerate(images):
        panel.paste(image, (index * width, 0))
    return panel


def select_representative_samples(
    model: nn.Module,
    samples: list[tuple[Path, int]],
    image_size: int,
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[int, list[SampleRecord]]:
    dataset = InferenceDataset(samples, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    candidates: dict[int, list[SampleRecord]] = {index: [] for index in range(len(class_names))}
    progress = tqdm(loader, leave=False, dynamic_ncols=True, file=sys.stdout)
    with torch.no_grad():
        for images, targets, paths in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
            probabilities = torch.softmax(outputs["logits"], dim=1)
            confidences, predictions = probabilities.max(dim=1)

            for sample_index, image_path in enumerate(paths):
                true_index = int(targets[sample_index].item())
                predicted_index = int(predictions[sample_index].item())
                if args.correct_only and predicted_index != true_index:
                    continue
                record = SampleRecord(
                    image_path=image_path,
                    true_index=true_index,
                    predicted_index=predicted_index,
                    confidence=float(confidences[sample_index].item()),
                )
                candidates[true_index].append(record)

    selected: dict[int, list[SampleRecord]] = {}
    for class_index, records in candidates.items():
        ranked = sorted(records, key=lambda item: item.confidence, reverse=True)
        selected[class_index] = ranked[: args.samples_per_class]
    return selected


def compute_explanations(
    model: nn.Module,
    image_path: Path,
    image_size: int,
    target_index: int,
    device: torch.device,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    image = Image.open(image_path).convert("RGBA")
    image = TF.resize(image, [image_size, image_size])
    input_tensor = normalize_tensor(TF.to_tensor(image)).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    hook = GradCAMHook(model.context_refine)
    model.zero_grad(set_to_none=True)
    outputs = model(input_tensor)
    target_score = outputs["logits"][0, target_index]
    target_score.backward()

    saliency = input_tensor.grad.detach()[0].abs().max(dim=0).values
    saliency = normalize_map(saliency)

    if hook.activations is None or hook.gradients is None:
        hook.close()
        raise RuntimeError("Grad-CAM hooks did not capture activations or gradients")

    weights = hook.gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * hook.activations).sum(dim=1, keepdim=True).relu()
    cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = normalize_map(cam)
    hook.close()

    base_rgb = TF.to_pil_image(denormalize_tensor(input_tensor[0].detach().cpu())[:3])
    _, saliency_overlay = heatmap_to_overlay(base_rgb, saliency)
    _, gradcam_overlay = heatmap_to_overlay(base_rgb, cam)
    panel = compose_panel(base_rgb, saliency_overlay, gradcam_overlay)
    return base_rgb, saliency_overlay, gradcam_overlay, panel


def build_markdown_report(
    run_name: str,
    dataset_name: str,
    class_names: list[str],
    records: list[dict[str, Any]],
) -> str:
    lines = [f"# Explainability Report: {run_name}", "", f"数据集: {dataset_name}", ""]
    for record in records:
        class_name = class_names[record["true_index"]]
        predicted_name = class_names[record["predicted_index"]]
        lines.extend(
            [
                f"## {class_name} | sample {record['sample_index']}",
                "",
                f"- 真值类别: {class_name}",
                f"- 预测类别: {predicted_name}",
                f"- 置信度: {record['confidence']:.4f}",
                f"- 图像路径: {record['image_path']}",
                "",
                f"![panel]({record['panel_path']})",
                "",
            ]
        )
    return "\n".join(lines)


def evaluate_checkpoint(checkpoint_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, args_dict, dataset_info = build_model_from_checkpoint(checkpoint, device)
    run_name = checkpoint_path.parent.name
    dataset_name = str(dataset_info["dataset_name"])
    if args.datasets and dataset_name not in set(args.datasets):
        return {}
    if args.full_only and not is_full_experiment(run_name, args_dict):
        print(f"[explainability] skip ablation run: {run_name}")
        return {}

    data_root = resolve_path(args.data_root)
    class_names, samples = scan_test_samples(data_root, dataset_name, max_samples=args.max_scan_samples)
    image_size = int(args_dict.get("image_size", 96))
    print(f"[explainability] run={run_name} dataset={dataset_name} samples={len(samples)}")

    selected = select_representative_samples(model, samples, image_size, class_names, args, device)
    run_dir = args.output_dir / run_name
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    artifact_records: list[dict[str, Any]] = []
    for class_index, records in selected.items():
        if not records:
            print(f"[explainability] no samples selected for class {class_names[class_index]} in {run_name}")
            continue
        for sample_index, record in enumerate(records, start=1):
            target_index = record.predicted_index if args.target_mode == "predicted" else record.true_index
            base, saliency_overlay, gradcam_overlay, panel = compute_explanations(
                model=model,
                image_path=Path(record.image_path),
                image_size=image_size,
                target_index=target_index,
                device=device,
            )
            prefix = f"{class_names[class_index]}_{sample_index:02d}"
            original_path = images_dir / f"{prefix}_original.png"
            saliency_path = images_dir / f"{prefix}_saliency.png"
            gradcam_path = images_dir / f"{prefix}_gradcam.png"
            panel_path = images_dir / f"{prefix}_panel.png"
            base.save(original_path)
            saliency_overlay.save(saliency_path)
            gradcam_overlay.save(gradcam_path)
            panel.save(panel_path)
            artifact_records.append(
                {
                    "class_name": class_names[class_index],
                    "sample_index": sample_index,
                    "image_path": record.image_path,
                    "true_index": record.true_index,
                    "predicted_index": record.predicted_index,
                    "confidence": record.confidence,
                    "target_index": target_index,
                    "original_path": str(original_path.relative_to(run_dir)),
                    "saliency_path": str(saliency_path.relative_to(run_dir)),
                    "gradcam_path": str(gradcam_path.relative_to(run_dir)),
                    "panel_path": str(panel_path.relative_to(run_dir)),
                }
            )
            print(
                f"[explainability] {run_name} class={class_names[class_index]} sample={sample_index} confidence={record.confidence:.4f}"
            )

    report_text = build_markdown_report(run_name, dataset_name, class_names, artifact_records)
    report_path = run_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")

    summary = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_name,
        "dataset_info": make_json_safe(dataset_info),
        "args": make_json_safe(args_dict),
        "samples_per_class": args.samples_per_class,
        "target_mode": args.target_mode,
        "correct_only": args.correct_only,
        "artifacts": artifact_records,
        "report_path": str(report_path),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    args.data_root = resolve_path(args.data_root)
    args.runs_root = resolve_path(args.runs_root)
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(args)
    summaries: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        summary = evaluate_checkpoint(checkpoint_path, args)
        if summary:
            summaries.append(summary)

    final_summary = {
        "runs_evaluated": len(summaries),
        "samples_per_class": args.samples_per_class,
        "target_mode": args.target_mode,
        "summaries": summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(make_json_safe(final_summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[explainability] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()