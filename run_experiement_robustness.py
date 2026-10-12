from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from PIL import Image
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from holonet.losses import CapsuleClassificationLoss
from holonet.metrics import compute_metrics, format_metrics, update_confusion_matrix
from holonet.model import HoloNet


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "RGBA"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "robustness"
DEFAULT_PERTURBATIONS = [
    "clean",
    "gaussian_noise",
    "gaussian_blur",
    "salt_pepper",
    "random_occlusion",
    "channel_dropout",
]


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest_name = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest_name, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--disable-{name}", dest=dest_name, action="store_false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HoloNet robustness on corrupted malware image test sets")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dirs", type=Path, nargs="*", default=None)
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=None)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--datasets", type=str, nargs="*", default=None)
    parser.add_argument("--include-runs", type=str, nargs="*", default=None)
    parser.add_argument("--exclude-runs", type=str, nargs="*", default=None)
    parser.add_argument("--perturbations", type=str, nargs="*", default=DEFAULT_PERTURBATIONS)
    parser.add_argument("--severities", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    add_bool_flag(parser, "amp", True, "Enable automatic mixed precision on CUDA during robustness evaluation.")
    add_bool_flag(parser, "full-only", True, "Evaluate only full models and skip ablation checkpoints by default.")
    return parser.parse_args()


def resolve_path(path: Path, base_dir: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        return [resolve_path(path) for path in args.checkpoints]

    if args.run_dirs:
        checkpoints = []
        for run_dir in args.run_dirs:
            resolved_run_dir = resolve_path(run_dir)
            if args.full_only and resolved_run_dir.name.startswith("ablation_"):
                continue
            checkpoint_path = resolved_run_dir / args.checkpoint_name
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
            checkpoints.append(checkpoint_path)
        if not checkpoints:
            raise FileNotFoundError("no full-model checkpoints found in the provided run directories")
        return checkpoints

    runs_root = resolve_path(args.runs_root)
    checkpoints: list[Path] = []
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    include_runs = set(args.include_runs or [])
    exclude_runs = set(args.exclude_runs or [])

    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if run_dir.name in {"launcher", "robustness"}:
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
        raise FileNotFoundError("no checkpoints found for robustness evaluation")
    return checkpoints


def stable_seed(base_seed: int, index: int, severity: int, perturbation: str) -> int:
    text_score = sum((position + 1) * ord(char) for position, char in enumerate(perturbation))
    return int(base_seed + index * 1009 + severity * 7919 + text_score * 31)


def build_generator(base_seed: int, index: int, severity: int, perturbation: str) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(stable_seed(base_seed, index, severity, perturbation))
    return generator


class RobustnessDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        image_size: int,
        perturbation: str,
        severity: int,
        seed: int,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.perturbation = perturbation
        self.severity = severity
        self.seed = seed
        self.normalize_mean = (0.5, 0.5, 0.5, 0.5)
        self.normalize_std = (0.5, 0.5, 0.5, 0.5)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGBA")
        image = TF.resize(image, [self.image_size, self.image_size])
        tensor = TF.to_tensor(image)
        tensor = apply_perturbation(tensor, self.perturbation, self.severity, self.seed, index)
        tensor = TF.normalize(tensor, mean=self.normalize_mean, std=self.normalize_std)
        return tensor, label


def apply_perturbation(
    image: torch.Tensor,
    perturbation: str,
    severity: int,
    seed: int,
    index: int,
) -> torch.Tensor:
    if perturbation == "clean":
        return image

    generator = build_generator(seed, index, severity, perturbation)
    image = image.clone()

    if perturbation == "gaussian_noise":
        sigma = 0.04 * severity
        noise = torch.randn(image.shape, generator=generator) * sigma
        return (image + noise).clamp(0.0, 1.0)

    if perturbation == "gaussian_blur":
        kernel_size = min(2 * severity + 1, 9)
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = 0.35 * severity
        return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    if perturbation == "salt_pepper":
        amount = 0.01 * severity
        salt_mask = torch.rand(image.shape[1:], generator=generator) < (amount / 2.0)
        pepper_mask = torch.rand(image.shape[1:], generator=generator) < (amount / 2.0)
        image[:, salt_mask] = 1.0
        image[:, pepper_mask] = 0.0
        return image

    if perturbation == "random_occlusion":
        height, width = image.shape[1:]
        ratio = min(0.12 + 0.08 * (severity - 1), 0.4)
        block_h = max(1, int(height * ratio))
        block_w = max(1, int(width * ratio))
        top = int(torch.randint(0, max(1, height - block_h + 1), (1,), generator=generator).item())
        left = int(torch.randint(0, max(1, width - block_w + 1), (1,), generator=generator).item())
        fill_value = float(torch.rand(1, generator=generator).item())
        image[:, top : top + block_h, left : left + block_w] = fill_value
        return image

    if perturbation == "channel_dropout":
        channel_count = 1 if severity <= 2 else 2
        channel_count = min(channel_count, image.size(0))
        permutation = torch.randperm(image.size(0), generator=generator)
        drop_channels = permutation[:channel_count]
        image[drop_channels] = 0.0
        return image

    raise ValueError(f"unsupported perturbation: {perturbation}")


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
            if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
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


def is_full_experiment(run_name: str, args_dict: dict[str, Any]) -> bool:
    experiment_name = str(args_dict.get("experiment_name", run_name))
    return not experiment_name.startswith("ablation_")


class TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def evaluate_with_live_logging(checkpoint_path: Path, args: argparse.Namespace, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        tee_stream = TeeStream(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee_stream), contextlib.redirect_stderr(tee_stream):
            print(f"[robustness] checkpoint={checkpoint_path}")
            print(f"[robustness] log={log_path}")
            return evaluate_checkpoint(checkpoint_path, args)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: CapsuleClassificationLoss,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    loss_meter = 0.0
    num_batches = 0
    confusion_matrix = torch.zeros((model.num_classes, model.num_classes), dtype=torch.long)

    progress = tqdm(loader, leave=False, dynamic_ncols=True, file=sys.stdout)
    with torch.no_grad():
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
                outputs = model(images)
                loss, loss_stats = criterion(outputs["logits"], outputs["capsule_lengths"], targets)

            predictions = outputs["logits"].argmax(dim=1).detach().cpu()
            update_confusion_matrix(confusion_matrix, targets.detach().cpu(), predictions)
            loss_meter += loss_stats["total_loss"]
            num_batches += 1
            progress.set_postfix(loss=f"{loss_stats['total_loss']:.4f}")

    metrics = compute_metrics(confusion_matrix)
    metrics["loss"] = loss_meter / max(num_batches, 1)
    return metrics


def aggregate_results(clean_metrics: dict[str, float], results: list[dict[str, Any]]) -> dict[str, float]:
    corrupted_results = [result for result in results if result["perturbation"] != "clean"]
    if not corrupted_results:
        return {
            "mean_accuracy": clean_metrics["accuracy"],
            "mean_macro_f1": clean_metrics["macro_f1"],
            "mean_accuracy_drop": 0.0,
            "mean_macro_f1_drop": 0.0,
            "worst_macro_f1": clean_metrics["macro_f1"],
        }

    mean_accuracy = sum(item["metrics"]["accuracy"] for item in corrupted_results) / len(corrupted_results)
    mean_macro_f1 = sum(item["metrics"]["macro_f1"] for item in corrupted_results) / len(corrupted_results)
    mean_accuracy_drop = clean_metrics["accuracy"] - mean_accuracy
    mean_macro_f1_drop = clean_metrics["macro_f1"] - mean_macro_f1
    worst_macro_f1 = min(item["metrics"]["macro_f1"] for item in corrupted_results)

    return {
        "mean_accuracy": mean_accuracy,
        "mean_macro_f1": mean_macro_f1,
        "mean_accuracy_drop": mean_accuracy_drop,
        "mean_macro_f1_drop": mean_macro_f1_drop,
        "worst_macro_f1": worst_macro_f1,
    }


def evaluate_checkpoint(checkpoint_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, args_dict, dataset_info = build_model_from_checkpoint(checkpoint, device)
    dataset_name = str(dataset_info["dataset_name"])
    run_name = checkpoint_path.parent.name

    if args.datasets and dataset_name not in set(args.datasets):
        return {}
    if args.full_only and not is_full_experiment(run_name, args_dict):
        print(f"[robustness] skip ablation run: {run_name}")
        return {}

    criterion = CapsuleClassificationLoss(
        label_smoothing=float(args_dict.get("label_smoothing", 0.0)),
        margin_weight=float(args_dict.get("margin_weight", 0.25)),
    )

    data_root = resolve_path(args.data_root)
    _, samples = scan_test_samples(data_root, dataset_name, max_samples=args.max_samples)
    image_size = int(args_dict.get("image_size", 96))
    print(f"[robustness] run={run_name} dataset={dataset_name} samples={len(samples)}")

    results: list[dict[str, Any]] = []
    clean_metrics: dict[str, float] | None = None

    for perturbation in args.perturbations:
        severities = [0] if perturbation == "clean" else list(args.severities)
        for severity in severities:
            dataset = RobustnessDataset(
                samples=samples,
                image_size=image_size,
                perturbation=perturbation,
                severity=severity,
                seed=args.seed,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
            )
            metrics = evaluate_loader(model, loader, criterion, device, args.amp)
            result = {
                "perturbation": perturbation,
                "severity": severity,
                "metrics": metrics,
            }
            if perturbation == "clean":
                clean_metrics = metrics
                result["accuracy_drop"] = 0.0
                result["macro_f1_drop"] = 0.0
            else:
                assert clean_metrics is not None
                result["accuracy_drop"] = clean_metrics["accuracy"] - metrics["accuracy"]
                result["macro_f1_drop"] = clean_metrics["macro_f1"] - metrics["macro_f1"]

            print(
                f"[robustness] {run_name} {perturbation} severity={severity} | {format_metrics(metrics)}"
            )
            results.append(result)

    assert clean_metrics is not None
    aggregate = aggregate_results(clean_metrics, results)
    summary = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_name,
        "dataset_info": dataset_info,
        "args": make_json_safe(args_dict),
        "clean_metrics": clean_metrics,
        "aggregate": aggregate,
        "results": results,
    }
    return summary


def main() -> None:
    args = parse_args()
    args.data_root = resolve_path(args.data_root)
    args.runs_root = resolve_path(args.runs_root)
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(args)
    summaries: list[dict[str, Any]] = []

    for checkpoint_path in checkpoints:
        run_name = checkpoint_path.parent.name
        log_path = logs_dir / f"{run_name}.log"
        summary = evaluate_with_live_logging(checkpoint_path, args, log_path)
        if not summary:
            continue
        run_name = summary["run_name"]
        summary["log_path"] = str(log_path)
        output_path = args.output_dir / f"{run_name}_robustness.json"
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)

    final_summary = {
        "runs_evaluated": len(summaries),
        "perturbations": args.perturbations,
        "severities": args.severities,
        "summaries": summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(f"[robustness] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()