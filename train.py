from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from holonet import CapsuleClassificationLoss, HoloNet, build_dataloaders, discover_datasets
from holonet.baselines import BASELINES, BaselineClassifier
from holonet.metrics import compute_metrics, format_metrics, update_confusion_matrix
from holonet.data import CHANNEL_MODES
from holonet.model import ROUTERS


def str2bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest_name = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest_name, action="store_true", default=default, help=help_text)
    parser.add_argument(
        f"--disable-{name}",
        dest=dest_name,
        action="store_false",
        help=f"Disable: {help_text.lower()}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HoloNet on RGBA malware image datasets")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    parser.add_argument("--data-root", type=Path, default=Path("RGBA"))
    parser.add_argument("--dataset", type=str, default="BODMAS")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="holonet")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stem-channels", type=int, nargs=3, default=(32, 64, 96))
    parser.add_argument("--primary-capsule-dim", type=int, default=16)
    parser.add_argument("--primary-capsule-types", type=int, default=12)
    parser.add_argument("--transformer-dim", type=int, default=128)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--ckan-grid-size", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--arch", type=str, default="holonet", choices=("holonet", *BASELINES))
    parser.add_argument(
        "--router",
        type=str,
        default=None,
        choices=ROUTERS,
        help="Routing head; overrides --use-transformer-router when given.",
    )
    parser.add_argument("--routing-iterations", type=int, default=3)
    parser.add_argument(
        "--channels",
        type=str,
        default="rgba",
        choices=tuple(CHANNEL_MODES),
        help="Input-channel ablation: rgba (default), rgb (drop alpha) or gray (mean of the four).",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    add_bool_flag(parser, "cache-images", False, "Cache resized images in RAM (forces num_workers=0); result-identical, ~10x faster.")
    add_bool_flag(parser, "balanced-sampling", True, "Use weighted sampling for imbalanced datasets.")
    add_bool_flag(parser, "amp", True, "Enable automatic mixed precision on CUDA.")
    add_bool_flag(parser, "use-ckan", True, "Enable the Conv-KAN fine-grained feature block.")
    add_bool_flag(parser, "use-transformer-router", True, "Enable transformer routing instead of pooling router.")
    add_bool_flag(parser, "save-last", True, "Save last checkpoint after every epoch.")
    parsed_args = parser.parse_args()
    return apply_config_overrides(parser, parsed_args)


def apply_config_overrides(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args

    config_data = json.loads(args.config.read_text(encoding="utf-8"))
    config_values = config_data.get("train_args", config_data)
    defaults = parser.parse_args([])

    for key, value in config_values.items():
        arg_key = key.replace("-", "_")
        if not hasattr(args, arg_key):
            raise ValueError(f"unknown config key: {key}")
        current_value = getattr(args, arg_key)
        default_value = getattr(defaults, arg_key)
        if current_value == default_value:
            if isinstance(default_value, Path):
                value = Path(value)
            setattr(args, arg_key, value)
    return args


def disable_margin_without_capsules(args: argparse.Namespace) -> None:
    if (args.arch != "holonet" or args.router == "token") and args.margin_weight > 0:
        args.margin_weight = 0.0


def save_atomic(obj: dict, path: Path) -> None:
    # write then rename, so a run stopped mid-save never leaves a truncated checkpoint
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # fixed 96 x 96 inputs, so autotuned convolution kernels pay off
    torch.backends.cudnn.benchmark = True


def build_model(num_classes: int, args: argparse.Namespace) -> nn.Module:
    if args.arch != "holonet":
        return BaselineClassifier(
            args.arch, num_classes=num_classes, image_size=args.image_size, in_channels=CHANNEL_MODES[args.channels]
        )
    return HoloNet(
        num_classes=num_classes,
        image_size=args.image_size,
        stem_channels=tuple(args.stem_channels),
        primary_capsule_dim=args.primary_capsule_dim,
        primary_capsule_types=args.primary_capsule_types,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        ckan_grid_size=args.ckan_grid_size,
        dropout=args.dropout,
        use_ckan=args.use_ckan,
        use_transformer_router=args.use_transformer_router,
        router=args.router,
        routing_iterations=args.routing_iterations,
        in_channels=CHANNEL_MODES[args.channels],
    )


def namespace_to_serializable_dict(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def resolve_run_name(args: argparse.Namespace, dataset_name: str) -> str:
    if args.run_name:
        return args.run_name

    components = [args.experiment_name, dataset_name.lower()]
    if not args.use_ckan:
        components.append("no_ckan")
    if not args.use_transformer_router:
        components.append("no_transformer")
    if args.margin_weight <= 0.0:
        components.append("no_margin")
    return "_".join(components)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    device: torch.device,
) -> tuple[int, float, int, list[dict[str, object]]]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    if "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    if "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])
    start_epoch = int(state.get("epoch", 0)) + 1
    best_val_f1 = float(state.get("best_val_f1", -1.0))
    best_epoch = int(state.get("best_epoch", -1))
    history = list(state.get("history", []))
    return start_epoch, best_val_f1, best_epoch, history


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: CapsuleClassificationLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: GradScaler,
    amp_enabled: bool,
) -> tuple[dict[str, float], float]:
    is_training = optimizer is not None
    model.train(is_training)

    loss_meter = 0.0
    num_batches = 0
    num_classes = model.num_classes
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    progress = tqdm(loader, leave=False, dynamic_ncols=True)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
            outputs = model(images)
            loss, loss_stats = criterion(outputs["logits"], outputs["capsule_lengths"], targets)

        if is_training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        predictions = outputs["logits"].argmax(dim=1).detach().cpu()
        update_confusion_matrix(confusion_matrix, targets.detach().cpu(), predictions)

        loss_meter += loss_stats["total_loss"]
        num_batches += 1
        progress.set_postfix(loss=f"{loss_stats['total_loss']:.4f}")

    metrics = compute_metrics(confusion_matrix)
    metrics["loss"] = loss_meter / max(num_batches, 1)
    return metrics, metrics["macro_f1"]


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: CapsuleClassificationLoss,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    scaler = GradScaler(enabled=False)
    with torch.no_grad():
        metrics, _ = run_epoch(model, loader, criterion, None, device, scaler, amp_enabled)
    return metrics


def train_single_dataset(dataset_name: str, args: argparse.Namespace) -> dict[str, object]:
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(
        data_root=args.data_root,
        dataset_name=dataset_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        balanced_sampling=args.balanced_sampling,
        cache_images=args.cache_images,
        channels=args.channels,
    )

    device = torch.device(args.device)
    model = build_model(num_classes=dataset_info.num_classes, args=args).to(device)
    criterion = CapsuleClassificationLoss(
        label_smoothing=args.label_smoothing,
        margin_weight=args.margin_weight,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    run_name = resolve_run_name(args, dataset_name)
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"
    history_path = run_dir / "history.json"
    config_path = run_dir / "resolved_args.json"
    config_path.write_text(json.dumps(namespace_to_serializable_dict(args), indent=2), encoding="utf-8")

    best_val_f1 = -1.0
    best_epoch = -1
    history: list[dict[str, object]] = []
    start_epoch = 1

    if args.resume is not None:
        start_epoch, best_val_f1, best_epoch, history = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(f"[{dataset_name}] resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        train_metrics, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=args.amp,
        )
        val_metrics = evaluate(model=model, loader=val_loader, criterion=criterion, device=device, amp_enabled=args.amp)
        scheduler.step()

        epoch_summary = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "seconds": time.time() - start_time,
        }
        history.append(epoch_summary)

        print(
            f"[{dataset_name}] epoch {epoch:03d} | "
            f"train: {format_metrics(train_metrics)} | "
            f"val: {format_metrics(val_metrics)}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            save_atomic(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "dataset_info": asdict(dataset_info),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "best_val_f1": best_val_f1,
                    "best_epoch": best_epoch,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "history": history,
                },
                checkpoint_path,
            )

        if args.save_last:
            save_atomic(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "dataset_info": asdict(dataset_info),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "best_val_f1": best_val_f1,
                    "best_epoch": best_epoch,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "history": history,
                },
                last_checkpoint_path,
            )

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_metrics = evaluate(model=model, loader=test_loader, criterion=criterion, device=device, amp_enabled=args.amp)

    summary = {
        "run_name": run_name,
        "experiment_name": args.experiment_name,
        "dataset": dataset_name,
        "dataset_info": asdict(dataset_info),
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "args": namespace_to_serializable_dict(args),
        "history": history,
    }
    history_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{dataset_name}] test: {format_metrics(test_metrics)}")
    return summary


def main() -> None:
    args = parse_args()
    disable_margin_without_capsules(args)
    seed_everything(args.seed)

    if args.all_datasets:
        dataset_names = discover_datasets(args.data_root)
    else:
        dataset_names = [args.dataset]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = [train_single_dataset(dataset_name, args) for dataset_name in dataset_names]
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()