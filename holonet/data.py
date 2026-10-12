from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CHANNEL_MODES = {"rgba": 4, "rgb": 3, "gray": 1}


class ChannelSelect:
    """Input-channel ablation on the stored RGBA images (applied after normalisation).

    rgb drops the alpha channel (every fourth byte); gray averages the four channels,
    so each pixel averages the same bytes a single-channel encoding at this
    resolution would.
    """

    def __init__(self, mode: str = "rgba") -> None:
        if mode not in CHANNEL_MODES:
            raise ValueError(f"unknown channel mode: {mode}")
        self.mode = mode

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "rgb":
            return x[:3]
        if self.mode == "gray":
            return x.mean(dim=0, keepdim=True)
        return x


@dataclass(slots=True)
class DatasetInfo:
    dataset_name: str
    num_classes: int
    class_names: list[str]
    train_size: int
    val_size: int
    test_size: int
    image_size: int


class MalwareImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: transforms.Compose,
        pre_transform: transforms.Compose | None = None,
        cache: bool = False,
    ) -> None:
        self.samples = samples
        self.transform = transform
        # When cache=True we run the deterministic pre_transform (e.g. Resize) once
        # and keep the small resized RGBA image in RAM; the stochastic transform
        # (augmentation/ToTensor/normalise) still runs every epoch, so outputs are
        # identical to the non-cached path. Eliminates per-epoch disk decode + resize.
        self.cache = cache
        self.pre_transform = pre_transform
        self._cached: list | None = None
        if cache:
            assert pre_transform is not None, "cache requires a pre_transform"
            self._cached = [
                pre_transform(Image.open(path).convert("RGBA")) for path, _ in samples
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        _, label = self.samples[index]
        if self.cache:
            return self.transform(self._cached[index]), label
        image = Image.open(self.samples[index][0]).convert("RGBA")
        return self.transform(image), label


def discover_datasets(root: str | Path) -> list[str]:
    root_path = Path(root)
    return sorted(
        path.name
        for path in root_path.iterdir()
        if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
    )


def _scan_split(
    split_dir: Path,
    class_names: Iterable[str] | None = None,
) -> tuple[list[str], list[tuple[Path, int]]]:
    if class_names is None:
        class_names = sorted(path.name for path in split_dir.iterdir() if path.is_dir())

    discovered = list(class_names)
    class_to_idx = {name: idx for idx, name in enumerate(discovered)}
    samples: list[tuple[Path, int]] = []

    for class_name in discovered:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, class_to_idx[class_name]))

    return discovered, samples


def _stratified_split(
    samples: list[tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    grouped: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)

    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    rng = random.Random(seed)

    for label, label_samples in grouped.items():
        rng.shuffle(label_samples)
        if len(label_samples) == 1:
            train_samples.extend(label_samples)
            continue

        val_count = max(1, round(len(label_samples) * val_ratio))
        val_count = min(val_count, len(label_samples) - 1)
        val_samples.extend(label_samples[:val_count])
        train_samples.extend(label_samples[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def _build_transforms(image_size: int, train: bool, channels: str = "rgba") -> transforms.Compose:
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5))

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=0, translate=(0.03, 0.03))],
                    p=0.25,
                ),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.15, scale=(0.01, 0.05), value="random"),
                ChannelSelect(channels),
            ]
        )

    return transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize, ChannelSelect(channels)]
    )


def _pre_transform(image_size: int) -> transforms.Compose:
    """Deterministic part (Resize) — cached once when cache_images is on."""
    return transforms.Compose([transforms.Resize((image_size, image_size))])


def _post_transform(image_size: int, train: bool, channels: str = "rgba") -> transforms.Compose:
    """Stochastic part — run every epoch; mirrors _build_transforms minus Resize."""
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5))
    if train:
        return transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=0, translate=(0.03, 0.03))],
                    p=0.25,
                ),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.15, scale=(0.01, 0.05), value="random"),
                ChannelSelect(channels),
            ]
        )
    return transforms.Compose([transforms.ToTensor(), normalize, ChannelSelect(channels)])


def _build_sampler(samples: list[tuple[Path, int]]) -> WeightedRandomSampler:
    counts = Counter(label for _, label in samples)
    weights = [1.0 / counts[label] for _, label in samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloaders(
    data_root: str | Path,
    dataset_name: str,
    image_size: int = 96,
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    balanced_sampling: bool = True,
    cache_images: bool = False,
    channels: str = "rgba",
) -> tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    root = Path(data_root) / dataset_name
    train_dir = root / "train"
    test_dir = root / "test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f"dataset split not found under {root}")

    class_names, train_val_samples = _scan_split(train_dir)
    _, test_samples = _scan_split(test_dir, class_names=class_names)

    train_samples, val_samples = _stratified_split(train_val_samples, val_ratio=val_ratio, seed=seed)

    if cache_images:
        # In-RAM cache of resized images; single-process loading avoids per-worker
        # duplication. Stochastic augmentation still applied per epoch (identical math).
        pre = _pre_transform(image_size)
        train_dataset = MalwareImageDataset(train_samples, _post_transform(image_size, True, channels), pre, cache=True)
        val_dataset = MalwareImageDataset(val_samples, _post_transform(image_size, False, channels), pre, cache=True)
        test_dataset = MalwareImageDataset(test_samples, _post_transform(image_size, False, channels), pre, cache=True)
        num_workers = 0
    else:
        train_dataset = MalwareImageDataset(train_samples, _build_transforms(image_size, True, channels))
        val_dataset = MalwareImageDataset(val_samples, _build_transforms(image_size, False, channels))
        test_dataset = MalwareImageDataset(test_samples, _build_transforms(image_size, False, channels))

    sampler = _build_sampler(train_samples) if balanced_sampling else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    info = DatasetInfo(
        dataset_name=dataset_name,
        num_classes=len(class_names),
        class_names=class_names,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        image_size=image_size,
    )
    return train_loader, val_loader, test_loader, info