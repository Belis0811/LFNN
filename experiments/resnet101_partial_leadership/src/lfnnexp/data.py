from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .distributed import DistributedContext


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data without the duplicate padding used by DDP sampler."""

    def __init__(self, dataset: Any, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.world_size - 1) // self.world_size


def build_transforms(config: dict[str, Any]) -> tuple[Any, Any]:
    input_size = int(config["input_size"])
    resize_size = int(config["resize_size"])
    train_steps: list[Any] = [
        transforms.RandomResizedCrop(
            input_size,
            scale=tuple(float(x) for x in config.get("train_scale", [0.08, 1.0])),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(
            p=float(config.get("horizontal_flip_probability", 0.5))
        ),
    ]
    color_jitter = float(config.get("color_jitter", 0.0))
    if color_jitter > 0:
        train_steps.append(
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            )
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    erasing = float(config.get("random_erasing_probability", 0.0))
    if erasing > 0:
        train_steps.append(transforms.RandomErasing(p=erasing))

    validation = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(train_steps), validation


def _loader_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    workers = int(config["num_workers"])
    values: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(config.get("pin_memory", True)),
        "persistent_workers": bool(config.get("persistent_workers", True))
        and workers > 0,
    }
    if workers > 0:
        values["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return values


def build_imagenet_loaders(
    data_config: dict[str, Any],
    global_batch_size: int,
    context: DistributedContext,
) -> tuple[DataLoader[Any], DataLoader[Any], DistributedSampler[Any] | None]:
    if global_batch_size % context.world_size != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by world size "
            f"{context.world_size}."
        )
    batch_size = global_batch_size // context.world_size
    root = Path(data_config["root"])
    train_root = root / data_config.get("train_subdir", "train")
    val_root = root / data_config.get("val_subdir", "val")
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(
            f"Expected ImageNet directories {train_root} and {val_root}."
        )

    train_transform, val_transform = build_transforms(data_config)
    train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_root, transform=val_transform)
    expected_classes = int(data_config.get("num_classes", 1000))
    if len(train_dataset.classes) != expected_classes:
        raise ValueError(
            f"Training set has {len(train_dataset.classes)} classes; expected "
            f"{expected_classes}."
        )
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Training and validation class-to-index mappings differ.")

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            drop_last=False,
        )
        if context.distributed
        else None
    )
    val_sampler = (
        DistributedEvalSampler(val_dataset, context.rank, context.world_size)
        if context.distributed
        else None
    )
    kwargs = _loader_kwargs(data_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **kwargs,
    )
    return train_loader, val_loader, train_sampler


class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, target = super().__getitem__(index)
        return image, target, index


def build_indexed_validation_loader(
    data_root: str | Path,
    *,
    val_subdir: str,
    input_size: int,
    resize_size: int,
    batch_size: int,
    workers: int,
    interpolation: str = "bilinear",
) -> tuple[IndexedImageFolder, DataLoader[Any]]:
    interpolation_modes = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
    }
    if interpolation not in interpolation_modes:
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    interpolation_mode = interpolation_modes[interpolation]
    transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation_mode),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = IndexedImageFolder(Path(data_root) / val_subdir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return dataset, loader
