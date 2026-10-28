#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torchvision import models

from lfnnexp.checkpoints import extract_state_dict
from lfnnexp.data import build_indexed_validation_loader
from lfnnexp.models import LFNNResNet101, build_bp_model, resolve_weights
from lfnnexp.provenance import sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ImageNet checkpoint.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--val-subdir", default="val")
    parser.add_argument(
        "--architecture",
        required=True,
        choices=(
            "resnet101",
            "resnet152",
            "alexnet",
            "vgg11",
            "vgg11_bn",
            "vgg13",
            "vgg13_bn",
            "vit_b_16",
            "lfnn_resnet101",
            "custom",
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--torchvision-weights")
    parser.add_argument("--state-dict-key", default="auto")
    parser.add_argument(
        "--model-factory",
        help="For --architecture custom: import path MODULE:FUNCTION.",
    )
    parser.add_argument(
        "--model-factory-kwargs-json",
        default="{}",
        help="JSON object passed to the custom model factory.",
    )
    parser.add_argument(
        "--logits-key",
        help="Dictionary key when a custom model returns a dictionary.",
    )
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--resize-size", type=int, required=True)
    parser.add_argument(
        "--interpolation", choices=("bilinear", "bicubic", "nearest"), default="bilinear"
    )
    parser.add_argument(
        "--pos-embed-policy",
        choices=("strict", "interpolate", "reinitialize"),
        default="strict",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-csv")
    return parser.parse_args()


def interpolate_positional_embedding(
    state: dict[str, torch.Tensor], input_size: int, patch_size: int = 16
) -> None:
    key = "encoder.pos_embedding"
    if key not in state:
        raise KeyError(f"State dictionary has no {key!r} tensor.")
    position = state[key]
    if position.ndim != 3 or position.shape[0] != 1:
        raise ValueError(f"Unexpected positional embedding shape: {position.shape}")
    token, image = position[:, :1], position[:, 1:]
    old_side = int(math.sqrt(image.shape[1]))
    if old_side * old_side != image.shape[1]:
        raise ValueError("Image positional tokens do not form a square grid.")
    if input_size % patch_size != 0:
        raise ValueError("Input size must be divisible by ViT patch size.")
    new_side = input_size // patch_size
    image = image.reshape(1, old_side, old_side, position.shape[2]).permute(0, 3, 1, 2)
    image = F.interpolate(
        image,
        size=(new_side, new_side),
        mode="bicubic",
        align_corners=True,
    )
    image = image.permute(0, 2, 3, 1).reshape(1, new_side * new_side, position.shape[2])
    state[key] = torch.cat([token, image], dim=1)


def build_model_and_state(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_object: Any | None = None
    checkpoint_metadata: dict[str, Any] = {}
    if args.checkpoint:
        checkpoint_object = torch.load(args.checkpoint, map_location="cpu")
        state = extract_state_dict(checkpoint_object, args.state_dict_key)
        checkpoint_metadata = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
        }
    else:
        if args.architecture == "custom":
            raise ValueError("Custom models require --checkpoint, not torchvision weights.")
        weights = resolve_weights(args.architecture, args.torchvision_weights)
        state = weights.get_state_dict(progress=True, check_hash=True)
        checkpoint_metadata = {
            "torchvision_weights": args.torchvision_weights,
            "torchvision_weights_enum": repr(weights),
        }

    if args.architecture == "custom":
        if not args.model_factory or ":" not in args.model_factory:
            raise ValueError("Pass --model-factory MODULE:FUNCTION for a custom model.")
        module_name, function_name = args.model_factory.split(":", maxsplit=1)
        factory = getattr(importlib.import_module(module_name), function_name)
        kwargs = json.loads(args.model_factory_kwargs_json)
        if not isinstance(kwargs, dict):
            raise ValueError("--model-factory-kwargs-json must decode to an object.")
        model = factory(**kwargs)
        model.load_state_dict(state, strict=True)
        checkpoint_metadata.update(
            {"model_factory": args.model_factory, "model_factory_kwargs": kwargs}
        )
        return model, checkpoint_metadata

    if args.architecture == "vit_b_16":
        model = models.vit_b_16(weights=None, image_size=args.input_size)
        pos_key = "encoder.pos_embedding"
        expected_shape = model.state_dict()[pos_key].shape
        found_shape = state.get(pos_key, torch.empty(0)).shape
        if args.pos_embed_policy == "strict":
            if expected_shape != found_shape:
                raise ValueError(
                    f"Strict positional embedding mismatch: checkpoint {found_shape}, "
                    f"model {expected_shape}."
                )
        elif args.pos_embed_policy == "interpolate":
            interpolate_positional_embedding(state, args.input_size)
        elif args.pos_embed_policy == "reinitialize":
            state.pop(pos_key, None)
        incompatible = model.load_state_dict(state, strict=False)
        allowed_missing = {pos_key} if args.pos_embed_policy == "reinitialize" else set()
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing != allowed_missing or unexpected:
            raise RuntimeError(
                f"State-dict mismatch. Missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        checkpoint_metadata.update(
            {
                "pos_embed_policy": args.pos_embed_policy,
                "checkpoint_pos_embedding_shape": list(found_shape),
                "model_pos_embedding_shape": list(expected_shape),
                "missing_keys": sorted(missing),
                "unexpected_keys": sorted(unexpected),
            }
        )
        return model, checkpoint_metadata

    if args.architecture == "lfnn_resnet101":
        if not isinstance(checkpoint_object, dict) or "config" not in checkpoint_object:
            raise ValueError(
                "LFNN ResNet evaluation requires a checkpoint produced by this package."
            )
        saved = checkpoint_object["config"]
        model_config = dict(saved["model"])
        model_config["num_classes"] = int(saved["data"]["num_classes"])
        model_config.pop("architecture", None)
        model = LFNNResNet101(**model_config)
        model.load_state_dict(state, strict=True)
        return model, checkpoint_metadata

    if args.torchvision_weights and not args.checkpoint:
        model_config = {
            "architecture": args.architecture,
            "torchvision_weights": args.torchvision_weights,
            "num_classes": 1000,
        }
        model = build_bp_model(model_config)
        return model, checkpoint_metadata
    model_config = {
        "architecture": args.architecture,
        "torchvision_weights": None,
        "num_classes": 1000,
    }
    model = build_bp_model(model_config)
    model.load_state_dict(state, strict=True)
    return model, checkpoint_metadata


def main() -> None:
    args = parse_args()
    if args.amp and not args.device.startswith("cuda"):
        raise ValueError("--amp is supported only on CUDA in this script.")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, source_metadata = build_model_and_state(args)
    model.to(device).eval()
    dataset, loader = build_indexed_validation_loader(
        args.data_root,
        val_subdir=args.val_subdir,
        input_size=args.input_size,
        resize_size=args.resize_size,
        batch_size=args.batch_size,
        workers=args.workers,
        interpolation=args.interpolation,
    )

    prediction_handle = None
    prediction_writer = None
    if args.predictions_csv:
        prediction_path = Path(args.predictions_csv)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_handle = prediction_path.open("w", encoding="utf-8", newline="")
        prediction_writer = csv.writer(prediction_handle)
        prediction_writer.writerow(
            ["sample_index", "relative_path", "target", "top1", "top5"]
        )

    loss_sum = 0.0
    correct1 = 0
    correct5 = 0
    count = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for images, targets, indices in loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    output = model(images)
                    if isinstance(output, dict):
                        if not args.logits_key:
                            raise ValueError(
                                "Model returned a dictionary; pass --logits-key."
                            )
                        logits = output[args.logits_key]
                    elif isinstance(output, (tuple, list)):
                        logits = output[0]
                    else:
                        logits = output
                    loss = F.cross_entropy(logits, targets, reduction="sum")
                top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
                matches = top5.eq(targets[:, None])
                loss_sum += float(loss)
                correct1 += int(matches[:, :1].sum().item())
                correct5 += int(matches.sum().item())
                count += targets.numel()
                if prediction_writer is not None:
                    for row_index, dataset_index in enumerate(indices.tolist()):
                        source_path = Path(dataset.samples[dataset_index][0])
                        relative_path = source_path.relative_to(
                            Path(args.data_root) / args.val_subdir
                        )
                        prediction_writer.writerow(
                            [
                                dataset_index,
                                str(relative_path),
                                int(targets[row_index].item()),
                                int(top5[row_index, 0].item()),
                                ";".join(str(int(x)) for x in top5[row_index].tolist()),
                            ]
                        )
    finally:
        if prediction_handle is not None:
            prediction_handle.close()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "architecture": args.architecture,
        "source": source_metadata,
        "protocol": {
            "data_root": str(Path(args.data_root).resolve()),
            "validation_subdir": args.val_subdir,
            "input_size": args.input_size,
            "resize_size": args.resize_size,
            "interpolation": args.interpolation,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "amp": args.amp,
            "device": str(device),
            "seed": args.seed,
        },
        "metrics": {
            "loss": loss_sum / count,
            "top1": 100.0 * correct1 / count,
            "top5": 100.0 * correct5 / count,
            "count": count,
            "elapsed_seconds": elapsed,
        },
        "predictions_csv": (
            str(Path(args.predictions_csv).resolve()) if args.predictions_csv else None
        ),
    }
    write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
