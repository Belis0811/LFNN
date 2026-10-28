from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torchvision import models


class LocalClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


class LFNNResNet101(nn.Module):
    """ResNet-101 split into four stage-level LFNN-l workers.

    The model preserves four disconnected backward graphs by detaching the input
    to each later worker. Local heads are used only during training. The output
    classifier consumes detached final features and is the inference output.
    """

    worker_names = ("stem_layer1", "layer2", "layer3", "layer4")

    def __init__(
        self,
        *,
        num_classes: int = 1000,
        local_head_hidden_dim: int = 256,
        detach_between_workers: bool = True,
        detach_output_classifier_input: bool = True,
        torchvision_weights: str | None = None,
    ):
        super().__init__()
        weights = resolve_weights("resnet101", torchvision_weights)
        base = models.resnet101(weights=weights)
        if num_classes != 1000:
            base.fc = nn.Linear(base.fc.in_features, num_classes)
        worker0 = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
        )
        self.workers = nn.ModuleList(
            [worker0, base.layer2, base.layer3, base.layer4]
        )
        channels = (256, 512, 1024, 2048)
        self.local_heads = nn.ModuleList(
            [
                LocalClassifier(channel, local_head_hidden_dim, num_classes)
                for channel in channels
            ]
        )
        self.output_pool = base.avgpool
        self.output_classifier = base.fc
        self.detach_between_workers = bool(detach_between_workers)
        self.detach_output_classifier_input = bool(detach_output_classifier_input)
        self._initialize_local_heads()

    def _initialize_local_heads(self) -> None:
        for head in self.local_heads:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        local_logits: list[torch.Tensor] = []
        features = images
        for index, (worker, head) in enumerate(zip(self.workers, self.local_heads)):
            if index > 0 and self.detach_between_workers:
                features = features.detach()
            features = worker(features)
            local_logits.append(head(features))
        output_features = (
            features.detach() if self.detach_output_classifier_input else features
        )
        output_logits = self.output_classifier(self.output_pool(output_features).flatten(1))
        return output_logits, local_logits

    def worker_parameters(self, index: int) -> Iterable[nn.Parameter]:
        yield from self.workers[index].parameters()
        yield from self.local_heads[index].parameters()


def resolve_weights(architecture: str, name: str | None) -> Any:
    if name is None or str(name).lower() in {"none", "null", ""}:
        return None
    enum_by_architecture = {
        "resnet101": models.ResNet101_Weights,
        "resnet152": models.ResNet152_Weights,
        "alexnet": models.AlexNet_Weights,
        "vgg11": models.VGG11_Weights,
        "vgg11_bn": models.VGG11_BN_Weights,
        "vgg13": models.VGG13_Weights,
        "vgg13_bn": models.VGG13_BN_Weights,
        "vit_b_16": models.ViT_B_16_Weights,
    }
    if architecture not in enum_by_architecture:
        raise ValueError(f"No torchvision weights mapping for {architecture}.")
    enum = enum_by_architecture[architecture]
    try:
        return enum[str(name)]
    except KeyError as error:
        allowed = ", ".join(item.name for item in enum)
        raise ValueError(
            f"Unknown weights {name!r} for {architecture}. Available: {allowed}"
        ) from error


def build_bp_model(config: dict[str, Any]) -> nn.Module:
    architecture = str(config["architecture"])
    weights = resolve_weights(architecture, config.get("torchvision_weights"))
    num_classes = int(config.get("num_classes", 1000))
    builders = {
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
        "alexnet": models.alexnet,
        "vgg11": models.vgg11,
        "vgg11_bn": models.vgg11_bn,
        "vgg13": models.vgg13,
        "vgg13_bn": models.vgg13_bn,
    }
    if architecture not in builders:
        raise ValueError(f"Unsupported BP architecture: {architecture}")
    model = builders[architecture](weights=weights)
    if num_classes != 1000:
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            final = model.classifier[-1]
            model.classifier[-1] = nn.Linear(final.in_features, num_classes)
        else:
            raise ValueError(f"Cannot replace classifier for {architecture}.")
    return model
