"""Image baselines trained under the same protocol as HoloNet.

Every model takes the 4-channel 96 x 96 Bin2Img input and is trained from scratch
(no ImageNet weights) with the same splits, augmentation, optimiser, schedule and
checkpoint rule, so differences come from the architecture alone.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from torch import nn
from torchvision import models

BASELINES = ("res2net50", "resnext50", "resnext_cbam", "convnext_tiny", "vit_s8")

# official Res2Net source (non-commercial licence), kept outside this repository
RES2NET_SOURCE = Path(
    os.environ.get(
        "RES2NET_SOURCE",
        r"D:\workspace\Res2Net\external\Res2Net-PretrainedModels\res2net_v1b.py",
    )
)


def _adapt_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True))
        return x * torch.sigmoid(weight)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(stats))


class CBAM(nn.Module):
    """Woo et al., 2018; applied to the residual branch before the addition."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


def _res2net50(num_classes: int, in_channels: int) -> nn.Module:
    spec = importlib.util.spec_from_file_location("res2net_v1b", RES2NET_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.res2net50_v1b_26w_4s(pretrained=False)
    model.conv1[0] = _adapt_conv(model.conv1[0], in_channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _resnext50(num_classes: int, in_channels: int, attention: bool) -> nn.Module:
    model = models.resnext50_32x4d(weights=None)
    model.conv1 = _adapt_conv(model.conv1, in_channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if attention:
        # ResNeXt+ (He et al.) = ResNeXt with plug-in attention in the blocks
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
            for block in layer:
                block.bn3 = nn.Sequential(block.bn3, CBAM(block.bn3.num_features))
    return model


def _convnext_tiny(num_classes: int, in_channels: int) -> nn.Module:
    model = models.convnext_tiny(weights=None)
    model.features[0][0] = _adapt_conv(model.features[0][0], in_channels)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def _vit_s8(num_classes: int, in_channels: int, image_size: int) -> nn.Module:
    # ViT-Small width/depth with 8 x 8 patches: 144 tokens on a 96 x 96 input
    model = models.VisionTransformer(
        image_size=image_size,
        patch_size=8,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
        num_classes=num_classes,
    )
    model.conv_proj = nn.Conv2d(in_channels, 384, kernel_size=8, stride=8)
    return model


class BaselineClassifier(nn.Module):
    """Wraps a backbone so it returns the dict the HoloNet training loop expects."""

    def __init__(self, name: str, num_classes: int, image_size: int = 96, in_channels: int = 4) -> None:
        super().__init__()
        if name not in BASELINES:
            raise ValueError(f"unknown baseline: {name}")
        self.name = name
        self.num_classes = num_classes
        if name == "res2net50":
            self.backbone = _res2net50(num_classes, in_channels)
        elif name == "resnext50":
            self.backbone = _resnext50(num_classes, in_channels, attention=False)
        elif name == "resnext_cbam":
            self.backbone = _resnext50(num_classes, in_channels, attention=True)
        elif name == "convnext_tiny":
            self.backbone = _convnext_tiny(num_classes, in_channels)
        else:
            self.backbone = _vit_s8(num_classes, in_channels, image_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.backbone(x)
        # probabilities stand in for capsule lengths; the margin term is off for baselines
        return {"logits": logits, "capsule_lengths": torch.softmax(logits.float(), dim=1)}
