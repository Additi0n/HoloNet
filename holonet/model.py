from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def squash(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    squared_norm = tensor.pow(2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + eps)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvKANBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        grid_size: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.local_mixer = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.local_norm = nn.BatchNorm2d(in_channels)
        self.base_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.output_norm = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.dropout = nn.Dropout2d(dropout)

        self.register_buffer("grid", torch.linspace(-2.0, 2.0, grid_size), persistent=False)
        self.bandwidth = nn.Parameter(torch.full((1, in_channels, 1, 1), 0.6))
        self.spline_weight = nn.Parameter(torch.empty(out_channels, in_channels, grid_size))
        nn.init.trunc_normal_(self.spline_weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = self.local_mixer(x)
        local_features = F.gelu(self.local_norm(local_features))

        base = self.base_projection(local_features)
        basis = torch.exp(
            -(
                (
                    local_features.unsqueeze(2) - self.grid.view(1, 1, -1, 1, 1)
                )
                / self.bandwidth.clamp_min(1e-3).unsqueeze(2)
            ).pow(2)
        )
        spline = torch.einsum("bcghw,ocg->bohw", basis, self.spline_weight)
        fused = self.output_norm(base + spline)
        fused = self.dropout(fused)
        return F.gelu(fused + self.shortcut(x))


class PrimaryCapsules(nn.Module):
    def __init__(
        self,
        in_channels: int,
        capsule_dim: int = 16,
        capsule_types: int = 12,
        kernel_size: int = 3,
        stride: int = 2,
        squash_output: bool = True,
    ) -> None:
        super().__init__()
        self.squash_output = squash_output
        self.capsule_dim = capsule_dim
        self.capsule_types = capsule_types
        self.conv = nn.Conv2d(
            in_channels,
            capsule_dim * capsule_types,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(capsule_dim * capsule_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.conv(x))
        batch_size, _, height, width = x.shape
        x = x.view(batch_size, self.capsule_types, self.capsule_dim, height, width)
        x = x.permute(0, 3, 4, 1, 2).reshape(batch_size, height * width * self.capsule_types, self.capsule_dim)
        return squash(x) if self.squash_output else x


class CapsuleTransformerRouter(nn.Module):
    def __init__(
        self,
        num_primary_tokens: int,
        num_classes: int,
        capsule_dim: int,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        feedforward_dim = int(model_dim * mlp_ratio)
        self.num_classes = num_classes
        self.input_projection = nn.Linear(capsule_dim, model_dim)
        self.primary_position = nn.Parameter(torch.zeros(1, num_primary_tokens, model_dim))
        self.class_tokens = nn.Parameter(torch.randn(1, num_classes, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.class_projection = nn.Linear(model_dim, capsule_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(12.0)))

    def forward(self, primary_capsules: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = primary_capsules.size(0)
        primary_tokens = self.input_projection(primary_capsules) + self.primary_position
        class_tokens = self.class_tokens.expand(batch_size, -1, -1)
        tokens = torch.cat([class_tokens, primary_tokens], dim=1)
        encoded = self.encoder(tokens)
        class_capsules = self.class_projection(encoded[:, : self.num_classes])
        class_capsules = squash(class_capsules)
        capsule_lengths = torch.linalg.norm(class_capsules, dim=-1)
        logits = capsule_lengths * self.logit_scale.exp()
        return logits, capsule_lengths


class CapsulePoolingRouter(nn.Module):
    def __init__(
        self,
        num_classes: int,
        capsule_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pool_norm = nn.LayerNorm(capsule_dim)
        self.class_projection = nn.Sequential(
            nn.Linear(capsule_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes * capsule_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, primary_capsules: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_capsule = self.pool_norm(primary_capsules.mean(dim=1))
        class_capsules = self.class_projection(pooled_capsule)
        class_capsules = class_capsules.view(primary_capsules.size(0), self.num_classes, -1)
        class_capsules = squash(class_capsules)
        capsule_lengths = torch.linalg.norm(class_capsules, dim=-1)
        logits = capsule_lengths * self.logit_scale.exp()
        return logits, capsule_lengths


class CapsuleDynamicRouter(nn.Module):
    """Routing-by-agreement (Sabour et al., 2017), used as the iterative-routing reference.

    Transformation matrices are shared across spatial positions and kept per capsule
    type, the usual convolutional-capsule convention; a separate matrix per primary
    capsule would need 432 x K x 16 x 16 weights, more than the rest of the network.
    """

    def __init__(
        self,
        num_classes: int,
        capsule_dim: int,
        capsule_types: int,
        iterations: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.capsule_types = capsule_types
        self.iterations = iterations
        self.transform = nn.Parameter(torch.randn(capsule_types, num_classes, capsule_dim, capsule_dim) * 0.05)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(12.0)))

    def forward(self, primary_capsules: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, capsule_dim = primary_capsules.shape
        # tokens are ordered (h, w, type), see PrimaryCapsules
        u = primary_capsules.view(batch_size, num_tokens // self.capsule_types, self.capsule_types, capsule_dim)
        u_hat = torch.einsum("bptd,tkde->bptke", u, self.transform)
        u_hat = u_hat.reshape(batch_size, num_tokens, self.num_classes, capsule_dim).float()

        routing_logits = u_hat.new_zeros(batch_size, num_tokens, self.num_classes)
        for iteration in range(self.iterations):
            coupling = torch.softmax(routing_logits, dim=2)
            class_capsules = squash((coupling.unsqueeze(-1) * u_hat).sum(dim=1))
            if iteration < self.iterations - 1:
                routing_logits = routing_logits + (u_hat * class_capsules.unsqueeze(1)).sum(dim=-1)

        capsule_lengths = torch.linalg.norm(class_capsules, dim=-1)
        logits = capsule_lengths * self.logit_scale.exp()
        return logits, capsule_lengths


class TokenTransformerHead(nn.Module):
    """No-capsule control: plain tokens, one class token and a linear head.

    Same encoder width and depth as CapsuleTransformerRouter, so the comparison
    isolates the capsule formulation (squash, class capsules, length-as-score).
    """

    def __init__(
        self,
        num_tokens: int,
        num_classes: int,
        token_dim: int,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(token_dim, model_dim)
        self.position = nn.Parameter(torch.zeros(1, num_tokens, model_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=int(model_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_projection(tokens) + self.position
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        logits = self.head(self.norm(self.encoder(x)[:, 0]))
        # no capsule lengths exist here; probabilities stand in so the caller's
        # interface is unchanged (the margin term is disabled for this variant)
        return logits, torch.softmax(logits.float(), dim=1)


ROUTERS = ("transformer", "pooling", "dynamic", "token")


class HoloNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int = 96,
        stem_channels: tuple[int, int, int] = (32, 64, 96),
        primary_capsule_dim: int = 16,
        primary_capsule_types: int = 12,
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        ckan_grid_size: int = 8,
        dropout: float = 0.1,
        use_ckan: bool = True,
        use_transformer_router: bool = True,
        router: str | None = None,
        routing_iterations: int = 3,
        in_channels: int = 4,
    ) -> None:
        super().__init__()
        if router is None:
            router = "transformer" if use_transformer_router else "pooling"
        if router not in ROUTERS:
            raise ValueError(f"unknown router: {router}")
        self.router_type = router
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16 to keep capsule tokenization aligned")

        self.image_size = image_size
        self.num_classes = num_classes
        self.use_ckan = use_ckan
        self.stem1 = ConvBNAct(in_channels, stem_channels[0], stride=2)
        self.stem2 = ConvBNAct(stem_channels[0], stem_channels[1], stride=2)
        self.ckan = ConvKANBlock(stem_channels[1], stem_channels[1], grid_size=ckan_grid_size, dropout=dropout)
        self.fusion = ConvBNAct(stem_channels[1] * 2, stem_channels[1], kernel_size=1, padding=0)
        self.stem3 = ConvBNAct(stem_channels[1], stem_channels[2], stride=2)
        self.context_refine = ConvBNAct(stem_channels[2], stem_channels[2])
        self.primary_capsules = PrimaryCapsules(
            stem_channels[2],
            capsule_dim=primary_capsule_dim,
            capsule_types=primary_capsule_types,
            stride=2,
            squash_output=router != "token",
        )

        capsule_grid = image_size // 16
        num_primary_tokens = capsule_grid * capsule_grid * primary_capsule_types
        if router == "transformer":
            self.router = CapsuleTransformerRouter(
                num_primary_tokens=num_primary_tokens,
                num_classes=num_classes,
                capsule_dim=primary_capsule_dim,
                model_dim=transformer_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout,
            )
        elif router == "pooling":
            self.router = CapsulePoolingRouter(
                num_classes=num_classes,
                capsule_dim=primary_capsule_dim,
                hidden_dim=transformer_dim,
                dropout=dropout,
            )
        elif router == "dynamic":
            self.router = CapsuleDynamicRouter(
                num_classes=num_classes,
                capsule_dim=primary_capsule_dim,
                capsule_types=primary_capsule_types,
                iterations=routing_iterations,
            )
        else:
            self.router = TokenTransformerHead(
                num_tokens=num_primary_tokens,
                num_classes=num_classes,
                token_dim=primary_capsule_dim,
                model_dim=transformer_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem1(x)
        x = self.stem2(x)
        if self.use_ckan:
            fine_grained = self.ckan(x)
            x = self.fusion(torch.cat([x, fine_grained], dim=1))
        x = self.context_refine(self.stem3(x))
        primary_capsules = self.primary_capsules(x)
        logits, capsule_lengths = self.router(primary_capsules)
        return {
            "logits": logits,
            "capsule_lengths": capsule_lengths,
            "primary_capsules": primary_capsules,
        }