from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _kernel_sizes(value: Sequence[int]) -> tuple[int, ...]:
    kernels = tuple(value)
    if not kernels:
        raise ValueError("kernel_sizes must contain at least one kernel")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0 or item % 2 == 0
        for item in kernels
    ):
        raise ValueError("kernel sizes must be odd positive integers")
    return kernels


def _dilations(value: Sequence[int], *, expected: int) -> tuple[int, ...]:
    dilations = tuple(value)
    if len(dilations) != expected:
        raise ValueError("kernel_sizes and dilations must have the same length")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in dilations):
        raise ValueError("dilations must be positive integers")
    return dilations


def _attention_order(value: str) -> str:
    if value not in {"ca_sa", "sa_ca"}:
        raise ValueError("order must be one of: ca_sa, sa_ca")
    return value


class MKSChannelAttention(nn.Module):
    """SENet-style channel attention with average and maximum pooling paths."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        reduction = _positive_int(reduction, "reduction")
        hidden_channels = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        self.gate = nn.Sigmoid()

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        avg_logits = self.shared_mlp(self.avg_pool(x))
        max_logits = self.shared_mlp(self.max_pool(x))
        return self.gate(0.5 * (avg_logits + max_logits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class MKSSpatialAttention(nn.Module):
    """Multi-kernel spatial attention with adaptive branch selection."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9),
        dilations: Sequence[int] = (1, 1, 2, 2),
        reduction: int = 16,
        branch_reduction: int = 4,
    ) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        reduction = _positive_int(reduction, "reduction")
        branch_reduction = _positive_int(branch_reduction, "branch_reduction")
        kernels = _kernel_sizes(kernel_sizes)
        dilations_tuple = _dilations(dilations, expected=len(kernels))
        branch_channels = max(8, channels // branch_reduction)
        hidden_channels = max(1, channels // reduction)

        self.branch_count = len(kernels)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel,
                        padding=dilation * (kernel // 2),
                        dilation=dilation,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.SiLU(inplace=True),
                )
                for kernel, dilation in zip(kernels, dilations_tuple, strict=True)
            ]
        )
        fused_channels = branch_channels * self.branch_count
        self.selection_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, self.branch_count, kernel_size=1, bias=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def _branch_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [branch(x) for branch in self.branches]

    def selection_weights(self, x: torch.Tensor) -> torch.Tensor:
        features = self._branch_features(x)
        fused = torch.cat(features, dim=1)
        return torch.softmax(self.selection_gate(fused), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._branch_features(x)
        fused = torch.cat(features, dim=1)
        weights = torch.softmax(self.selection_gate(fused), dim=1)
        stacked = torch.stack(features, dim=1)
        selected = (stacked * weights[:, :, None]).sum(dim=1)
        avg_pool = fused.mean(dim=1, keepdim=True)
        max_pool = fused.amax(dim=1, keepdim=True)
        spatial = self.spatial_gate(torch.cat((avg_pool, max_pool), dim=1))
        return self.project(selected * spatial)


class MKSBlock(nn.Module):
    """MKS block combining channel and spatial attention with a residual path."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9),
        dilations: Sequence[int] = (1, 1, 2, 2),
        reduction: int = 16,
        order: str = "ca_sa",
    ) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        self.order = _attention_order(order)
        self.channel_attention = MKSChannelAttention(channels, reduction=reduction)
        self.spatial_attention = MKSSpatialAttention(
            channels,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            reduction=reduction,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == "ca_sa":
            y = self.spatial_attention(self.channel_attention(x))
        else:
            y = self.channel_attention(self.spatial_attention(x))
        return x + y
