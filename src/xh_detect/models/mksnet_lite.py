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
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 or item % 2 == 0 for item in kernels):
        raise ValueError("kernel sizes must be odd positive integers")
    return kernels


class MKSNetLiteBlock(nn.Module):
    """Channel-preserving multi-kernel spatial/channel attention block."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        reduction: int = 16,
    ) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        reduction = _positive_int(reduction, "reduction")
        kernels = _kernel_sizes(kernel_sizes)
        hidden_channels = max(1, channels // reduction)

        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=channels,
                    bias=False,
                )
                for kernel in kernels
            ]
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU(inplace=True)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = torch.stack([branch(x) for branch in self.branches], dim=0).mean(dim=0)
        fused = self.activation(self.norm(self.project(fused)))
        fused = fused * self.channel_attention(fused)
        avg_pool = fused.mean(dim=1, keepdim=True)
        max_pool = fused.amax(dim=1, keepdim=True)
        spatial_gate = self.spatial_attention(torch.cat((avg_pool, max_pool), dim=1))
        return x + fused * spatial_gate
