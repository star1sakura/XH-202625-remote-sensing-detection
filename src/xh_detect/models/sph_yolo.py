from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


class NAMBlock(nn.Module):
    """Shape-preserving normalization-based attention block for SPH-YOLO heads."""

    def __init__(self, channels: int, use_spatial: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        eps = _positive_float(eps, "eps")
        if not isinstance(use_spatial, bool):
            raise TypeError("use_spatial must be a boolean")

        self.channels = channels
        self.use_spatial = use_spatial
        self.eps = eps
        self.channel_norm = nn.BatchNorm2d(channels, affine=True)
        self.spatial_norm = nn.BatchNorm2d(1, affine=True) if use_spatial else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_scale = self.channel_norm.weight.abs()
        channel_scale = channel_scale / (channel_scale.sum() + self.eps)
        channel_gate = torch.sigmoid(
            self.channel_norm(x) * channel_scale.view(1, self.channels, 1, 1) * self.channels
        )
        y = x * channel_gate
        if self.spatial_norm is not None:
            spatial = y.mean(dim=1, keepdim=True)
            y = y * torch.sigmoid(self.spatial_norm(spatial))
        return x + y


class SwinPredictionBlock(nn.Module):
    """Window-attention prediction block that preserves BCHW tensor shape."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        num_heads = _positive_int(num_heads, "num_heads")
        window_size = _positive_int(window_size, "window_size")
        mlp_ratio = _positive_float(mlp_ratio, "mlp_ratio")
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        hidden_channels = max(channels, int(math.ceil(channels * mlp_ratio)))
        self.channels = channels
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )

    def _partition_windows(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        b, c, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        valid_tokens = torch.ones((b, 1, h, w), dtype=torch.bool, device=x.device)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        valid_tokens = F.pad(valid_tokens, (0, pad_w, 0, pad_h), value=False)
        padded_h = h + pad_h
        padded_w = w + pad_w
        x = x.permute(0, 2, 3, 1).contiguous()
        valid_tokens = valid_tokens.permute(0, 2, 3, 1).contiguous()
        windows = x.view(
            b,
            padded_h // self.window_size,
            self.window_size,
            padded_w // self.window_size,
            self.window_size,
            c,
        )
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        key_padding_mask = valid_tokens.view(
            b,
            padded_h // self.window_size,
            self.window_size,
            padded_w // self.window_size,
            self.window_size,
            1,
        )
        key_padding_mask = key_padding_mask.permute(0, 1, 3, 2, 4, 5).contiguous()
        key_padding_mask = ~key_padding_mask.view(-1, self.window_size * self.window_size)
        return (
            windows.view(-1, self.window_size * self.window_size, c),
            key_padding_mask,
            padded_h,
            padded_w,
        )

    def _merge_windows(
        self,
        windows: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        padded_h: int,
        padded_w: int,
    ) -> torch.Tensor:
        x = windows.view(
            batch_size,
            padded_h // self.window_size,
            padded_w // self.window_size,
            self.window_size,
            self.window_size,
            self.channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch_size, padded_h, padded_w, self.channels)
        x = x[:, :height, :width, :]
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        windows, key_padding_mask, padded_h, padded_w = self._partition_windows(x)
        attn_input = self.norm1(windows)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        windows = windows + attn_output
        windows = windows + self.mlp(self.norm2(windows))
        return self._merge_windows(windows, b, h, w, padded_h, padded_w)
