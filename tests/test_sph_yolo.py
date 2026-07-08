from __future__ import annotations

import pytest
import torch
import ultralytics.nn.tasks as tasks

from xh_detect.models.sph_yolo import NAMBlock, SwinPredictionBlock
from xh_detect.models.ultralytics import register_custom_modules


def test_nam_block_preserves_shape_and_allows_gradients() -> None:
    block = NAMBlock(16)
    x = torch.randn(2, 16, 12, 10, requires_grad=True)

    y = block(x)
    y.mean().backward()

    assert y.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(y).all()


def test_nam_block_can_disable_spatial_attention() -> None:
    block = NAMBlock(8, use_spatial=False)
    x = torch.randn(1, 8, 6, 6)

    y = block(x)

    assert y.shape == x.shape


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"channels": 0}, "channels must be a positive integer"),
        ({"channels": True}, "channels must be a positive integer"),
        ({"channels": 8, "eps": 0.0}, "eps must be positive"),
    ],
)
def test_nam_block_validates_arguments(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        NAMBlock(**kwargs)


def test_swin_prediction_block_preserves_shape_when_padding_is_needed() -> None:
    block = SwinPredictionBlock(24, num_heads=4, window_size=7, mlp_ratio=2.0)
    x = torch.randn(2, 24, 15, 13, requires_grad=True)

    y = block(x)
    y.sum().backward()

    assert y.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"channels": 0}, "channels must be a positive integer"),
        ({"channels": 16, "num_heads": 0}, "num_heads must be a positive integer"),
        ({"channels": 16, "num_heads": 3}, "channels must be divisible by num_heads"),
        ({"channels": 16, "window_size": 0}, "window_size must be a positive integer"),
        ({"channels": 16, "mlp_ratio": 0.0}, "mlp_ratio must be positive"),
    ],
)
def test_swin_prediction_block_validates_arguments(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        SwinPredictionBlock(**kwargs)


def test_register_custom_modules_exposes_sph_blocks_to_ultralytics() -> None:
    original_nam = getattr(tasks, "NAMBlock", None)
    original_swin = getattr(tasks, "SwinPredictionBlock", None)
    try:
        if hasattr(tasks, "NAMBlock"):
            delattr(tasks, "NAMBlock")
        if hasattr(tasks, "SwinPredictionBlock"):
            delattr(tasks, "SwinPredictionBlock")

        register_custom_modules()

        assert tasks.NAMBlock is NAMBlock
        assert tasks.SwinPredictionBlock is SwinPredictionBlock
    finally:
        if original_nam is not None:
            tasks.NAMBlock = original_nam
        elif hasattr(tasks, "NAMBlock"):
            delattr(tasks, "NAMBlock")
        if original_swin is not None:
            tasks.SwinPredictionBlock = original_swin
        elif hasattr(tasks, "SwinPredictionBlock"):
            delattr(tasks, "SwinPredictionBlock")
