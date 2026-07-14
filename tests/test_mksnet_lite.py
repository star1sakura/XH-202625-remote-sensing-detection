from __future__ import annotations

import pytest
import torch

from xh_detect.models.mksnet_lite import MKSNetLiteBlock
from xh_detect.models.ultralytics import register_custom_modules


def test_mksnet_lite_block_preserves_shape_and_allows_gradients() -> None:
    torch.manual_seed(7)
    block = MKSNetLiteBlock(16, kernel_sizes=(3, 5, 7), reduction=4)
    x = torch.randn(2, 16, 24, 32, requires_grad=True)

    y = block(x)
    loss = y.mean()
    loss.backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_mksnet_lite_block_supports_single_kernel() -> None:
    block = MKSNetLiteBlock(8, kernel_sizes=(3,), reduction=4)
    x = torch.randn(1, 8, 10, 10)

    assert block(x).shape == x.shape


@pytest.mark.parametrize(
    ("channels", "kernel_sizes", "reduction", "message"),
    [
        (0, (3, 5), 16, "channels must be a positive integer"),
        (8, (), 16, "kernel_sizes must contain at least one kernel"),
        (8, (2, 3), 16, "kernel sizes must be odd positive integers"),
        (8, (3,), 0, "reduction must be a positive integer"),
    ],
)
def test_mksnet_lite_block_validates_arguments(
    channels: int,
    kernel_sizes: tuple[int, ...],
    reduction: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MKSNetLiteBlock(channels, kernel_sizes=kernel_sizes, reduction=reduction)


def test_register_custom_modules_exposes_block_to_ultralytics() -> None:
    import ultralytics.nn.tasks as tasks

    original = getattr(tasks, "MKSNetLiteBlock", None)
    if hasattr(tasks, "MKSNetLiteBlock"):
        delattr(tasks, "MKSNetLiteBlock")
    try:
        register_custom_modules()

        assert tasks.MKSNetLiteBlock is MKSNetLiteBlock
    finally:
        if original is not None:
            tasks.MKSNetLiteBlock = original
        elif hasattr(tasks, "MKSNetLiteBlock"):
            delattr(tasks, "MKSNetLiteBlock")
