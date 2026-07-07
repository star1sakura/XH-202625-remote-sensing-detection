from __future__ import annotations

import pytest
import torch

from xh_detect.models.ultralytics import register_custom_modules
from xh_detect.models.mksnet_v2 import (
    MKSBlock,
    MKSChannelAttention,
    MKSNetBackbone,
    MKSStage,
    MKSSpatialAttention,
)


def test_channel_attention_preserves_shape_and_uses_avg_and_max_paths() -> None:
    torch.manual_seed(7)
    module = MKSChannelAttention(16, reduction=4)
    x = torch.randn(2, 16, 12, 14, requires_grad=True)

    y = module(x)
    y.mean().backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert module.avg_pool.output_size == 1
    assert module.max_pool.output_size == 1


def test_spatial_attention_selection_weights_sum_to_one() -> None:
    torch.manual_seed(7)
    module = MKSSpatialAttention(
        24,
        kernel_sizes=(3, 5, 7),
        dilations=(1, 1, 2),
        reduction=4,
        branch_reduction=2,
    )
    x = torch.randn(2, 24, 16, 16)

    weights = module.selection_weights(x)
    y = module(x)

    assert weights.shape == (2, 3, 1, 1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 1, 1), atol=1e-6)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mks_block_supports_both_attention_orders() -> None:
    x = torch.randn(1, 16, 10, 10)

    ca_sa = MKSBlock(16, kernel_sizes=(3, 5), dilations=(1, 2), reduction=4, order="ca_sa")
    sa_ca = MKSBlock(16, kernel_sizes=(3, 5), dilations=(1, 2), reduction=4, order="sa_ca")

    assert ca_sa(x).shape == x.shape
    assert sa_ca(x).shape == x.shape


def test_mks_stage_preserves_shape_and_depth() -> None:
    stage = MKSStage(16, depth=2, kernel_sizes=(3, 5), dilations=(1, 2), reduction=4)
    x = torch.randn(1, 16, 12, 12)

    y = stage(x)

    assert y.shape == x.shape
    assert len(stage.blocks) == 2


def test_mksnet_backbone_returns_p3_p4_p5_feature_maps() -> None:
    backbone = MKSNetBackbone(
        channels=(16, 32, 64, 128, 192),
        depths=(1, 1, 1, 1),
        kernel_sizes=(3, 5),
        dilations=(1, 2),
        reduction=4,
    )
    x = torch.randn(1, 3, 128, 128)

    p3, p4, p5 = backbone(x)

    assert p3.shape == (1, 64, 16, 16)
    assert p4.shape == (1, 128, 8, 8)
    assert p5.shape == (1, 192, 4, 4)


def test_mksnet_backbone_validates_channel_and_depth_lengths() -> None:
    with pytest.raises(ValueError, match="channels must contain five stage widths"):
        MKSNetBackbone(channels=(16, 32, 64, 128))
    with pytest.raises(ValueError, match="depths must contain four stage depths"):
        MKSNetBackbone(depths=(1, 1, 1))


def test_register_custom_modules_exposes_mksnet_v2_to_ultralytics() -> None:
    import ultralytics.nn.tasks as tasks

    names = ("MKSChannelAttention", "MKSSpatialAttention", "MKSBlock", "MKSStage")
    originals = {name: getattr(tasks, name, None) for name in names}
    for name in names:
        if hasattr(tasks, name):
            delattr(tasks, name)
    try:
        register_custom_modules()

        assert tasks.MKSChannelAttention is MKSChannelAttention
        assert tasks.MKSSpatialAttention is MKSSpatialAttention
        assert tasks.MKSBlock is MKSBlock
        assert tasks.MKSStage is MKSStage
    finally:
        for name, original in originals.items():
            if original is not None:
                setattr(tasks, name, original)
            elif hasattr(tasks, name):
                delattr(tasks, name)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: MKSChannelAttention(0), "channels must be a positive integer"),
        (lambda: MKSChannelAttention(8, reduction=0), "reduction must be a positive integer"),
        (
            lambda: MKSSpatialAttention(8, kernel_sizes=(3,), dilations=(1, 2)),
            "kernel_sizes and dilations must have the same length",
        ),
        (
            lambda: MKSSpatialAttention(8, kernel_sizes=(2,), dilations=(1,)),
            "kernel sizes must be odd positive integers",
        ),
        (
            lambda: MKSSpatialAttention(8, kernel_sizes=(3,), dilations=(0,)),
            "dilations must be positive integers",
        ),
        (lambda: MKSBlock(8, order="bad"), "order must be one of: ca_sa, sa_ca"),
    ],
)
def test_mksnet_v2_validates_arguments(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
