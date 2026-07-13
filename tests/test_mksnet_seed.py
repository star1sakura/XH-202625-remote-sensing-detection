from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch import nn

from xh_detect.mksnet_seed import (
    MAIN_TO_MKS_LAYER_MAP,
    MKS_IDENTITY_LAYER_INDICES,
    initialize_mksnet_lite_from_main,
)
from xh_detect.models.mksnet_lite import MKSNetLiteBlock


class _FakeModel:
    def __init__(self, layers: nn.ModuleList) -> None:
        self.model = layers
        self.names = {0: "ship"}


class _FakeYOLO:
    source: _FakeModel
    target: _FakeModel

    def __init__(self, path: str) -> None:
        self.model = self.source if path.endswith("main.pt") else self.target

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"single-mksnet-checkpoint")


def _conv(value: float) -> nn.Conv2d:
    layer = nn.Conv2d(4, 4, kernel_size=1, bias=False)
    nn.init.constant_(layer.weight, value)
    return layer


@patch("xh_detect.mksnet_seed.register_custom_modules")
@patch("xh_detect.mksnet_seed.YOLO", _FakeYOLO)
def test_initialize_mksnet_lite_from_main_maps_all_shared_layers(
    register_custom_modules: Mock,
    tmp_path: Path,
) -> None:
    main_checkpoint = tmp_path / "main.pt"
    model_yaml = tmp_path / "mks.yaml"
    output_checkpoint = tmp_path / "seeded.pt"
    main_checkpoint.write_bytes(b"main")
    model_yaml.write_text("model", encoding="utf-8")
    source_layers = nn.ModuleList([_conv(index + 1.0) for index in range(24)])
    target_items: list[nn.Module] = [_conv(0.0) for _ in range(26)]
    for index in MKS_IDENTITY_LAYER_INDICES:
        target_items[index] = MKSNetLiteBlock(4)
    target_layers = nn.ModuleList(target_items)
    _FakeYOLO.source = _FakeModel(source_layers)
    _FakeYOLO.target = _FakeModel(target_layers)

    result = initialize_mksnet_lite_from_main(
        main_checkpoint,
        model_yaml,
        output_checkpoint,
    )

    register_custom_modules.assert_called_once_with()
    assert result.transferred_layers == 24
    assert result.identity_blocks == (17, 21)
    assert output_checkpoint.read_bytes() == b"single-mksnet-checkpoint"
    for target_index, source_index in MAIN_TO_MKS_LAYER_MAP.items():
        assert torch.equal(
            target_layers[target_index].weight,
            source_layers[source_index].weight,
        )
    for index in MKS_IDENTITY_LAYER_INDICES:
        block = target_layers[index]
        assert isinstance(block, MKSNetLiteBlock)
        assert torch.count_nonzero(block.norm.weight) == 0
        assert torch.count_nonzero(block.norm.bias) == 0


def test_initialize_mksnet_lite_from_main_refuses_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "main.pt"
    model_yaml = tmp_path / "mks.yaml"
    output = tmp_path / "seeded.pt"
    for path in (source, model_yaml, output):
        path.write_bytes(b"x")

    try:
        initialize_mksnet_lite_from_main(source, model_yaml, output)
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("existing output must be rejected")
