from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from xh_detect.mksnet_seed import (
    MAIN_TO_MKS_LAYER_MAP,
    MKS_IDENTITY_LAYER_INDICES,
    initialize_mksnet_lite_from_main,
    interpolate_checkpoints,
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


@patch("xh_detect.mksnet_seed.register_custom_modules")
def test_interpolate_checkpoints_uses_base_model_and_tuned_ema(
    register_custom_modules: Mock,
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.pt"
    tuned_path = tmp_path / "tuned.pt"
    output_path = tmp_path / "merged.pt"
    base = nn.Sequential(nn.Linear(2, 1, bias=False), nn.BatchNorm1d(1))
    tuned = nn.Sequential(nn.Linear(2, 1, bias=False), nn.BatchNorm1d(1))
    nn.init.constant_(base[0].weight, 0.0)
    nn.init.constant_(tuned[0].weight, 2.0)
    tuned[1].num_batches_tracked.fill_(7)
    torch.save({"model": base}, base_path)
    torch.save({"model": None, "ema": tuned}, tuned_path)

    result = interpolate_checkpoints(base_path, tuned_path, output_path, 0.5)

    register_custom_modules.assert_called_once_with()
    payload = torch.load(output_path, map_location="cpu", weights_only=False)
    assert torch.equal(payload["model"][0].weight.float(), torch.ones(1, 2))
    assert payload["model"][1].num_batches_tracked.item() == 7
    assert payload["ema"] is None
    assert payload["interpolation"]["alpha"] == 0.5
    assert result.state_tensors == len(base.state_dict())


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), True])
def test_interpolate_checkpoints_rejects_invalid_alpha(alpha: object, tmp_path: Path) -> None:
    base = tmp_path / "base.pt"
    tuned = tmp_path / "tuned.pt"
    base.write_bytes(b"base")
    tuned.write_bytes(b"tuned")

    with pytest.raises(ValueError, match="alpha"):
        interpolate_checkpoints(base, tuned, tmp_path / "out.pt", alpha)  # type: ignore[arg-type]
