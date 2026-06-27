from __future__ import annotations

from pathlib import Path

import yaml

from xh_detect.config import PipelineConfig


def test_mksnet_lite_model_yaml_contains_custom_blocks() -> None:
    path = Path("configs/models/xh25-yolo26s-mksnet-lite.yaml")
    model = yaml.safe_load(path.read_text(encoding="utf-8"))
    layers = model["backbone"] + model["head"]

    custom_layers = [layer for layer in layers if layer[2] == "MKSNetLiteBlock"]

    assert model["nc"] == 25
    assert model["scale"] == "s"
    assert len(custom_layers) == 2
    assert custom_layers[0] == [-1, 1, "MKSNetLiteBlock", [128]]
    assert custom_layers[1] == [-1, 1, "MKSNetLiteBlock", [256]]
    assert layers[-1] == [[17, 21, 24], 1, "Detect", ["nc"]]


def test_mksnet_lite_pipeline_config_loads() -> None:
    config = PipelineConfig.from_yaml("configs/xh25-mksnet-lite.yaml")

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.model_path == "runs/train/xh25-mksnet-lite/weights/best.pt"
    assert config.image_size == 1024
    assert config.batch_size == 8
    assert set(config.class_thresholds) == set(range(25))
