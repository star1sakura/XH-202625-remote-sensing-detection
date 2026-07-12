from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from ultralytics import YOLO

from xh_detect.config import PipelineConfig
from xh_detect.models.ultralytics import register_custom_modules


def _load_model_yaml(path: str) -> dict[str, object]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _layers(model: dict[str, object]) -> list[list[object]]:
    return list(model["backbone"]) + list(model["head"])


def test_sph_p2_model_yaml_adds_four_scale_detect() -> None:
    model = _load_model_yaml("configs/models/xh25-yolo26s-sph-p2.yaml")
    layers = _layers(model)

    assert model["nc"] == 25
    assert model["scale"] == "s"
    assert model["end2end"] is True
    assert model["reg_max"] == 1
    assert layers[17] == [-1, 1, "nn.Upsample", [None, 2, "nearest"]]
    assert layers[18] == [[-1, 2], 1, "Concat", [1]]
    assert layers[19] == [-1, 2, "C3k2", [128, True]]
    assert layers[-1] == [[19, 22, 25, 28], 1, "Detect", ["nc"]]


def test_vehicle_expert_sph_p2_model_and_pipeline_are_one_class() -> None:
    model = _load_model_yaml("configs/models/vehicle-yolo26s-sph-p2.yaml")
    config = PipelineConfig.from_yaml("configs/vehicle-expert-sph-p2.yaml")

    assert model["nc"] == 1
    assert _layers(model)[-1] == [[19, 22, 25, 28], 1, "Detect", ["nc"]]
    assert config.taxonomy == "vehicle1"
    assert config.model_path == "runs/train/vehicle-expert-sph-p2/weights/best.pt"
    assert config.class_thresholds == {0: 0.25}


def test_sph_p2_pipeline_config_loads() -> None:
    config = PipelineConfig.from_yaml("configs/xh25-sph-p2.yaml")

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.model_path == "runs/train/xh25-sph-p2/weights/best.pt"
    assert config.device == "0"
    assert config.image_size == 1024
    assert config.tile_size == 1024
    assert config.overlap == 0.2
    assert config.batch_size == 8
    assert config.merge_iou == 0.3
    assert config.edge_margin == 16
    assert config.half is True
    assert set(config.class_thresholds) == set(range(25))
    assert all(threshold == 0.25 for threshold in config.class_thresholds.values())


def test_sph_p2_model_smoke_loads_with_detection_model() -> None:
    register_custom_modules()

    model = YOLO("configs/models/xh25-yolo26s-sph-p2.yaml")

    assert model.model.__class__.__name__ == "DetectionModel"


def test_sph_p2_nam_model_yaml_adds_nam_blocks() -> None:
    model = _load_model_yaml("configs/models/xh25-yolo26s-sph-p2-nam.yaml")
    layers = _layers(model)

    nam_layers = [layer for layer in layers if layer[2] == "NAMBlock"]

    assert len(nam_layers) == 2
    assert layers[20] == [-1, 1, "NAMBlock", [64]]
    assert layers[24] == [-1, 1, "NAMBlock", [128]]
    assert layers[-1] == [[20, 24, 27, 30], 1, "Detect", ["nc"]]


def test_sph_full_model_yaml_adds_swin_prediction_blocks() -> None:
    model = _load_model_yaml("configs/models/xh25-yolo26s-sph-full.yaml")
    layers = _layers(model)

    swin_layers = [layer for layer in layers if layer[2] == "SwinPredictionBlock"]

    assert len(swin_layers) == 4
    assert layers[21] == [-1, 1, "SwinPredictionBlock", [64, 4, 7, 2.0]]
    assert layers[26] == [-1, 1, "SwinPredictionBlock", [128, 4, 7, 2.0]]
    assert layers[30] == [-1, 1, "SwinPredictionBlock", [256, 8, 7, 2.0]]
    assert layers[34] == [-1, 1, "SwinPredictionBlock", [512, 8, 7, 2.0]]
    assert layers[-1] == [[21, 26, 30, 34], 1, "Detect", ["nc"]]


def test_sph_p2_nam_pipeline_config_loads() -> None:
    config = PipelineConfig.from_yaml("configs/xh25-sph-p2-nam.yaml")

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.model_path == "runs/train/xh25-sph-p2-nam/weights/best.pt"
    assert config.device == "0"
    assert config.image_size == 1024
    assert config.tile_size == 1024
    assert config.overlap == 0.2
    assert config.batch_size == 8
    assert config.merge_iou == 0.3
    assert config.edge_margin == 16
    assert config.half is True
    assert set(config.class_thresholds) == set(range(25))
    assert all(threshold == 0.25 for threshold in config.class_thresholds.values())


def test_sph_full_pipeline_config_loads() -> None:
    config = PipelineConfig.from_yaml("configs/xh25-sph-full.yaml")

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.model_path == "runs/train/xh25-sph-full/weights/best.pt"
    assert config.device == "0"
    assert config.image_size == 1024
    assert config.tile_size == 1024
    assert config.overlap == 0.2
    assert config.batch_size == 8
    assert config.merge_iou == 0.3
    assert config.edge_margin == 16
    assert config.half is True
    assert set(config.class_thresholds) == set(range(25))
    assert all(threshold == 0.25 for threshold in config.class_thresholds.values())


@pytest.mark.parametrize(
    "path",
    [
        "configs/models/xh25-yolo26s-sph-p2-nam.yaml",
        "configs/models/xh25-yolo26s-sph-full.yaml",
    ],
)
def test_sph_custom_model_variants_smoke_load_with_detection_model(path: str) -> None:
    register_custom_modules()

    model = YOLO(path)

    assert model.model.__class__.__name__ == "DetectionModel"
