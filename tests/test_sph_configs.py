from __future__ import annotations

from pathlib import Path

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


def test_sph_p2_pipeline_config_loads() -> None:
    config = PipelineConfig.from_yaml("configs/xh25-sph-p2.yaml")

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.model_path == "runs/train/xh25-sph-p2/weights/best.pt"
    assert config.image_size == 1024
    assert config.batch_size == 8
    assert set(config.class_thresholds) == set(range(25))


def test_sph_p2_model_smoke_loads_with_detection_model() -> None:
    register_custom_modules()

    model = YOLO("configs/models/xh25-yolo26s-sph-p2.yaml")

    assert model.model.__class__.__name__ == "DetectionModel"
