from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from xh_detect.training import export_tensorrt, train_model


@patch("xh_detect.training.YOLO")
def test_train_model_passes_reproducible_arguments(yolo_class: Mock) -> None:
    model = yolo_class.return_value

    train_model("dataset.yaml", "yolo26s-obb.pt", epochs=2, image_size=1024, device="0")

    yolo_class.assert_called_once_with("yolo26s-obb.pt")
    model.train.assert_called_once_with(
        data="dataset.yaml",
        epochs=2,
        imgsz=1024,
        device="0",
        batch=8,
        workers=4,
        amp=False,
        seed=42,
        deterministic=True,
        project="runs/train",
        name="xh25-baseline",
        exist_ok=True,
        resume=False,
    )


@patch("xh_detect.training.YOLO")
def test_train_model_passes_official_baseline_options(yolo_class: Mock) -> None:
    model = yolo_class.return_value

    train_model(
        "datasets/xh25/dataset.yaml",
        "yolo26s.pt",
        epochs=1,
        image_size=1024,
        device="0",
        batch=8,
        workers=4,
        amp=False,
        project="runs/train",
        name="xh25-baseline",
        resume=False,
    )

    model.train.assert_called_once_with(
        data="datasets/xh25/dataset.yaml",
        epochs=1,
        imgsz=1024,
        device="0",
        batch=8,
        workers=4,
        amp=False,
        seed=42,
        deterministic=True,
        project="runs/train",
        name="xh25-baseline",
        exist_ok=True,
        resume=False,
    )


@patch("xh_detect.training.YOLO")
def test_export_tensorrt_returns_exported_path(yolo_class: Mock) -> None:
    model = yolo_class.return_value
    model.export.return_value = "runs/model.engine"

    result = export_tensorrt("best.pt", image_size=1024, device="0")

    assert result == "runs/model.engine"
    model.export.assert_called_once_with(
        format="engine",
        imgsz=1024,
        half=True,
        device="0",
        batch=1,
    )


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (train_model, ("", "model.pt", 1, 640, "cpu")),
        (train_model, ("data.yaml", "", 1, 640, "cpu")),
        (train_model, ("data.yaml", "model.pt", 0, 640, "cpu")),
        (train_model, ("data.yaml", "model.pt", 1, 0, "cpu")),
        (export_tensorrt, ("", 640, "cpu")),
        (export_tensorrt, ("model.pt", 0, "cpu")),
    ],
)
def test_training_wrappers_validate_arguments(function, args) -> None:
    with pytest.raises((TypeError, ValueError)):
        function(*args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch": 0},
        {"workers": -1},
        {"amp": "false"},
        {"project": ""},
        {"name": ""},
        {"resume": "false"},
    ],
)
def test_train_model_validates_reproducible_options(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        train_model("data.yaml", "model.pt", 1, 640, "cpu", **kwargs)
