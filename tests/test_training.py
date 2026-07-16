from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from xh_detect.training import export_tensorrt, train_model


@patch("xh_detect.training.YOLO")
def test_train_model_passes_reproducible_arguments(yolo_class: Mock) -> None:
    model = yolo_class.return_value
    expected_project = str((Path.cwd() / "runs/train").resolve())

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
        project=expected_project,
        name="xh25-baseline",
        exist_ok=True,
        resume=False,
    )


@patch("xh_detect.training.YOLO")
def test_train_model_forwards_custom_seed(yolo_class: Mock) -> None:
    train_model(
        "dataset.yaml",
        "yolo26s.pt",
        epochs=1,
        image_size=1024,
        device="0",
        seed=44,
    )

    assert yolo_class.return_value.train.call_args.kwargs["seed"] == 44
    assert yolo_class.return_value.train.call_args.kwargs["deterministic"] is True


@pytest.mark.parametrize("seed", [-1, True])
def test_train_model_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises((TypeError, ValueError), match="seed must be a non-negative integer"):
        train_model(
            "dataset.yaml",
            "yolo26s.pt",
            epochs=1,
            image_size=1024,
            device="0",
            seed=seed,  # type: ignore[arg-type]
        )


@patch("xh_detect.training.YOLO")
def test_train_model_resolves_relative_project_against_cwd(
    yolo_class: Mock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    model = yolo_class.return_value

    train_model("dataset.yaml", "yolo26s-obb.pt", epochs=2, image_size=1024, device="0")

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
        project=str((tmp_path / "runs/train").resolve()),
        name="xh25-baseline",
        exist_ok=True,
        resume=False,
    )


@patch("xh_detect.training.YOLO")
def test_train_model_preserves_absolute_project_path(
    yolo_class: Mock,
    tmp_path: Path,
) -> None:
    model = yolo_class.return_value
    project = (tmp_path / "custom-runs").resolve()

    train_model(
        "dataset.yaml",
        "yolo26s-obb.pt",
        epochs=2,
        image_size=1024,
        device="0",
        project=str(project),
    )

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
        project=str(project),
        name="xh25-baseline",
        exist_ok=True,
        resume=False,
    )


@patch("xh_detect.training.YOLO")
def test_train_model_passes_official_baseline_options(yolo_class: Mock) -> None:
    model = yolo_class.return_value
    expected_project = str((Path.cwd() / "runs/train").resolve())

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
        project=expected_project,
        name="xh25-baseline",
        exist_ok=True,
        resume=False,
    )


@patch("xh_detect.training.register_custom_modules")
@patch("xh_detect.training.YOLO")
def test_train_model_registers_custom_modules_before_model_load(
    yolo_class: Mock,
    register_custom_modules: Mock,
) -> None:
    events: list[str] = []
    register_custom_modules.side_effect = lambda: events.append("register")
    yolo_class.side_effect = lambda model_path: events.append(f"yolo:{model_path}") or Mock()

    train_model("dataset.yaml", "configs/models/xh25-mksnet-lite.yaml", 1, 640, "cpu")

    assert events[:2] == ["register", "yolo:configs/models/xh25-mksnet-lite.yaml"]


@patch("xh_detect.training.register_custom_modules")
@patch("xh_detect.training.YOLO")
def test_train_model_loads_optional_pretrained_weights(
    yolo_class: Mock,
    register_custom_modules: Mock,
) -> None:
    model = yolo_class.return_value
    model.load.return_value = model

    train_model(
        "dataset.yaml",
        "configs/models/xh25-mksnet-lite.yaml",
        1,
        640,
        "cpu",
        pretrained="yolo26s.pt",
    )

    register_custom_modules.assert_called_once_with()
    yolo_class.assert_called_once_with("configs/models/xh25-mksnet-lite.yaml")
    model.load.assert_called_once_with("yolo26s.pt")
    model.train.assert_called_once()


@patch("xh_detect.training.YOLO")
def test_train_model_uses_density_aware_trainer_when_enabled(yolo_class: Mock) -> None:
    from xh_detect.models.density_assigner import DensityAwareDetectionTrainer

    model = yolo_class.return_value

    train_model(
        "dataset.yaml",
        "yolo26s.pt",
        1,
        1024,
        "0",
        density_assignment=True,
        density_constant=12.0,
        density_threshold=0.25,
    )

    assert model.train.call_args.kwargs["trainer"] is DensityAwareDetectionTrainer
    assert DensityAwareDetectionTrainer.density_config.constant == 12.0
    assert DensityAwareDetectionTrainer.density_config.threshold == 0.25


@patch("xh_detect.training.YOLO")
def test_train_model_forwards_explicit_finetuning_options(yolo_class: Mock) -> None:
    model = yolo_class.return_value

    train_model(
        "dataset.yaml",
        "model.yaml",
        2,
        1024,
        "0",
        optimizer="AdamW",
        learning_rate=1e-4,
        freeze=11,
        save_period=1,
        warmup_epochs=0.0,
        warmup_bias_lr=0.0,
    )

    assert model.train.call_args.kwargs["optimizer"] == "AdamW"
    assert model.train.call_args.kwargs["lr0"] == 1e-4
    assert model.train.call_args.kwargs["freeze"] == 11
    assert model.train.call_args.kwargs["save_period"] == 1
    assert model.train.call_args.kwargs["warmup_epochs"] == 0.0
    assert model.train.call_args.kwargs["warmup_bias_lr"] == 0.0


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
        {"optimizer": ""},
        {"learning_rate": 0},
        {"freeze": -1},
        {"save_period": 0},
    ],
)
def test_train_model_validates_reproducible_options(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        train_model("data.yaml", "model.pt", 1, 640, "cpu", **kwargs)


@pytest.mark.parametrize(
    ("gcd_loss", "gcd_assignment"),
    [(True, False), (False, True), (True, True)],
)
def test_train_model_uses_gcd_trainer_for_selected_mode(
    gcd_loss: bool,
    gcd_assignment: bool,
) -> None:
    from xh_detect.models.gcd import GCDDetectionTrainer

    with patch("xh_detect.training.YOLO") as yolo_class:
        model = yolo_class.return_value

        train_model(
            "dataset.yaml",
            "yolo26s.pt",
            1,
            1024,
            "0",
            gcd_loss=gcd_loss,
            gcd_assignment=gcd_assignment,
        )

    assert model.train.call_args.kwargs["trainer"] is GCDDetectionTrainer
    assert GCDDetectionTrainer.gcd_config.use_loss is gcd_loss
    assert GCDDetectionTrainer.gcd_config.use_assignment is gcd_assignment


@patch("xh_detect.training.YOLO")
def test_train_model_rejects_density_and_gcd_combination(yolo_class: Mock) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        train_model(
            "dataset.yaml",
            "yolo26s.pt",
            1,
            1024,
            "0",
            density_assignment=True,
            gcd_loss=True,
        )

    yolo_class.assert_not_called()


@pytest.mark.parametrize("option", ["gcd_loss", "gcd_assignment"])
def test_train_model_validates_gcd_flags(option: str) -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        train_model(
            "dataset.yaml",
            "yolo26s.pt",
            1,
            1024,
            "0",
            **{option: "true"},
        )
