from __future__ import annotations

import math
from pathlib import Path

from ultralytics import YOLO

from xh_detect.models.density_assigner import (
    DensityAssignerConfig,
    DensityAwareDetectionTrainer,
)
from xh_detect.models.gcd import GCDDetectionTrainer, GCDTrainingConfig
from xh_detect.models.ultralytics import register_custom_modules


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _project_path(value: object, name: str = "project") -> str:
    project = Path(_non_empty(value, name)).expanduser()
    if not project.is_absolute():
        project = Path.cwd() / project
    return str(project.resolve())


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def train_model(
    dataset_yaml: str,
    model_path: str,
    epochs: int,
    image_size: int,
    device: str,
    *,
    batch: int = 8,
    workers: int = 4,
    amp: bool = False,
    project: str = "runs/train",
    name: str = "xh25-baseline",
    resume: bool = False,
    seed: int = 42,
    pretrained: str | None = None,
    density_assignment: bool = False,
    density_constant: float = 12.0,
    density_threshold: float = 0.25,
    gcd_loss: bool = False,
    gcd_assignment: bool = False,
    optimizer: str | None = None,
    learning_rate: float | None = None,
    freeze: int | None = None,
    save_period: int | None = None,
    warmup_epochs: float | None = None,
    warmup_bias_lr: float | None = None,
) -> None:
    dataset = _non_empty(dataset_yaml, "dataset_yaml")
    model_source = _non_empty(model_path, "model_path")
    epochs = _positive_int(epochs, "epochs")
    image_size = _positive_int(image_size, "image_size")
    device = _non_empty(device, "device")
    batch = _positive_int(batch, "batch")
    workers = _non_negative_int(workers, "workers")
    amp = _bool(amp, "amp")
    project = _project_path(project, "project")
    name = _non_empty(name, "name")
    resume = _bool(resume, "resume")
    seed = _non_negative_int(seed, "seed")
    density_assignment = _bool(density_assignment, "density_assignment")
    gcd_loss = _bool(gcd_loss, "gcd_loss")
    gcd_assignment = _bool(gcd_assignment, "gcd_assignment")
    if density_assignment and (gcd_loss or gcd_assignment):
        raise ValueError("density assignment cannot be combined with GCD training")

    pretrained_model = None if pretrained is None else _non_empty(pretrained, "pretrained")
    optimizer_name = None if optimizer is None else _non_empty(optimizer, "optimizer")
    if learning_rate is not None and (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be positive")
    if freeze is not None:
        freeze = _non_negative_int(freeze, "freeze")
    if save_period is not None:
        save_period = _positive_int(save_period, "save_period")
    for value, option in (
        (warmup_epochs, "warmup_epochs"),
        (warmup_bias_lr, "warmup_bias_lr"),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{option} must be finite and non-negative")

    register_custom_modules()
    model = YOLO(model_source)
    if pretrained_model is not None:
        model = model.load(pretrained_model)
    train_arguments: dict[str, object] = {
        "data": dataset,
        "epochs": epochs,
        "imgsz": image_size,
        "device": device,
        "batch": batch,
        "workers": workers,
        "amp": amp,
        "seed": seed,
        "deterministic": True,
        "project": project,
        "name": name,
        "exist_ok": True,
        "resume": resume,
    }
    if density_assignment:
        DensityAwareDetectionTrainer.density_config = DensityAssignerConfig(
            constant=density_constant,
            threshold=density_threshold,
        )
        train_arguments["trainer"] = DensityAwareDetectionTrainer
    if gcd_loss or gcd_assignment:
        GCDDetectionTrainer.gcd_config = GCDTrainingConfig(
            use_loss=gcd_loss,
            use_assignment=gcd_assignment,
        )
        train_arguments["trainer"] = GCDDetectionTrainer
    if optimizer_name is not None:
        train_arguments["optimizer"] = optimizer_name
    if learning_rate is not None:
        train_arguments["lr0"] = float(learning_rate)
    if freeze is not None:
        train_arguments["freeze"] = freeze
    if save_period is not None:
        train_arguments["save_period"] = save_period
    if warmup_epochs is not None:
        train_arguments["warmup_epochs"] = float(warmup_epochs)
    if warmup_bias_lr is not None:
        train_arguments["warmup_bias_lr"] = float(warmup_bias_lr)
    model.train(
        **train_arguments,
    )


def export_tensorrt(model_path: str, image_size: int, device: str) -> str:
    model_source = _non_empty(model_path, "model_path")
    image_size = _positive_int(image_size, "image_size")
    device = _non_empty(device, "device")

    model = YOLO(model_source)
    result = model.export(
        format="engine",
        imgsz=image_size,
        half=True,
        device=device,
        batch=1,
    )
    return str(result)
