from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from xh_detect.models.density_assigner import (
    DensityAssignerConfig,
    DensityAwareDetectionTrainer,
)
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
    pretrained: str | None = None,
    density_assignment: bool = False,
    density_constant: float = 12.0,
    density_threshold: float = 0.25,
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
    density_assignment = _bool(density_assignment, "density_assignment")
    pretrained_model = None if pretrained is None else _non_empty(pretrained, "pretrained")

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
        "seed": 42,
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
