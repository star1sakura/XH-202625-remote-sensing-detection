from __future__ import annotations

from ultralytics import YOLO


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def train_model(
    dataset_yaml: str,
    model_path: str,
    epochs: int,
    image_size: int,
    device: str,
) -> None:
    dataset = _non_empty(dataset_yaml, "dataset_yaml")
    model_source = _non_empty(model_path, "model_path")
    epochs = _positive_int(epochs, "epochs")
    image_size = _positive_int(image_size, "image_size")
    device = _non_empty(device, "device")

    model = YOLO(model_source)
    model.train(
        data=dataset,
        epochs=epochs,
        imgsz=image_size,
        device=device,
        seed=42,
        deterministic=True,
        project="runs/train",
        name="baseline",
        exist_ok=True,
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
