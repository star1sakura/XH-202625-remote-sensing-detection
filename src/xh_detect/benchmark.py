from __future__ import annotations

import math
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from xh_detect.pipeline import InferencePipeline
from xh_detect.types import ImageArray, StageTimings


def summarize_durations(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("at least one duration is required")
    normalized = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in normalized):
        raise ValueError("durations must be finite and non-negative")
    return {
        "median_s": float(median(normalized)),
        "p95_s": float(np.percentile(normalized, 95)),
    }


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def create_synthetic_image(
    destination: Path,
    width: int = 10_000,
    height: int = 10_000,
) -> Path:
    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    destination = Path(destination)

    image = np.full((height, width, 3), 96, dtype=np.uint8)
    box_width = max(2, min(300, width // 5))
    box_height = max(2, min(200, height // 8))
    x_step = max(box_width * 2, width // 6)
    y_step = max(box_height * 2, height // 6)
    for y in range(max(1, y_step // 2), height, y_step):
        for x in range(max(1, x_step // 2), width, x_step):
            cv2.rectangle(
                image,
                (x, y),
                (min(x + box_width, width - 1), min(y + box_height, height - 1)),
                (140, 140, 140),
                -1,
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"failed to write synthetic benchmark image: {destination}")
    return destination


def _stage_values(
    timings: list[StageTimings],
    attribute: str,
) -> list[float]:
    return [float(getattr(item, attribute)) for item in timings]


def benchmark_pipeline(
    pipeline: InferencePipeline,
    image: ImageArray,
    image_id: str,
    repeats: int,
) -> dict[str, float]:
    repeats = _positive_int(repeats, "repeats")
    if not isinstance(image_id, str) or not image_id.strip():
        raise ValueError("image_id must be a non-empty string")

    pipeline.run(image, f"{image_id}-warmup")
    timings = [
        pipeline.run(image, f"{image_id}-{index}").timings
        for index in range(repeats)
    ]
    summary = summarize_durations(_stage_values(timings, "total_s"))
    for attribute, prefix in (
        ("preprocess_s", "preprocess"),
        ("inference_s", "inference"),
        ("postprocess_s", "postprocess"),
    ):
        stage_summary = summarize_durations(_stage_values(timings, attribute))
        summary[f"{prefix}_median_s"] = stage_summary["median_s"]
        summary[f"{prefix}_p95_s"] = stage_summary["p95_s"]
    return summary
