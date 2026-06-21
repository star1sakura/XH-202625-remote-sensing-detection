from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from xh_detect.benchmark import (
    benchmark_pipeline,
    create_synthetic_image,
    summarize_durations,
)
from xh_detect.types import InferenceResult, StageTimings


def test_summarize_durations_reports_median_and_p95() -> None:
    summary = summarize_durations([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary["median_s"] == 3.0
    assert summary["p95_s"] == pytest.approx(4.8)


@pytest.mark.parametrize("values", [[], [1.0, -1.0], [float("nan")], [float("inf")]])
def test_summarize_durations_rejects_invalid_values(values: list[float]) -> None:
    with pytest.raises(ValueError):
        summarize_durations(values)


def test_create_synthetic_image_supports_small_test_dimensions(tmp_path: Path) -> None:
    destination = tmp_path / "synthetic.png"

    result = create_synthetic_image(destination, width=320, height=240)

    image = cv2.imread(str(result), cv2.IMREAD_COLOR)
    assert result == destination
    assert image is not None
    assert image.shape == (240, 320, 3)
    assert int(image.max()) > int(image.min())


def test_benchmark_pipeline_runs_one_warmup_and_aggregates_stages() -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run(self, image: np.ndarray, image_id: str) -> InferenceResult:
            self.calls.append(image_id)
            value = float(len(self.calls))
            return InferenceResult(
                detections=(),
                timings=StageTimings(
                    preprocess_s=value * 0.1,
                    inference_s=value * 0.2,
                    postprocess_s=value * 0.05,
                    total_s=value,
                ),
            )

    pipeline = FakePipeline()
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    summary = benchmark_pipeline(pipeline, image, "scene", repeats=3)  # type: ignore[arg-type]

    assert pipeline.calls == ["scene-warmup", "scene-0", "scene-1", "scene-2"]
    assert summary["median_s"] == 3.0
    assert summary["p95_s"] == pytest.approx(3.9)
    assert summary["inference_median_s"] == pytest.approx(0.6)


@pytest.mark.parametrize("repeats", [0, -1, True, 1.5])
def test_benchmark_pipeline_rejects_invalid_repeats(repeats: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        benchmark_pipeline(
            object(),  # type: ignore[arg-type]
            np.zeros((4, 4, 3), dtype=np.uint8),
            "scene",
            repeats,  # type: ignore[arg-type]
        )
