from __future__ import annotations

import numpy as np
import pytest

from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import Detection
from xh_detect.visualize import CLASS_NAMES, class_counts, draw_detections


def _detection(class_id: int = 0, score: float = 0.9) -> Detection:
    return Detection(
        "img",
        class_id,
        score,
        ((10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0)),
    )


@pytest.mark.parametrize("mode", ["obb", "hbb"])
def test_draw_detections_changes_pixels_without_mutating_input(mode: str) -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    original = image.copy()

    rendered = draw_detections(image, [_detection()], mode=mode)

    assert rendered.shape == image.shape
    assert rendered.dtype == image.dtype
    assert rendered.sum() > 0
    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(rendered, image)


def test_class_counts_has_all_classes_and_accepts_generator() -> None:
    detections = (_detection(class_id) for class_id in [0, 2, 2])

    assert class_counts(detections) == {
        "coarse": {"aircraft": 1, "ship": 0, "vehicle": 2},
        "fine": {"aircraft": 1, "ship": 0, "vehicle": 2},
    }
    assert CLASS_NAMES == {0: "aircraft", 1: "ship", 2: "vehicle"}


def test_xh25_counts_include_coarse_and_fine_views() -> None:
    detections = [_detection(0), _detection(4), _detection(24)]

    counts = class_counts(detections, taxonomy=get_taxonomy("xh25"))

    assert counts["coarse"] == {"aircraft": 1, "ship": 1, "vehicle": 1}
    assert counts["fine"]["HM"] == 1
    assert counts["fine"]["A1_SU-35"] == 1
    assert counts["fine"]["FSC"] == 1


def test_draw_detections_accepts_generator_and_empty_list() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    rendered = draw_detections(image, (_detection(1) for _ in range(1)), mode="obb")
    empty = draw_detections(image, [], mode="hbb")

    assert rendered.sum() > 0
    np.testing.assert_array_equal(empty, image)


@pytest.mark.parametrize("mode", ["", "rotated", "OBB"])
def test_draw_detections_rejects_invalid_mode_even_when_empty(mode: str) -> None:
    with pytest.raises(ValueError, match="mode"):
        draw_detections(np.zeros((10, 10, 3), dtype=np.uint8), [], mode=mode)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_draw_detections_rejects_unsupported_images(image: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError), match="image"):
        draw_detections(image, [], mode="obb")  # type: ignore[arg-type]


def test_visualization_rejects_unknown_class() -> None:
    detection = _detection(9)
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="class"):
        draw_detections(image, [detection])
    with pytest.raises(ValueError, match="class"):
        class_counts([detection])


def test_draw_detections_accepts_xh25_class_24() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    rendered = draw_detections(image, [_detection(24)], taxonomy=get_taxonomy("xh25"))

    assert rendered.sum() > 0
