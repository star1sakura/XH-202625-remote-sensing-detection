from __future__ import annotations

import pytest

from xh_detect.types import Detection


def _box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def test_diou_suppression_keeps_high_score_and_non_overlapping_detection() -> None:
    from xh_detect.postprocess import SuppressionRule, suppress_class_detections

    detections = [
        Detection("img", 3, 0.95, _box(0, 0, 10, 10)),
        Detection("img", 3, 0.90, _box(1, 0, 11, 10)),
        Detection("img", 3, 0.80, _box(30, 0, 40, 10)),
    ]

    kept = suppress_class_detections(
        detections,
        {3: SuppressionRule(method="diou", threshold=0.30)},
    )

    assert [(item.score, item.polygon) for item in kept] == [
        (0.95, _box(0, 0, 10, 10)),
        (0.80, _box(30, 0, 40, 10)),
    ]


def test_unconfigured_class_is_not_suppressed() -> None:
    from xh_detect.postprocess import SuppressionRule, suppress_class_detections

    detections = [
        Detection("img", 24, 0.70, _box(0, 0, 10, 10)),
        Detection("img", 24, 0.60, _box(0, 0, 10, 10)),
    ]

    assert (
        suppress_class_detections(
            detections,
            {3: SuppressionRule(method="iou", threshold=0.30)},
        )
        == detections
    )


@pytest.mark.parametrize(
    ("method", "threshold"),
    [("bad", 0.3), ("iou", -0.1), ("iou", 1.1), ("diou", -1.1), ("diou", 1.1)],
)
def test_suppression_rule_rejects_invalid_values(method: str, threshold: float) -> None:
    from xh_detect.postprocess import SuppressionRule

    with pytest.raises(ValueError):
        SuppressionRule(method=method, threshold=threshold)


def test_low_score_area_filter_requires_both_conditions_and_matching_class() -> None:
    from xh_detect.postprocess import (
        LowScoreAreaRule,
        filter_low_score_area_detections,
    )

    detections = [
        Detection("img", 24, 0.20, _box(0, 0, 40, 20)),
        Detection("img", 24, 0.22, _box(0, 0, 10, 10)),
        Detection("img", 24, 0.20, _box(0, 0, 10, 10)),
        Detection("img", 3, 0.20, _box(0, 0, 10, 10)),
    ]

    kept = filter_low_score_area_detections(
        detections,
        {24: LowScoreAreaRule(score_ceiling=0.21, min_area=700)},
    )

    assert kept == [detections[0], detections[1], detections[3]]


def test_low_score_area_filter_keeps_boundary_values() -> None:
    from xh_detect.postprocess import (
        LowScoreAreaRule,
        filter_low_score_area_detections,
    )

    detections = [
        Detection("img", 24, 0.21, _box(0, 0, 10, 10)),
        Detection("img", 24, 0.20, _box(0, 0, 35, 20)),
    ]

    assert (
        filter_low_score_area_detections(
            detections,
            {24: LowScoreAreaRule(score_ceiling=0.21, min_area=700)},
        )
        == detections
    )


@pytest.mark.parametrize(
    ("score_ceiling", "min_area"),
    [(-0.1, 700), (1.1, 700), (float("nan"), 700), (0.21, -1), (0.21, float("inf"))],
)
def test_low_score_area_rule_rejects_invalid_values(
    score_ceiling: float,
    min_area: float,
) -> None:
    from xh_detect.postprocess import LowScoreAreaRule

    with pytest.raises(ValueError):
        LowScoreAreaRule(score_ceiling=score_ceiling, min_area=min_area)
