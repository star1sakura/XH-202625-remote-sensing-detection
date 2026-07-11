from __future__ import annotations

import pytest

from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import Detection, ObjectAnnotation


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


def test_vehicle_pair_reports_candidate_only_recoverable_truth() -> None:
    from xh_detect.complementarity import analyze_complementarity

    truth = [
        ObjectAnnotation("img", 24, _box(0, 0, 10, 10)),
        ObjectAnnotation("img", 24, _box(20, 0, 30, 10)),
    ]
    predictions = {
        "main": [Detection("img", 24, 0.9, _box(0, 0, 10, 10))],
        "sph-p2": [
            Detection("img", 24, 0.9, _box(0, 0, 10, 10)),
            Detection("img", 24, 0.8, _box(20, 0, 30, 10)),
            Detection("img", 24, 0.7, _box(40, 0, 50, 10)),
        ],
    }

    report = analyze_complementarity(
        predictions,
        truth,
        taxonomy=get_taxonomy("xh25"),
        baseline_name="main",
    )

    vehicle = report.pairwise["sph-p2"]["vehicle"]
    assert vehicle.shared_tp == 1
    assert vehicle.baseline_only_tp == 0
    assert vehicle.candidate_only_tp == 1
    assert vehicle.oracle_tp == 2
    assert vehicle.oracle_recall == 1.0
    assert vehicle.candidate_fp == 1


def test_ship_matching_uses_point_five_iou_and_score_order() -> None:
    from xh_detect.complementarity import analyze_complementarity

    truth = [ObjectAnnotation("img", 3, _box(0, 0, 10, 10))]
    predictions = {
        "main": [Detection("img", 3, 0.9, _box(0, 0, 4.9, 10))],
        "candidate": [
            Detection("img", 3, 0.9, _box(0, 0, 5, 10)),
            Detection("img", 3, 0.8, _box(0, 0, 10, 10)),
        ],
    }

    report = analyze_complementarity(
        predictions,
        truth,
        taxonomy=get_taxonomy("xh25"),
        baseline_name="main",
    )

    assert report.models["main"]["ship"].tp == 0
    assert report.models["main"]["ship"].fp == 1
    assert report.models["candidate"]["ship"].tp == 1
    assert report.models["candidate"]["ship"].fp == 1
    assert report.pairwise["candidate"]["ship"].candidate_only_tp == 1


@pytest.mark.parametrize(
    ("predictions", "baseline_name", "message"),
    [
        ({"main": []}, "main", "at least two"),
        ({"main": [], "candidate": []}, "missing", "baseline"),
    ],
)
def test_complementarity_rejects_invalid_model_set(
    predictions: dict[str, list[Detection]],
    baseline_name: str,
    message: str,
) -> None:
    from xh_detect.complementarity import analyze_complementarity

    with pytest.raises(ValueError, match=message):
        analyze_complementarity(
            predictions,
            [],
            taxonomy=get_taxonomy("xh25"),
            baseline_name=baseline_name,
        )
