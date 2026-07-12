from __future__ import annotations

import pytest

from xh_detect.types import Detection, ObjectAnnotation, Polygon4
from xh_detect.vehicle_expert import analyze_vehicle_expert_holdout


def _box(x: float) -> Polygon4:
    return ((x, 0.0), (x + 10, 0.0), (x + 10, 10.0), (x, 10.0))


def test_selects_feasible_vehicle_expert_threshold() -> None:
    truth = [ObjectAnnotation("1", 24, _box(x)) for x in (0, 20, 40, 60)]
    main = [Detection("1", 24, 0.9, _box(0))]
    expert = [
        Detection("1", 0, 0.90, _box(0)),
        Detection("1", 0, 0.80, _box(20)),
        Detection("1", 0, 0.70, _box(40)),
        Detection("1", 0, 0.60, _box(60)),
        Detection("1", 0, 0.55, _box(100)),
    ]

    report = analyze_vehicle_expert_holdout(
        main,
        expert,
        truth,
        image_ids={"1"},
        thresholds=(0.50, 0.65),
    )

    assert report.baseline_tp == 1
    assert report.baseline_fp == 0
    assert report.baseline_fn == 3
    assert report.selected is not None
    assert report.selected.threshold == 0.50
    assert report.selected.added_tp == 3
    assert report.selected.added_fp == 1
    assert report.selected.fused_fdr == 0.2


def test_counts_main_false_positive_on_holdout_image_without_truth() -> None:
    report = analyze_vehicle_expert_holdout(
        [Detection("empty", 24, 0.8, _box(0))],
        [],
        [],
        image_ids={"empty"},
        thresholds=(0.5,),
        minimum_added_tp=0,
    )

    assert report.baseline_fp == 1


def test_rejects_nonzero_expert_class() -> None:
    with pytest.raises(ValueError, match="class 0"):
        analyze_vehicle_expert_holdout(
            [],
            [Detection("1", 1, 0.8, _box(0))],
            [],
            image_ids={"1"},
            thresholds=(0.5,),
        )
