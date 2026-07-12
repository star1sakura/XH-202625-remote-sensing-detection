from __future__ import annotations

import math

import pytest

from xh_detect.types import Detection, ObjectAnnotation, Polygon4
from xh_detect.vehicle_confirmation.proposals import (
    label_vehicle_proposals,
    satisfies_vehicle_fdr,
)


def _box(x1: float, y1: float, x2: float, y2: float) -> Polygon4:
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _truth(
    image_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    class_id: int = 24,
    difficult: bool = False,
) -> ObjectAnnotation:
    return ObjectAnnotation(image_id, class_id, _box(x1, y1, x2, y2), difficult)


def _detection(
    image_id: str,
    score: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    class_id: int = 24,
) -> Detection:
    return Detection(image_id, class_id, score, _box(x1, y1, x2, y2))


def test_labels_only_recoverable_vehicle_as_positive() -> None:
    truth = [_truth("img", 0, 0, 10, 10), _truth("img", 30, 0, 40, 10)]
    main = [_detection("img", 0.95, 0, 0, 10, 10)]
    proposals = [
        _detection("img", 0.90, 0, 0, 10, 10),
        _detection("img", 0.80, 30, 0, 40, 10),
        _detection("img", 0.70, 30, 0, 40, 10),
        _detection("img", 0.60, 70, 0, 80, 10),
    ]

    labels, report = label_vehicle_proposals(main, proposals, truth)

    assert [(item.label, item.reason) for item in labels] == [
        (0, "duplicate_main"),
        (1, "recoverable_truth"),
        (0, "duplicate_proposal"),
        (0, "background"),
    ]
    assert labels[1].matched_truth_index == 1
    assert labels[0].duplicate_main
    assert not labels[1].duplicate_main
    assert report.main_vehicle_tp == 1
    assert report.main_vehicle_fp == 0
    assert report.recoverable_tp == 1
    assert report.proposal_fp == 3
    assert report.duplicate_main == 1
    assert report.duplicate_proposal == 1


def test_ignores_nonvehicle_predictions_and_truth() -> None:
    labels, report = label_vehicle_proposals(
        [_detection("img", 0.9, 0, 0, 10, 10, class_id=3)],
        [
            _detection("img", 0.9, 0, 0, 10, 10, class_id=3),
            _detection("img", 0.8, 20, 0, 30, 10),
        ],
        [
            _truth("img", 0, 0, 10, 10, class_id=3),
            _truth("img", 20, 0, 30, 10),
        ],
    )

    assert len(labels) == 1
    assert labels[0].proposal_index == 1
    assert labels[0].label == 1
    assert report.recoverable_tp == 1


def test_exact_point_three_five_iou_matches_vehicle_truth() -> None:
    labels, _ = label_vehicle_proposals(
        [],
        [_detection("img", 0.8, 0, 0, 3.5, 10)],
        [_truth("img", 0, 0, 10, 10)],
    )

    assert labels[0].label == 1
    assert labels[0].reason == "recoverable_truth"


def test_score_order_and_original_index_stably_claim_truth() -> None:
    proposals = [
        _detection("img", 0.8, 0, 0, 10, 10),
        _detection("img", 0.9, 0, 0, 10, 10),
        _detection("img", 0.9, 0, 0, 10, 10),
    ]

    labels, _ = label_vehicle_proposals([], proposals, [_truth("img", 0, 0, 10, 10)])

    assert [item.proposal_index for item in labels] == [1, 2, 0]
    assert [item.label for item in labels] == [1, 0, 0]


def test_difficult_truth_is_excluded_and_images_are_isolated() -> None:
    labels, report = label_vehicle_proposals(
        [_detection("other", 0.95, 0, 0, 10, 10)],
        [
            _detection("img", 0.9, 0, 0, 10, 10),
            _detection("other", 0.8, 20, 0, 30, 10),
        ],
        [
            _truth("img", 0, 0, 10, 10, difficult=True),
            _truth("other", 20, 0, 30, 10),
        ],
    )

    assert [(item.label, item.reason) for item in labels] == [
        (0, "background"),
        (1, "recoverable_truth"),
    ]
    assert report.main_vehicle_fp == 1


@pytest.mark.parametrize("vehicle_class_id", [-1, 25, True, 1.5])
def test_rejects_invalid_vehicle_class_id(vehicle_class_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        label_vehicle_proposals([], [], [], vehicle_class_id=vehicle_class_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("iou_threshold", [-0.1, 1.1, math.nan, math.inf, True, "0.35"])
def test_rejects_invalid_iou_threshold(iou_threshold: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        label_vehicle_proposals([], [], [], iou_threshold=iou_threshold)  # type: ignore[arg-type]


def test_rejects_out_of_taxonomy_detection_class() -> None:
    with pytest.raises(ValueError, match="class_id"):
        label_vehicle_proposals([], [_detection("img", 0.9, 0, 0, 10, 10, class_id=25)], [])


def test_returns_immutable_tuple() -> None:
    labels, _ = label_vehicle_proposals(
        [],
        [_detection("img", 0.9, 0, 0, 10, 10)],
        [_truth("img", 0, 0, 10, 10)],
    )

    assert isinstance(labels, tuple)
    with pytest.raises(AttributeError):
        labels.append(labels[0])  # type: ignore[attr-defined]


def test_vehicle_fdr_constraint_uses_fused_counts() -> None:
    assert satisfies_vehicle_fdr(55, 14, 4, 1)
    assert not satisfies_vehicle_fdr(55, 14, 3, 1)


@pytest.mark.parametrize(
    ("args", "ceiling"),
    [
        ((-1, 14, 4, 1), 0.202899),
        ((55, True, 4, 1), 0.202899),
        ((55, 14, 4, 1), math.nan),
        ((55, 14, 4, 1), 1.1),
    ],
)
def test_vehicle_fdr_rejects_invalid_counts_and_ceiling(
    args: tuple[object, object, object, object],
    ceiling: float,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        satisfies_vehicle_fdr(*args, ceiling=ceiling)  # type: ignore[arg-type]
