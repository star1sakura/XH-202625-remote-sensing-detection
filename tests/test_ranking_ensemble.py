from __future__ import annotations

import pytest

from xh_detect.ranking_ensemble import RankingEnsemblePolicy, fuse_ranking_ensemble
from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import Detection


def _box(x1: float, y1: float, x2: float, y2: float):
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def test_fuse_ranking_ensemble_selects_role_specific_classes_and_thresholds() -> None:
    taxonomy = get_taxonomy("xh25")
    aircraft = [
        Detection("1", 4, 0.25, _box(0, 0, 10, 10)),
        Detection("1", 3, 0.99, _box(20, 0, 30, 10)),
    ]
    ship = [
        Detection("1", 3, 0.31, _box(40, 0, 50, 10)),
        Detection("1", 3, 0.30, _box(60, 0, 70, 10)),
    ]
    vehicle = [Detection("1", 24, 0.25, _box(80, 0, 90, 10))]

    result = fuse_ranking_ensemble(
        aircraft_predictions=aircraft,
        ship_predictions=ship,
        vehicle_primary_predictions=vehicle,
        vehicle_supplement_predictions=[],
        taxonomy=taxonomy,
    )

    assert [(item.class_id, item.score) for item in result] == [
        (3, 0.31),
        (4, 0.25),
        (24, 0.25),
    ]


def test_vehicle_supplement_adds_new_proposal_and_rejects_duplicates() -> None:
    taxonomy = get_taxonomy("xh25")
    primary = [Detection("1", 24, 0.80, _box(0, 0, 10, 10))]
    supplement = [
        Detection("1", 24, 0.90, _box(1, 0, 11, 10)),
        Detection("1", 24, 0.70, _box(30, 0, 40, 10)),
        Detection("1", 24, 0.63, _box(50, 0, 60, 10)),
    ]

    result = fuse_ranking_ensemble(
        aircraft_predictions=[],
        ship_predictions=[],
        vehicle_primary_predictions=primary,
        vehicle_supplement_predictions=supplement,
        taxonomy=taxonomy,
    )

    assert [(item.score, item.polygon) for item in result] == [
        (0.80, _box(0, 0, 10, 10)),
        (0.70, _box(30, 0, 40, 10)),
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aircraft_threshold", -0.1),
        ("ship_threshold", 1.1),
        ("vehicle_primary_threshold", float("nan")),
        ("vehicle_supplement_threshold", True),
        ("vehicle_duplicate_iou", "0.3"),
    ],
)
def test_ranking_ensemble_policy_rejects_invalid_values(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises((TypeError, ValueError)):
        RankingEnsemblePolicy(**kwargs)
