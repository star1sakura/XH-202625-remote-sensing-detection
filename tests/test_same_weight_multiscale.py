from __future__ import annotations

from pathlib import Path

import pytest

from xh_detect.same_weight_multiscale import (
    SameWeightMultiscalePolicy,
    fuse_same_weight_multiscale,
    load_same_weight_multiscale_policy,
)
from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import Detection


def _box(x1: float, y1: float, x2: float, y2: float):
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _policy(**overrides: object) -> SameWeightMultiscalePolicy:
    values: dict[str, object] = {
        "class_thresholds": dict.fromkeys(range(25), 0.10),
    }
    values.update(overrides)
    return SameWeightMultiscalePolicy(**values)  # type: ignore[arg-type]


def test_fuse_same_weight_multiscale_uses_role_specific_scales() -> None:
    taxonomy = get_taxonomy("xh25")
    predictions_1024 = [
        Detection("1", 4, 0.20, _box(0, 0, 10, 10)),
        Detection("1", 4, 0.05, _box(20, 0, 30, 10)),
        Detection("1", 3, 0.90, _box(60, 0, 70, 10)),
        Detection("1", 3, 0.57, _box(100, 0, 110, 10)),
        Detection("1", 3, 0.55, _box(120, 0, 130, 10)),
    ]
    predictions_1280 = [
        Detection("1", 4, 0.90, _box(1, 0, 11, 10)),
        Detection("1", 4, 0.80, _box(30, 0, 40, 10)),
        Detection("1", 4, 0.74, _box(45, 0, 55, 10)),
    ]
    predictions_1536 = [
        Detection("1", 3, 0.20, _box(60, 0, 70, 10)),
        Detection("1", 24, 0.20, _box(140, 0, 150, 10)),
        Detection("1", 24, 0.30, _box(160, 0, 170, 10)),
        Detection("1", 24, 0.20, _box(180, 0, 210, 30)),
    ]

    result = fuse_same_weight_multiscale(
        predictions_1024=predictions_1024,
        predictions_1280=predictions_1280,
        predictions_1536=predictions_1536,
        taxonomy=taxonomy,
        policy=_policy(),
    )

    assert [(item.class_id, item.score, item.polygon) for item in result] == [
        (3, 0.57, _box(100, 0, 110, 10)),
        (3, 0.20, _box(60, 0, 70, 10)),
        (4, 0.80, _box(30, 0, 40, 10)),
        (4, 0.20, _box(0, 0, 10, 10)),
        (24, 0.30, _box(160, 0, 170, 10)),
        (24, 0.20, _box(180, 0, 210, 30)),
    ]


def test_fuse_same_weight_multiscale_requires_all_class_thresholds() -> None:
    with pytest.raises(ValueError, match="exactly"):
        fuse_same_weight_multiscale(
            predictions_1024=[],
            predictions_1280=[],
            predictions_1536=[],
            taxonomy=get_taxonomy("xh25"),
            policy=SameWeightMultiscalePolicy(class_thresholds={0: 0.1}),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aircraft_supplement_threshold", -0.1),
        ("aircraft_duplicate_iou", 1.1),
        ("ship_supplement_threshold", float("nan")),
        ("ship_duplicate_iou", True),
        ("vehicle_score_ceiling", "0.21"),
        ("vehicle_min_area", -1.0),
    ],
)
def test_same_weight_multiscale_policy_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _policy(**{field: value})


def test_load_same_weight_multiscale_policy_normalizes_class_ids(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(
        "class_thresholds:\n  '0': 0.12\naircraft_supplement_threshold: 0.8\n",
        encoding="utf-8",
    )

    policy = load_same_weight_multiscale_policy(path)

    assert policy.class_thresholds == {0: 0.12}
    assert policy.aircraft_supplement_threshold == 0.8
