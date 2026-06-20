from __future__ import annotations

from collections.abc import Iterable

import pytest

from xh_detect.geometry import obb_to_hbb
from xh_detect.merge import keep_tile_prediction, merge_detections, project_prediction
from xh_detect.types import BoxPrediction, Detection, TileMeta


def _prediction(
    class_id: int = 0,
    score: float = 0.9,
    polygon: tuple[tuple[float, float], ...] = (
        (10.0, 10.0),
        (30.0, 10.0),
        (30.0, 30.0),
        (10.0, 30.0),
    ),
) -> BoxPrediction:
    return BoxPrediction(class_id=class_id, score=score, polygon=polygon)  # type: ignore[arg-type]


def _detection(
    image_id: str = "scene",
    class_id: int = 0,
    score: float = 0.9,
    polygon: tuple[tuple[float, float], ...] = (
        (10.0, 10.0),
        (30.0, 10.0),
        (30.0, 30.0),
        (10.0, 30.0),
    ),
) -> Detection:
    return Detection(image_id=image_id, class_id=class_id, score=score, polygon=polygon)  # type: ignore[arg-type]


def test_project_prediction_offsets_and_clips_to_image_bounds() -> None:
    prediction = _prediction(
        polygon=(
            (-5.0, -7.0),
            (30.0, -7.0),
            (30.0, 25.0),
            (-5.0, 25.0),
        ),
    )
    meta = TileMeta(
        image_id="scene",
        tile_id="scene__x100_y200_s128",
        x=100,
        y=200,
        width=128,
        height=128,
        valid_width=64,
        valid_height=20,
    )

    result = project_prediction(prediction, meta, image_width=180, image_height=220)

    assert result.image_id == "scene"
    assert result.class_id == prediction.class_id
    assert result.score == prediction.score
    assert result.polygon == (
        (95.0, 193.0),
        (130.0, 193.0),
        (130.0, 220.0),
        (95.0, 220.0),
    )


@pytest.mark.parametrize(
    ("image_width", "image_height", "exc_type"),
    [
        (0, 10, ValueError),
        (10, 0, ValueError),
        (-1, 10, ValueError),
        (10, -1, ValueError),
        (10.5, 10, TypeError),
        (10, 10.5, TypeError),
        (True, 10, TypeError),
        (10, False, TypeError),
    ],
)
def test_project_prediction_validates_image_dimensions(
    image_width: object,
    image_height: object,
    exc_type: type[BaseException],
) -> None:
    prediction = _prediction()
    meta = TileMeta("scene", "tile", 0, 0, 128, 128, 128, 128)

    with pytest.raises(exc_type):
        project_prediction(prediction, meta, image_width=image_width, image_height=image_height)


@pytest.mark.parametrize(
    "meta",
    [
        TileMeta("scene", "tile", -1, 0, 128, 128, 128, 128),
        TileMeta("scene", "tile", 0, -1, 128, 128, 128, 128),
        TileMeta("scene", "tile", 0, 0, 128, 128, 0, 128),
        TileMeta("scene", "tile", 0, 0, 128, 128, 128, 0),
        TileMeta("scene", "tile", 0, 0, 128, 128, 129, 128),
        TileMeta("scene", "tile", 0, 0, 128, 128, 128, 129),
        TileMeta("scene", "tile", 10, 0, 128, 128, 128, 128),
        TileMeta("scene", "tile", 0, 20, 128, 128, 128, 128),
    ],
)
def test_project_prediction_rejects_inconsistent_tile_meta(meta: TileMeta) -> None:
    prediction = _prediction()

    with pytest.raises(ValueError):
        project_prediction(prediction, meta, image_width=128, image_height=128)


@pytest.mark.parametrize(
    ("margin", "exc_type"),
    [
        (-1, ValueError),
        (-0.5, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("16", TypeError),
    ],
)
def test_keep_tile_prediction_validates_margin(
    margin: object, exc_type: type[BaseException]
) -> None:
    prediction = _prediction()
    meta = TileMeta("scene", "tile", 0, 0, 128, 128, 128, 128)

    with pytest.raises(exc_type):
        keep_tile_prediction(prediction, meta, image_width=256, image_height=256, margin=margin)


@pytest.mark.parametrize(
    ("iou_threshold", "exc_type"),
    [
        (-0.1, ValueError),
        (1.1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("0.3", TypeError),
    ],
)
def test_merge_detections_validates_iou_threshold(
    iou_threshold: object,
    exc_type: type[BaseException],
) -> None:
    with pytest.raises(exc_type):
        merge_detections([_detection()], iou_threshold=iou_threshold)


@pytest.mark.parametrize(
    ("meta", "prediction", "margin", "expected"),
    [
        (
            TileMeta("scene", "left-internal", 64, 0, 128, 128, 64, 128),
            _prediction(polygon=((0.0, 10.0), (20.0, 10.0), (20.0, 30.0), (0.0, 30.0))),
            16,
            False,
        ),
        (
            TileMeta("scene", "left-external", 0, 0, 128, 128, 64, 128),
            _prediction(polygon=((0.0, 10.0), (20.0, 10.0), (20.0, 30.0), (0.0, 30.0))),
            16,
            True,
        ),
        (
            TileMeta("scene", "top-internal", 0, 64, 128, 128, 128, 64),
            _prediction(polygon=((10.0, 0.0), (30.0, 0.0), (30.0, 20.0), (10.0, 20.0))),
            16,
            False,
        ),
        (
            TileMeta("scene", "top-external", 0, 0, 128, 128, 128, 64),
            _prediction(polygon=((10.0, 0.0), (30.0, 0.0), (30.0, 20.0), (10.0, 20.0))),
            16,
            True,
        ),
        (
            TileMeta("scene", "right-internal", 32, 0, 128, 128, 64, 128),
            _prediction(polygon=((45.0, 10.0), (63.0, 10.0), (63.0, 30.0), (45.0, 30.0))),
            16,
            False,
        ),
        (
            TileMeta("scene", "right-external", 192, 0, 128, 128, 64, 128),
            _prediction(polygon=((45.0, 10.0), (63.0, 10.0), (63.0, 30.0), (45.0, 30.0))),
            16,
            True,
        ),
        (
            TileMeta("scene", "bottom-internal", 0, 32, 128, 128, 128, 64),
            _prediction(polygon=((10.0, 45.0), (30.0, 45.0), (30.0, 63.0), (10.0, 63.0))),
            16,
            False,
        ),
        (
            TileMeta("scene", "bottom-external", 0, 192, 128, 128, 128, 64),
            _prediction(polygon=((10.0, 45.0), (30.0, 45.0), (30.0, 63.0), (10.0, 63.0))),
            16,
            True,
        ),
    ],
)
def test_keep_tile_prediction_distinguishes_internal_and_outer_edges(
    meta: TileMeta,
    prediction: BoxPrediction,
    margin: int,
    expected: bool,
) -> None:
    assert keep_tile_prediction(
        prediction,
        meta,
        image_width=256,
        image_height=256,
        margin=margin,
    ) is expected


def test_keep_tile_prediction_rejects_padded_center() -> None:
    prediction = _prediction(polygon=((84.0, 10.0), (95.0, 10.0), (95.0, 30.0), (84.0, 30.0)))
    meta = TileMeta("scene", "tile", 0, 0, 128, 128, 80, 80)

    assert keep_tile_prediction(
        prediction,
        meta,
        image_width=256,
        image_height=256,
        margin=16,
    ) is False


def test_keep_tile_prediction_margin_zero_only_checks_center() -> None:
    prediction = _prediction(polygon=((0.0, 10.0), (20.0, 10.0), (20.0, 30.0), (0.0, 30.0)))
    meta = TileMeta("scene", "tile", 64, 0, 128, 128, 64, 128)

    assert keep_tile_prediction(
        prediction,
        meta,
        image_width=256,
        image_height=256,
        margin=0,
    ) is True


def test_merge_detections_suppresses_same_class_duplicate_and_keeps_other_classes() -> None:
    polygon = ((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0))
    detections = [
        _detection(image_id="scene", class_id=2, score=0.9, polygon=polygon),
        _detection(image_id="scene", class_id=2, score=0.7, polygon=polygon),
        _detection(image_id="scene", class_id=1, score=0.8, polygon=polygon),
    ]

    merged = merge_detections(detections, iou_threshold=0.3)

    assert [(item.class_id, item.score) for item in merged] == [(2, 0.9), (1, 0.8)]


def test_merge_detections_suppresses_when_polygon_iou_meets_threshold() -> None:
    polygon = ((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0))
    detections = [
        _detection(image_id="scene", class_id=0, score=0.9, polygon=polygon),
        _detection(image_id="scene", class_id=0, score=0.7, polygon=polygon),
    ]

    merged = merge_detections(detections, iou_threshold=1.0)

    assert merged == [detections[0]]


def test_merge_detections_keeps_different_images_and_classes() -> None:
    polygon = ((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0))
    detections = [
        _detection(image_id="image-a", class_id=0, score=0.9, polygon=polygon),
        _detection(image_id="image-b", class_id=0, score=0.8, polygon=polygon),
        _detection(image_id="image-a", class_id=1, score=0.7, polygon=polygon),
    ]

    merged = merge_detections(detections, iou_threshold=0.3)

    assert merged == [detections[0], detections[1], detections[2]]


def test_merge_detections_preserves_original_order_for_equal_scores_across_groups() -> None:
    group_a_left = _detection(
        image_id="scene",
        class_id=0,
        score=0.8,
        polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
    )
    group_b = _detection(
        image_id="other",
        class_id=1,
        score=0.8,
        polygon=((100.0, 100.0), (110.0, 100.0), (110.0, 110.0), (100.0, 110.0)),
    )
    group_a_right = _detection(
        image_id="scene",
        class_id=0,
        score=0.8,
        polygon=((30.0, 30.0), (40.0, 30.0), (40.0, 40.0), (30.0, 40.0)),
    )
    detections = [group_a_left, group_b, group_a_right]

    merged = merge_detections(detections, iou_threshold=0.3)

    assert merged == detections


def test_merge_detections_skips_polygon_iou_when_hbbs_do_not_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon_a = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    polygon_b = ((20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0))
    detections = [
        _detection(image_id="scene", class_id=0, score=0.9, polygon=polygon_a),
        _detection(image_id="scene", class_id=0, score=0.8, polygon=polygon_b),
    ]

    calls: list[tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]] = []

    def spy(left: tuple[tuple[float, float], ...], right: tuple[tuple[float, float], ...]) -> float:
        calls.append((left, right))
        raise AssertionError("polygon_iou should not be called when HBB IoU is zero")

    monkeypatch.setattr("xh_detect.merge.polygon_iou", spy)

    merged = merge_detections(detections, iou_threshold=0.3)

    assert merged == detections
    assert calls == []


def test_merge_detections_keeps_invalid_or_self_intersecting_polygon_when_iou_is_zero() -> None:
    bow_tie = ((0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0))
    square = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    detections = [
        _detection(image_id="scene", class_id=0, score=0.9, polygon=bow_tie),
        _detection(image_id="scene", class_id=0, score=0.8, polygon=square),
    ]

    merged = merge_detections(detections, iou_threshold=0.3)

    assert merged == detections


def test_merge_detections_accepts_generator_input() -> None:
    polygon = ((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0))

    def items() -> Iterable[Detection]:
        yield _detection(image_id="scene", class_id=1, score=0.9, polygon=polygon)
        yield _detection(image_id="scene", class_id=1, score=0.7, polygon=polygon)

    merged = merge_detections(items(), iou_threshold=0.3)

    assert len(merged) == 1
    assert merged[0].score == 0.9


def test_merge_detections_is_stable_and_does_not_modify_input() -> None:
    polygon_a = ((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0))
    polygon_b = ((50.0, 50.0), (70.0, 50.0), (70.0, 70.0), (50.0, 70.0))
    detections = [
        _detection(image_id="scene", class_id=0, score=0.8, polygon=polygon_a),
        _detection(image_id="scene", class_id=1, score=0.8, polygon=polygon_b),
    ]
    original = list(detections)

    merged = merge_detections(detections, iou_threshold=0.3)

    assert detections == original
    assert merged == original


def test_merge_detections_returns_empty_list_for_empty_input() -> None:
    assert merge_detections([], iou_threshold=0.3) == []


def test_project_prediction_keeps_detection_fields_and_center() -> None:
    prediction = _prediction(class_id=2, score=0.42)
    meta = TileMeta("scene", "tile", 3, 4, 128, 128, 128, 128)

    result = project_prediction(prediction, meta, image_width=256, image_height=256)

    assert result.image_id == "scene"
    assert result.class_id == 2
    assert result.score == 0.42
    assert obb_to_hbb(result.polygon) == (13.0, 14.0, 33.0, 34.0)
