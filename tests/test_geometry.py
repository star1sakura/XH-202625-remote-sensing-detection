import math

import pytest

from xh_detect.geometry import clip_polygon, hbb_iou, obb_to_hbb, polygon_iou

SQUARE = ((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0))


def test_obb_to_hbb() -> None:
    assert obb_to_hbb(SQUARE) == (10.0, 10.0, 30.0, 30.0)


def test_polygon_iou_identical() -> None:
    assert polygon_iou(SQUARE, SQUARE) == pytest.approx(1.0)


def test_polygon_iou_disjoint_returns_zero() -> None:
    left = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    right = ((20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0))
    assert polygon_iou(left, right) == 0.0


def test_polygon_iou_invalid_self_intersecting_returns_zero() -> None:
    bow_tie = ((0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0))
    assert polygon_iou(bow_tie, SQUARE) == 0.0


def test_hbb_iou_partial_overlap() -> None:
    assert hbb_iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)) == pytest.approx(1 / 3)


def test_hbb_iou_normalizes_reversed_coordinates() -> None:
    forward = (0.0, 0.0, 10.0, 10.0)
    reversed_box = (10.0, 10.0, 0.0, 0.0)

    assert hbb_iou(forward, reversed_box) == pytest.approx(1.0)


def test_hbb_iou_zero_area_returns_zero() -> None:
    assert hbb_iou((0.0, 0.0, 0.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 0.0


def test_polygon_iou_non_finite_coordinates_returns_zero() -> None:
    nan_polygon = ((math.nan, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))

    assert polygon_iou(nan_polygon, SQUARE) == 0.0


@pytest.mark.parametrize("width,height", [(0, 16), (16, 0), (-1, 16), (16, -1)])
def test_clip_polygon_rejects_non_positive_dimensions(width: int, height: int) -> None:
    polygon = ((-5.0, 5.0), (20.0, 5.0), (20.0, 25.0), (-5.0, 25.0))

    with pytest.raises(ValueError, match="width and height must be positive"):
        clip_polygon(polygon, width=width, height=height)


def test_clip_polygon_to_image() -> None:
    polygon = ((-5.0, 5.0), (20.0, 5.0), (20.0, 25.0), (-5.0, 25.0))
    assert clip_polygon(polygon, width=16, height=16) == (
        (0.0, 5.0),
        (15.0, 5.0),
        (15.0, 15.0),
        (0.0, 15.0),
    )
