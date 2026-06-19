from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from xh_detect.types import (
    BoxPrediction,
    Detection,
    ImageArray,
    InferenceResult,
    ObjectAnnotation,
    Point,
    Polygon4,
    StageTimings,
    Tile,
    TileMeta,
)


def test_polygon_type_aliases_match_expected_shapes() -> None:
    assert Point == tuple[float, float]
    assert Polygon4 == tuple[Point, Point, Point, Point]
    assert ImageArray is not None
    assert np.asarray([[1]], dtype=np.uint8).dtype == np.dtype("uint8")


def test_object_annotation_and_related_records_are_frozen() -> None:
    polygon = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    annotation = ObjectAnnotation("image", 1, polygon)

    with pytest.raises(FrozenInstanceError):
        annotation.class_id = 2

    assert [field.name for field in fields(ObjectAnnotation)] == [
        "image_id",
        "class_id",
        "polygon",
        "difficult",
    ]
    assert [field.name for field in fields(BoxPrediction)] == ["class_id", "score", "polygon"]
    assert [field.name for field in fields(Detection)] == [
        "image_id",
        "class_id",
        "score",
        "polygon",
    ]
    assert [field.name for field in fields(TileMeta)] == [
        "image_id",
        "tile_id",
        "x",
        "y",
        "width",
        "height",
        "valid_width",
        "valid_height",
    ]
    assert [field.name for field in fields(Tile)] == ["image", "meta"]
    assert [field.name for field in fields(StageTimings)] == [
        "preprocess_s",
        "inference_s",
        "postprocess_s",
        "total_s",
    ]
    assert [field.name for field in fields(InferenceResult)] == ["detections", "timings"]
