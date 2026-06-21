from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xh_detect.exporters import export_coco_results, validate_coco_results
from xh_detect.types import Detection


def _detection(
    image_id: str = "scene-1",
    class_id: int = 1,
    score: float = 0.9,
    polygon: tuple[tuple[float, float], ...] = (
        (10.0, 20.0),
        (30.0, 20.0),
        (30.0, 50.0),
        (10.0, 50.0),
    ),
) -> Detection:
    return Detection(image_id=image_id, class_id=class_id, score=score, polygon=polygon)  # type: ignore[arg-type]


def test_validate_coco_results_accepts_strict_records() -> None:
    records = [
        {
            "image_id": 3,
            "category_id": 0,
            "bbox": [10.0, -2.5, 20.0, 30.0],
            "score": 1.0,
        }
    ]

    validate_coco_results(records)


@pytest.mark.parametrize(
    ("record", "match"),
    [
        ({"image_id": 1, "category_id": 0, "bbox": [1.0, 2.0, 3.0, 4.0]}, "score"),
        (
            {
                "image_id": 1,
                "category_id": 0,
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "score": 0.5,
                "extra": 1,
            },
            "unexpected",
        ),
        (
            {"image_id": True, "category_id": 0, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.5},
            "image_id",
        ),
        (
            {"image_id": 1, "category_id": 3, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.5},
            "category_id",
        ),
        (
            {"image_id": 1, "category_id": 0, "bbox": (1.0, 2.0, 3.0, 4.0), "score": 0.5},
            "bbox",
        ),
        (
            {"image_id": 1, "category_id": 0, "bbox": [1.0, 2.0, 3.0], "score": 0.5},
            "bbox",
        ),
        (
            {"image_id": 1, "category_id": 0, "bbox": [1.0, 2.0, float("nan"), 4.0], "score": 0.5},
            "finite",
        ),
        (
            {"image_id": 1, "category_id": 0, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 1.5},
            "score",
        ),
    ],
)
def test_validate_coco_results_rejects_bad_records(record: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        validate_coco_results([record])


def test_validate_coco_results_rejects_duplicate_boxes_even_with_different_scores() -> None:
    records = [
        {"image_id": 1, "category_id": 0, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9},
        {"image_id": 1, "category_id": 0, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.7},
    ]

    with pytest.raises(ValueError, match="duplicate"):
        validate_coco_results(records)


def test_validate_coco_results_allows_same_bbox_on_different_class_or_image() -> None:
    records = [
        {"image_id": 1, "category_id": 0, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9},
        {"image_id": 2, "category_id": 0, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.8},
        {"image_id": 1, "category_id": 1, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.7},
    ]

    validate_coco_results(records)


def test_export_coco_results_preserves_order_and_writes_unicode_json(tmp_path: Path) -> None:
    destination = tmp_path / "结果" / "detections.json"
    detections = [
        _detection(image_id="img-a", class_id=2, score=0.95),
        _detection(
            image_id="img-b",
            class_id=0,
            score=0.5,
            polygon=(
                (-3.0, 1.0),
                (4.0, 1.0),
                (4.0, 8.0),
                (-3.0, 8.0),
            ),
        ),
    ]

    result = export_coco_results(detections, {"img-a": 10, "img-b": 11}, destination)

    assert result == destination
    payload = destination.read_text(encoding="utf-8")
    assert "结果" in str(destination)
    assert payload.startswith('[\n  {\n    "image_id": 10')
    assert "\n  }\n]" in payload
    assert json.loads(payload) == [
        {"image_id": 10, "category_id": 2, "bbox": [10.0, 20.0, 20.0, 30.0], "score": 0.95},
        {"image_id": 11, "category_id": 0, "bbox": [-3.0, 1.0, 7.0, 7.0], "score": 0.5},
    ]


def test_export_coco_results_normalizes_numpy_scalars_for_json(tmp_path: Path) -> None:
    destination = tmp_path / "numpy-scalars.json"
    detection = _detection(
        image_id="scene-1",
        class_id=np.int64(2),  # type: ignore[arg-type]
        score=np.float32(0.75),  # type: ignore[arg-type]
    )

    export_coco_results([detection], {"scene-1": np.int64(7)}, destination)

    records = json.loads(destination.read_text(encoding="utf-8"))
    assert records[0]["image_id"] == 7
    assert records[0]["category_id"] == 2
    assert records[0]["score"] == pytest.approx(0.75)


def test_export_coco_results_accepts_generator_and_empty_input(tmp_path: Path) -> None:
    destination = tmp_path / "results.json"

    def items() -> object:
        if False:
            yield _detection()  # pragma: no cover

    result = export_coco_results(items(), {"scene-1": 1}, destination)

    assert result == destination
    assert destination.read_text(encoding="utf-8") == "[]"


def test_export_coco_results_rejects_unknown_image_id_and_duplicate_map_values(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "results.json"

    with pytest.raises(ValueError, match="non-empty"):
        export_coco_results([_detection()], {"": 1}, destination)

    with pytest.raises(TypeError, match="image_id_map"):
        export_coco_results([_detection()], {"scene-1": True}, destination)

    with pytest.raises(ValueError, match="unknown image_id"):
        export_coco_results([_detection(image_id="missing")], {"scene-1": 1}, destination)

    with pytest.raises(ValueError, match="unique"):
        export_coco_results([_detection()], {"scene-1": 1, "scene-2": 1}, destination)


def test_export_coco_results_rejects_nearby_bboxes_only_when_exact_duplicate(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "results.json"
    detections = [
        _detection(
            image_id="scene-1",
            polygon=((10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)),
        ),
        _detection(
            image_id="scene-1",
            score=0.8,
            polygon=((10.0, 10.0), (20.0000001, 10.0), (20.0000001, 20.0), (10.0, 20.0)),
        ),
    ]

    result = export_coco_results(detections, {"scene-1": 1}, destination)

    assert result == destination
    assert len(json.loads(destination.read_text(encoding="utf-8"))) == 2


@pytest.mark.parametrize(
    ("polygon", "match"),
    [
        (
            (
                (0.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (10.0, 0.0),
            ),
            "scene-1",
        ),
        (
            (
                (0.0, 0.0),
                (5.0, 5.0),
                (10.0, 10.0),
                (15.0, 15.0),
            ),
            "scene-1",
        ),
        (
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, float("inf")),
                (0.0, 1.0),
            ),
            "scene-1",
        ),
    ],
)
def test_export_coco_results_rejects_invalid_detection_polygon_and_keeps_existing_file(
    tmp_path: Path,
    polygon: tuple[tuple[float, float], ...],
    match: str,
) -> None:
    destination = tmp_path / "results.json"
    destination.write_text("old content", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        export_coco_results([_detection(polygon=polygon)], {"scene-1": 1}, destination)

    assert destination.read_text(encoding="utf-8") == "old content"


def test_export_coco_results_accepts_valid_rotated_rectangle(tmp_path: Path) -> None:
    destination = tmp_path / "results.json"
    detection = _detection(
        polygon=(
            (1.0, 0.0),
            (3.0, 2.0),
            (2.0, 3.0),
            (0.0, 1.0),
        ),
    )

    result = export_coco_results([detection], {"scene-1": 7}, destination)

    assert result == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == [
        {"image_id": 7, "category_id": 1, "bbox": [0.0, 0.0, 3.0, 3.0], "score": 0.9},
    ]


def test_export_coco_results_uses_atomic_replace_and_keeps_old_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "results.json"
    destination.write_text("old content", encoding="utf-8")
    calls: list[tuple[Path, Path]] = []

    def fail_replace(source: str | bytes | Path, target: str | bytes | Path) -> None:
        calls.append((Path(source), Path(target)))
        raise RuntimeError("replace failed")

    monkeypatch.setattr("xh_detect.exporters.os.replace", fail_replace)

    with pytest.raises(RuntimeError, match="replace failed"):
        export_coco_results([_detection()], {"scene-1": 1}, destination)

    assert destination.read_text(encoding="utf-8") == "old content"
    assert len(calls) == 1
