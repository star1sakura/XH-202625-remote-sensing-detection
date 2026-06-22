from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from PIL import Image

from xh_detect.data.xh25 import (
    DatasetAudit,
    ImageRecord,
    audit_dataset,
    parse_yolo_hbb_label,
    source_group_id,
)
from xh_detect.types import ObjectAnnotation


def _write_image(
    path: Path,
    *,
    size: tuple[int, int] = (100, 80),
    color: tuple[int, int, int] = (10, 20, 30),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_label(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_parse_yolo_hbb_label_returns_hbb_polygon(tmp_path: Path) -> None:
    label_path = tmp_path / "sample.txt"
    _write_label(label_path, "24 0.5 0.5 0.2 0.25\n")

    annotations = parse_yolo_hbb_label(label_path, "sample", 100, 80)

    assert annotations == (
        ObjectAnnotation(
            image_id="sample",
            class_id=24,
            polygon=((40, 30), (60, 30), (60, 50), (40, 50)),
        ),
    )


def test_parse_clamps_six_decimal_rounding_at_normalized_boundary(
    tmp_path: Path,
) -> None:
    label_path = tmp_path / "rounded.txt"
    _write_label(label_path, "3 0.969870 0.616857 0.060261 0.124593\n")

    (annotation,) = parse_yolo_hbb_label(label_path, "rounded", 1228, 1000)

    assert annotation.polygon[1][0] == 1228
    assert annotation.polygon[2][0] == 1228


def test_parse_rejects_box_outside_normalized_boundary_tolerance(
    tmp_path: Path,
) -> None:
    label_path = tmp_path / "outside.txt"
    _write_label(label_path, "3 0.969870 0.616857 0.060265 0.124593\n")

    with pytest.raises(ValueError, match="outside image"):
        parse_yolo_hbb_label(label_path, "outside", 1228, 1000)


@pytest.mark.parametrize("dimensions", [(0, 80), (100, 0), (-1, 80), (100, -1)])
def test_parse_rejects_non_positive_image_dimensions(
    tmp_path: Path, dimensions: tuple[int, int]
) -> None:
    label_path = tmp_path / "dimensions.txt"
    _write_label(label_path, "0 0.5 0.5 0.2 0.2\n")

    with pytest.raises(ValueError, match="width and height") as error:
        parse_yolo_hbb_label(label_path, "dimensions", *dimensions)

    assert str(label_path) in str(error.value)


def test_parse_rejects_non_positive_height_parameter(tmp_path: Path) -> None:
    label_path = tmp_path / "height.txt"
    _write_label(label_path, "0 0.5 0.5 0.2 0.2\n")

    with pytest.raises(ValueError, match="width and height must be positive") as error:
        parse_yolo_hbb_label(label_path, "height", 100, -1)

    assert f"{label_path}:0" in str(error.value)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("25 0.5 0.5 0.2 0.2", "class ID 25"),
        ("0 0.5 0.5 0.2", "five fields"),
        ("zero 0.5 0.5 0.2 0.2", "numeric"),
        ("0 center 0.5 0.2 0.2", "numeric"),
        ("0 nan 0.5 0.2 0.2", "finite"),
        ("0 0.5 inf 0.2 0.2", "finite"),
        ("0 1.1 0.5 0.2 0.2", "outside image"),
        ("0 0.5 -0.1 0.2 0.2", "outside image"),
        ("0 0.5 0.5 0 0.2", "width and height must be positive"),
        ("0 0.5 0.5 -0.1 0.2", "width and height must be positive"),
        ("0 0.5 0.5 0.2 0", "width and height must be positive"),
        ("0 0.5 0.5 0.2 -0.1", "width and height must be positive"),
        ("0 0.5 0.5 1.1 0.2", "outside image"),
        ("0 0.1 0.5 0.4 0.2", "outside image"),
    ],
)
def test_parse_rejects_invalid_lines_with_path_and_line_number(
    tmp_path: Path, line: str, message: str
) -> None:
    label_path = tmp_path / "invalid.txt"
    _write_label(label_path, f"0 0.5 0.5 0.2 0.2\n{line}\n")

    with pytest.raises(ValueError) as error:
        parse_yolo_hbb_label(label_path, "invalid", 100, 80)

    error_message = str(error.value)
    assert f"{label_path}:2" in error_message
    assert message in error_message


def test_parse_rejects_blank_line_between_annotations(tmp_path: Path) -> None:
    label_path = tmp_path / "blank-line.txt"
    _write_label(
        label_path,
        "0 0.5 0.5 0.2 0.2\n\n24 0.5 0.5 0.2 0.2\n",
    )

    with pytest.raises(ValueError) as error:
        parse_yolo_hbb_label(label_path, "blank-line", 100, 80)

    assert f"{label_path}:2" in str(error.value)
    assert "five fields" in str(error.value)


def test_parse_allows_completely_empty_label_file(tmp_path: Path) -> None:
    label_path = tmp_path / "empty.txt"
    _write_label(label_path)

    assert parse_yolo_hbb_label(label_path, "empty", 100, 80) == ()


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("scene_crop1", "scene"),
        ("scene_crop0002", "scene"),
        ("scene_CROP42", "scene"),
        ("MAR20_1002", "MAR20_1002"),
        ("crop1_scene", "crop1_scene"),
        ("_crop1", "_crop1"),
    ],
)
def test_source_group_id_only_removes_trailing_crop_suffix(stem: str, expected: str) -> None:
    assert source_group_id(stem) == expected


@pytest.mark.parametrize("missing_subdir", [Path("images/train"), Path("labels/train")])
def test_audit_dataset_requires_train_directories(
    tmp_path: Path,
    missing_subdir: Path,
) -> None:
    source_root = tmp_path / "dataset"
    for subdir in (Path("images/train"), Path("labels/train")):
        if subdir != missing_subdir:
            (source_root / subdir).mkdir(parents=True)

    with pytest.raises(ValueError) as error:
        audit_dataset(source_root)

    assert str(source_root / missing_subdir) in str(error.value)
    assert "directory" in str(error.value)


@pytest.mark.parametrize("file_subdir", [Path("images/train"), Path("labels/train")])
def test_audit_dataset_rejects_train_path_that_is_not_directory(
    tmp_path: Path,
    file_subdir: Path,
) -> None:
    source_root = tmp_path / "dataset"
    for subdir in (Path("images/train"), Path("labels/train")):
        path = source_root / subdir
        path.parent.mkdir(parents=True, exist_ok=True)
        if subdir == file_subdir:
            path.write_text("not a directory", encoding="utf-8")
        else:
            path.mkdir()

    with pytest.raises(ValueError) as error:
        audit_dataset(source_root)

    assert str(source_root / file_subdir) in str(error.value)
    assert "directory" in str(error.value)


def test_audit_dataset_rejects_empty_dataset(tmp_path: Path) -> None:
    source_root = tmp_path / "dataset"
    (source_root / "images" / "train").mkdir(parents=True)
    (source_root / "labels" / "train").mkdir(parents=True)

    with pytest.raises(ValueError, match="empty"):
        audit_dataset(source_root)


def test_audit_dataset_reports_expected_statistics(tmp_path: Path) -> None:
    source_root = tmp_path / "dataset"
    _write_image(source_root / "images" / "train" / "scene_crop1.jpg")
    _write_image(
        source_root / "images" / "train" / "scene_crop0002.jpg",
        color=(40, 50, 60),
    )
    _write_label(
        source_root / "labels" / "train" / "scene_crop1.txt",
        "0 0.5 0.5 0.2 0.25\n",
    )
    _write_label(
        source_root / "labels" / "train" / "scene_crop0002.txt",
        "24 0.5 0.5 0.2 0.25\n",
    )

    report = audit_dataset(source_root)

    assert report.images == 2
    assert report.labels == 2
    assert report.targets == {0: 1, 24: 1}
    assert report.images_per_class == {0: 1, 24: 1}
    assert report.dimensions == {"100x80": 2}
    assert report.modes == {"RGB": 2}
    assert report.source_groups == 1
    assert report.invalid_lines == 0
    assert report.near_duplicate_candidates == ()
    assert tuple(record.stem for record in report.records) == (
        "scene_crop0002",
        "scene_crop1",
    )


def test_audit_dataset_reports_exact_cross_group_hash_matches(tmp_path: Path) -> None:
    source_root = tmp_path / "dataset"
    for stem in ("second", "first"):
        _write_image(source_root / "images" / "train" / f"{stem}.jpg")
        _write_label(source_root / "labels" / "train" / f"{stem}.txt")

    report = audit_dataset(source_root)

    assert report.near_duplicate_candidates == (("first", "second"),)
    assert {record.group_id for record in report.records} == {"first", "second"}


def test_audit_dataset_buckets_duplicate_candidates_by_hash_and_group(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "dataset"
    image_dir = source_root / "images" / "train"
    label_dir = source_root / "labels" / "train"
    vertical = Image.new("L", (8, 8))
    vertical.putdata([0 if x < 4 else 255 for _y in range(8) for x in range(8)])
    horizontal = Image.new("L", (8, 8))
    horizontal.putdata([0 if y < 4 else 255 for y in range(8) for _x in range(8)])
    solid = Image.new("L", (8, 8), 64)
    for stem, image in (
        ("first", vertical),
        ("second", vertical),
        ("different", horizontal),
        ("scene_crop1", solid),
        ("scene_crop2", solid),
    ):
        image_dir.mkdir(parents=True, exist_ok=True)
        image.save(image_dir / f"{stem}.jpg")
        _write_label(label_dir / f"{stem}.txt")

    report = audit_dataset(source_root)

    assert report.near_duplicate_candidates == (("first", "second"),)


def test_audit_dataset_aggregates_pairing_image_and_label_errors(tmp_path: Path) -> None:
    source_root = tmp_path / "dataset"
    _write_image(source_root / "images" / "train" / "missing_label.jpg")
    _write_label(source_root / "labels" / "train" / "missing_image.txt")
    broken_image = source_root / "images" / "train" / "broken.jpg"
    broken_image.parent.mkdir(parents=True, exist_ok=True)
    broken_image.write_bytes(b"not a jpeg")
    _write_label(source_root / "labels" / "train" / "broken.txt")
    _write_image(source_root / "images" / "train" / "invalid.jpg")
    invalid_label = source_root / "labels" / "train" / "invalid.txt"
    _write_label(invalid_label, "25 0.5 0.5 0.2 0.2\n")

    with pytest.raises(ValueError) as error:
        audit_dataset(source_root)

    error_message = str(error.value)
    assert "missing_label" in error_message
    assert "missing_image" in error_message
    assert str(broken_image) in error_message
    assert f"{invalid_label}:1" in error_message
    assert "class ID 25" in error_message


def test_audit_dataset_aggregates_decompression_bomb_as_image_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "dataset"
    image_path = source_root / "images" / "train" / "bomb.jpg"
    _write_image(image_path)
    _write_label(source_root / "labels" / "train" / "bomb.txt")

    def raise_decompression_bomb(_path: Path) -> None:
        raise Image.DecompressionBombError("image is too large")

    monkeypatch.setattr("xh_detect.data.xh25.Image.open", raise_decompression_bomb)

    with pytest.raises(ValueError) as error:
        audit_dataset(source_root)

    assert f"{image_path}: damaged image" in str(error.value)
    assert "image is too large" in str(error.value)


def test_audit_dataset_checks_invalid_label_when_paired_image_is_damaged(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "dataset"
    image_path = source_root / "images" / "train" / "same.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"not a jpeg")
    label_path = source_root / "labels" / "train" / "same.txt"
    _write_label(label_path, "25 0.5 0.5 0.2 0.2\n")

    with pytest.raises(ValueError) as error:
        audit_dataset(source_root)

    error_message = str(error.value)
    assert f"{image_path}: damaged image" in error_message
    assert f"{label_path}:1" in error_message
    assert "class ID 25" in error_message


def test_audit_structures_are_frozen_and_defensively_copy_collections() -> None:
    annotations = [
        ObjectAnnotation(
            image_id="sample",
            class_id=0,
            polygon=((0, 0), (1, 0), (1, 1), (0, 1)),
        )
    ]
    record = ImageRecord(
        stem="sample",
        image_path=Path("sample.jpg"),
        label_path=Path("sample.txt"),
        width=1,
        height=1,
        mode="L",
        group_id="sample",
        perceptual_hash="0000000000000000",
        annotations=annotations,
    )
    targets = {0: 1}
    images_per_class = {0: 1}
    dimensions = {"1x1": 1}
    modes = {"L": 1}
    duplicates = [["first", "second"]]
    records = [record]
    report = DatasetAudit(
        images=1,
        labels=1,
        targets=targets,
        images_per_class=images_per_class,
        dimensions=dimensions,
        modes=modes,
        source_groups=1,
        invalid_lines=0,
        near_duplicate_candidates=duplicates,
        records=records,
    )

    annotations.clear()
    targets[0] = 99
    images_per_class[0] = 99
    dimensions["1x1"] = 99
    modes["L"] = 99
    duplicates[0][0] = "changed"
    records.clear()

    assert len(record.annotations) == 1
    assert report.targets == {0: 1}
    assert report.images_per_class == {0: 1}
    assert report.dimensions == {"1x1": 1}
    assert report.modes == {"L": 1}
    assert report.near_duplicate_candidates == (("first", "second"),)
    assert report.records == (record,)
    with pytest.raises(TypeError):
        report.targets[0] = 2
    with pytest.raises(FrozenInstanceError):
        report.images = 2
