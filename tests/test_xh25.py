from __future__ import annotations

import ctypes
import json
import os
import subprocess
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from itertools import combinations
from pathlib import Path
from time import perf_counter

import pytest
import yaml
from PIL import Image

from xh_detect.data.xh25 import (
    DatasetAudit,
    ImageRecord,
    PreparedDataset,
    _link_or_copy,
    _select_split,
    _select_validation_groups,
    audit_dataset,
    parse_yolo_hbb_label,
    prepare_dataset,
    source_group_id,
)
from xh_detect.taxonomy import get_taxonomy
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


def _write_class_sample(source_root: Path, stem: str, class_id: int) -> None:
    _write_image(
        source_root / "images" / "train" / f"{stem}.jpg",
        size=(16, 12),
        color=(
            (class_id * 37 + len(stem)) % 256,
            (class_id * 53 + len(stem) * 3) % 256,
            (class_id * 71 + len(stem) * 5) % 256,
        ),
    )
    _write_label(
        source_root / "labels" / "train" / f"{stem}.txt",
        f"{class_id} 0.5 0.5 0.5 0.5\n",
    )


def _write_complete_source(source_root: Path) -> None:
    for class_id in range(25):
        for group_index in range(3):
            group = f"class{class_id:02d}_group{group_index}"
            if class_id == 0 and group_index == 0:
                _write_class_sample(source_root, f"{group}_crop1", class_id)
                _write_class_sample(source_root, f"{group}_crop2", class_id)
            else:
                _write_class_sample(source_root, group, class_id)


def _write_multilabel_sample(
    source_root: Path,
    stem: str,
    class_ids: range | tuple[int, ...],
) -> None:
    _write_image(source_root / "images" / "train" / f"{stem}.jpg", size=(16, 12))
    _write_label(
        source_root / "labels" / "train" / f"{stem}.txt",
        "".join(f"{class_id} 0.5 0.5 0.5 0.5\n" for class_id in class_ids),
    )


def _markdown_json_section(markdown: str, heading: str) -> object:
    marker = f"## {heading}\n\n```json\n"
    section = markdown.split(marker, maxsplit=1)[1].split("\n```", maxsplit=1)[0]
    return json.loads(section)


def _recursive_tree_snapshot(root: Path) -> dict[str, tuple[str, bytes | None]]:
    return {
        path.relative_to(root).as_posix(): (
            "directory" if path.is_dir() else "file",
            None if path.is_dir() else path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
    }


def _memory_record(
    group_id: str,
    class_box_counts: dict[int, int],
) -> ImageRecord:
    polygon = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    return ImageRecord(
        stem=group_id,
        image_path=Path(f"{group_id}.jpg"),
        label_path=Path(f"{group_id}.txt"),
        width=1,
        height=1,
        mode="L",
        group_id=group_id,
        perceptual_hash="0000000000000000",
        annotations=tuple(
            ObjectAnnotation(
                image_id=group_id,
                class_id=class_id,
                polygon=polygon,
            )
            for class_id, box_count in sorted(class_box_counts.items())
            for _ in range(box_count)
        ),
    )


def _quality_objective(
    all_records: tuple[ImageRecord, ...],
    val_records: tuple[ImageRecord, ...],
    val_ratio: float,
) -> tuple[float, float, float, float]:
    taxonomy = get_taxonomy("xh25")

    def counts(
        records: tuple[ImageRecord, ...],
    ) -> tuple[Counter[int], Counter[int], Counter[str]]:
        boxes = Counter(
            annotation.class_id for record in records for annotation in record.annotations
        )
        images = Counter(
            class_id
            for record in records
            for class_id in {annotation.class_id for annotation in record.annotations}
        )
        coarse = Counter(
            taxonomy.coarse_name(annotation.class_id)
            for record in records
            for annotation in record.annotations
        )
        return boxes, images, coarse

    total_boxes, total_images, total_coarse = counts(all_records)
    val_boxes, val_images, val_coarse = counts(val_records)
    fine_box_deviations = [
        abs(val_boxes[class_id] / total_boxes[class_id] - val_ratio) for class_id in range(25)
    ]
    remaining_deviations = [
        abs(val_images[class_id] / total_images[class_id] - val_ratio) for class_id in range(25)
    ] + [
        abs(val_coarse[coarse_name] / total_coarse[coarse_name] - val_ratio)
        for coarse_name in ("ship", "aircraft", "vehicle")
    ]
    all_deviations = fine_box_deviations + remaining_deviations
    return (
        max(fine_box_deviations),
        sum(deviation * deviation for deviation in all_deviations),
        sum(all_deviations),
        abs(len(val_records) / len(all_records) - val_ratio),
    )


def test_data_package_exports_prepare_dataset_api() -> None:
    from xh_detect.data import PreparedDataset as ExportedPreparedDataset
    from xh_detect.data import prepare_dataset as exported_prepare_dataset

    assert ExportedPreparedDataset is PreparedDataset
    assert exported_prepare_dataset is prepare_dataset


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


def test_prepare_dataset_is_deterministic_group_safe_and_writes_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    _write_complete_source(source_root)

    first = prepare_dataset(source_root, first_output, val_ratio=0.15, seed=42)
    second = prepare_dataset(source_root, second_output, val_ratio=0.15, seed=42)

    assert first.train_stems == second.train_stems
    assert first.val_stems == second.val_stems
    assert first.train_groups == second.train_groups
    assert first.val_groups == second.val_groups
    assert first.train_groups.isdisjoint(first.val_groups)
    assert first.train_stems.isdisjoint(first.val_stems)
    assert all(first.train_class_counts[class_id] > 0 for class_id in range(25))
    assert all(first.val_class_counts[class_id] > 0 for class_id in range(25))
    assert (
        "class00_group0_crop1" in first.train_stems and "class00_group0_crop2" in first.train_stems
    ) or ("class00_group0_crop1" in first.val_stems and "class00_group0_crop2" in first.val_stems)

    train_manifest = (
        (first_output / "manifests" / "train.txt").read_text(encoding="utf-8").splitlines()
    )
    val_manifest = (first_output / "manifests" / "val.txt").read_text(encoding="utf-8").splitlines()
    assert train_manifest == sorted(train_manifest)
    assert val_manifest == sorted(val_manifest)
    assert train_manifest == [f"images/train/{stem}.jpg" for stem in sorted(first.train_stems)]
    assert val_manifest == [f"images/val/{stem}.jpg" for stem in sorted(first.val_stems)]
    assert all(not Path(entry).is_absolute() for entry in train_manifest + val_manifest)

    source_groups = json.loads(
        (first_output / "manifests" / "source-groups.json").read_text(encoding="utf-8")
    )
    assert list(source_groups) == sorted(source_groups)
    assert source_groups["class00_group0_crop1"]["group"] == "class00_group0"
    assert source_groups["class00_group0_crop1"]["split"] in {"train", "val"}

    first_image_map = json.loads(
        (first_output / "manifests" / "val-image-map.json").read_text(encoding="utf-8")
    )
    second_image_map = json.loads(
        (second_output / "manifests" / "val-image-map.json").read_text(encoding="utf-8")
    )
    assert first_image_map == second_image_map
    assert list(first_image_map) == sorted(first.val_stems)
    assert list(first_image_map.values()) == list(range(1, len(first.val_stems) + 1))

    demo_samples = json.loads(
        (first_output / "manifests" / "demo-samples.json").read_text(encoding="utf-8")
    )
    assert set(demo_samples) == {"ship", "aircraft", "vehicle"}
    assert all(path in val_manifest for path in demo_samples.values())

    taxonomy = get_taxonomy("xh25")
    dataset_yaml = yaml.safe_load((first_output / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset_yaml == {
        "path": str(first_output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(taxonomy.names),
    }

    coco = json.loads(
        (first_output / "reports" / "val-ground-truth.json").read_text(encoding="utf-8")
    )
    assert coco["categories"] == [
        {"id": class_id, "name": taxonomy.names[class_id]} for class_id in range(25)
    ]
    assert [image["id"] for image in coco["images"]] == list(range(1, len(first.val_stems) + 1))
    assert [image["file_name"] for image in coco["images"]] == val_manifest
    assert all(image["width"] == 16 and image["height"] == 12 for image in coco["images"])
    assert [annotation["id"] for annotation in coco["annotations"]] == list(
        range(1, len(coco["annotations"]) + 1)
    )
    assert all(annotation["bbox"] == [4.0, 3.0, 8.0, 6.0] for annotation in coco["annotations"])
    assert all(annotation["area"] == 48.0 for annotation in coco["annotations"])
    assert all(annotation["iscrowd"] == 0 for annotation in coco["annotations"])
    assert {annotation["image_id"] for annotation in coco["annotations"]} == set(
        first_image_map.values()
    )

    analysis = json.loads(
        (first_output / "reports" / "dataset-analysis.json").read_text(encoding="utf-8")
    )
    assert analysis["source"]["images"] == 76
    assert analysis["split"]["group_overlap"] == 0
    assert analysis["val_ratio"] == 0.15
    assert analysis["seed"] == 42
    assert set(analysis["link_mode_counts"]) <= {"hardlink", "symlink", "copy"}
    analysis_markdown = (first_output / "reports" / "dataset-analysis.md").read_text(
        encoding="utf-8"
    )
    for heading in (
        "Dimensions",
        "Modes",
        "Coarse counts",
        "Near-duplicate candidates",
        "Link modes",
        "Source target counts",
        "Target counts",
        "Source group counts",
    ):
        assert heading in analysis_markdown
    source_target_counts = _markdown_json_section(
        analysis_markdown,
        "Source target counts",
    )
    split_target_counts = _markdown_json_section(analysis_markdown, "Target counts")
    source_group_counts = _markdown_json_section(
        analysis_markdown,
        "Source group counts",
    )
    assert source_target_counts == analysis["source"]["targets"]
    assert source_target_counts["0"] == analysis["source"]["targets"]["0"]
    assert source_target_counts["24"] == analysis["source"]["targets"]["24"]
    assert split_target_counts == {
        "train": analysis["split"]["train"]["targets"],
        "val": analysis["split"]["val"]["targets"],
    }
    assert split_target_counts["train"]["0"] == analysis["split"]["train"]["targets"]["0"]
    assert split_target_counts["val"]["24"] == analysis["split"]["val"]["targets"]["24"]
    assert source_group_counts == {
        "source_groups": analysis["source"]["source_groups"],
        "train_groups": analysis["split"]["train"]["source_groups"],
        "val_groups": analysis["split"]["val"]["source_groups"],
    }


def test_prepare_dataset_rejects_class_with_one_source_group(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_class_sample(source_root, "class00_only", 0)
    for class_id in range(1, 25):
        _write_class_sample(source_root, f"class{class_id:02d}_group0", class_id)
        _write_class_sample(source_root, f"class{class_id:02d}_group1", class_id)

    with pytest.raises(ValueError, match=r"class 0.*source groups"):
        prepare_dataset(source_root, tmp_path / "output")


def test_prepare_dataset_fill_preserves_one_train_group_for_multilabel_classes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    _write_multilabel_sample(source_root, "g1", (0, 1))
    _write_multilabel_sample(source_root, "x1", (1,))
    _write_multilabel_sample(source_root, "g2", (0, 2))
    _write_multilabel_sample(source_root, "x2", (2,))
    _write_multilabel_sample(source_root, "g3", (0,))
    for group_id in ("h1", "h2", "h3"):
        _write_multilabel_sample(source_root, group_id, tuple(range(3, 25)))
    for index in range(20):
        _write_multilabel_sample(source_root, f"empty{index:02d}", ())

    prepared = prepare_dataset(
        source_root,
        tmp_path / "output",
        val_ratio=0.49,
        seed=11,
    )

    class_zero_groups = {"g1", "g2", "g3"}
    assert len(class_zero_groups & prepared.val_groups) == 2
    assert len(class_zero_groups & prepared.train_groups) == 1
    assert all(prepared.train_class_counts[class_id] > 0 for class_id in range(25))
    assert all(prepared.val_class_counts[class_id] > 0 for class_id in range(25))


def test_prepare_dataset_backtracks_to_find_valid_multilabel_split(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    group_classes = {
        "g0": (0, 1),
        "g1": (0, 1),
        "g2": (0, 2),
        "g3": (1, 2),
        "g4": (2,),
        "h0": tuple(range(3, 25)),
        "h1": tuple(range(3, 25)),
        "h2": tuple(range(3, 25)),
    }
    for group_id, class_ids in group_classes.items():
        _write_multilabel_sample(source_root, group_id, class_ids)

    prepared = prepare_dataset(
        source_root,
        tmp_path / "output",
        val_ratio=0.49,
        seed=0,
    )

    assert prepared.val_groups == frozenset({"g1", "g2", "g3", "h0", "h1"})
    assert prepared.train_groups == frozenset({"g0", "g4", "h2"})
    assert all(prepared.train_class_counts[class_id] > 0 for class_id in range(25))
    assert all(prepared.val_class_counts[class_id] > 0 for class_id in range(25))


def test_prepare_dataset_allows_safe_validation_below_image_target(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    all_classes = tuple(range(25))
    _write_multilabel_sample(source_root, "small", all_classes)
    for crop_index in range(1, 10):
        _write_multilabel_sample(source_root, f"large_crop{crop_index}", all_classes)

    prepared = prepare_dataset(
        source_root,
        tmp_path / "output",
        val_ratio=0.49,
        seed=1,
    )

    assert prepared.val_groups == frozenset({"small"})
    assert prepared.train_groups == frozenset({"large"})
    assert len(prepared.val_stems) == 1
    assert len(prepared.train_stems) == 9
    assert all(prepared.train_class_counts[class_id] > 0 for class_id in range(25))
    assert all(prepared.val_class_counts[class_id] > 0 for class_id in range(25))


def test_select_split_handles_thousands_of_groups_without_recursion() -> None:
    polygon = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

    def record(group_id: str, class_ids: tuple[int, ...]) -> ImageRecord:
        return ImageRecord(
            stem=group_id,
            image_path=Path(f"{group_id}.jpg"),
            label_path=Path(f"{group_id}.txt"),
            width=1,
            height=1,
            mode="L",
            group_id=group_id,
            perceptual_hash="0000000000000000",
            annotations=tuple(
                ObjectAnnotation(
                    image_id=group_id,
                    class_id=class_id,
                    polygon=polygon,
                )
                for class_id in class_ids
            ),
        )

    records = tuple(record(f"group{index:04d}", (0,)) for index in range(3000)) + (
        record("other0000", tuple(range(1, 25))),
        record("other0001", tuple(range(1, 25))),
    )
    audit = DatasetAudit(
        images=len(records),
        labels=len(records),
        targets={0: 3000, **{class_id: 2 for class_id in range(1, 25)}},
        images_per_class={0: 3000, **{class_id: 2 for class_id in range(1, 25)}},
        dimensions={"1x1": len(records)},
        modes={"L": len(records)},
        source_groups=len(records),
        invalid_lines=0,
        near_duplicate_candidates=(),
        records=records,
    )

    started = perf_counter()
    train_records, val_records = _select_split(audit, val_ratio=0.4, seed=42)
    elapsed = perf_counter() - started

    train_class_zero = sum(
        any(annotation.class_id == 0 for annotation in item.annotations) for item in train_records
    )
    val_class_zero = sum(
        any(annotation.class_id == 0 for annotation in item.annotations) for item in val_records
    )
    assert val_class_zero == 1200
    assert train_class_zero == 1800
    assert len(val_records) == 1201
    assert len(train_records) == 1801
    assert elapsed < 5.0


def test_select_split_optimizes_stratification_to_exhaustive_best_objective() -> None:
    records = tuple(
        _memory_record(
            group_id,
            dict.fromkeys(range(25), weight),
        )
        for group_id, weight in (
            ("weight08", 8),
            ("weight04", 4),
            ("weight03", 3),
            ("weight02", 2),
            ("weight01", 1),
        )
    )
    audit = DatasetAudit(
        images=len(records),
        labels=len(records),
        targets=dict.fromkeys(range(25), 18),
        images_per_class=dict.fromkeys(range(25), 5),
        dimensions={"1x1": len(records)},
        modes={"L": len(records)},
        source_groups=len(records),
        invalid_lines=0,
        near_duplicate_candidates=(),
        records=records,
    )
    oracle = min(
        _quality_objective(records, tuple(candidate), 0.4) for candidate in combinations(records, 2)
    )

    objectives = []
    for seed in (0, 3):
        _, val_records = _select_split(audit, val_ratio=0.4, seed=seed)
        objective = _quality_objective(records, val_records, 0.4)
        objectives.append(objective)
        assert objective == oracle
        assert objective[0] < 0.02
        assert objective[3] == 0.0

    assert objectives[0] == objectives[1]


def test_split_objective_matches_non_collinear_manual_and_exhaustive_oracle() -> None:
    group_counts = {
        "group-a": {**dict.fromkeys(range(25), 1), 0: 6, 24: 0},
        "group-b": {**dict.fromkeys(range(25), 1), 4: 0, 24: 2},
        "group-c": {**dict.fromkeys(range(25), 1), 0: 1, 4: 6, 24: 1},
        "group-d": {**dict.fromkeys(range(25), 1), 0: 0, 4: 1, 24: 5},
        "group-e": dict.fromkeys(range(25), 1),
    }
    records = tuple(
        _memory_record(
            group_id,
            {class_id: count for class_id, count in counts.items() if count},
        )
        for group_id, counts in group_counts.items()
    )
    manual_candidate = tuple(
        record for record in records if record.group_id in {"group-a", "group-d"}
    )
    assert _quality_objective(records, manual_candidate, 0.4) == pytest.approx(
        (
            4 / 15,
            65813 / 304200,
            1487 / 1170,
            0.0,
        )
    )

    targets = Counter(
        annotation.class_id for record in records for annotation in record.annotations
    )
    images_per_class = Counter(
        class_id
        for record in records
        for class_id in {annotation.class_id for annotation in record.annotations}
    )
    audit = DatasetAudit(
        images=len(records),
        labels=len(records),
        targets=dict(targets),
        images_per_class=dict(images_per_class),
        dimensions={"1x1": len(records)},
        modes={"L": len(records)},
        source_groups=len(records),
        invalid_lines=0,
        near_duplicate_candidates=(),
        records=records,
    )
    class_groups = {
        class_id: {
            record.group_id
            for record in records
            if any(annotation.class_id == class_id for annotation in record.annotations)
        }
        for class_id in range(25)
    }
    required = {
        class_id: max(
            2 if len(groups) >= 3 else 1,
            min(len(groups) - 1, round(len(groups) * 0.4)),
        )
        for class_id, groups in class_groups.items()
    }
    oracle = min(
        _quality_objective(records, tuple(candidate), 0.4)
        for candidate in combinations(records, 2)
        if all(
            len({record.group_id for record in candidate} & class_groups[class_id])
            >= required[class_id]
            for class_id in range(25)
        )
    )

    _, val_records = _select_split(audit, val_ratio=0.4, seed=0)

    assert _quality_objective(records, val_records, 0.4) == pytest.approx(oracle)


def test_validation_group_search_matches_exhaustive_feasibility_oracle() -> None:
    group_classes = {
        "g0": {0, 1},
        "g1": {0, 2},
        "g2": {0, 3},
        "g3": {1, 2},
        "g4": {1, 3},
        "g5": {2, 3},
        "g6": {0, 1, 2},
        "g7": {1, 2, 3},
        "g8": {0, 3},
        "h0": set(range(4, 25)),
        "h1": set(range(4, 25)),
        "h2": set(range(4, 25)),
    }
    class_groups = {class_id: set() for class_id in range(25)}
    for group_id, class_ids in group_classes.items():
        for class_id in class_ids:
            class_groups[class_id].add(group_id)
    required = {
        class_id: max(
            2 if len(groups) >= 3 else 1,
            min(len(groups) - 1, round(len(groups) * 0.49)),
        )
        for class_id, groups in class_groups.items()
    }
    all_groups = tuple(group_classes)
    feasible = {
        frozenset(candidate)
        for count in range(len(all_groups) + 1)
        for candidate in combinations(all_groups, count)
        if all(
            required[class_id]
            <= len(set(candidate) & class_groups[class_id])
            <= len(class_groups[class_id]) - 1
            for class_id in range(25)
        )
    }
    assert feasible

    for seed in range(6):
        selected = _select_validation_groups(
            class_groups,
            group_classes,
            val_ratio=0.49,
            seed=seed,
        )
        assert selected in feasible


@pytest.mark.parametrize("val_ratio", [float("nan"), float("inf"), -0.1, 0.0, 0.5, 1.0])
def test_prepare_dataset_rejects_invalid_val_ratio(tmp_path: Path, val_ratio: float) -> None:
    with pytest.raises(ValueError, match="val_ratio"):
        prepare_dataset(tmp_path / "source", tmp_path / "output", val_ratio=val_ratio)


@pytest.mark.parametrize("seed", [True, -1, 1.5, "42"])
def test_prepare_dataset_rejects_invalid_seed(tmp_path: Path, seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        prepare_dataset(tmp_path / "source", tmp_path / "output", seed=seed)


def test_link_or_copy_falls_back_to_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "nested" / "destination.txt"
    source.write_text("new content", encoding="utf-8")
    destination.parent.mkdir()
    destination.write_text("old content", encoding="utf-8")

    def fail_hardlink(_source: Path, _destination: Path) -> None:
        raise OSError("hard links unavailable")

    def fail_symlink(_self: Path, _target: Path, target_is_directory: bool = False) -> None:
        del target_is_directory
        raise OSError("symbolic links unavailable")

    monkeypatch.setattr(os, "link", fail_hardlink)
    monkeypatch.setattr(Path, "symlink_to", fail_symlink)

    assert _link_or_copy(source, destination) == "copy"
    assert destination.read_bytes() == source.read_bytes()


def test_prepare_dataset_replaces_previous_split_without_orphans(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_complete_source(source_root)
    first = prepare_dataset(source_root, output_root, seed=42)

    stale_image = output_root / "images" / "val" / "stale.jpg"
    stale_label = output_root / "labels" / "train" / "stale.txt"
    stale_image.write_bytes(b"stale")
    stale_label.write_text("stale", encoding="utf-8")

    second = prepare_dataset(source_root, output_root, seed=43)

    assert (first.train_stems, first.val_stems) != (
        second.train_stems,
        second.val_stems,
    )
    assert {path.stem for path in (output_root / "images" / "train").glob("*.jpg")} == set(
        second.train_stems
    )
    assert {path.stem for path in (output_root / "images" / "val").glob("*.jpg")} == set(
        second.val_stems
    )
    assert {path.stem for path in (output_root / "labels" / "train").glob("*.txt")} == set(
        second.train_stems
    )
    assert {path.stem for path in (output_root / "labels" / "val").glob("*.txt")} == set(
        second.val_stems
    )
    assert not stale_image.exists()
    assert not stale_label.exists()


def test_prepare_dataset_materialization_failure_preserves_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_complete_source(source_root)
    prepare_dataset(source_root, output_root, seed=42)
    before = _recursive_tree_snapshot(output_root)
    original_link_or_copy = _link_or_copy
    calls = 0

    def fail_seventh_copy(source: Path, destination: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 7:
            raise OSError("injected materialization failure")
        return original_link_or_copy(source, destination)

    monkeypatch.setattr("xh_detect.data.xh25._link_or_copy", fail_seventh_copy)

    with pytest.raises(OSError, match="injected materialization failure"):
        prepare_dataset(source_root, output_root, seed=43)

    assert calls == 7
    assert _recursive_tree_snapshot(output_root) == before
    assert [
        path.name
        for path in tmp_path.iterdir()
        if path.name.startswith(".output.") and path.name != output_root.name
    ] == []


def test_prepare_dataset_holds_directory_locks_during_materialization_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_complete_source(source_root)
    from xh_detect.data import xh25

    active_locks = 0
    lock_scopes: list[tuple[Path, ...]] = []
    original_link_or_copy = xh25._link_or_copy
    original_atomic_write_text = xh25._atomic_write_text

    @contextmanager
    def track_directory_locks(paths: tuple[Path, ...]):
        nonlocal active_locks
        normalized = tuple(Path(path) for path in paths)
        lock_scopes.append(normalized)
        active_locks += 1
        try:
            yield
        finally:
            active_locks -= 1

    def assert_locked_link(source: Path, destination: Path) -> str:
        assert active_locks > 0
        return original_link_or_copy(source, destination)

    def assert_locked_write(path: Path, text: str, output: Path) -> None:
        assert active_locks > 0
        original_atomic_write_text(path, text, output)

    monkeypatch.setattr(
        xh25,
        "_locked_directories",
        track_directory_locks,
        raising=False,
    )
    monkeypatch.setattr(xh25, "_link_or_copy", assert_locked_link)
    monkeypatch.setattr(xh25, "_atomic_write_text", assert_locked_write)

    prepare_dataset(source_root, output_root, seed=42)
    prepare_dataset(source_root, output_root, seed=43)

    assert any(
        {path.name for path in scope} >= {"images", "labels", "manifests", "reports"}
        for scope in lock_scopes
    )
    if os.name != "nt":
        assert any(
            len(scope) == 1 and scope[0].name.startswith(".output.backup-") for scope in lock_scopes
        )


@pytest.mark.skipif(os.name != "nt", reason="NTFS junction attack is Windows-only")
def test_prepare_dataset_directory_lock_blocks_real_junction_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_target = tmp_path / "probe-target"
    probe_link = tmp_path / "probe-link"
    probe_target.mkdir()
    probe = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(probe_link), str(probe_target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip(f"junction creation unavailable: {probe.stderr or probe.stdout}")
    subprocess.run(
        ["cmd", "/c", "rmdir", str(probe_link)],
        capture_output=True,
        check=True,
    )

    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    external_root = tmp_path / "external"
    _write_complete_source(source_root)
    external_root.mkdir()
    original_link_or_copy = _link_or_copy
    attack_results: dict[str, int] = {}
    attacked = False

    def attack_train_directory(train_directory: Path) -> None:
        removal = subprocess.run(
            ["cmd", "/c", "rmdir", str(train_directory)],
            capture_output=True,
            text=True,
            check=False,
        )
        attack_results["remove"] = removal.returncode
        if removal.returncode == 0:
            junction = subprocess.run(
                [
                    "cmd",
                    "/c",
                    "mklink",
                    "/J",
                    str(train_directory),
                    str(external_root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            attack_results["junction"] = junction.returncode

    def attack_before_first_write(source: Path, destination: Path) -> str:
        nonlocal attacked
        if not attacked and destination.parent.parts[-2:] == ("images", "train"):
            attacked = True
            thread = threading.Thread(
                target=attack_train_directory,
                args=(destination.parent,),
            )
            thread.start()
            thread.join(timeout=10)
            assert not thread.is_alive()
        return original_link_or_copy(source, destination)

    monkeypatch.setattr("xh_detect.data.xh25._link_or_copy", attack_before_first_write)

    prepare_dataset(source_root, output_root)

    assert attacked
    assert attack_results["remove"] != 0
    assert not list(external_root.glob("*.jpg"))
    assert not list(external_root.glob("*.txt"))


@pytest.mark.skipif(os.name != "nt", reason="NTFS junction attack is Windows-only")
def test_safe_remove_tree_lock_blocks_root_junction_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_target = tmp_path / "probe-target"
    probe_link = tmp_path / "probe-link"
    probe_target.mkdir()
    probe = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(probe_link), str(probe_target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip(f"junction creation unavailable: {probe.stderr or probe.stdout}")
    subprocess.run(
        ["cmd", "/c", "rmdir", str(probe_link)],
        capture_output=True,
        check=True,
    )

    from xh_detect.data import xh25

    cleanup_root = tmp_path / "cleanup-root"
    external_root = tmp_path / "external"
    cleanup_root.mkdir()
    external_root.mkdir()
    marker = external_root / "marker.txt"
    marker.write_text("do not delete", encoding="utf-8")
    original_scandir = os.scandir
    attack_results: dict[str, int] = {}
    attacked = False

    def attack_cleanup_root() -> None:
        removal = subprocess.run(
            ["cmd", "/c", "rmdir", str(cleanup_root)],
            capture_output=True,
            text=True,
            check=False,
        )
        attack_results["remove"] = removal.returncode
        if removal.returncode == 0:
            junction = subprocess.run(
                [
                    "cmd",
                    "/c",
                    "mklink",
                    "/J",
                    str(cleanup_root),
                    str(external_root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            attack_results["junction"] = junction.returncode

    def attack_during_scandir(path: Path | str):
        nonlocal attacked
        if not attacked and Path(path) == cleanup_root:
            attacked = True
            thread = threading.Thread(target=attack_cleanup_root)
            thread.start()
            thread.join(timeout=10)
            assert not thread.is_alive()
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", attack_during_scandir)

    xh25._safe_remove_tree(cleanup_root)

    assert attacked
    assert attack_results["remove"] != 0
    assert marker.read_text(encoding="utf-8") == "do not delete"
    assert not cleanup_root.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle cleanup is Windows-only")
def test_safe_remove_tree_windows_deletes_nested_directories_without_shutil_rmtree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect.data import xh25

    cleanup_root = tmp_path / "cleanup-root"
    nested = cleanup_root / "nested"
    nested.mkdir(parents=True)
    (nested / "marker.txt").write_text("remove me", encoding="utf-8")
    rmtree_calls: list[Path] = []

    def fail_rmtree(path: Path | str, *args: object, **kwargs: object) -> None:
        del args, kwargs
        rmtree_calls.append(Path(path))
        raise AssertionError("Windows cleanup must not delegate nested trees to shutil.rmtree")

    monkeypatch.setattr(xh25.shutil, "rmtree", fail_rmtree)

    xh25._safe_remove_tree(cleanup_root)

    assert rmtree_calls == []
    assert not cleanup_root.exists()


@pytest.mark.skipif(os.name != "nt", reason="NTFS junction cleanup is Windows-only")
def test_safe_remove_tree_removes_nested_junction_without_touching_target(
    tmp_path: Path,
) -> None:
    from xh_detect.data import xh25

    cleanup_root = tmp_path / "cleanup-root"
    nested = cleanup_root / "nested"
    external_root = tmp_path / "external"
    nested.mkdir(parents=True)
    external_root.mkdir()
    marker = external_root / "marker.txt"
    marker.write_text("do not delete", encoding="utf-8")
    junction = nested / "junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"junction creation unavailable: {created.stderr or created.stdout}")

    xh25._safe_remove_tree(cleanup_root)

    assert marker.read_text(encoding="utf-8") == "do not delete"
    assert not cleanup_root.exists()


def test_open_windows_directory_createfile_failure_preserves_winerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect.data import xh25_fs

    class FakeFunction:
        argtypes: object
        restype: object

        def __init__(self, result: object) -> None:
            self.result = result

        def __call__(self, *args: object) -> object:
            del args
            return self.result

    class FakeKernel32:
        def __init__(self) -> None:
            self.CreateFileW = FakeFunction(ctypes.c_void_p(-1).value)
            self.GetFileInformationByHandle = FakeFunction(0)
            self.CloseHandle = FakeFunction(1)

    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: FakeKernel32(), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 3, raising=False)

    with pytest.raises(OSError) as error:
        xh25_fs._open_windows_directory(Path("missing-directory"))

    assert getattr(error.value, "winerror", None) == 3
    assert not isinstance(error.value, ProcessLookupError)


def test_prepare_dataset_publish_failure_restores_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_complete_source(source_root)
    prepare_dataset(source_root, output_root, seed=42)
    before = _recursive_tree_snapshot(output_root)
    original_replace = os.replace

    def fail_stage_publish(source: Path | str, destination: Path | str) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == output_root and source_path.name.startswith(".output.stage-"):
            raise OSError("injected publish failure")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_stage_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        prepare_dataset(source_root, output_root, seed=43)

    assert _recursive_tree_snapshot(output_root) == before
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".output.")] == []


@pytest.mark.parametrize("existing_output", [False, True])
def test_prepare_dataset_recognizes_replace_that_completed_before_wrapper_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_output: bool,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_complete_source(source_root)
    before = None
    if existing_output:
        prepare_dataset(source_root, output_root, seed=42)
        before = _recursive_tree_snapshot(output_root)
    original_replace = os.replace
    injected = False

    def replace_then_raise(source: Path | str, destination: Path | str) -> None:
        nonlocal injected
        source_path = Path(source)
        destination_path = Path(destination)
        original_replace(source, destination)
        if (
            not injected
            and destination_path == output_root
            and source_path.name.startswith(".output.stage-")
        ):
            injected = True
            raise RuntimeError("wrapper failed after successful replace")

    monkeypatch.setattr(os, "replace", replace_then_raise)

    prepared = prepare_dataset(source_root, output_root, seed=43)

    assert injected
    assert prepared.output_root == output_root
    if before is not None:
        assert _recursive_tree_snapshot(output_root) != before
    assert not (output_root / ".xh25-transaction").exists()
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".output.")] == []


def test_prepare_dataset_retains_backup_if_safe_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_complete_source(source_root)
    prepare_dataset(source_root, output_root, seed=42)
    before = _recursive_tree_snapshot(output_root)
    from xh_detect.data import xh25

    original_safe_remove_tree = xh25._safe_remove_tree

    def fail_backup_cleanup(path: Path) -> None:
        if Path(path).name.startswith(".output.backup-"):
            raise PermissionError("injected backup cleanup failure")
        original_safe_remove_tree(path)

    monkeypatch.setattr(xh25, "_safe_remove_tree", fail_backup_cleanup)

    with pytest.raises(RuntimeError, match="retained backup"):
        prepare_dataset(source_root, output_root, seed=43)

    backups = [path for path in tmp_path.iterdir() if path.name.startswith(".output.backup-")]
    assert len(backups) == 1
    assert _recursive_tree_snapshot(backups[0]) == before
    assert _recursive_tree_snapshot(output_root) != before


def test_prepare_dataset_retains_backup_if_rollback_cleanup_is_safety_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_complete_source(source_root)
    prepare_dataset(source_root, output_root, seed=42)
    before = _recursive_tree_snapshot(output_root)
    from xh_detect.data import xh25

    original_safe_remove_tree = xh25._safe_remove_tree

    def fail_output_cleanup(path: Path) -> None:
        if Path(path).name == output_root.name:
            raise ValueError("unsafe cleanup")
        original_safe_remove_tree(path)

    def fail_after_publish(root: Path, *args: object, **kwargs: object) -> None:
        if Path(root).name == output_root.name:
            raise RuntimeError("injected post-publish validation failure")

    monkeypatch.setattr(xh25, "_safe_remove_tree", fail_output_cleanup)
    monkeypatch.setattr(xh25, "_validate_materialized_dataset", fail_after_publish)

    with pytest.raises(
        RuntimeError,
        match=r"retained backup after rollback cleanup failure: .*\.output\.backup-",
    ):
        prepare_dataset(source_root, output_root, seed=43)

    backups = [path for path in tmp_path.iterdir() if path.name.startswith(".output.backup-")]
    assert len(backups) == 1
    assert _recursive_tree_snapshot(backups[0]) == before


@pytest.mark.parametrize("reparse_name", ["output", "manifests"])
def test_prepare_dataset_rejects_mocked_reparse_points_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reparse_name: str,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_complete_source(source_root)
    calls = 0
    original_link_or_copy = _link_or_copy

    def track_link_or_copy(source: Path, destination: Path) -> str:
        nonlocal calls
        calls += 1
        return original_link_or_copy(source, destination)

    def fake_is_reparse_point(path: Path) -> bool:
        path = Path(path)
        return path == output_root if reparse_name == "output" else path.name == "manifests"

    monkeypatch.setattr("xh_detect.data.xh25._link_or_copy", track_link_or_copy)
    monkeypatch.setattr(
        "xh_detect.data.xh25._is_reparse_point",
        fake_is_reparse_point,
        raising=False,
    )

    with pytest.raises(ValueError, match="reparse"):
        prepare_dataset(source_root, output_root)

    assert calls == 0
    assert not output_root.exists()
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".output.")] == []


def test_prepare_dataset_rechecks_reparse_points_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_complete_source(source_root)
    prepare_dataset(source_root, output_root, seed=42)
    before = _recursive_tree_snapshot(output_root)
    output_checks = 0

    def become_reparse_before_publish(path: Path) -> bool:
        nonlocal output_checks
        if Path(path) == output_root:
            output_checks += 1
            return output_checks >= 2
        return False

    monkeypatch.setattr(
        "xh_detect.data.xh25._is_reparse_point",
        become_reparse_before_publish,
        raising=False,
    )

    with pytest.raises(ValueError, match="reparse"):
        prepare_dataset(source_root, output_root, seed=43)

    assert output_checks >= 2
    assert _recursive_tree_snapshot(output_root) == before
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".output.")] == []


def test_prepare_dataset_rejects_symlink_output_without_external_writes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    external_root = tmp_path / "external"
    _write_complete_source(source_root)
    external_root.mkdir()
    marker = external_root / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")
    try:
        output_root.symlink_to(external_root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    before = _recursive_tree_snapshot(external_root)

    with pytest.raises(ValueError, match="reparse"):
        prepare_dataset(source_root, output_root)

    assert _recursive_tree_snapshot(external_root) == before


@pytest.mark.parametrize(
    "path_relation",
    ["equal", "output_inside_source", "source_inside_output"],
)
def test_prepare_dataset_rejects_overlapping_paths_without_side_effects(
    tmp_path: Path,
    path_relation: str,
) -> None:
    if path_relation == "equal":
        source_root = tmp_path / "source"
        output_root = source_root
    elif path_relation == "output_inside_source":
        source_root = tmp_path / "source"
        output_root = source_root / "prepared"
    else:
        output_root = tmp_path / "output"
        source_root = output_root / "source"
    _write_complete_source(source_root)
    before = _recursive_tree_snapshot(tmp_path)

    with pytest.raises(ValueError) as error:
        prepare_dataset(source_root, output_root)

    assert "overlap" in str(error.value)
    assert _recursive_tree_snapshot(tmp_path) == before
    assert not (source_root / "manifests").exists()
    assert not (source_root / "reports").exists()
    assert not (source_root / "dataset.yaml").exists()


@pytest.mark.parametrize("metadata_directory", ["manifests", "reports"])
@pytest.mark.parametrize("target_kind", ["source", "external"])
def test_prepare_dataset_rejects_metadata_directory_link_outside_output(
    tmp_path: Path,
    metadata_directory: str,
    target_kind: str,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    external_root = tmp_path / "external"
    _write_complete_source(source_root)
    output_root.mkdir()
    external_root.mkdir()
    target = source_root if target_kind == "source" else external_root
    linked_directory = output_root / metadata_directory
    try:
        linked_directory.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    before = {path.relative_to(target) for path in target.rglob("*")}

    with pytest.raises(ValueError, match="outside output_root"):
        prepare_dataset(source_root, output_root)

    assert {path.relative_to(target) for path in target.rglob("*")} == before


def test_prepare_dataset_accepts_ordinary_metadata_directories(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_complete_source(source_root)
    (output_root / "manifests").mkdir(parents=True)
    (output_root / "reports").mkdir()

    prepared = prepare_dataset(source_root, output_root)

    assert prepared.output_root == output_root
    assert (output_root / "manifests" / "train.txt").is_file()
    assert (output_root / "reports" / "dataset-analysis.json").is_file()
    assert (output_root / "dataset.yaml").is_file()


def test_prepare_dataset_rejects_metadata_parent_resolving_outside_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    external_root = tmp_path / "external"
    _write_complete_source(source_root)
    output_root.mkdir()
    external_root.mkdir()
    original_resolve = Path.resolve

    def resolve_metadata_link(path: Path, *args: object, **kwargs: object) -> Path:
        if path == output_root / "manifests":
            return original_resolve(external_root)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_metadata_link)

    with pytest.raises(ValueError, match="outside output_root"):
        prepare_dataset(source_root, output_root)

    assert list(external_root.iterdir()) == []


def test_prepared_dataset_is_frozen_and_defensively_copies_collections() -> None:
    train_stems = {"train"}
    val_stems = {"val"}
    train_groups = {"train-group"}
    val_groups = {"val-group"}
    train_counts = {0: 1}
    val_counts = {24: 2}
    prepared = PreparedDataset(
        output_root=Path("output"),
        train_stems=train_stems,
        val_stems=val_stems,
        train_groups=train_groups,
        val_groups=val_groups,
        train_class_counts=train_counts,
        val_class_counts=val_counts,
    )

    train_stems.add("changed")
    val_stems.add("changed")
    train_groups.add("changed")
    val_groups.add("changed")
    train_counts[0] = 99
    val_counts[24] = 99

    assert prepared.train_stems == frozenset({"train"})
    assert prepared.val_stems == frozenset({"val"})
    assert prepared.train_groups == frozenset({"train-group"})
    assert prepared.val_groups == frozenset({"val-group"})
    assert prepared.train_class_counts == {
        **dict.fromkeys(range(25), 0),
        0: 1,
    }
    assert prepared.val_class_counts == {
        **dict.fromkeys(range(25), 0),
        24: 2,
    }
    with pytest.raises(TypeError):
        prepared.train_class_counts[0] = 2
    with pytest.raises(FrozenInstanceError):
        prepared.output_root = Path("changed")
