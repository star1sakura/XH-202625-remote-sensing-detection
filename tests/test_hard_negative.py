from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from xh_detect.data.hard_negative import (
    HardNegativePolicy,
    build_main_hn_dataset,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for split, stems in {"train": ("train-a", "train-b"), "val": ("val-a",)}.items():
        for stem in stems:
            image_path = root / "images" / split / f"{stem}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (128, 128), (30, 40, 50)).save(image_path)
            label_path = root / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                "24 0.1171875 0.1171875 0.078125 0.078125\n" if stem == "train-a" else "",
                encoding="utf-8",
            )
    (root / "manifests").mkdir(parents=True)
    (root / "manifests" / "train.txt").write_text(
        "images/train/train-a.jpg\nimages/train/train-b.jpg\n",
        encoding="utf-8",
    )
    (root / "manifests" / "val.txt").write_text("images/val/val-a.jpg\n", encoding="utf-8")
    _write_json(root / "manifests" / "train-image-map.json", {"train-a": 10, "train-b": 11})
    _write_json(root / "manifests" / "val-image-map.json", {"val-a": 20})
    _write_json(
        root / "manifests" / "source-groups.json",
        {
            "train-a": {"group": "group-a", "split": "train"},
            "train-b": {"group": "group-b", "split": "train"},
            "val-a": {"group": "group-v", "split": "val"},
        },
    )
    _write_json(
        root / "reports" / "train-ground-truth.json",
        {
            "images": [
                {"id": 10, "file_name": "images/train/train-a.jpg"},
                {"id": 11, "file_name": "images/train/train-b.jpg"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 10,
                    "category_id": 24,
                    "bbox": [10, 10, 10, 10],
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": class_id, "name": str(class_id)} for class_id in range(25)],
        },
    )
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": {class_id: str(class_id) for class_id in range(25)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return root


def _predictions(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "predictions.json"
    _write_json(path, records)
    return path


def test_builder_is_seed_deterministic_and_writes_empty_labels(tmp_path: Path) -> None:
    source = _source(tmp_path)
    predictions = _predictions(
        tmp_path,
        [{"image_id": 11, "category_id": 24, "bbox": [80, 80, 10, 10], "score": 0.9}],
    )
    policy = HardNegativePolicy(0.60, 64, 8, 1, 2, 42)

    first = build_main_hn_dataset(source, predictions, tmp_path / "out-a", policy)
    second = build_main_hn_dataset(source, predictions, tmp_path / "out-b", policy)

    assert first.selected_hard_negatives == second.selected_hard_negatives == 1
    assert first.vehicle_upsampled_images == 1
    assert (tmp_path / "out-a" / "labels" / "train" / "train-b__hn01.txt").read_text(
        encoding="utf-8"
    ) == ""
    first_manifest = (tmp_path / "out-a" / "manifests" / "train.txt").read_text(encoding="utf-8")
    second_manifest = (tmp_path / "out-b" / "manifests" / "train.txt").read_text(encoding="utf-8")
    assert first_manifest == second_manifest
    assert "images/train/train-a__vehup01.jpg" in first_manifest
    assert len(list((tmp_path / "out-a" / "images" / "val").glob("*.jpg"))) == 1


def test_builder_rejects_crop_that_overlaps_any_train_target(tmp_path: Path) -> None:
    source = _source(tmp_path)
    predictions = _predictions(
        tmp_path,
        [{"image_id": 10, "category_id": 3, "bbox": [25, 25, 8, 8], "score": 0.9}],
    )

    with pytest.raises(ValueError, match="no label-safe hard negatives"):
        build_main_hn_dataset(
            source,
            predictions,
            tmp_path / "out",
            HardNegativePolicy(crop_size=64, object_margin=8),
        )


def test_builder_caps_candidates_by_source_group(tmp_path: Path) -> None:
    source = _source(tmp_path)
    groups_path = source / "manifests" / "source-groups.json"
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    groups["train-b"]["group"] = "group-a"
    _write_json(groups_path, groups)
    predictions = _predictions(
        tmp_path,
        [
            {"image_id": 10, "category_id": 3, "bbox": [90, 90, 8, 8], "score": 0.95},
            {"image_id": 11, "category_id": 24, "bbox": [80, 80, 8, 8], "score": 0.90},
        ],
    )

    result = build_main_hn_dataset(
        source,
        predictions,
        tmp_path / "out",
        HardNegativePolicy(crop_size=32, max_crops_per_group=1),
    )

    assert result.selected_hard_negatives == 1


def test_builder_rejects_non_train_prediction_id(tmp_path: Path) -> None:
    source = _source(tmp_path)
    predictions = _predictions(
        tmp_path,
        [{"image_id": 20, "category_id": 24, "bbox": [80, 80, 8, 8], "score": 0.9}],
    )

    with pytest.raises(ValueError, match="not mapped to a train image"):
        build_main_hn_dataset(source, predictions, tmp_path / "out", HardNegativePolicy())


def test_builder_rejects_overlapping_or_nonempty_output(tmp_path: Path) -> None:
    source = _source(tmp_path)
    predictions = _predictions(
        tmp_path,
        [{"image_id": 11, "category_id": 3, "bbox": [80, 80, 8, 8], "score": 0.9}],
    )

    with pytest.raises(ValueError, match="overlap"):
        build_main_hn_dataset(source, predictions, source / "derived", HardNegativePolicy())
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        build_main_hn_dataset(source, predictions, output, HardNegativePolicy())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_floor": 1.1},
        {"crop_size": 0},
        {"object_margin": -1},
        {"max_crops_per_group": 0},
        {"vehicle_multiplier": 0},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_hard_negative_policy_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        HardNegativePolicy(**kwargs)
