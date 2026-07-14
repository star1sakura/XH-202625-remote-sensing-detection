from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from xh_detect.data.hard_example import (
    HardExamplePolicy,
    build_hard_example_dataset,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    labels = {
        "train-a": "3 0.15625 0.15625 0.15625 0.15625\n",
        "train-b": "24 0.234375 0.234375 0.15625 0.15625\n",
        "val-a": "",
    }
    for split, stems in {"train": ("train-a", "train-b"), "val": ("val-a",)}.items():
        for stem in stems:
            image_path = root / "images" / split / f"{stem}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (128, 128), (30, 40, 50)).save(image_path)
            label_path = root / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(labels[stem], encoding="utf-8")
    _write_json(root / "manifests" / "train-image-map.json", {"train-a": 10, "train-b": 11})
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
            "annotations": [
                {"image_id": 10, "category_id": 3, "bbox": [10, 10, 20, 20]},
                {"image_id": 11, "category_id": 24, "bbox": [20, 20, 20, 20]},
            ]
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


def test_builder_writes_missed_positive_crops_and_safe_negative(tmp_path: Path) -> None:
    source = _source(tmp_path)
    predictions = tmp_path / "predictions.json"
    _write_json(
        predictions,
        [
            {"image_id": 10, "category_id": 3, "bbox": [80, 80, 10, 10], "score": 0.9},
        ],
    )

    result = build_hard_example_dataset(
        source,
        predictions,
        tmp_path / "out",
        HardExamplePolicy(crop_size=64, vehicle_positive_multiplier=2),
    )

    assert result.missed_truth_by_coarse_class == {"ship": 1, "vehicle": 1}
    assert result.selected_positive_by_coarse_class == {"ship": 1, "vehicle": 2}
    assert result.selected_negative_by_coarse_class == {"ship": 1, "vehicle": 0}
    positive_labels = list((tmp_path / "out" / "labels" / "train").glob("*__hp_*.txt"))
    assert len(positive_labels) == 3
    assert any(path.read_text(encoding="utf-8").startswith("3 ") for path in positive_labels)
    assert sum(path.read_text(encoding="utf-8").startswith("24 ") for path in positive_labels) == 2
    negative_labels = list((tmp_path / "out" / "labels" / "train").glob("*__hn_*.txt"))
    assert len(negative_labels) == 1
    assert negative_labels[0].read_text(encoding="utf-8") == ""
    assert len(list((tmp_path / "out" / "images" / "val").glob("*.jpg"))) == 1


def test_builder_rejects_validation_prediction_id(tmp_path: Path) -> None:
    source = _source(tmp_path)
    predictions = tmp_path / "predictions.json"
    _write_json(
        predictions,
        [{"image_id": 99, "category_id": 24, "bbox": [80, 80, 10, 10], "score": 0.9}],
    )

    with pytest.raises(ValueError, match="not mapped to train"):
        build_hard_example_dataset(
            source,
            predictions,
            tmp_path / "out",
            HardExamplePolicy(crop_size=64),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crop_size": 0},
        {"background_score_floor": 1.1},
        {"max_positive_crops_per_group": 0},
        {"max_negative_crops_per_group": 0},
        {"vehicle_positive_multiplier": 0},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_hard_example_policy_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        HardExamplePolicy(**kwargs)
