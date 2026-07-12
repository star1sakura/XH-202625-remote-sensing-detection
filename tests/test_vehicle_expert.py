from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from xh_detect.data.vehicle_expert import (
    VehicleExpertPolicy,
    build_vehicle_expert_dataset,
)


def _write_source(root: Path) -> Path:
    image_map: dict[str, int] = {}
    groups: dict[str, dict[str, str]] = {}
    annotations: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for image_id in range(1, 7):
        stem = f"image-{image_id}"
        image_map[stem] = image_id
        groups[stem] = {"group": f"group-{image_id}", "split": "train"}
        image_path = root / "images" / "train" / f"{stem}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_path), np.full((128, 128, 3), image_id * 20, dtype=np.uint8))
        label_path = root / "labels" / "train" / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("", encoding="utf-8")
        if image_id <= 4:
            annotations.append({"image_id": image_id, "category_id": 24, "bbox": [56, 56, 8, 8]})
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": 24,
                    "bbox": [56, 56, 8, 8],
                    "score": 0.9,
                }
            )
        else:
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": 24,
                    "bbox": [8, 8, 8, 8],
                    "score": 0.8,
                }
            )
    manifests = root / "manifests"
    reports = root / "reports"
    manifests.mkdir()
    reports.mkdir()
    (manifests / "train-image-map.json").write_text(json.dumps(image_map), encoding="utf-8")
    (manifests / "source-groups.json").write_text(json.dumps(groups), encoding="utf-8")
    (reports / "train-ground-truth.json").write_text(
        json.dumps({"annotations": annotations}), encoding="utf-8"
    )
    prediction_path = root.parent / "sph.json"
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
    return prediction_path


def test_builds_one_class_positive_and_hard_negative_crops(tmp_path: Path) -> None:
    source = tmp_path / "source"
    predictions = _write_source(source)
    output = tmp_path / "vehicle-expert"

    result = build_vehicle_expert_dataset(
        source,
        predictions,
        output,
        VehicleExpertPolicy(crop_size=64, holdout_ratio=0.5, max_negatives_per_group=2),
    )

    labels = sorted((output / "labels").rglob("*.txt"))
    nonempty = [path for path in labels if path.read_text(encoding="utf-8").strip()]
    empty = [path for path in labels if not path.read_text(encoding="utf-8").strip()]
    assert result.positive_crops == 4
    assert result.negative_crops == 2
    assert len(nonempty) == 4
    assert len(empty) == 2
    assert all(
        all(line.startswith("0 ") for line in path.read_text(encoding="utf-8").splitlines())
        for path in nonempty
    )
    assert result.train_groups.isdisjoint(result.val_groups)
    assert result.train_positive > 0 and result.val_positive > 0
    assert (output / "dataset.yaml").is_file()
    assert len(list((output / "images" / "train").glob("*.jpg"))) == result.train_crops
    assert len(list((output / "images" / "val").glob("*.jpg"))) == result.val_crops
    source_val_map = json.loads(
        (output / "manifests" / "source-val-image-map.json").read_text(encoding="utf-8")
    )
    source_groups = json.loads(
        (source / "manifests" / "source-groups.json").read_text(encoding="utf-8")
    )
    assert {source_groups[stem]["group"] for stem in source_val_map} == result.val_groups
    source_val_truth = json.loads(
        (output / "reports" / "source-val-ground-truth.json").read_text(encoding="utf-8")
    )
    assert {item["image_id"] for item in source_val_truth["annotations"]} <= set(
        source_val_map.values()
    )


def test_vehicle_expert_dataset_is_seed_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    predictions = _write_source(source)
    first = tmp_path / "first"
    second = tmp_path / "second"
    policy = VehicleExpertPolicy(crop_size=64, holdout_ratio=0.5, seed=7)

    build_vehicle_expert_dataset(source, predictions, first, policy)
    build_vehicle_expert_dataset(source, predictions, second, policy)

    assert (first / "manifests" / "source-groups.json").read_bytes() == (
        second / "manifests" / "source-groups.json"
    ).read_bytes()
    assert (first / "reports" / "vehicle-expert-dataset.json").read_bytes() == (
        second / "reports" / "vehicle-expert-dataset.json"
    ).read_bytes()
