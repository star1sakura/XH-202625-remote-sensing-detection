from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from xh_detect.vehicle_confirmation.data import (
    VehicleCropPolicy,
    build_vehicle_confirmer_dataset,
)


def _prediction(image_id: int, score: float, bbox: list[float]) -> dict[str, object]:
    return {"image_id": image_id, "category_id": 24, "bbox": bbox, "score": score}


def _write_fixture(root: Path) -> tuple[Path, Path]:
    stems = ["a_img", "b_img", "c_img", "d_img"]
    image_map = {stem: index for index, stem in enumerate(stems, start=1)}
    groups = {
        stem: {"group": f"group-{stem[0]}", "split": "train"} for stem in stems
    }
    for stem in stems:
        image_path = root / "images" / "train" / f"{stem}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_path), np.full((100, 100, 3), 255, dtype=np.uint8))
        label_path = root / "labels" / "train" / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("", encoding="utf-8")
    manifests = root / "manifests"
    reports = root / "reports"
    manifests.mkdir()
    reports.mkdir()
    (manifests / "train-image-map.json").write_text(json.dumps(image_map), encoding="utf-8")
    (manifests / "source-groups.json").write_text(json.dumps(groups), encoding="utf-8")
    (reports / "train-ground-truth.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {"image_id": 1, "category_id": 24, "bbox": [40, 40, 10, 10]},
                    {"image_id": 1, "category_id": 24, "bbox": [70, 70, 10, 10]},
                    {"image_id": 3, "category_id": 24, "bbox": [40, 40, 10, 10]},
                ]
            }
        ),
        encoding="utf-8",
    )
    main_path = root.parent / "main.json"
    sph_path = root.parent / "sph.json"
    main_path.write_text(
        json.dumps([_prediction(1, 0.95, [70, 70, 10, 10])]), encoding="utf-8"
    )
    sph_path.write_text(
        json.dumps(
            [
                _prediction(1, 0.99, [70, 70, 10, 10]),
                _prediction(1, 0.90, [40, 40, 10, 10]),
                _prediction(1, 0.80, [40, 40, 10, 10]),
                _prediction(2, 0.70, [0, 0, 10, 10]),
                _prediction(3, 0.90, [40, 40, 10, 10]),
                _prediction(4, 0.70, [0, 0, 10, 10]),
            ]
        ),
        encoding="utf-8",
    )
    return main_path, sph_path


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_builds_deterministic_group_isolated_vehicle_crops(tmp_path: Path) -> None:
    source = tmp_path / "source"
    main, sph = _write_fixture(source)
    first = tmp_path / "first"
    second = tmp_path / "second"
    policy = VehicleCropPolicy(output_size=32, holdout_ratio=0.5, seed=42)

    result = build_vehicle_confirmer_dataset(source, main, sph, first, policy)
    second_result = build_vehicle_confirmer_dataset(source, main, sph, second, policy)

    train = _records(first / "manifests" / "train.jsonl")
    holdout = _records(first / "manifests" / "holdout.jsonl")
    records = train + holdout
    assert {record["reason"] for record in records} == {
        "recoverable_truth",
        "duplicate_proposal",
        "background",
    }
    assert all(record["proposal_index"] != 0 for record in records)
    assert sum(record["label"] == 1 for record in records) == 2
    assert sum(record["label"] == 0 for record in records) == 3
    assert result.train_groups.isdisjoint(result.holdout_groups)
    assert result.train_positive >= 1 and result.train_negative >= 1
    assert result.holdout_positive >= 1 and result.holdout_negative >= 1
    assert result.train_groups == second_result.train_groups
    assert result.holdout_groups == second_result.holdout_groups
    assert (first / "manifests" / "train.jsonl").read_bytes() == (
        second / "manifests" / "train.jsonl"
    ).read_bytes()
    assert (first / "manifests" / "holdout.jsonl").read_bytes() == (
        second / "manifests" / "holdout.jsonl"
    ).read_bytes()

    edge_record = next(record for record in records if record["image_id"] == "2")
    assert edge_record["width_norm"] == 0.1
    assert edge_record["height_norm"] == 0.1
    edge_crop = cv2.imread(str(first / str(edge_record["crop"])), cv2.IMREAD_COLOR)
    assert edge_crop is not None and edge_crop.shape == (32, 32, 3)
    assert int(edge_crop[0, 0].max()) == 0
    assert int(edge_crop[16, 16].min()) > 200

    report = json.loads(
        (first / "reports" / "vehicle-confirmer-dataset.json").read_text(encoding="utf-8")
    )
    assert report["train"]["examples"] == len(train)
    assert report["holdout"]["examples"] == len(holdout)
    assert set(report["train"]["groups"]) == result.train_groups
    assert set(report["holdout"]["groups"]) == result.holdout_groups


def test_crop_side_uses_context_scale_and_clamps() -> None:
    policy = VehicleCropPolicy(context_scale=2.0, min_side=64, max_side=256)

    assert policy.crop_side(10, 20) == 64
    assert policy.crop_side(60, 40) == 120
    assert policy.crop_side(200, 100) == 256


def test_rejects_validation_group_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    main, sph = _write_fixture(source)
    groups_path = source / "manifests" / "source-groups.json"
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    groups["a_img"]["split"] = "val"
    groups_path.write_text(json.dumps(groups), encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="train"):
        build_vehicle_confirmer_dataset(source, main, sph, output, VehicleCropPolicy())

    assert not output.exists()


def test_rejects_unknown_prediction_image_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    main, sph = _write_fixture(source)
    payload = json.loads(sph.read_text(encoding="utf-8"))
    payload.append(_prediction(99, 0.5, [0, 0, 10, 10]))
    sph.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="not mapped"):
        build_vehicle_confirmer_dataset(source, main, sph, output, VehicleCropPolicy())

    assert not output.exists()


@pytest.mark.parametrize("missing", ["image", "label"])
def test_rejects_missing_train_asset_before_writing(tmp_path: Path, missing: str) -> None:
    source = tmp_path / "source"
    main, sph = _write_fixture(source)
    path = source / f"{missing}s" / "train" / ("a_img.jpg" if missing == "image" else "a_img.txt")
    path.unlink()
    output = tmp_path / "output"

    with pytest.raises(ValueError, match=missing):
        build_vehicle_confirmer_dataset(source, main, sph, output, VehicleCropPolicy())

    assert not output.exists()


def test_rejects_overlapping_or_nonempty_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    main, sph = _write_fixture(source)

    with pytest.raises(ValueError, match="overlap"):
        build_vehicle_confirmer_dataset(
            source,
            main,
            sph,
            source / "nested-output",
            VehicleCropPolicy(),
        )

    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        build_vehicle_confirmer_dataset(source, main, sph, output, VehicleCropPolicy())
    assert (output / "existing.txt").read_text(encoding="utf-8") == "keep"
