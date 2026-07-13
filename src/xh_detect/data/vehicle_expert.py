from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import mkdtemp

import cv2
import numpy as np
import yaml

from xh_detect.evaluator import load_coco_ground_truth, load_coco_predictions
from xh_detect.geometry import HBB, obb_to_hbb
from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import ObjectAnnotation
from xh_detect.vehicle_confirmation.proposals import label_vehicle_proposals


@dataclass(frozen=True)
class VehicleExpertPolicy:
    crop_size: int = 512
    holdout_ratio: float = 0.20
    max_negatives_per_group: int = 8
    background_score_floor: float = 0.25
    negative_to_positive_ratio: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("crop_size", "max_negatives_per_group"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.holdout_ratio, bool)
            or not isinstance(self.holdout_ratio, (int, float))
            or not math.isfinite(self.holdout_ratio)
            or not 0.0 < self.holdout_ratio < 1.0
        ):
            raise ValueError("holdout_ratio must be finite and within (0, 1)")
        if (
            isinstance(self.background_score_floor, bool)
            or not isinstance(self.background_score_floor, (int, float))
            or not math.isfinite(self.background_score_floor)
            or not 0.0 <= self.background_score_floor <= 1.0
        ):
            raise ValueError("background_score_floor must be finite and within [0, 1]")
        if (
            isinstance(self.negative_to_positive_ratio, bool)
            or not isinstance(self.negative_to_positive_ratio, (int, float))
            or not math.isfinite(self.negative_to_positive_ratio)
            or self.negative_to_positive_ratio < 0.0
        ):
            raise ValueError("negative_to_positive_ratio must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True)
class VehicleExpertDatasetResult:
    output_root: Path
    positive_crops: int
    negative_crops: int
    train_crops: int
    val_crops: int
    train_positive: int
    val_positive: int
    train_groups: frozenset[str]
    val_groups: frozenset[str]


@dataclass(frozen=True)
class _CropCandidate:
    image_id: str
    stem: str
    source_group: str
    source_index: int
    center: tuple[float, float]
    positive: bool
    score: float


def _load_mapping(path: Path, description: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"source dataset is missing {description}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return {str(key): value for key, value in payload.items()}


def _validate_roots(source_root: Path, output_root: Path) -> None:
    source = source_root.resolve()
    output = output_root.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError(f"source_root and output_root overlap: {source} / {output}")
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise ValueError(f"output_root already exists and is not empty: {output_root}")


def _train_layout(
    source_root: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    image_map = _load_mapping(source_root / "manifests" / "train-image-map.json", "train image map")
    source_groups = _load_mapping(source_root / "manifests" / "source-groups.json", "source groups")
    id_to_stem: dict[str, str] = {}
    group_by_stem: dict[str, str] = {}
    stem_to_id: dict[str, int] = {}
    for stem, image_id in image_map.items():
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError("train image IDs must be integers")
        normalized_id = str(image_id)
        if normalized_id in id_to_stem:
            raise ValueError("train image map contains duplicate IDs")
        details = source_groups.get(stem)
        if not isinstance(details, Mapping) or details.get("split") != "train":
            raise ValueError(f"source group for {stem} is not a train record")
        group = details.get("group")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"source group for {stem} is invalid")
        image_path = source_root / "images" / "train" / f"{stem}.jpg"
        label_path = source_root / "labels" / "train" / f"{stem}.txt"
        if not image_path.is_file() or not label_path.is_file():
            raise ValueError(f"train image or label is missing for {stem}")
        id_to_stem[normalized_id] = stem
        group_by_stem[stem] = group
        stem_to_id[stem] = image_id
    return id_to_stem, group_by_stem, stem_to_id


def _split_groups(
    all_groups: set[str],
    vehicle_counts: Mapping[str, int],
    policy: VehicleExpertPolicy,
) -> tuple[frozenset[str], frozenset[str]]:
    ranked = sorted(
        all_groups,
        key=lambda group: hashlib.sha256(f"{policy.seed}:{group}".encode()).hexdigest(),
    )
    target = max(1, min(len(ranked) - 1, round(len(ranked) * policy.holdout_ratio)))
    total_vehicles = sum(vehicle_counts.values())
    minimum_val_vehicles = min(3, total_vehicles - 1)
    val = set(ranked[:target])
    val_vehicles = sum(vehicle_counts.get(group, 0) for group in val)
    if total_vehicles - val_vehicles == 0:
        group_to_return = next(
            group for group in reversed(ranked[:target]) if vehicle_counts.get(group, 0) > 0
        )
        val.remove(group_to_return)
        val_vehicles -= vehicle_counts[group_to_return]
    for group in ranked[target:]:
        group_vehicles = vehicle_counts.get(group, 0)
        if val_vehicles >= minimum_val_vehicles:
            break
        if group_vehicles > 0 and total_vehicles - val_vehicles - group_vehicles > 0:
            val.add(group)
            val_vehicles += group_vehicles
    if val_vehicles < minimum_val_vehicles or total_vehicles - val_vehicles <= 0:
        raise ValueError("cannot create source-group split with vehicle truth on both sides")
    return frozenset(all_groups - val), frozenset(val)


def _crop_box(center: tuple[float, float], width: int, height: int, size: int) -> HBB:
    crop_width = min(size, width)
    crop_height = min(size, height)
    left = round(center[0] - crop_width / 2)
    top = round(center[1] - crop_height / 2)
    left = max(0, min(left, width - crop_width))
    top = max(0, min(top, height - crop_height))
    return float(left), float(top), float(left + crop_width), float(top + crop_height)


def _intersects(left: HBB, right: HBB) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(
        left[1], right[1]
    )


def _center(box: HBB) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _background_center(stem: str, width: int, height: int, policy: VehicleExpertPolicy):
    digest = hashlib.sha256(f"{policy.seed}:background:{stem}".encode()).digest()
    crop_width = min(policy.crop_size, width)
    crop_height = min(policy.crop_size, height)
    left_range = width - crop_width
    top_range = height - crop_height
    left = int.from_bytes(digest[:8], "big") % (left_range + 1)
    top = int.from_bytes(digest[8:16], "big") % (top_range + 1)
    return left + crop_width / 2.0, top + crop_height / 2.0


def _yolo_labels(truth: list[ObjectAnnotation], crop: HBB) -> str:
    crop_width = crop[2] - crop[0]
    crop_height = crop[3] - crop[1]
    lines: list[str] = []
    for item in truth:
        box = obb_to_hbb(item.polygon)
        center_x, center_y = _center(box)
        if not (crop[0] <= center_x < crop[2] and crop[1] <= center_y < crop[3]):
            continue
        left = max(box[0], crop[0]) - crop[0]
        top = max(box[1], crop[1]) - crop[1]
        right = min(box[2], crop[2]) - crop[0]
        bottom = min(box[3], crop[3]) - crop[1]
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            continue
        lines.append(
            "0 "
            f"{(left + right) / 2 / crop_width:.8f} "
            f"{(top + bottom) / 2 / crop_height:.8f} "
            f"{width / crop_width:.8f} "
            f"{height / crop_height:.8f}\n"
        )
    return "".join(lines)


def build_vehicle_expert_dataset(
    source_root: Path,
    sph_predictions_json: Path,
    output_root: Path,
    policy: VehicleExpertPolicy,
) -> VehicleExpertDatasetResult:
    if not isinstance(policy, VehicleExpertPolicy):
        raise TypeError("policy must be a VehicleExpertPolicy")
    source_root = Path(source_root)
    output_root = Path(output_root)
    _validate_roots(source_root, output_root)
    id_to_stem, group_by_stem, stem_to_id = _train_layout(source_root)
    taxonomy = get_taxonomy("xh25")
    truth = load_coco_ground_truth(
        source_root / "reports" / "train-ground-truth.json", taxonomy=taxonomy
    )
    predictions = load_coco_predictions(sph_predictions_json, taxonomy=taxonomy)
    for item in (*truth, *predictions):
        if item.image_id not in id_to_stem:
            raise ValueError(f"image ID {item.image_id} is not mapped to train")

    vehicle_truth = [item for item in truth if item.class_id == 24 and not item.difficult]
    truth_by_image: dict[str, list[ObjectAnnotation]] = defaultdict(list)
    vehicle_counts: Counter[str] = Counter()
    positives: list[_CropCandidate] = []
    for truth_index, item in enumerate(vehicle_truth):
        truth_by_image[item.image_id].append(item)
        stem = id_to_stem[item.image_id]
        group = group_by_stem[stem]
        vehicle_counts[group] += 1
        positives.append(
            _CropCandidate(
                item.image_id,
                stem,
                group,
                truth_index,
                _center(obb_to_hbb(item.polygon)),
                True,
                1.0,
            )
        )
    if len(vehicle_truth) < 2:
        raise ValueError("at least two vehicle truths are required")

    all_groups = set(group_by_stem.values())
    train_groups, val_groups = _split_groups(all_groups, vehicle_counts, policy)
    labeled, _ = label_vehicle_proposals([], predictions, truth)
    background = [
        item
        for item in labeled
        if item.reason == "background" and item.detection.score >= policy.background_score_floor
    ]
    background.sort(key=lambda item: (-item.detection.score, item.proposal_index))

    image_cache: dict[str, np.ndarray] = {}

    def image_for(stem: str) -> np.ndarray:
        if stem not in image_cache:
            path = source_root / "images" / "train" / f"{stem}.jpg"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read train image: {path}")
            image_cache[stem] = image
        return image_cache[stem]

    negatives: list[_CropCandidate] = []
    negative_group_counts: Counter[str] = Counter()
    for item in background:
        stem = id_to_stem[item.detection.image_id]
        group = group_by_stem[stem]
        if negative_group_counts[group] >= policy.max_negatives_per_group:
            continue
        image = image_for(stem)
        height, width = image.shape[:2]
        crop = _crop_box(
            _center(obb_to_hbb(item.detection.polygon)),
            width,
            height,
            policy.crop_size,
        )
        if any(
            _intersects(crop, obb_to_hbb(target.polygon))
            for target in truth_by_image[item.detection.image_id]
        ):
            continue
        negatives.append(
            _CropCandidate(
                item.detection.image_id,
                stem,
                group,
                item.proposal_index,
                _center(obb_to_hbb(item.detection.polygon)),
                False,
                item.detection.score,
            )
        )
        negative_group_counts[group] += 1

    target_negatives = math.ceil(len(positives) * policy.negative_to_positive_ratio)
    background_stems = sorted(
        (stem for stem in group_by_stem if not truth_by_image.get(str(stem_to_id[stem]))),
        key=lambda stem: hashlib.sha256(f"{policy.seed}:fill:{stem}".encode()).hexdigest(),
    )
    for fill_index, stem in enumerate(background_stems):
        if len(negatives) >= target_negatives:
            break
        group = group_by_stem[stem]
        image = image_for(stem)
        height, width = image.shape[:2]
        negatives.append(
            _CropCandidate(
                str(stem_to_id[stem]),
                stem,
                group,
                1_000_000 + fill_index,
                _background_center(stem, width, height, policy),
                False,
                0.0,
            )
        )

    candidates = positives + negatives
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent))
    generated_groups: dict[str, dict[str, str]] = {}
    split_stems: dict[str, list[str]] = {"train": [], "val": []}
    split_positive: Counter[str] = Counter()
    try:
        for candidate in sorted(
            candidates,
            key=lambda item: (item.source_group, item.stem, not item.positive, item.source_index),
        ):
            split = "val" if candidate.source_group in val_groups else "train"
            kind = "veh" if candidate.positive else "bg"
            generated_stem = f"{candidate.stem}__{kind}_{candidate.source_index:06d}"
            image = image_for(candidate.stem)
            height, width = image.shape[:2]
            crop = _crop_box(candidate.center, width, height, policy.crop_size)
            left, top, right, bottom = (int(value) for value in crop)
            image_path = stage / "images" / split / f"{generated_stem}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), image[top:bottom, left:right]):
                raise RuntimeError(f"failed to write crop: {image_path}")
            labels = _yolo_labels(truth_by_image.get(candidate.image_id, []), crop)
            if candidate.positive and not labels:
                raise RuntimeError(f"positive crop has no vehicle label: {generated_stem}")
            label_path = stage / "labels" / split / f"{generated_stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(labels, encoding="utf-8")
            split_stems[split].append(generated_stem)
            split_positive[split] += candidate.positive
            generated_groups[generated_stem] = {
                "group": candidate.source_group,
                "split": split,
            }

        manifests = stage / "manifests"
        reports = stage / "reports"
        manifests.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            (manifests / f"{split}.txt").write_text(
                "".join(f"images/{split}/{stem}.jpg\n" for stem in split_stems[split]),
                encoding="utf-8",
            )
        (manifests / "source-groups.json").write_text(
            json.dumps(generated_groups, indent=2, sort_keys=True), encoding="utf-8"
        )
        source_val_image_map = {
            stem: stem_to_id[stem]
            for stem in sorted(stem_to_id)
            if group_by_stem[stem] in val_groups
        }
        (manifests / "source-val-image-map.json").write_text(
            json.dumps(source_val_image_map, indent=2), encoding="utf-8"
        )
        source_val_ids = {str(image_id) for image_id in source_val_image_map.values()}
        source_val_truth = {
            "annotations": [
                {
                    "image_id": int(item.image_id),
                    "category_id": item.class_id,
                    "bbox": [
                        obb_to_hbb(item.polygon)[0],
                        obb_to_hbb(item.polygon)[1],
                        obb_to_hbb(item.polygon)[2] - obb_to_hbb(item.polygon)[0],
                        obb_to_hbb(item.polygon)[3] - obb_to_hbb(item.polygon)[1],
                    ],
                    "iscrowd": int(item.difficult),
                }
                for item in truth
                if item.image_id in source_val_ids
            ]
        }
        (reports / "source-val-ground-truth.json").write_text(
            json.dumps(source_val_truth, indent=2), encoding="utf-8"
        )
        report = {
            "policy": asdict(policy),
            "positive_crops": len(positives),
            "negative_crops": len(negatives),
            "train": {"crops": len(split_stems["train"]), "positive": split_positive["train"]},
            "val": {"crops": len(split_stems["val"]), "positive": split_positive["val"]},
            "train_groups": sorted(train_groups),
            "val_groups": sorted(val_groups),
        }
        (reports / "vehicle-expert-dataset.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        (stage / "dataset.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": str(output_root.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "names": {0: "vehicle"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        if output_root.exists():
            output_root.rmdir()
        stage.replace(output_root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return VehicleExpertDatasetResult(
        output_root,
        len(positives),
        len(negatives),
        len(split_stems["train"]),
        len(split_stems["val"]),
        split_positive["train"],
        split_positive["val"],
        train_groups,
        val_groups,
    )
