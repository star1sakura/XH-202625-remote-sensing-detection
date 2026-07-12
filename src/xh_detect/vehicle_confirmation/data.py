from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import combinations
from numbers import Real
from pathlib import Path
from tempfile import mkdtemp

import cv2
import numpy as np

from xh_detect.evaluator import load_coco_ground_truth, load_coco_predictions
from xh_detect.geometry import obb_to_hbb
from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import Detection, ObjectAnnotation
from xh_detect.vehicle_confirmation.proposals import (
    LabeledVehicleProposal,
    label_vehicle_proposals,
)


@dataclass(frozen=True)
class VehicleCropPolicy:
    context_scale: float = 2.0
    min_side: int = 64
    max_side: int = 256
    output_size: int = 160
    holdout_ratio: float = 0.20
    seed: int = 42

    def __post_init__(self) -> None:
        if (
            isinstance(self.context_scale, bool)
            or not isinstance(self.context_scale, Real)
            or not math.isfinite(float(self.context_scale))
            or self.context_scale <= 0
        ):
            raise ValueError("context_scale must be finite and positive")
        for name in ("min_side", "max_side", "output_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_side > self.max_side:
            raise ValueError("min_side must not exceed max_side")
        if (
            isinstance(self.holdout_ratio, bool)
            or not isinstance(self.holdout_ratio, Real)
            or not math.isfinite(float(self.holdout_ratio))
            or not 0.0 < self.holdout_ratio < 1.0
        ):
            raise ValueError("holdout_ratio must be finite and within (0, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    def crop_side(self, width: float, height: float) -> int:
        if not all(math.isfinite(value) and value >= 0 for value in (width, height)):
            raise ValueError("crop width and height must be finite and non-negative")
        scaled = round(float(self.context_scale) * max(width, height))
        return max(self.min_side, min(self.max_side, scaled))


@dataclass(frozen=True)
class VehicleConfirmerDatasetResult:
    output_root: Path
    train_examples: int
    holdout_examples: int
    train_positive: int
    train_negative: int
    holdout_positive: int
    holdout_negative: int
    train_groups: frozenset[str]
    holdout_groups: frozenset[str]


@dataclass(frozen=True)
class _Candidate:
    labeled: LabeledVehicleProposal
    stem: str
    source_group: str


def _load_mapping(path: Path, *, description: str) -> dict[str, object]:
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


def _load_train_layout(source_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    raw_image_map = _load_mapping(
        source_root / "manifests" / "train-image-map.json",
        description="train image map",
    )
    source_groups = _load_mapping(
        source_root / "manifests" / "source-groups.json",
        description="source groups",
    )
    id_to_stem: dict[str, str] = {}
    group_by_stem: dict[str, str] = {}
    for stem, image_id in raw_image_map.items():
        if not stem.strip() or isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError("train image map contains an invalid stem or image ID")
        normalized_id = str(image_id)
        if normalized_id in id_to_stem:
            raise ValueError("train image map contains duplicate image IDs")
        details = source_groups.get(stem)
        if not isinstance(details, Mapping) or details.get("split") != "train":
            raise ValueError(f"source group for {stem} is not a train record")
        group = details.get("group")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"source group for {stem} has an invalid group ID")
        image_path = source_root / "images" / "train" / f"{stem}.jpg"
        label_path = source_root / "labels" / "train" / f"{stem}.txt"
        if not image_path.is_file():
            raise ValueError(f"train image is missing: {image_path}")
        if not label_path.is_file():
            raise ValueError(f"train label is missing: {label_path}")
        id_to_stem[normalized_id] = stem
        group_by_stem[stem] = group
    if not id_to_stem:
        raise ValueError("train image map must not be empty")
    return id_to_stem, group_by_stem


def _validate_image_ids(
    items: Iterable[Detection | ObjectAnnotation],
    id_to_stem: Mapping[str, str],
    description: str,
) -> None:
    for item in items:
        image_id = str(item.image_id)
        if image_id not in id_to_stem:
            raise ValueError(f"{description} image ID {image_id} is not mapped to a train image")


def _group_counts(candidates: list[_Candidate]) -> tuple[int, int]:
    positive = sum(item.labeled.label == 1 for item in candidates)
    return positive, len(candidates) - positive


def _split_groups(
    candidates: list[_Candidate], policy: VehicleCropPolicy
) -> tuple[frozenset[str], frozenset[str]]:
    by_group: dict[str, list[_Candidate]] = defaultdict(list)
    for item in candidates:
        by_group[item.source_group].append(item)
    ranked = sorted(
        by_group,
        key=lambda group: hashlib.sha256(f"{policy.seed}:{group}".encode()).hexdigest(),
    )
    if len(ranked) < 2:
        raise ValueError("at least two source groups are required")
    total_positive, total_negative = _group_counts(candidates)
    if total_positive < 2 or total_negative < 2:
        raise ValueError("at least two positive and two negative proposals are required")

    def valid_holdout(groups: set[str]) -> bool:
        holdout_items = [item for group in groups for item in by_group[group]]
        holdout_positive, holdout_negative = _group_counts(holdout_items)
        return (
            holdout_positive > 0
            and holdout_negative > 0
            and total_positive - holdout_positive > 0
            and total_negative - holdout_negative > 0
        )

    anchors: set[str] | None = None
    for width in (1, 2):
        for selected in combinations(ranked, width):
            candidate = set(selected)
            if valid_holdout(candidate):
                anchors = candidate
                break
        if anchors is not None:
            break
    if anchors is None:
        raise ValueError("cannot isolate positive and negative proposals in both partitions")

    target = max(1, min(len(ranked) - 1, round(len(ranked) * policy.holdout_ratio)))
    holdout = set(anchors)
    for group in ranked:
        if len(holdout) >= target:
            break
        candidate = holdout | {group}
        if valid_holdout(candidate):
            holdout = candidate
    train = set(ranked) - holdout
    if not valid_holdout(holdout):
        raise AssertionError("internal source-group split is invalid")
    return frozenset(train), frozenset(holdout)


def _extract_crop(image: np.ndarray, labeled: LabeledVehicleProposal, policy: VehicleCropPolicy):
    x1, y1, x2, y2 = obb_to_hbb(labeled.detection.polygon)
    side = policy.crop_side(x2 - x1, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = math.floor(center_x - side / 2.0)
    top = math.floor(center_y - side / 2.0)
    right = left + side
    bottom = top + side
    height, width = image.shape[:2]
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(width, right)
    source_bottom = min(height, bottom)
    canvas = np.zeros((side, side, 3), dtype=image.dtype)
    if source_right > source_left and source_bottom > source_top:
        destination_left = source_left - left
        destination_top = source_top - top
        canvas[
            destination_top : destination_top + source_bottom - source_top,
            destination_left : destination_left + source_right - source_left,
        ] = image[source_top:source_bottom, source_left:source_right]
    return cv2.resize(
        canvas,
        (policy.output_size, policy.output_size),
        interpolation=cv2.INTER_LINEAR,
    )


def _materialize_partition(
    stage: Path,
    split: str,
    candidates: list[_Candidate],
    source_root: Path,
    policy: VehicleCropPolicy,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    image_cache: dict[str, np.ndarray] = {}
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.source_group,
            item.labeled.detection.image_id,
            item.labeled.proposal_index,
        ),
    )
    for index, candidate in enumerate(ordered, start=1):
        if candidate.stem not in image_cache:
            image_path = source_root / "images" / "train" / f"{candidate.stem}.jpg"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read train image: {image_path}")
            image_cache[candidate.stem] = image
        image = image_cache[candidate.stem]
        crop = _extract_crop(image, candidate.labeled, policy)
        relative_crop = Path("crops") / split / f"{index:06d}.png"
        destination = stage / relative_crop
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), crop):
            raise RuntimeError(f"failed to write vehicle crop: {destination}")
        x1, y1, x2, y2 = obb_to_hbb(candidate.labeled.detection.polygon)
        height, width = image.shape[:2]
        records.append(
            {
                "crop": relative_crop.as_posix(),
                "image_id": candidate.labeled.detection.image_id,
                "proposal_index": candidate.labeled.proposal_index,
                "label": candidate.labeled.label,
                "reason": candidate.labeled.reason,
                "sph_score": candidate.labeled.detection.score,
                "width_norm": (x2 - x1) / width,
                "height_norm": (y2 - y1) / height,
                "source_group": candidate.source_group,
            }
        )
    manifest = stage / "manifests" / f"{split}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return records


def _partition_report(records: list[dict[str, object]], groups: frozenset[str]):
    positive = sum(record["label"] == 1 for record in records)
    return {
        "examples": len(records),
        "positive": positive,
        "negative": len(records) - positive,
        "groups": sorted(groups),
        "group_hashes": [hashlib.sha256(group.encode()).hexdigest() for group in sorted(groups)],
    }


def build_vehicle_confirmer_dataset(
    source_root: Path,
    main_predictions_json: Path,
    sph_predictions_json: Path,
    output_root: Path,
    policy: VehicleCropPolicy,
) -> VehicleConfirmerDatasetResult:
    if not isinstance(policy, VehicleCropPolicy):
        raise TypeError("policy must be a VehicleCropPolicy")
    source_root = Path(source_root)
    output_root = Path(output_root)
    _validate_roots(source_root, output_root)
    id_to_stem, group_by_stem = _load_train_layout(source_root)
    taxonomy = get_taxonomy("xh25")
    main_predictions = load_coco_predictions(main_predictions_json, taxonomy=taxonomy)
    sph_predictions = load_coco_predictions(sph_predictions_json, taxonomy=taxonomy)
    truth = load_coco_ground_truth(
        source_root / "reports" / "train-ground-truth.json",
        taxonomy=taxonomy,
    )
    _validate_image_ids(main_predictions, id_to_stem, "main prediction")
    _validate_image_ids(sph_predictions, id_to_stem, "SPH prediction")
    _validate_image_ids(truth, id_to_stem, "ground-truth")
    labeled, _ = label_vehicle_proposals(main_predictions, sph_predictions, truth)
    candidates = [
        _Candidate(
            item,
            id_to_stem[item.detection.image_id],
            group_by_stem[id_to_stem[item.detection.image_id]],
        )
        for item in labeled
        if not item.duplicate_main
    ]
    if not candidates:
        raise ValueError("no runtime vehicle proposals are available")
    train_groups, holdout_groups = _split_groups(candidates, policy)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent))
    try:
        train_records = _materialize_partition(
            stage,
            "train",
            [item for item in candidates if item.source_group in train_groups],
            source_root,
            policy,
        )
        holdout_records = _materialize_partition(
            stage,
            "holdout",
            [item for item in candidates if item.source_group in holdout_groups],
            source_root,
            policy,
        )
        report = {
            "policy": asdict(policy),
            "train": _partition_report(train_records, train_groups),
            "holdout": _partition_report(holdout_records, holdout_groups),
        }
        report_path = stage / "reports" / "vehicle-confirmer-dataset.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (stage / "reports" / "vehicle-confirmer-dataset.md").write_text(
            "# Vehicle Confirmer Dataset\n\n"
            f"- Train: {len(train_records)} examples\n"
            f"- Holdout: {len(holdout_records)} examples\n"
            f"- Train groups: {len(train_groups)}\n"
            f"- Holdout groups: {len(holdout_groups)}\n",
            encoding="utf-8",
        )
        if output_root.exists():
            output_root.rmdir()
        stage.replace(output_root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    train_positive = sum(record["label"] == 1 for record in train_records)
    holdout_positive = sum(record["label"] == 1 for record in holdout_records)
    return VehicleConfirmerDatasetResult(
        output_root=output_root,
        train_examples=len(train_records),
        holdout_examples=len(holdout_records),
        train_positive=train_positive,
        train_negative=len(train_records) - train_positive,
        holdout_positive=holdout_positive,
        holdout_negative=len(holdout_records) - holdout_positive,
        train_groups=train_groups,
        holdout_groups=holdout_groups,
    )
