from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import yaml

from xh_detect.evaluator import load_coco_ground_truth, load_coco_predictions
from xh_detect.geometry import HBB, hbb_iou, obb_to_hbb
from xh_detect.taxonomy import Taxonomy, get_taxonomy
from xh_detect.types import Detection, ObjectAnnotation


@dataclass(frozen=True)
class HardExamplePolicy:
    crop_size: int = 768
    background_score_floor: float = 0.60
    max_positive_crops_per_group: int = 8
    max_negative_crops_per_group: int = 2
    vehicle_positive_multiplier: int = 2
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "crop_size",
            "max_positive_crops_per_group",
            "max_negative_crops_per_group",
            "vehicle_positive_multiplier",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.background_score_floor, bool)
            or not isinstance(self.background_score_floor, (int, float))
            or not math.isfinite(self.background_score_floor)
            or not 0.0 <= self.background_score_floor <= 1.0
        ):
            raise ValueError("background_score_floor must be finite and in [0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True)
class HardExampleResult:
    output_root: Path
    original_train_images: int
    hard_positive_crops: int
    hard_negative_crops: int
    missed_truth_by_coarse_class: dict[str, int]
    selected_positive_by_coarse_class: dict[str, int]
    selected_negative_by_coarse_class: dict[str, int]


@dataclass(frozen=True)
class _PositiveCandidate:
    truth_index: int
    truth: ObjectAnnotation
    stem: str
    group: str
    coarse_class: str


@dataclass(frozen=True)
class _NegativeCandidate:
    prediction_index: int
    prediction: Detection
    stem: str
    group: str
    coarse_class: str


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


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _source_stems(source_root: Path, split: str) -> list[str]:
    images = {path.stem for path in (source_root / "images" / split).glob("*.jpg")}
    labels = {path.stem for path in (source_root / "labels" / split).glob("*.txt")}
    if images != labels:
        raise ValueError(f"source {split} image and label stems are inconsistent")
    return sorted(images)


def _materialize_originals(source_root: Path, output_root: Path, split: str) -> list[str]:
    stems = _source_stems(source_root, split)
    for stem in stems:
        _copy_or_link(
            source_root / "images" / split / f"{stem}.jpg",
            output_root / "images" / split / f"{stem}.jpg",
        )
        _copy_or_link(
            source_root / "labels" / split / f"{stem}.txt",
            output_root / "labels" / split / f"{stem}.txt",
        )
    return stems


def _train_layout(source_root: Path) -> tuple[dict[str, str], dict[str, str], dict[str, object]]:
    image_map = _load_mapping(source_root / "manifests" / "train-image-map.json", "train image map")
    groups = _load_mapping(source_root / "manifests" / "source-groups.json", "source groups")
    id_to_stem: dict[str, str] = {}
    group_by_stem: dict[str, str] = {}
    for stem, image_id in image_map.items():
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError("train image IDs must be integers")
        details = groups.get(stem)
        if not isinstance(details, Mapping) or details.get("split") != "train":
            raise ValueError(f"source group for {stem} is not a train record")
        group = details.get("group")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"source group for {stem} is invalid")
        normalized_id = str(image_id)
        if normalized_id in id_to_stem:
            raise ValueError("train image map contains duplicate image IDs")
        id_to_stem[normalized_id] = stem
        group_by_stem[stem] = group
    return id_to_stem, group_by_stem, groups


def _matched_truth_indices(
    predictions: list[tuple[int, Detection]],
    truths: list[tuple[int, ObjectAnnotation]],
    threshold: float,
) -> tuple[set[int], list[tuple[int, Detection]]]:
    matched: set[int] = set()
    false_positives: list[tuple[int, Detection]] = []
    for prediction_index, prediction in sorted(predictions, key=lambda item: -item[1].score):
        prediction_box = obb_to_hbb(prediction.polygon)
        best_truth_index = -1
        best_iou = -1.0
        for truth_index, truth in truths:
            if truth_index in matched:
                continue
            overlap = hbb_iou(prediction_box, obb_to_hbb(truth.polygon))
            if overlap >= threshold and overlap > best_iou:
                best_truth_index = truth_index
                best_iou = overlap
        if best_truth_index >= 0:
            matched.add(best_truth_index)
        else:
            false_positives.append((prediction_index, prediction))
    return matched, false_positives


def _mine_candidates(
    predictions: list[Detection],
    truth: list[ObjectAnnotation],
    *,
    id_to_stem: Mapping[str, str],
    group_by_stem: Mapping[str, str],
    taxonomy: Taxonomy,
) -> tuple[list[_PositiveCandidate], list[_NegativeCandidate], dict[str, list[ObjectAnnotation]]]:
    truth_by_key: dict[tuple[str, str], list[tuple[int, ObjectAnnotation]]] = defaultdict(list)
    predictions_by_key: dict[tuple[str, str], list[tuple[int, Detection]]] = defaultdict(list)
    truth_by_image: dict[str, list[ObjectAnnotation]] = defaultdict(list)
    for index, item in enumerate(truth):
        if item.difficult:
            continue
        truth_by_image[item.image_id].append(item)
        coarse = taxonomy.coarse_name(item.class_id)
        if coarse in {"ship", "vehicle"}:
            truth_by_key[(item.image_id, coarse)].append((index, item))
    for index, item in enumerate(predictions):
        coarse = taxonomy.coarse_name(item.class_id)
        if coarse in {"ship", "vehicle"}:
            predictions_by_key[(item.image_id, coarse)].append((index, item))

    positives: list[_PositiveCandidate] = []
    negatives: list[_NegativeCandidate] = []
    for key in sorted(set(truth_by_key) | set(predictions_by_key)):
        image_id, coarse = key
        stem = id_to_stem.get(image_id)
        if stem is None:
            raise ValueError(f"prediction or truth image ID {image_id} is not mapped to train")
        threshold = 0.35 if coarse == "vehicle" else 0.50
        matched, false_positives = _matched_truth_indices(
            predictions_by_key.get(key, []),
            truth_by_key.get(key, []),
            threshold,
        )
        for truth_index, item in truth_by_key.get(key, []):
            if truth_index not in matched:
                positives.append(
                    _PositiveCandidate(
                        truth_index,
                        item,
                        stem,
                        group_by_stem[stem],
                        coarse,
                    )
                )
        for prediction_index, item in false_positives:
            negatives.append(
                _NegativeCandidate(
                    prediction_index,
                    item,
                    stem,
                    group_by_stem[stem],
                    coarse,
                )
            )
    return positives, negatives, truth_by_image


def _center(box: HBB) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _crop_box(center: tuple[float, float], width: int, height: int, size: int) -> HBB:
    crop_width = min(size, width)
    crop_height = min(size, height)
    left = max(0, min(round(center[0] - crop_width / 2), width - crop_width))
    top = max(0, min(round(center[1] - crop_height / 2), height - crop_height))
    return float(left), float(top), float(left + crop_width), float(top + crop_height)


def _intersects(left: HBB, right: HBB) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(
        left[1], right[1]
    )


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
        if width <= 0.0 or height <= 0.0:
            continue
        lines.append(
            f"{item.class_id} "
            f"{(left + right) / 2 / crop_width:.8f} "
            f"{(top + bottom) / 2 / crop_height:.8f} "
            f"{width / crop_width:.8f} "
            f"{height / crop_height:.8f}\n"
        )
    return "".join(lines)


def _write_manifest(output_root: Path, split: str, stems: list[str]) -> None:
    path = output_root / "manifests" / f"{split}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"images/{split}/{stem}.jpg\n" for stem in sorted(stems)),
        encoding="utf-8",
    )


def _candidate_rank(seed: int, label: str, stem: str, index: int) -> str:
    return hashlib.sha256(f"{seed}:{label}:{stem}:{index}".encode()).hexdigest()


def build_hard_example_dataset(
    source_root: Path,
    predictions_json: Path,
    output_root: Path,
    policy: HardExamplePolicy,
) -> HardExampleResult:
    if not isinstance(policy, HardExamplePolicy):
        raise TypeError("policy must be a HardExamplePolicy")
    source_root = Path(source_root)
    predictions_json = Path(predictions_json)
    output_root = Path(output_root)
    _validate_roots(source_root, output_root)
    dataset_path = source_root / "dataset.yaml"
    if not dataset_path.is_file():
        raise ValueError(f"source dataset is missing dataset.yaml: {dataset_path}")
    dataset_payload = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_payload, Mapping) or not isinstance(
        dataset_payload.get("names"), Mapping
    ):
        raise ValueError("source dataset.yaml must contain names mapping")

    id_to_stem, group_by_stem, source_groups = _train_layout(source_root)
    taxonomy = get_taxonomy("xh25")
    predictions = load_coco_predictions(predictions_json, taxonomy=taxonomy)
    truth = load_coco_ground_truth(
        source_root / "reports" / "train-ground-truth.json",
        taxonomy=taxonomy,
    )
    positives, negatives, truth_by_image = _mine_candidates(
        predictions,
        truth,
        id_to_stem=id_to_stem,
        group_by_stem=group_by_stem,
        taxonomy=taxonomy,
    )
    missed_counts = Counter(item.coarse_class for item in positives)

    positives.sort(
        key=lambda item: (
            (obb_to_hbb(item.truth.polygon)[2] - obb_to_hbb(item.truth.polygon)[0])
            * (obb_to_hbb(item.truth.polygon)[3] - obb_to_hbb(item.truth.polygon)[1]),
            _candidate_rank(policy.seed, "positive", item.stem, item.truth_index),
        )
    )
    selected_positives: list[_PositiveCandidate] = []
    positive_group_counts: Counter[tuple[str, str]] = Counter()
    for candidate in positives:
        key = (candidate.group, candidate.coarse_class)
        if positive_group_counts[key] >= policy.max_positive_crops_per_group:
            continue
        selected_positives.append(candidate)
        positive_group_counts[key] += 1

    negatives = [
        item
        for item in negatives
        if item.prediction.score >= policy.background_score_floor
        and not any(
            _intersects(
                obb_to_hbb(item.prediction.polygon),
                obb_to_hbb(target.polygon),
            )
            for target in truth_by_image[item.prediction.image_id]
        )
    ]
    negatives.sort(
        key=lambda item: (
            -item.prediction.score,
            _candidate_rank(policy.seed, "negative", item.stem, item.prediction_index),
        )
    )
    selected_negatives: list[_NegativeCandidate] = []
    negative_group_counts: Counter[tuple[str, str]] = Counter()
    for candidate in negatives:
        key = (candidate.group, candidate.coarse_class)
        if negative_group_counts[key] >= policy.max_negative_crops_per_group:
            continue
        selected_negatives.append(candidate)
        negative_group_counts[key] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    train_stems = _materialize_originals(source_root, output_root, "train")
    val_stems = _materialize_originals(source_root, output_root, "val")
    output_groups = {
        key: dict(value) for key, value in source_groups.items() if isinstance(value, Mapping)
    }
    image_cache: dict[str, object] = {}

    def image_for(stem: str):
        if stem not in image_cache:
            path = source_root / "images" / "train" / f"{stem}.jpg"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read train image: {path}")
            image_cache[stem] = image
        return image_cache[stem]

    positive_counts: Counter[str] = Counter()
    hard_positive_crops = 0
    for candidate in selected_positives:
        repeats = policy.vehicle_positive_multiplier if candidate.coarse_class == "vehicle" else 1
        for repeat in range(repeats):
            alias = (
                f"{candidate.stem}__hp_{candidate.coarse_class}_"
                f"{candidate.truth_index:06d}_{repeat:02d}"
            )
            image = image_for(candidate.stem)
            height, width = image.shape[:2]
            crop = _crop_box(
                _center(obb_to_hbb(candidate.truth.polygon)),
                width,
                height,
                policy.crop_size,
            )
            left, top, right, bottom = (int(value) for value in crop)
            image_path = output_root / "images" / "train" / f"{alias}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), image[top:bottom, left:right]):
                raise RuntimeError(f"failed to write hard-positive image: {image_path}")
            labels = _yolo_labels(truth_by_image[candidate.truth.image_id], crop)
            if not labels:
                raise RuntimeError(f"hard-positive crop has no labels: {alias}")
            label_path = output_root / "labels" / "train" / f"{alias}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(labels, encoding="utf-8")
            train_stems.append(alias)
            output_groups[alias] = {"group": candidate.group, "split": "train"}
            positive_counts[candidate.coarse_class] += 1
            hard_positive_crops += 1

    negative_counts: Counter[str] = Counter()
    for candidate in selected_negatives:
        alias = f"{candidate.stem}__hn_{candidate.coarse_class}_{candidate.prediction_index:06d}"
        image = image_for(candidate.stem)
        height, width = image.shape[:2]
        crop = _crop_box(
            _center(obb_to_hbb(candidate.prediction.polygon)),
            width,
            height,
            policy.crop_size,
        )
        if any(
            _intersects(crop, obb_to_hbb(item.polygon))
            for item in truth_by_image[candidate.prediction.image_id]
        ):
            continue
        left, top, right, bottom = (int(value) for value in crop)
        image_path = output_root / "images" / "train" / f"{alias}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), image[top:bottom, left:right]):
            raise RuntimeError(f"failed to write hard-negative image: {image_path}")
        label_path = output_root / "labels" / "train" / f"{alias}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("", encoding="utf-8")
        train_stems.append(alias)
        output_groups[alias] = {"group": candidate.group, "split": "train"}
        negative_counts[candidate.coarse_class] += 1

    _write_manifest(output_root, "train", train_stems)
    _write_manifest(output_root, "val", val_stems)
    manifests = output_root / "manifests"
    (manifests / "source-groups.json").write_text(
        json.dumps(output_groups, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_dataset = dict(dataset_payload)
    output_dataset.update(
        {"path": str(output_root.resolve()), "train": "images/train", "val": "images/val"}
    )
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(output_dataset, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    result = HardExampleResult(
        output_root=output_root,
        original_train_images=len(_source_stems(source_root, "train")),
        hard_positive_crops=hard_positive_crops,
        hard_negative_crops=sum(negative_counts.values()),
        missed_truth_by_coarse_class={name: missed_counts[name] for name in ("ship", "vehicle")},
        selected_positive_by_coarse_class={
            name: positive_counts[name] for name in ("ship", "vehicle")
        },
        selected_negative_by_coarse_class={
            name: negative_counts[name] for name in ("ship", "vehicle")
        },
    )
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report = {**asdict(result), "output_root": str(output_root.resolve()), "policy": asdict(policy)}
    (reports / "hard-example.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result
