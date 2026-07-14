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
from xh_detect.geometry import hbb_iou, obb_to_hbb
from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import Detection, ObjectAnnotation


@dataclass(frozen=True)
class HardNegativePolicy:
    confidence_floor: float = 0.60
    crop_size: int = 512
    object_margin: int = 16
    max_crops_per_group: int = 2
    vehicle_multiplier: int = 2
    seed: int = 42

    def __post_init__(self) -> None:
        if (
            isinstance(self.confidence_floor, bool)
            or not isinstance(self.confidence_floor, (int, float))
            or not math.isfinite(self.confidence_floor)
            or not 0.0 <= self.confidence_floor <= 1.0
        ):
            raise ValueError("confidence_floor must be finite and between 0 and 1")
        for name in ("crop_size", "max_crops_per_group", "vehicle_multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("object_margin", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class HardNegativeResult:
    output_root: Path
    original_train_images: int
    vehicle_upsampled_images: int
    selected_hard_negatives: int
    rejected_target_overlap: int
    selected_by_coarse_class: dict[str, int]


@dataclass(frozen=True)
class _Candidate:
    prediction_index: int
    detection: Detection
    stem: str
    group: str
    crop: tuple[int, int, int, int]
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


def _class_ids(path: Path) -> set[int]:
    result: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path} contains an invalid YOLO label line")
        result.add(int(fields[0]))
    return result


def _crop_box(
    detection: Detection, width: int, height: int, size: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = obb_to_hbb(detection.polygon)
    crop_width = min(size, width)
    crop_height = min(size, height)
    left = round((x1 + x2 - crop_width) / 2)
    top = round((y1 + y2 - crop_height) / 2)
    left = max(0, min(left, width - crop_width))
    top = max(0, min(top, height - crop_height))
    return left, top, left + crop_width, top + crop_height


def _intersects(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(
        left[1], right[1]
    )


def _is_unmatched_background(
    detection: Detection,
    truths: list[ObjectAnnotation],
    coarse_class: str,
) -> bool:
    threshold = 0.35 if coarse_class == "vehicle" else 0.50
    prediction_box = obb_to_hbb(detection.polygon)
    return not any(
        hbb_iou(prediction_box, obb_to_hbb(truth.polygon)) >= threshold
        for truth in truths
        if not truth.difficult
    )


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


def _write_manifest(output_root: Path, split: str, stems: list[str]) -> None:
    path = output_root / "manifests" / f"{split}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"images/{split}/{stem}.jpg\n" for stem in sorted(stems)),
        encoding="utf-8",
    )


def build_main_hn_dataset(
    source_root: Path,
    predictions_json: Path,
    output_root: Path,
    policy: HardNegativePolicy,
) -> HardNegativeResult:
    if not isinstance(policy, HardNegativePolicy):
        raise TypeError("policy must be a HardNegativePolicy")
    source_root = Path(source_root)
    output_root = Path(output_root)
    predictions_json = Path(predictions_json)
    _validate_roots(source_root, output_root)

    dataset_path = source_root / "dataset.yaml"
    if not dataset_path.is_file():
        raise ValueError(f"source dataset is missing dataset.yaml: {dataset_path}")
    dataset_payload = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_payload, Mapping) or not isinstance(
        dataset_payload.get("names"), Mapping
    ):
        raise ValueError("source dataset.yaml must contain names mapping")

    image_map = _load_mapping(source_root / "manifests" / "train-image-map.json", "train image map")
    id_to_stem = {str(image_id): stem for stem, image_id in image_map.items()}
    if len(id_to_stem) != len(image_map):
        raise ValueError("train image map contains duplicate image IDs")
    source_groups = _load_mapping(source_root / "manifests" / "source-groups.json", "source groups")
    taxonomy = get_taxonomy("xh25")
    predictions = load_coco_predictions(predictions_json, taxonomy=taxonomy)
    truth = load_coco_ground_truth(
        source_root / "reports" / "train-ground-truth.json", taxonomy=taxonomy
    )
    truths_by_image: dict[str, list[ObjectAnnotation]] = defaultdict(list)
    for annotation in truth:
        truths_by_image[annotation.image_id].append(annotation)

    image_shapes: dict[str, tuple[int, int]] = {}
    candidates: list[_Candidate] = []
    rejected_target_overlap = 0
    for prediction_index, detection in enumerate(predictions):
        stem = id_to_stem.get(detection.image_id)
        if stem is None:
            raise ValueError(
                f"prediction image ID {detection.image_id} is not mapped to a train image"
            )
        group_payload = source_groups.get(stem)
        if not isinstance(group_payload, Mapping) or group_payload.get("split") != "train":
            raise ValueError(f"source group for {stem} is not a train record")
        coarse_class = taxonomy.coarse_name(detection.class_id)
        if coarse_class not in {"ship", "vehicle"} or detection.score < policy.confidence_floor:
            continue
        same_coarse_truth = [
            item
            for item in truths_by_image[detection.image_id]
            if taxonomy.coarse_name(item.class_id) == coarse_class
        ]
        if not _is_unmatched_background(detection, same_coarse_truth, coarse_class):
            continue
        image_path = source_root / "images" / "train" / f"{stem}.jpg"
        if stem not in image_shapes:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read train image: {image_path}")
            image_shapes[stem] = (image.shape[1], image.shape[0])
        width, height = image_shapes[stem]
        crop = _crop_box(detection, width, height, policy.crop_size)
        safety = (
            crop[0] - policy.object_margin,
            crop[1] - policy.object_margin,
            crop[2] + policy.object_margin,
            crop[3] + policy.object_margin,
        )
        if any(
            not item.difficult and _intersects(safety, obb_to_hbb(item.polygon))
            for item in truths_by_image[detection.image_id]
        ):
            rejected_target_overlap += 1
            continue
        candidates.append(
            _Candidate(
                prediction_index,
                detection,
                stem,
                str(group_payload["group"]),
                crop,
                coarse_class,
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.detection.score,
            hashlib.sha256(
                f"{policy.seed}:{item.stem}:{item.prediction_index}".encode()
            ).hexdigest(),
        )
    )
    selected: list[_Candidate] = []
    group_counts: Counter[str] = Counter()
    for candidate in candidates:
        if group_counts[candidate.group] >= policy.max_crops_per_group:
            continue
        selected.append(candidate)
        group_counts[candidate.group] += 1
    if not selected:
        raise ValueError("no label-safe hard negatives satisfy the policy")

    output_root.mkdir(parents=True, exist_ok=True)
    train_stems = _materialize_originals(source_root, output_root, "train")
    val_stems = _materialize_originals(source_root, output_root, "val")
    output_groups = {
        key: dict(value) for key, value in source_groups.items() if isinstance(value, Mapping)
    }

    vehicle_upsampled_images = 0
    for stem in list(train_stems):
        label_path = source_root / "labels" / "train" / f"{stem}.txt"
        if 24 not in _class_ids(label_path):
            continue
        for copy_index in range(1, policy.vehicle_multiplier):
            alias = f"{stem}__vehup{copy_index:02d}"
            _copy_or_link(
                source_root / "images" / "train" / f"{stem}.jpg",
                output_root / "images" / "train" / f"{alias}.jpg",
            )
            _copy_or_link(label_path, output_root / "labels" / "train" / f"{alias}.txt")
            train_stems.append(alias)
            output_groups[alias] = {"group": output_groups[stem]["group"], "split": "train"}
            vehicle_upsampled_images += 1

    selected_by_class: Counter[str] = Counter()
    selected_per_stem: Counter[str] = Counter()
    for candidate in selected:
        selected_per_stem[candidate.stem] += 1
        alias = f"{candidate.stem}__hn{selected_per_stem[candidate.stem]:02d}"
        image = cv2.imread(
            str(source_root / "images" / "train" / f"{candidate.stem}.jpg"),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError(f"cannot read train image for crop: {candidate.stem}")
        left, top, right, bottom = candidate.crop
        image_path = output_root / "images" / "train" / f"{alias}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), image[top:bottom, left:right]):
            raise RuntimeError(f"failed to write hard-negative image: {image_path}")
        label_path = output_root / "labels" / "train" / f"{alias}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("", encoding="utf-8")
        train_stems.append(alias)
        output_groups[alias] = {"group": candidate.group, "split": "train"}
        selected_by_class[candidate.coarse_class] += 1

    _write_manifest(output_root, "train", train_stems)
    _write_manifest(output_root, "val", val_stems)
    manifests_dir = output_root / "manifests"
    (manifests_dir / "source-groups.json").write_text(
        json.dumps(output_groups, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_dataset = dict(dataset_payload)
    output_dataset.update(
        {"path": str(output_root.resolve()), "train": "images/train", "val": "images/val"}
    )
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(output_dataset, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    result = HardNegativeResult(
        output_root=output_root,
        original_train_images=len(_source_stems(source_root, "train")),
        vehicle_upsampled_images=vehicle_upsampled_images,
        selected_hard_negatives=len(selected),
        rejected_target_overlap=rejected_target_overlap,
        selected_by_coarse_class={
            name: selected_by_class.get(name, 0) for name in ("ship", "vehicle")
        },
    )
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        **asdict(result),
        "output_root": str(output_root.resolve()),
        "policy": asdict(policy),
    }
    (reports_dir / "hard-negative.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (reports_dir / "hard-negative.md").write_text(
        "\n".join(
            [
                "# Main Hard-Negative Dataset",
                "",
                f"- Original train images: {result.original_train_images}",
                f"- Vehicle upsample aliases: {result.vehicle_upsampled_images}",
                f"- Hard negatives: {result.selected_hard_negatives}",
                f"- Rejected target overlaps: {result.rejected_target_overlap}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return result
