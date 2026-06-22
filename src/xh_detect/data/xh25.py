from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType

import yaml
from PIL import Image, UnidentifiedImageError

from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import ObjectAnnotation, Polygon4

_CROP_SUFFIX = re.compile(r"_crop\d+$", re.IGNORECASE)
_BOUNDARY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ImageRecord:
    stem: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    mode: str
    group_id: str
    perceptual_hash: str
    annotations: tuple[ObjectAnnotation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "annotations", tuple(self.annotations))


@dataclass(frozen=True)
class DatasetAudit:
    images: int
    labels: int
    targets: Mapping[int, int]
    images_per_class: Mapping[int, int]
    dimensions: Mapping[str, int]
    modes: Mapping[str, int]
    source_groups: int
    invalid_lines: int
    near_duplicate_candidates: tuple[tuple[str, str], ...]
    records: tuple[ImageRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))
        object.__setattr__(
            self,
            "images_per_class",
            MappingProxyType(dict(self.images_per_class)),
        )
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))
        object.__setattr__(self, "modes", MappingProxyType(dict(self.modes)))
        object.__setattr__(
            self,
            "near_duplicate_candidates",
            tuple(tuple(pair) for pair in self.near_duplicate_candidates),
        )
        object.__setattr__(self, "records", tuple(self.records))


@dataclass(frozen=True)
class PreparedDataset:
    output_root: Path
    train_stems: frozenset[str]
    val_stems: frozenset[str]
    train_groups: frozenset[str]
    val_groups: frozenset[str]
    train_class_counts: Mapping[int, int]
    val_class_counts: Mapping[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "train_stems", frozenset(self.train_stems))
        object.__setattr__(self, "val_stems", frozenset(self.val_stems))
        object.__setattr__(self, "train_groups", frozenset(self.train_groups))
        object.__setattr__(self, "val_groups", frozenset(self.val_groups))
        object.__setattr__(
            self,
            "train_class_counts",
            MappingProxyType(
                {class_id: self.train_class_counts.get(class_id, 0) for class_id in range(25)}
            ),
        )
        object.__setattr__(
            self,
            "val_class_counts",
            MappingProxyType(
                {class_id: self.val_class_counts.get(class_id, 0) for class_id in range(25)}
            ),
        )


def source_group_id(stem: str) -> str:
    group_id = _CROP_SUFFIX.sub("", stem)
    return group_id or stem


def _line_error(path: Path, line_number: int, message: str) -> ValueError:
    return ValueError(f"{path}:{line_number}: {message}")


def _read_normalized_hbb(
    path: Path,
) -> tuple[tuple[int, float, float, float, float], ...]:
    boxes: list[tuple[int, float, float, float, float]] = []
    text = path.read_text(encoding="utf-8-sig")
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise _line_error(path, line_number, "YOLO HBB labels require five fields")

        try:
            class_id = int(fields[0])
        except ValueError:
            raise _line_error(path, line_number, "class ID must be numeric") from None
        if class_id not in get_taxonomy("xh25").valid_ids:
            raise _line_error(path, line_number, f"invalid class ID {class_id}")

        try:
            x_center, y_center, box_width, box_height = map(float, fields[1:])
        except ValueError:
            raise _line_error(path, line_number, "coordinates must be numeric") from None
        coordinates = (x_center, y_center, box_width, box_height)
        if not all(math.isfinite(value) for value in coordinates):
            raise _line_error(path, line_number, "coordinates must be finite")
        if box_width <= 0.0 or box_height <= 0.0:
            raise _line_error(
                path,
                line_number,
                "bounding box width and height must be positive",
            )
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and box_width <= 1.0
            and box_height <= 1.0
        ):
            raise _line_error(path, line_number, "bounding box is outside image")

        left = x_center - box_width / 2.0
        right = x_center + box_width / 2.0
        top = y_center - box_height / 2.0
        bottom = y_center + box_height / 2.0
        if (
            left < -_BOUNDARY_TOLERANCE
            or top < -_BOUNDARY_TOLERANCE
            or right > 1.0 + _BOUNDARY_TOLERANCE
            or bottom > 1.0 + _BOUNDARY_TOLERANCE
        ):
            raise _line_error(path, line_number, "bounding box is outside image")
        boxes.append(
            (
                class_id,
                max(0.0, left),
                max(0.0, top),
                min(1.0, right),
                min(1.0, bottom),
            )
        )

    return tuple(boxes)


def parse_yolo_hbb_label(
    path: Path,
    image_id: str,
    width: int,
    height: int,
) -> tuple[ObjectAnnotation, ...]:
    if width <= 0 or height <= 0:
        raise _line_error(path, 0, "width and height must be positive")

    annotations: list[ObjectAnnotation] = []
    for class_id, left, top, right, bottom in _read_normalized_hbb(path):
        polygon: Polygon4 = (
            (left * width, top * height),
            (right * width, top * height),
            (right * width, bottom * height),
            (left * width, bottom * height),
        )
        annotations.append(
            ObjectAnnotation(
                image_id=image_id,
                class_id=class_id,
                polygon=polygon,
            )
        )

    return tuple(annotations)


def _average_hash(image: Image.Image) -> str:
    resized = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    get_flattened_data = getattr(resized, "get_flattened_data", None)
    pixels = list(resized.getdata()) if get_flattened_data is None else list(get_flattened_data())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | int(pixel >= mean)
    return f"{bits:016x}"


def audit_dataset(source_root: Path) -> DatasetAudit:
    images_dir = source_root / "images" / "train"
    labels_dir = source_root / "labels" / "train"
    errors: list[str] = []
    for directory in (images_dir, labels_dir):
        if not directory.exists():
            errors.append(f"required directory does not exist: {directory}")
        elif not directory.is_dir():
            errors.append(f"required path is not a directory: {directory}")
    if errors:
        raise ValueError("dataset audit failed:\n" + "\n".join(errors))

    image_paths = {path.stem: path for path in sorted(images_dir.glob("*.jpg"))}
    label_paths = {path.stem: path for path in sorted(labels_dir.glob("*.txt"))}
    if not image_paths:
        errors.append(f"dataset is empty: no .jpg images in {images_dir}")
    if not label_paths:
        errors.append(f"dataset is empty: no .txt labels in {labels_dir}")

    for stem in sorted(image_paths.keys() - label_paths.keys()):
        errors.append(f"missing label for image: {image_paths[stem]}")
    for stem in sorted(label_paths.keys() - image_paths.keys()):
        errors.append(f"missing image for label: {label_paths[stem]}")

    records: list[ImageRecord] = []
    targets: Counter[int] = Counter()
    images_per_class: Counter[int] = Counter()
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()

    for stem in sorted(image_paths.keys() & label_paths.keys()):
        image_path = image_paths[stem]
        label_path = label_paths[stem]
        image_details: tuple[int, int, str, str] | None = None
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                if width <= 0 or height <= 0:
                    errors.append(f"{image_path}: width and height must be positive")
                else:
                    image_details = (
                        width,
                        height,
                        str(image.mode),
                        _average_hash(image),
                    )
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            errors.append(f"{image_path}: damaged image: {error}")

        if image_details is None:
            try:
                _read_normalized_hbb(label_path)
            except (OSError, UnicodeError, ValueError) as error:
                errors.append(str(error))
            continue

        width, height, mode, perceptual_hash = image_details
        try:
            annotations = parse_yolo_hbb_label(
                label_path,
                image_id=stem,
                width=width,
                height=height,
            )
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(str(error))
            continue

        for annotation in annotations:
            targets[annotation.class_id] += 1
        for class_id in {annotation.class_id for annotation in annotations}:
            images_per_class[class_id] += 1
        dimensions[f"{width}x{height}"] += 1
        modes[mode] += 1
        records.append(
            ImageRecord(
                stem=stem,
                image_path=image_path,
                label_path=label_path,
                width=width,
                height=height,
                mode=mode,
                group_id=source_group_id(stem),
                perceptual_hash=perceptual_hash,
                annotations=annotations,
            )
        )

    if errors:
        raise ValueError("dataset audit failed:\n" + "\n".join(errors))

    records_by_hash: dict[str, list[ImageRecord]] = {}
    for record in records:
        records_by_hash.setdefault(record.perceptual_hash, []).append(record)
    duplicate_candidates = {
        tuple(sorted((first.stem, second.stem)))
        for bucket in records_by_hash.values()
        for first, second in combinations(bucket, 2)
        if first.group_id != second.group_id
    }
    return DatasetAudit(
        images=len(image_paths),
        labels=len(label_paths),
        targets=dict(targets),
        images_per_class=dict(images_per_class),
        dimensions=dict(dimensions),
        modes=dict(modes),
        source_groups=len({record.group_id for record in records}),
        invalid_lines=0,
        near_duplicate_candidates=tuple(sorted(duplicate_candidates)),
        records=tuple(records),
    )


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _class_counts(records: tuple[ImageRecord, ...]) -> dict[int, int]:
    counts: Counter[int] = Counter(
        annotation.class_id for record in records for annotation in record.annotations
    )
    return {class_id: counts[class_id] for class_id in range(25)}


def _coarse_counts(class_counts: Mapping[int, int]) -> dict[str, int]:
    taxonomy = get_taxonomy("xh25")
    counts = {"ship": 0, "aircraft": 0, "vehicle": 0}
    for class_id in range(25):
        counts[taxonomy.coarse_name(class_id)] += class_counts[class_id]
    return counts


def _safe_unlink_file(path: Path) -> None:
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            raise ValueError(f"refusing to unlink directory as a file: {path}")
        path.unlink()


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink_file(destination)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        _safe_unlink_file(destination)

    try:
        destination.symlink_to(source.resolve())
        return "symlink"
    except OSError:
        _safe_unlink_file(destination)

    shutil.copy2(source, destination)
    return "copy"


def _validate_output_target_parent(path: Path, output_root: Path) -> None:
    resolved_output = output_root.resolve()
    resolved_parent = path.parent.resolve()
    if resolved_parent != resolved_output and not resolved_parent.is_relative_to(resolved_output):
        raise ValueError(f"refusing to write outside output_root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve()
    if resolved_parent != resolved_output and not resolved_parent.is_relative_to(resolved_output):
        raise ValueError(f"refusing to write outside output_root: {path}")


def _atomic_write_text(path: Path, text: str, output_root: Path) -> None:
    _validate_output_target_parent(path, output_root)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, value: object, output_root: Path) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        output_root,
    )


def _split_directories(output_root: Path) -> tuple[Path, ...]:
    return (
        output_root / "images" / "train",
        output_root / "images" / "val",
        output_root / "labels" / "train",
        output_root / "labels" / "val",
    )


def _reset_split_directories(source_root: Path, output_root: Path) -> None:
    resolved_output = output_root.resolve()
    source_data_directories = (
        (source_root / "images" / "train").resolve(),
        (source_root / "labels" / "train").resolve(),
    )
    for directory in _split_directories(output_root):
        resolved_directory = directory.resolve()
        if resolved_directory == resolved_output or not resolved_directory.is_relative_to(
            resolved_output
        ):
            raise ValueError(f"refusing to clean split directory outside output_root: {directory}")
        if any(
            source_directory == resolved_directory
            or source_directory.is_relative_to(resolved_directory)
            or resolved_directory.is_relative_to(source_directory)
            for source_directory in source_data_directories
        ):
            raise ValueError(f"output split directory would contain source data: {directory}")
        if directory.is_symlink():
            raise ValueError(f"refusing to recursively clean symlink: {directory}")
        if directory.exists():
            if not directory.is_dir():
                raise ValueError(f"output split path is not a directory: {directory}")
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def _select_split(
    audit: DatasetAudit,
    val_ratio: float,
    seed: int,
) -> tuple[tuple[ImageRecord, ...], tuple[ImageRecord, ...]]:
    records_by_group: dict[str, list[ImageRecord]] = {}
    class_groups = {class_id: set() for class_id in range(25)}
    group_classes: dict[str, set[int]] = {}
    for record in audit.records:
        records_by_group.setdefault(record.group_id, []).append(record)
        classes = {annotation.class_id for annotation in record.annotations}
        group_classes.setdefault(record.group_id, set()).update(classes)
        for class_id in classes:
            class_groups[class_id].add(record.group_id)

    required_val_groups: dict[int, int] = {}
    for class_id in range(25):
        groups = class_groups[class_id]
        if len(groups) < 2:
            raise ValueError(
                f"class {class_id} requires at least 2 source groups; "
                f"found {len(groups)} source groups"
            )
        minimum = 2 if len(groups) >= 3 else 1
        required_val_groups[class_id] = max(
            minimum,
            min(len(groups) - 1, round(len(groups) * val_ratio)),
        )

    ranked_class_groups = {
        class_id: tuple(
            sorted(
                groups,
                key=lambda group_id: _stable_rank(seed, group_id),
            )
        )
        for class_id, groups in class_groups.items()
    }

    def preserves_train_groups(candidate: str, selected: frozenset[str] | set[str]) -> bool:
        selected_with_candidate = selected | {candidate}
        return all(
            len(class_groups[class_id] & selected_with_candidate) <= len(class_groups[class_id]) - 1
            for class_id in group_classes[candidate]
        )

    failed_states: set[frozenset[str]] = set()

    def select_required_groups(selected: frozenset[str]) -> frozenset[str] | None:
        if selected in failed_states:
            return None

        unmet_classes: list[tuple[int, int, tuple[str, ...]]] = []
        for class_id in range(25):
            selected_count = len(class_groups[class_id] & selected)
            required = required_val_groups[class_id]
            if selected_count >= required:
                continue
            candidates = tuple(
                group_id
                for group_id in ranked_class_groups[class_id]
                if group_id not in selected and preserves_train_groups(group_id, selected)
            )
            if len(candidates) < required - selected_count:
                failed_states.add(selected)
                return None
            unmet_classes.append((len(candidates), class_id, candidates))

        if not unmet_classes:
            return selected

        _, _, candidates = min(unmet_classes, key=lambda item: (item[0], item[1]))
        for candidate in candidates:
            result = select_required_groups(selected | {candidate})
            if result is not None:
                return result

        failed_states.add(selected)
        return None

    selected_groups = select_required_groups(frozenset())
    if selected_groups is None:
        raise ValueError(
            "validation source-group targets cannot be met while preserving "
            "at least one train source group for every class"
        )
    val_groups = set(selected_groups)

    target_val_images = round(len(audit.records) * val_ratio)
    val_images = sum(len(records_by_group[group_id]) for group_id in val_groups)
    remaining_groups = sorted(
        records_by_group.keys() - val_groups,
        key=lambda group_id: _stable_rank(seed, group_id),
    )
    for group_id in remaining_groups:
        if val_images >= target_val_images:
            break
        if preserves_train_groups(group_id, val_groups):
            val_groups.add(group_id)
            val_images += len(records_by_group[group_id])

    train_records = tuple(record for record in audit.records if record.group_id not in val_groups)
    val_records = tuple(record for record in audit.records if record.group_id in val_groups)
    train_counts = _class_counts(train_records)
    val_counts = _class_counts(val_records)
    for class_id in range(25):
        if train_counts[class_id] <= 0 or val_counts[class_id] <= 0:
            raise ValueError(
                f"class {class_id} is missing from "
                f"{'train' if train_counts[class_id] <= 0 else 'val'} split"
            )

    train_stems = {record.stem for record in train_records}
    val_stems = {record.stem for record in val_records}
    train_groups = {record.group_id for record in train_records}
    selected_val_groups = {record.group_id for record in val_records}
    if not train_stems.isdisjoint(val_stems):
        raise ValueError("train and val stems overlap")
    if not train_groups.isdisjoint(selected_val_groups):
        raise ValueError("train and val source groups overlap")
    return train_records, val_records


def _relative_image_path(split: str, stem: str) -> str:
    return (Path("images") / split / f"{stem}.jpg").as_posix()


def _demo_samples(val_records: tuple[ImageRecord, ...]) -> dict[str, str]:
    taxonomy = get_taxonomy("xh25")
    samples: dict[str, str] = {}
    for coarse_name in ("ship", "aircraft", "vehicle"):
        candidates = sorted(
            record.stem
            for record in val_records
            if any(
                taxonomy.coarse_name(annotation.class_id) == coarse_name
                for annotation in record.annotations
            )
        )
        if not candidates:
            raise ValueError(f"validation split is missing a {coarse_name} demo sample")
        samples[coarse_name] = _relative_image_path("val", candidates[0])
    return samples


def _coco_ground_truth(
    val_records: tuple[ImageRecord, ...],
    image_map: Mapping[str, int],
) -> dict[str, object]:
    taxonomy = get_taxonomy("xh25")
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    for record in sorted(val_records, key=lambda item: item.stem):
        image_id = image_map[record.stem]
        images.append(
            {
                "id": image_id,
                "file_name": _relative_image_path("val", record.stem),
                "width": record.width,
                "height": record.height,
            }
        )
        for annotation in record.annotations:
            xs = [
                min(float(record.width), max(0.0, float(point[0]))) for point in annotation.polygon
            ]
            ys = [
                min(float(record.height), max(0.0, float(point[1]))) for point in annotation.polygon
            ]
            xmin = min(xs)
            ymin = min(ys)
            box_width = max(xs) - xmin
            box_height = max(ys) - ymin
            if box_width <= 0.0 or box_height <= 0.0:
                raise ValueError(f"{record.stem}: clamped bounding box must have positive size")
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": annotation.class_id,
                    "bbox": [xmin, ymin, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "images": images,
        "categories": [
            {"id": class_id, "name": taxonomy.names[class_id]} for class_id in range(25)
        ],
        "annotations": annotations,
    }


def _analysis_report(
    audit: DatasetAudit,
    train_records: tuple[ImageRecord, ...],
    val_records: tuple[ImageRecord, ...],
    val_ratio: float,
    seed: int,
    link_mode_counts: Mapping[str, int],
) -> dict[str, object]:
    source_counts = {class_id: audit.targets.get(class_id, 0) for class_id in range(25)}
    train_counts = _class_counts(train_records)
    val_counts = _class_counts(val_records)
    train_groups = {record.group_id for record in train_records}
    val_groups = {record.group_id for record in val_records}
    return {
        "source": {
            "images": audit.images,
            "labels": audit.labels,
            "targets": source_counts,
            "dimensions": dict(audit.dimensions),
            "modes": dict(audit.modes),
            "source_groups": audit.source_groups,
            "near_duplicate_candidates": list(audit.near_duplicate_candidates),
        },
        "split": {
            "train": {
                "images": len(train_records),
                "targets": train_counts,
                "coarse_counts": _coarse_counts(train_counts),
                "source_groups": len(train_groups),
            },
            "val": {
                "images": len(val_records),
                "targets": val_counts,
                "coarse_counts": _coarse_counts(val_counts),
                "source_groups": len(val_groups),
            },
            "group_overlap": len(train_groups & val_groups),
        },
        "val_ratio": val_ratio,
        "seed": seed,
        "link_mode_counts": {
            mode: link_mode_counts.get(mode, 0) for mode in ("hardlink", "symlink", "copy")
        },
    }


def _analysis_markdown(report: Mapping[str, object]) -> str:
    source = report["source"]
    split = report["split"]
    assert isinstance(source, Mapping)
    assert isinstance(split, Mapping)
    train = split["train"]
    val = split["val"]
    assert isinstance(train, Mapping)
    assert isinstance(val, Mapping)
    target_counts = {"train": train["targets"], "val": val["targets"]}
    coarse_counts = {
        "train": train["coarse_counts"],
        "val": val["coarse_counts"],
    }
    source_group_counts = {
        "source_groups": source["source_groups"],
        "train_groups": train["source_groups"],
        "val_groups": val["source_groups"],
    }
    return (
        "# XH25 Dataset Analysis\n\n"
        f"- Source images: {source['images']}\n"
        f"- Source labels: {source['labels']}\n"
        f"- Source groups: {source['source_groups']}\n"
        f"- Train images: {train['images']}\n"
        f"- Validation images: {val['images']}\n"
        f"- Group overlap: {split['group_overlap']}\n"
        f"- Validation ratio: {report['val_ratio']}\n"
        f"- Seed: {report['seed']}\n\n"
        "## Dimensions\n\n"
        f"```json\n{json.dumps(source['dimensions'], indent=2)}\n```\n\n"
        "## Modes\n\n"
        f"```json\n{json.dumps(source['modes'], indent=2)}\n```\n\n"
        "## Coarse counts\n\n"
        f"```json\n{json.dumps(coarse_counts, indent=2)}\n```\n\n"
        "## Near-duplicate candidates\n\n"
        f"```json\n{json.dumps(source['near_duplicate_candidates'], indent=2)}\n"
        "```\n\n"
        "## Link modes\n\n"
        f"```json\n{json.dumps(report['link_mode_counts'], indent=2)}\n```\n\n"
        "## Source target counts\n\n"
        f"```json\n{json.dumps(source['targets'], indent=2)}\n```\n\n"
        "## Target counts\n\n"
        f"```json\n{json.dumps(target_counts, indent=2)}\n```\n\n"
        "## Source group counts\n\n"
        f"```json\n{json.dumps(source_group_counts, indent=2)}\n```\n"
    )


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> PreparedDataset:
    if (
        not isinstance(val_ratio, float)
        or not math.isfinite(val_ratio)
        or not 0.0 < val_ratio < 0.5
    ):
        raise ValueError("val_ratio must be a finite float between 0 and 0.5")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    source_root = Path(source_root)
    output_root = Path(output_root)
    audit = audit_dataset(source_root)
    train_records, val_records = _select_split(audit, val_ratio, seed)
    demo_samples = _demo_samples(val_records)
    val_image_map = {
        record.stem: image_id
        for image_id, record in enumerate(
            sorted(val_records, key=lambda item: item.stem),
            start=1,
        )
    }
    coco = _coco_ground_truth(val_records, val_image_map)

    manifests_dir = output_root / "manifests"
    reports_dir = output_root / "reports"
    metadata_paths = (
        manifests_dir / "train.txt",
        manifests_dir / "val.txt",
        manifests_dir / "source-groups.json",
        manifests_dir / "val-image-map.json",
        manifests_dir / "demo-samples.json",
        output_root / "dataset.yaml",
        reports_dir / "dataset-analysis.json",
        reports_dir / "dataset-analysis.md",
        reports_dir / "val-ground-truth.json",
    )
    for metadata_path in metadata_paths:
        _validate_output_target_parent(metadata_path, output_root)

    _reset_split_directories(source_root, output_root)
    link_mode_counts: Counter[str] = Counter()
    for split, records in (("train", train_records), ("val", val_records)):
        for record in records:
            link_mode_counts[
                _link_or_copy(
                    record.image_path,
                    output_root / "images" / split / record.image_path.name,
                )
            ] += 1
            link_mode_counts[
                _link_or_copy(
                    record.label_path,
                    output_root / "labels" / split / record.label_path.name,
                )
            ] += 1

    sorted_train = sorted(record.stem for record in train_records)
    sorted_val = sorted(record.stem for record in val_records)
    train_manifest = "".join(f"{_relative_image_path('train', stem)}\n" for stem in sorted_train)
    val_manifest = "".join(f"{_relative_image_path('val', stem)}\n" for stem in sorted_val)
    source_groups = {
        record.stem: {
            "group": record.group_id,
            "split": "val" if record.stem in val_image_map else "train",
        }
        for record in sorted(audit.records, key=lambda item: item.stem)
    }
    taxonomy = get_taxonomy("xh25")
    dataset_yaml = yaml.safe_dump(
        {
            "path": str(output_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": dict(taxonomy.names),
        },
        allow_unicode=True,
        sort_keys=False,
    )
    analysis = _analysis_report(
        audit,
        train_records,
        val_records,
        val_ratio,
        seed,
        link_mode_counts,
    )

    _atomic_write_text(manifests_dir / "train.txt", train_manifest, output_root)
    _atomic_write_text(manifests_dir / "val.txt", val_manifest, output_root)
    _atomic_write_json(
        manifests_dir / "source-groups.json",
        source_groups,
        output_root,
    )
    _atomic_write_json(
        manifests_dir / "val-image-map.json",
        val_image_map,
        output_root,
    )
    _atomic_write_json(
        manifests_dir / "demo-samples.json",
        demo_samples,
        output_root,
    )
    _atomic_write_text(output_root / "dataset.yaml", dataset_yaml, output_root)
    _atomic_write_json(
        reports_dir / "dataset-analysis.json",
        analysis,
        output_root,
    )
    _atomic_write_text(
        reports_dir / "dataset-analysis.md",
        _analysis_markdown(analysis),
        output_root,
    )
    _atomic_write_json(
        reports_dir / "val-ground-truth.json",
        coco,
        output_root,
    )

    train_groups = frozenset(record.group_id for record in train_records)
    val_groups = frozenset(record.group_id for record in val_records)
    return PreparedDataset(
        output_root=output_root,
        train_stems=frozenset(sorted_train),
        val_stems=frozenset(sorted_val),
        train_groups=train_groups,
        val_groups=val_groups,
        train_class_counts=_class_counts(train_records),
        val_class_counts=_class_counts(val_records),
    )
