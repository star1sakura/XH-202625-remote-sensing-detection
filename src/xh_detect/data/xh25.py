from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import MappingProxyType

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
