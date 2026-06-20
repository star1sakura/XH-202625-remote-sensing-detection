from __future__ import annotations

import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml
from PIL import Image, UnidentifiedImageError
from shapely.errors import GEOSException
from shapely.geometry import Polygon

from xh_detect.types import ObjectAnnotation, Polygon4

CLASS_MAP = {
    "plane": 0,
    "ship": 1,
    "small-vehicle": 2,
    "large-vehicle": 2,
    "small vehicle": 2,
    "large vehicle": 2,
}

CLASS_TOKEN_PATTERNS = tuple(
    sorted(
        ((tuple(class_name.split()), class_id) for class_name, class_id in CLASS_MAP.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)

VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class ConversionStats:
    images: int
    targets: Mapping[int, int]
    invalid_lines: int
    skipped_images: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))


def parse_label_file(path: Path, image_id: str) -> tuple[ObjectAnnotation, ...]:
    annotations, _ = _parse_label_file(path, image_id)
    return annotations


def _parse_label_file(path: Path, image_id: str) -> tuple[tuple[ObjectAnnotation, ...], int]:
    annotations: list[ObjectAnnotation] = []
    invalid_lines = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("imagesource:") or parts[0].startswith("gsd:"):
            continue
        if len(parts) < 9:
            continue

        class_tokens = [token.lower() for token in parts[8:]]
        class_id: int | None = None
        difficult_token: str | None = None
        for candidate_tokens, candidate_class_id in CLASS_TOKEN_PATTERNS:
            if tuple(class_tokens[: len(candidate_tokens)]) != candidate_tokens:
                continue
            class_id = candidate_class_id
            remainder = class_tokens[len(candidate_tokens) :]
            if len(remainder) != 1:
                invalid_lines += 1
                break
            difficult_token = remainder[0]
            if difficult_token not in {"0", "1"}:
                invalid_lines += 1
                break
            try:
                coordinates = [float(value) for value in parts[:8]]
            except ValueError:
                invalid_lines += 1
                break
            polygon: Polygon4 = (
                (coordinates[0], coordinates[1]),
                (coordinates[2], coordinates[3]),
                (coordinates[4], coordinates[5]),
                (coordinates[6], coordinates[7]),
            )
            annotations.append(
                ObjectAnnotation(
                    image_id=image_id,
                    class_id=class_id,
                    polygon=polygon,
                    difficult=difficult_token == "1",
                )
            )
            break
        else:
            continue
    return tuple(annotations), invalid_lines


def _to_yolo_line(annotation: ObjectAnnotation, width: int, height: int) -> str:
    values = [str(annotation.class_id)]
    for x, y in annotation.polygon:
        values.append(f"{x / width:.8f}")
        values.append(f"{y / height:.8f}")
    return " ".join(values)


def _is_valid_annotation(annotation: ObjectAnnotation, width: int, height: int) -> bool:
    if not all(math.isfinite(coord) for point in annotation.polygon for coord in point):
        return False
    if not all(0.0 <= x < width and 0.0 <= y < height for x, y in annotation.polygon):
        return False
    try:
        polygon = Polygon(annotation.polygon)
        return polygon.is_valid and not polygon.is_empty and polygon.area > 0
    except GEOSException:
        return False


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
        return
    except OSError:
        pass
    try:
        destination.symlink_to(source.resolve())
        return
    except OSError:
        pass
    shutil.copy2(source, destination)


def convert_split(
    images_dir: Path,
    labels_dir: Path,
    output_root: Path,
    split: str,
) -> ConversionStats:
    if split not in VALID_SPLITS:
        raise ValueError(f"invalid split {split!r}; expected one of {sorted(VALID_SPLITS)}")

    converted_images = 0
    invalid_lines = 0
    skipped_images = 0
    targets = {0: 0, 1: 0, 2: 0}
    labels_output_dir = output_root / "labels" / split
    images_output_dir = output_root / "images" / split
    labels_output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(images_dir.glob("*.png")):
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
        except (OSError, UnidentifiedImageError):
            skipped_images += 1
            continue
        _link_or_copy(image_path, images_output_dir / image_path.name)
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            annotations, parse_invalid_lines = _parse_label_file(
                label_path, image_id=image_path.stem
            )
            invalid_lines += parse_invalid_lines
        else:
            annotations = ()
        valid_annotations = [
            annotation
            for annotation in annotations
            if _is_valid_annotation(annotation, width, height)
        ]
        invalid_lines += len(annotations) - len(valid_annotations)
        usable = [annotation for annotation in valid_annotations if not annotation.difficult]
        for annotation in usable:
            targets[annotation.class_id] += 1
        text = "\n".join(_to_yolo_line(annotation, width, height) for annotation in usable)
        (labels_output_dir / f"{image_path.stem}.txt").write_text(
            f"{text}\n" if text else "",
            encoding="utf-8",
        )
        converted_images += 1

    return ConversionStats(
        images=converted_images,
        targets=targets,
        invalid_lines=invalid_lines,
        skipped_images=skipped_images,
    )


def write_dataset_yaml(output_root: Path) -> Path:
    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "aircraft", 1: "ship", 2: "vehicle"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return dataset_yaml
