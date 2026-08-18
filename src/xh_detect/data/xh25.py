from __future__ import annotations

import csv
import ctypes
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import zipfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, mkdtemp
from types import MappingProxyType
from typing import BinaryIO
from uuid import uuid4

import numpy as np
import yaml
from PIL import Image

from xh_detect.data.xh25_fs import (
    _windows_error,
)
from xh_detect.data.xh25_fs import (
    locked_directories as _locked_directories,
)
from xh_detect.data.xh25_split import _required_val_group_counts, optimize_validation_groups
from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import ObjectAnnotation, Polygon4

_CROP_SUFFIX = re.compile(r"_crop\d+$", re.IGNORECASE)
_BOUNDARY_TOLERANCE = 1e-6
_PHASH_DISTANCE_THRESHOLD = 6
_TRANSACTION_MARKER_NAME = ".xh25-transaction"
_WINDOWS_DELETE_RETRIES = 3


def _dct_basis(size: int = 32, coefficients: int = 8) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64) + 0.5
    frequencies = np.arange(coefficients, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * frequencies * positions / size)
    basis[0] *= math.sqrt(1.0 / size)
    basis[1:] *= math.sqrt(2.0 / size)
    return basis


_PHASH_DCT_BASIS = _dct_basis()


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
    reviewed_core_stems: frozenset[str] = frozenset()
    added_val_stems: frozenset[str] = frozenset()
    duplicate_group_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "train_stems", frozenset(self.train_stems))
        object.__setattr__(self, "val_stems", frozenset(self.val_stems))
        object.__setattr__(self, "train_groups", frozenset(self.train_groups))
        object.__setattr__(self, "val_groups", frozenset(self.val_groups))
        object.__setattr__(self, "reviewed_core_stems", frozenset(self.reviewed_core_stems))
        object.__setattr__(self, "added_val_stems", frozenset(self.added_val_stems))
        object.__setattr__(
            self,
            "duplicate_group_pairs",
            tuple(tuple(pair) for pair in self.duplicate_group_pairs),
        )
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


@dataclass
class _ValidationSearchFrame:
    selected: frozenset[str]
    class_id: int
    candidates: tuple[str, ...]
    next_index: int = 0


@dataclass(frozen=True)
class _ReviewedArchive:
    core_stems: frozenset[str]
    label_overrides: Mapping[str, Path]
    sha256: str
    target_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "core_stems", frozenset(self.core_stems))
        object.__setattr__(self, "label_overrides", MappingProxyType(dict(self.label_overrides)))


@dataclass(frozen=True)
class _PreparationMetadata:
    reviewed_core_stems: frozenset[str] = frozenset()
    reviewed_archive_sha256: str | None = None
    reviewed_target_count: int = 0
    duplicate_group_pairs: tuple[tuple[str, str], ...] = ()
    raw_source_groups: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "reviewed_core_stems", frozenset(self.reviewed_core_stems))
        object.__setattr__(
            self,
            "duplicate_group_pairs",
            tuple(tuple(pair) for pair in self.duplicate_group_pairs),
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


def _perceptual_hash(image: Image.Image) -> str:
    resized = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float64)
    coefficients = _PHASH_DCT_BASIS @ pixels @ _PHASH_DCT_BASIS.T
    flattened = coefficients.reshape(-1)
    median = float(np.median(flattened[1:]))
    bits = 0
    for coefficient in flattened:
        bits = (bits << 1) | int(coefficient >= median)
    return f"{bits:016x}"


def _near_duplicate_candidates(
    records: tuple[ImageRecord, ...] | list[ImageRecord],
) -> tuple[tuple[str, str], ...]:
    hash_values = [int(record.perceptual_hash, 16) for record in records]
    candidate_indexes: set[tuple[int, int]] = set()
    for shift in range(0, 64, 8):
        buckets: dict[int, list[int]] = {}
        for index, value in enumerate(hash_values):
            buckets.setdefault((value >> shift) & 0xFF, []).append(index)
        for bucket in buckets.values():
            candidate_indexes.update(combinations(bucket, 2))

    candidates = {
        tuple(sorted((records[first].stem, records[second].stem)))
        for first, second in candidate_indexes
        if records[first].group_id != records[second].group_id
        and (hash_values[first] ^ hash_values[second]).bit_count() <= _PHASH_DISTANCE_THRESHOLD
    }
    return tuple(sorted(candidates))


def _sha256_binary(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_binary(stream)


def _safe_archive_parts(info: zipfile.ZipInfo) -> tuple[str, ...]:
    if "\\" in info.filename:
        raise ValueError(f"reviewed archive contains an unsafe path: {info.filename}")
    archive_path = PurePosixPath(info.filename)
    if archive_path.is_absolute() or ".." in archive_path.parts:
        raise ValueError(f"reviewed archive contains an unsafe path: {info.filename}")
    if info.flag_bits & 0x1:
        raise ValueError(f"reviewed archive contains an encrypted entry: {info.filename}")
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise ValueError(f"reviewed archive contains a symbolic link: {info.filename}")
    return archive_path.parts


def _load_reviewed_archive(
    source_root: Path,
    archive_path: Path,
    temporary_root: Path,
) -> _ReviewedArchive:
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise ValueError(f"reviewed archive does not exist: {archive_path}")

    source_images = source_root / "images" / "train"
    source_labels = source_root / "labels" / "train"
    extracted_labels = temporary_root / "labels"
    extracted_labels.mkdir(parents=True)
    image_members: dict[str, zipfile.ZipInfo] = {}
    label_members: dict[str, zipfile.ZipInfo] = {}
    archive_roots: set[tuple[str, ...]] = set()

    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot open reviewed archive: {archive_path}: {error}") from error

    with archive:
        for info in archive.infolist():
            parts = _safe_archive_parts(info)
            if info.is_dir() or len(parts) < 3:
                continue
            parent = tuple(part.casefold() for part in parts[-3:-1])
            suffix = PurePosixPath(parts[-1]).suffix.casefold()
            if parent == ("images", "val") and suffix == ".jpg":
                destination = image_members
            elif parent == ("labels", "val") and suffix == ".txt":
                destination = label_members
            else:
                continue
            stem = PurePosixPath(parts[-1]).stem
            if not stem or stem in destination:
                raise ValueError(f"reviewed archive contains a duplicate stem: {stem}")
            destination[stem] = info
            archive_roots.add(parts[:-3])

        if not image_members or not label_members:
            raise ValueError("reviewed archive must contain images/val and labels/val")
        if set(image_members) != set(label_members):
            missing_labels = sorted(set(image_members) - set(label_members))
            missing_images = sorted(set(label_members) - set(image_members))
            raise ValueError(
                "reviewed archive image/label stems are inconsistent: "
                f"missing_labels={missing_labels[:5]}, missing_images={missing_images[:5]}"
            )
        if len(archive_roots) != 1:
            raise ValueError("reviewed archive images and labels must share one package root")

        target_count = 0
        label_overrides: dict[str, Path] = {}
        for stem in sorted(image_members):
            source_image = source_images / f"{stem}.jpg"
            source_label = source_labels / f"{stem}.txt"
            if not source_image.is_file() or not source_label.is_file():
                raise ValueError(f"reviewed archive stem is missing from source dataset: {stem}")
            with archive.open(image_members[stem]) as reviewed_image:
                reviewed_hash = _sha256_binary(reviewed_image)
            if reviewed_hash != _sha256_file(source_image):
                raise ValueError(f"reviewed image differs from source image: {stem}")

            label_path = extracted_labels / f"{stem}.txt"
            with archive.open(label_members[stem]) as reviewed_label:
                label_path.write_bytes(reviewed_label.read())
            target_count += len(_read_normalized_hbb(label_path))
            label_overrides[stem] = label_path

    return _ReviewedArchive(
        core_stems=frozenset(image_members),
        label_overrides=label_overrides,
        sha256=_sha256_file(archive_path),
        target_count=target_count,
    )


def audit_dataset(
    source_root: Path,
    *,
    label_overrides: Mapping[str, Path] | None = None,
) -> DatasetAudit:
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
    override_paths = dict(label_overrides or {})
    for stem, path in sorted(override_paths.items()):
        if stem not in image_paths or stem not in label_paths:
            errors.append(f"label override does not match a source image and label: {stem}")
        elif not Path(path).is_file():
            errors.append(f"label override does not exist: {path}")
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
        label_path = Path(override_paths.get(stem, label_paths[stem]))
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
                        _perceptual_hash(image),
                    )
        except Exception as error:
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

    return DatasetAudit(
        images=len(image_paths),
        labels=len(label_paths),
        targets=dict(targets),
        images_per_class=dict(images_per_class),
        dimensions=dict(dimensions),
        modes=dict(modes),
        source_groups=len({record.group_id for record in records}),
        invalid_lines=0,
        near_duplicate_candidates=_near_duplicate_candidates(records),
        records=tuple(records),
    )


def _read_duplicate_group_pairs(
    csv_path: Path | None,
    available_stems: set[str],
) -> tuple[tuple[str, str], ...]:
    if csv_path is None:
        return ()
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise ValueError(f"duplicate group CSV does not exist: {csv_path}")

    pairs: set[tuple[str, str]] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"left", "right"}.issubset(reader.fieldnames):
            raise ValueError("duplicate group CSV requires left,right columns")
        for row_number, row in enumerate(reader, start=2):
            left_value = (row.get("left") or "").strip()
            right_value = (row.get("right") or "").strip()
            if not left_value or not right_value:
                raise ValueError(f"duplicate group CSV row {row_number} has an empty stem")
            left = Path(left_value).stem
            right = Path(right_value).stem
            if left == right:
                raise ValueError(f"duplicate group CSV row {row_number} is a self-pair: {left}")
            missing = sorted({left, right} - available_stems)
            if missing:
                raise ValueError(
                    f"duplicate group CSV row {row_number} has unknown stems: {missing}"
                )
            pair = tuple(sorted((left, right)))
            if pair in pairs:
                raise ValueError(f"duplicate group CSV row {row_number} repeats pair: {pair}")
            pairs.add(pair)
    return tuple(sorted(pairs))


def _merge_duplicate_groups(
    audit: DatasetAudit,
    pairs: tuple[tuple[str, str], ...],
) -> DatasetAudit:
    if not pairs:
        return audit
    records_by_stem = {record.stem: record for record in audit.records}
    parent = {record.group_id: record.group_id for record in audit.records}

    def find(group_id: str) -> str:
        while parent[group_id] != group_id:
            parent[group_id] = parent[parent[group_id]]
            group_id = parent[group_id]
        return group_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        canonical, merged = sorted((left_root, right_root))
        parent[merged] = canonical

    for left_stem, right_stem in pairs:
        union(records_by_stem[left_stem].group_id, records_by_stem[right_stem].group_id)

    merged_records = tuple(
        replace(record, group_id=find(record.group_id)) for record in audit.records
    )
    return replace(
        audit,
        source_groups=len({record.group_id for record in merged_records}),
        records=merged_records,
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


def _is_reparse_point(path: Path) -> bool:
    path = Path(path)
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except (FileNotFoundError, NotADirectoryError):
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _assert_no_reparse_points(path: Path) -> None:
    path = Path(path)
    for candidate in (path, *path.parents):
        if _is_reparse_point(candidate):
            raise ValueError(f"refusing reparse point in output path: {candidate}")


def _remove_reparse_point(path: Path) -> None:
    try:
        path.unlink()
    except (IsADirectoryError, PermissionError):
        path.rmdir()


def _open_windows_delete_handle(path: Path) -> int:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE

    file_read_attributes = 0x0080
    delete_access = 0x00010000
    file_share_read = 0x0001
    file_share_write = 0x0002
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    file_flag_open_reparse_point = 0x00200000
    invalid_handle = ctypes.c_void_p(-1).value
    handle = create_file(
        str(path),
        file_read_attributes | delete_access,
        file_share_read | file_share_write,
        None,
        open_existing,
        file_flag_backup_semantics | file_flag_open_reparse_point,
        None,
    )
    if handle == invalid_handle:
        error = ctypes.get_last_error()
        if error in {2, 3} and not os.path.lexists(path):
            return 0
        raise _windows_error(error, "cannot open cleanup entry", path)
    return int(handle)


def _close_windows_delete_handle(handle: int, path: Path) -> None:
    from ctypes import wintypes

    if handle == 0:
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    if not close_handle(handle):
        raise _windows_error(ctypes.get_last_error(), "cannot close cleanup entry", path)


def _windows_handle_file_attributes(handle: int, path: Path) -> int:
    from ctypes import wintypes

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("reparse_tag", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_information = kernel32.GetFileInformationByHandleEx
    get_information.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    get_information.restype = wintypes.BOOL

    file_attribute_tag_info = 9
    information = FileAttributeTagInfo()
    if not get_information(
        handle,
        file_attribute_tag_info,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        raise _windows_error(ctypes.get_last_error(), "cannot inspect cleanup entry", path)
    return int(information.file_attributes)


def _mark_windows_handle_for_deletion(handle: int, path: Path) -> None:
    from ctypes import wintypes

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("delete_file", ctypes.c_ubyte)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    set_information.restype = wintypes.BOOL

    file_disposition_info = 4
    disposition = FileDispositionInfo(1)
    if not set_information(
        handle,
        file_disposition_info,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise _windows_error(ctypes.get_last_error(), "cannot delete cleanup entry", path)


def _remove_windows_tree_entry(path: Path, *, reject_reparse_root: bool = False) -> None:
    path = Path(path)
    if not os.path.lexists(path):
        return

    file_attribute_directory = 0x0010
    file_attribute_reparse_point = 0x0400
    error_dir_not_empty = 145
    last_error: OSError | None = None
    for attempt in range(_WINDOWS_DELETE_RETRIES):
        handle = _open_windows_delete_handle(path)
        if handle == 0:
            return
        try:
            attributes = _windows_handle_file_attributes(handle, path)
            if reject_reparse_root and attributes & file_attribute_reparse_point:
                raise ValueError(f"refusing to clean reparse point root: {path}")
            is_directory = bool(attributes & file_attribute_directory)
            is_reparse_point = bool(attributes & file_attribute_reparse_point)
            if is_directory and not is_reparse_point:
                with os.scandir(path) as entries:
                    children = [Path(entry.path) for entry in entries]
                for child in children:
                    _remove_windows_tree_entry(child)
            try:
                _mark_windows_handle_for_deletion(handle, path)
                return
            except OSError as error:
                if (
                    is_directory
                    and not is_reparse_point
                    and getattr(error, "winerror", None) == error_dir_not_empty
                    and attempt + 1 < _WINDOWS_DELETE_RETRIES
                ):
                    last_error = error
                    continue
                raise
        finally:
            _close_windows_delete_handle(handle, path)
    if last_error is not None:
        raise last_error


def _safe_remove_tree(path: Path) -> None:
    path = Path(path)
    if not os.path.lexists(path):
        return
    if os.name == "nt":
        _remove_windows_tree_entry(path, reject_reparse_root=True)
        return
    if _is_reparse_point(path):
        raise ValueError(f"refusing to clean reparse point root: {path}")
    if not path.is_dir():
        path.unlink()
        return
    with _locked_directories((path,)):
        if _is_reparse_point(path):
            raise ValueError(f"refusing to clean reparse point root: {path}")
        with os.scandir(path) as entries:
            children = [Path(entry.path) for entry in entries]
        for child in children:
            if _is_reparse_point(child):
                _remove_reparse_point(child)
            elif stat.S_ISDIR(os.lstat(child).st_mode):
                shutil.rmtree(child)
            else:
                child.unlink()
    if _is_reparse_point(path):
        raise ValueError(f"refusing to clean replaced reparse point root: {path}")
    path.rmdir()


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


def _prepare_output_target_parent(path: Path, output_root: Path) -> None:
    _validate_output_target_parent(path, output_root)
    _assert_no_reparse_points(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_points(path.parent)
    _validate_output_target_parent(path, output_root)


def _atomic_write_text(path: Path, text: str, output_root: Path) -> None:
    _prepare_output_target_parent(path, output_root)
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


def _select_validation_groups(
    class_groups: Mapping[int, set[str]],
    group_classes: Mapping[str, set[int]],
    val_ratio: float,
    seed: int,
    initial_selected: frozenset[str] | set[str] = frozenset(),
) -> frozenset[str]:
    class_group_counts: dict[int, int] = {}
    for class_id in range(25):
        groups = class_groups[class_id]
        group_count = len(groups)
        if group_count < 2:
            raise ValueError(
                f"class {class_id} requires at least 2 source groups; "
                f"found {group_count} source groups"
            )
        class_group_counts[class_id] = group_count
    required_val_groups = _required_val_group_counts(class_groups, val_ratio)

    ranked_class_groups = {
        class_id: tuple(
            sorted(
                groups,
                key=lambda group_id: _stable_rank(seed, group_id),
            )
        )
        for class_id, groups in class_groups.items()
    }
    failed_states: set[frozenset[str]] = set()
    stack: list[_ValidationSearchFrame] = []
    selected = frozenset(initial_selected)
    unknown_groups = selected - group_classes.keys()
    if unknown_groups:
        raise ValueError(f"required validation groups do not exist: {sorted(unknown_groups)}")

    while True:
        if selected not in failed_states:
            selected_counts = dict.fromkeys(range(25), 0)
            for group_id in selected:
                for class_id in group_classes[group_id]:
                    selected_counts[class_id] += 1

            dead_end = any(
                selected_counts[class_id] > class_group_counts[class_id] - 1
                for class_id in range(25)
            )
            unmet_classes: list[tuple[int, int, int, tuple[str, ...]]] = []
            if not dead_end:
                for class_id in range(25):
                    selected_count = selected_counts[class_id]
                    required = required_val_groups[class_id]
                    if selected_count >= required:
                        continue
                    candidates = tuple(
                        group_id
                        for group_id in ranked_class_groups[class_id]
                        if group_id not in selected
                        and all(
                            selected_counts[related_class] < class_group_counts[related_class] - 1
                            for related_class in group_classes[group_id]
                        )
                    )
                    if selected_count + len(candidates) < required:
                        dead_end = True
                        break
                    unmet_classes.append(
                        (
                            len(candidates),
                            class_id,
                            required - selected_count,
                            candidates,
                        )
                    )

            if not dead_end and not unmet_classes:
                return selected

            if dead_end:
                failed_states.add(selected)
            else:
                _, class_id, deficit, candidates = min(
                    unmet_classes,
                    key=lambda item: (item[0], item[1]),
                )
                if all(group_classes[group_id] == {class_id} for group_id in candidates):
                    selected = selected.union(candidates[:deficit])
                    continue
                stack.append(
                    _ValidationSearchFrame(
                        selected=selected,
                        class_id=class_id,
                        candidates=candidates,
                    )
                )

        while stack:
            frame = stack[-1]
            if frame.next_index >= len(frame.candidates):
                failed_states.add(frame.selected)
                stack.pop()
                continue
            candidate = frame.candidates[frame.next_index]
            frame.next_index += 1
            next_state = frame.selected | {candidate}
            if next_state not in failed_states:
                selected = next_state
                break
        else:
            raise ValueError(
                "validation source-group targets cannot be met while preserving "
                "at least one train source group for every class"
            )


def _select_split(
    audit: DatasetAudit,
    val_ratio: float,
    seed: int,
    required_val_stems: frozenset[str] | set[str] = frozenset(),
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

    def preserves_train_groups(candidate: str, selected: frozenset[str] | set[str]) -> bool:
        selected_with_candidate = selected | {candidate}
        return all(
            len(class_groups[class_id] & selected_with_candidate) <= len(class_groups[class_id]) - 1
            for class_id in group_classes[candidate]
        )

    records_by_stem = {record.stem: record for record in audit.records}
    required_stems = frozenset(required_val_stems)
    missing_required = sorted(required_stems - records_by_stem.keys())
    if missing_required:
        raise ValueError(f"required validation stems do not exist: {missing_required[:10]}")
    frozen_val_groups = frozenset(records_by_stem[stem].group_id for stem in required_stems)

    val_groups = set(
        _select_validation_groups(
            class_groups,
            group_classes,
            val_ratio,
            seed,
            frozen_val_groups,
        )
    )

    target_val_images = math.ceil(len(audit.records) * val_ratio)
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
    if val_images < target_val_images:
        raise ValueError(
            "validation image target cannot be met while preserving at least one "
            f"train source group for every class: target={target_val_images}, actual={val_images}"
        )
    largest_group_size = max(len(group_records) for group_records in records_by_group.values())
    max_val_images = max(
        val_images,
        target_val_images + largest_group_size - 1,
    )
    val_groups = set(
        optimize_validation_groups(
            audit.records,
            val_groups,
            class_groups,
            val_ratio,
            seed,
            _stable_rank,
            frozen_selected=frozen_val_groups,
            min_val_images=target_val_images,
            max_val_images=max_val_images,
        )
    )

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
    if not required_stems.issubset(val_stems):
        raise ValueError("required validation stems were moved out of validation")
    if len(val_records) < target_val_images:
        raise ValueError("validation image count is below the requested minimum ratio")
    if len(val_records) > max_val_images:
        raise ValueError("validation image count exceeds the permitted group overshoot")
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
    records: tuple[ImageRecord, ...],
    image_map: Mapping[str, int],
    split: str,
) -> dict[str, object]:
    taxonomy = get_taxonomy("xh25")
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    for record in sorted(records, key=lambda item: item.stem):
        image_id = image_map[record.stem]
        images.append(
            {
                "id": image_id,
                "file_name": _relative_image_path(split, record.stem),
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
    metadata: _PreparationMetadata,
) -> dict[str, object]:
    source_counts = {class_id: audit.targets.get(class_id, 0) for class_id in range(25)}
    train_counts = _class_counts(train_records)
    val_counts = _class_counts(val_records)
    train_groups = {record.group_id for record in train_records}
    val_groups = {record.group_id for record in val_records}
    val_stems = {record.stem for record in val_records}
    reviewed_core = metadata.reviewed_core_stems
    added_val = val_stems - reviewed_core
    return {
        "source": {
            "images": audit.images,
            "labels": audit.labels,
            "targets": source_counts,
            "dimensions": dict(audit.dimensions),
            "modes": dict(audit.modes),
            "source_groups": audit.source_groups,
            "raw_source_groups": metadata.raw_source_groups or audit.source_groups,
            "effective_source_groups": audit.source_groups,
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
        "minimum_val_images": math.ceil(audit.images * val_ratio),
        "actual_val_ratio": len(val_records) / audit.images,
        "seed": seed,
        "reviewed": {
            "archive_sha256": metadata.reviewed_archive_sha256,
            "core_images": len(reviewed_core),
            "targets": metadata.reviewed_target_count,
            "added_val_images": len(added_val),
        },
        "accepted_duplicate_unions": [list(pair) for pair in metadata.duplicate_group_pairs],
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
    reviewed = report["reviewed"]
    assert isinstance(train, Mapping)
    assert isinstance(val, Mapping)
    assert isinstance(reviewed, Mapping)
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
        f"- Requested validation ratio: {report['val_ratio']}\n"
        f"- Actual validation ratio: {report['actual_val_ratio']}\n"
        f"- Minimum validation images: {report['minimum_val_images']}\n"
        f"- Reviewed validation core: {reviewed['core_images']}\n"
        f"- Added validation images: {reviewed['added_val_images']}\n"
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


def _metadata_paths(output_root: Path) -> tuple[Path, ...]:
    manifests_dir = output_root / "manifests"
    reports_dir = output_root / "reports"
    return (
        manifests_dir / "train.txt",
        manifests_dir / "val.txt",
        manifests_dir / "val-reviewed-core.txt",
        manifests_dir / "val-added.txt",
        manifests_dir / "near-duplicate-unions.csv",
        manifests_dir / "source-groups.json",
        manifests_dir / "train-image-map.json",
        manifests_dir / "val-image-map.json",
        manifests_dir / "demo-samples.json",
        output_root / "dataset.yaml",
        reports_dir / "dataset-analysis.json",
        reports_dir / "dataset-analysis.md",
        reports_dir / "train-ground-truth.json",
        reports_dir / "val-ground-truth.json",
    )


def _fixed_output_directories(output_root: Path) -> tuple[Path, ...]:
    return (
        output_root / "images",
        output_root / "images" / "train",
        output_root / "images" / "val",
        output_root / "labels",
        output_root / "labels" / "train",
        output_root / "labels" / "val",
        output_root / "manifests",
        output_root / "reports",
    )


def _validate_output_tree_paths(output_root: Path) -> None:
    _assert_no_reparse_points(output_root)
    for metadata_path in _metadata_paths(output_root):
        _validate_output_target_parent(metadata_path, output_root)
    for directory in _fixed_output_directories(output_root):
        if _is_reparse_point(directory):
            raise ValueError(f"refusing reparse point in output path: {directory}")


def _validate_existing_output_root(output_root: Path) -> None:
    if _is_reparse_point(output_root):
        raise ValueError(f"refusing reparse point in output path: {output_root}")
    try:
        output_mode = os.lstat(output_root).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(output_mode):
        raise ValueError(f"output_root must be a directory when it already exists: {output_root}")


def _create_stage_directories(stage_root: Path) -> None:
    _validate_output_tree_paths(stage_root)
    for directory in _fixed_output_directories(stage_root):
        directory.mkdir(parents=True, exist_ok=True)
        if _is_reparse_point(directory):
            raise ValueError(f"refusing reparse point in stage path: {directory}")


def _validate_materialized_dataset(
    root: Path,
    train_records: tuple[ImageRecord, ...],
    val_records: tuple[ImageRecord, ...],
    metadata: _PreparationMetadata | None = None,
) -> None:
    metadata = metadata or _PreparationMetadata()
    _validate_output_tree_paths(root)
    expected = {
        "train": {record.stem for record in train_records},
        "val": {record.stem for record in val_records},
    }
    for split in ("train", "val"):
        image_stems = {
            path.stem for path in (root / "images" / split).iterdir() if path.suffix == ".jpg"
        }
        label_stems = {
            path.stem for path in (root / "labels" / split).iterdir() if path.suffix == ".txt"
        }
        if image_stems != expected[split] or label_stems != expected[split]:
            raise ValueError(f"materialized {split} image/label stems are inconsistent")
        manifest_stems = {
            Path(line).stem
            for line in (root / "manifests" / f"{split}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        }
        if manifest_stems != expected[split]:
            raise ValueError(f"materialized {split} manifest stems are inconsistent")

    reviewed_core = {
        Path(line).stem
        for line in (root / "manifests" / "val-reviewed-core.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    added_val = {
        Path(line).stem
        for line in (root / "manifests" / "val-added.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if reviewed_core != set(metadata.reviewed_core_stems):
        raise ValueError("materialized reviewed validation manifest is inconsistent")
    if reviewed_core & added_val or reviewed_core | added_val != expected["val"]:
        raise ValueError("materialized added validation manifest is inconsistent")

    duplicate_pairs = _read_duplicate_group_pairs(
        root / "manifests" / "near-duplicate-unions.csv",
        expected["train"] | expected["val"],
    )
    if duplicate_pairs != metadata.duplicate_group_pairs:
        raise ValueError("materialized duplicate group manifest is inconsistent")

    source_groups = json.loads(
        (root / "manifests" / "source-groups.json").read_text(encoding="utf-8")
    )
    expected_all = expected["train"] | expected["val"]
    if set(source_groups) != expected_all:
        raise ValueError("materialized source-group manifest stems are inconsistent")
    for stem, details in source_groups.items():
        split = details.get("split")
        if split not in expected or stem not in expected[split]:
            raise ValueError("materialized source-group split is inconsistent")

    for split in ("train", "val"):
        image_map = json.loads(
            (root / "manifests" / f"{split}-image-map.json").read_text(encoding="utf-8")
        )
        sorted_stems = sorted(expected[split])
        expected_map = {stem: image_id for image_id, stem in enumerate(sorted_stems, start=1)}
        if image_map != expected_map:
            raise ValueError(f"materialized {split} image map is inconsistent")

        coco = json.loads(
            (root / "reports" / f"{split}-ground-truth.json").read_text(encoding="utf-8")
        )
        coco_images = coco.get("images")
        if not isinstance(coco_images, list):
            raise ValueError("materialized COCO images are invalid")
        coco_map = {
            Path(str(image["file_name"])).stem: image["id"]
            for image in coco_images
            if isinstance(image, Mapping)
        }
        if coco_map != expected_map:
            raise ValueError("materialized COCO image IDs are inconsistent")
        if any(
            Path(str(image["file_name"])).parts[:2] != ("images", split)
            for image in coco_images
            if isinstance(image, Mapping)
        ):
            raise ValueError("materialized COCO image split is inconsistent")


def _materialize_locked_stage(
    audit: DatasetAudit,
    train_records: tuple[ImageRecord, ...],
    val_records: tuple[ImageRecord, ...],
    stage_root: Path,
    published_root: Path,
    val_ratio: float,
    seed: int,
    transaction_id: str,
    metadata: _PreparationMetadata,
) -> None:
    demo_samples = _demo_samples(val_records)
    train_image_map = {
        record.stem: image_id
        for image_id, record in enumerate(
            sorted(train_records, key=lambda item: item.stem),
            start=1,
        )
    }
    val_image_map = {
        record.stem: image_id
        for image_id, record in enumerate(
            sorted(val_records, key=lambda item: item.stem),
            start=1,
        )
    }
    train_coco = _coco_ground_truth(train_records, train_image_map, "train")
    val_coco = _coco_ground_truth(val_records, val_image_map, "val")
    link_mode_counts: Counter[str] = Counter()
    for split, records in (("train", train_records), ("val", val_records)):
        for record in records:
            image_destination = stage_root / "images" / split / record.image_path.name
            label_destination = stage_root / "labels" / split / record.label_path.name
            _assert_no_reparse_points(image_destination.parent)
            link_mode_counts[_link_or_copy(record.image_path, image_destination)] += 1
            _assert_no_reparse_points(label_destination.parent)
            if record.stem in metadata.reviewed_core_stems:
                shutil.copy2(record.label_path, label_destination)
                link_mode_counts["copy"] += 1
            else:
                link_mode_counts[_link_or_copy(record.label_path, label_destination)] += 1

    sorted_train = sorted(record.stem for record in train_records)
    sorted_val = sorted(record.stem for record in val_records)
    train_manifest = "".join(f"{_relative_image_path('train', stem)}\n" for stem in sorted_train)
    val_manifest = "".join(f"{_relative_image_path('val', stem)}\n" for stem in sorted_val)
    reviewed_core_manifest = "".join(
        f"{_relative_image_path('val', stem)}\n" for stem in sorted(metadata.reviewed_core_stems)
    )
    added_val_manifest = "".join(
        f"{_relative_image_path('val', stem)}\n"
        for stem in sorted(set(sorted_val) - metadata.reviewed_core_stems)
    )
    duplicate_csv_stream = io.StringIO(newline="")
    duplicate_writer = csv.writer(duplicate_csv_stream, lineterminator="\n")
    duplicate_writer.writerow(("left", "right"))
    duplicate_writer.writerows(metadata.duplicate_group_pairs)
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
            "path": str(published_root.resolve()),
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
        metadata,
    )
    manifests_dir = stage_root / "manifests"
    reports_dir = stage_root / "reports"
    _atomic_write_text(manifests_dir / "train.txt", train_manifest, stage_root)
    _atomic_write_text(manifests_dir / "val.txt", val_manifest, stage_root)
    _atomic_write_text(
        manifests_dir / "val-reviewed-core.txt",
        reviewed_core_manifest,
        stage_root,
    )
    _atomic_write_text(manifests_dir / "val-added.txt", added_val_manifest, stage_root)
    _atomic_write_text(
        manifests_dir / "near-duplicate-unions.csv",
        duplicate_csv_stream.getvalue(),
        stage_root,
    )
    _atomic_write_json(manifests_dir / "source-groups.json", source_groups, stage_root)
    _atomic_write_json(manifests_dir / "train-image-map.json", train_image_map, stage_root)
    _atomic_write_json(manifests_dir / "val-image-map.json", val_image_map, stage_root)
    _atomic_write_json(manifests_dir / "demo-samples.json", demo_samples, stage_root)
    _atomic_write_text(stage_root / "dataset.yaml", dataset_yaml, stage_root)
    _atomic_write_json(reports_dir / "dataset-analysis.json", analysis, stage_root)
    _atomic_write_text(
        reports_dir / "dataset-analysis.md",
        _analysis_markdown(analysis),
        stage_root,
    )
    _atomic_write_json(reports_dir / "train-ground-truth.json", train_coco, stage_root)
    _atomic_write_json(reports_dir / "val-ground-truth.json", val_coco, stage_root)
    _atomic_write_text(
        stage_root / _TRANSACTION_MARKER_NAME,
        transaction_id,
        stage_root,
    )
    _validate_materialized_dataset(stage_root, train_records, val_records, metadata)


def _materialize_into_stage(
    audit: DatasetAudit,
    train_records: tuple[ImageRecord, ...],
    val_records: tuple[ImageRecord, ...],
    stage_root: Path,
    published_root: Path,
    val_ratio: float,
    seed: int,
    transaction_id: str,
    metadata: _PreparationMetadata,
) -> None:
    _create_stage_directories(stage_root)
    with _locked_directories((stage_root, *_fixed_output_directories(stage_root))):
        _validate_output_tree_paths(stage_root)
        _materialize_locked_stage(
            audit,
            train_records,
            val_records,
            stage_root,
            published_root,
            val_ratio,
            seed,
            transaction_id,
            metadata,
        )


def publish_train_mining_artifacts(dataset_root: Path) -> tuple[Path, Path]:
    dataset_root = Path(dataset_root)
    _validate_output_tree_paths(dataset_root)
    audit = audit_dataset(dataset_root)
    records = tuple(sorted(audit.records, key=lambda item: item.stem))
    train_stems = {record.stem for record in records}
    manifest_path = dataset_root / "manifests" / "train.txt"
    source_groups_path = dataset_root / "manifests" / "source-groups.json"
    if not manifest_path.is_file() or not source_groups_path.is_file():
        raise ValueError("prepared dataset is missing train manifest or source groups")
    manifest_stems = {
        Path(line).stem
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if manifest_stems != train_stems:
        raise ValueError("prepared train manifest is inconsistent with train images")
    source_groups = json.loads(source_groups_path.read_text(encoding="utf-8"))
    if not isinstance(source_groups, Mapping) or any(
        not isinstance(source_groups.get(stem), Mapping)
        or source_groups[stem].get("split") != "train"
        for stem in train_stems
    ):
        raise ValueError("prepared source groups are inconsistent with train images")

    image_map = {record.stem: image_id for image_id, record in enumerate(records, start=1)}
    truth = _coco_ground_truth(records, image_map, "train")
    image_map_path = dataset_root / "manifests" / "train-image-map.json"
    truth_path = dataset_root / "reports" / "train-ground-truth.json"
    _atomic_write_json(image_map_path, image_map, dataset_root)
    _atomic_write_json(truth_path, truth, dataset_root)
    return image_map_path, truth_path


def _reserve_sibling_path(output_root: Path, kind: str) -> Path:
    path = Path(
        mkdtemp(
            prefix=f".{output_root.name}.{kind}-",
            dir=output_root.parent,
        )
    )
    path.rmdir()
    return path


def _cleanup_or_report(path: Path, kind: str) -> None:
    try:
        _safe_remove_tree(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"retained {kind} after safe cleanup failure: {path}") from error


def _transaction_marker_matches(root: Path, transaction_id: str) -> bool:
    try:
        _assert_no_reparse_points(root)
    except ValueError:
        return False
    marker = root / _TRANSACTION_MARKER_NAME
    if _is_reparse_point(marker):
        return False
    try:
        if not stat.S_ISREG(os.lstat(marker).st_mode):
            return False
        return marker.read_text(encoding="utf-8") == transaction_id
    except OSError:
        return False


def _prepare_audited_dataset(
    audit: DatasetAudit,
    output_root: Path,
    val_ratio: float,
    seed: int,
    metadata: _PreparationMetadata,
) -> PreparedDataset:
    output_root = Path(output_root)
    _validate_existing_output_root(output_root)
    _validate_output_tree_paths(output_root)
    train_records, val_records = _select_split(
        audit,
        val_ratio,
        seed,
        metadata.reviewed_core_stems,
    )
    _assert_no_reparse_points(output_root.parent)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_points(output_root.parent)
    stage_root = Path(
        mkdtemp(
            prefix=f".{output_root.name}.stage-",
            dir=output_root.parent,
        )
    )
    backup_root: Path | None = None
    failed_root: Path | None = None
    published = False
    publication_validated = False
    transaction_id = uuid4().hex
    try:
        _materialize_into_stage(
            audit,
            train_records,
            val_records,
            stage_root,
            output_root,
            val_ratio,
            seed,
            transaction_id,
            metadata,
        )
        _validate_output_tree_paths(output_root)
        _validate_existing_output_root(output_root)
        _validate_output_tree_paths(stage_root)
        if os.path.lexists(output_root):
            backup_root = _reserve_sibling_path(output_root, "backup")
            os.replace(output_root, backup_root)
        try:
            os.replace(stage_root, output_root)
            published = True
        except BaseException:
            if (
                not os.path.lexists(stage_root)
                and os.path.lexists(output_root)
                and _transaction_marker_matches(output_root, transaction_id)
            ):
                published = True
            else:
                if os.path.lexists(output_root):
                    failed_root = _reserve_sibling_path(output_root, "failed")
                    os.replace(output_root, failed_root)
                raise
        marker_path = output_root / _TRANSACTION_MARKER_NAME
        if not _transaction_marker_matches(output_root, transaction_id):
            raise RuntimeError("published output transaction marker does not match")
        marker_path.unlink()
        _validate_materialized_dataset(output_root, train_records, val_records, metadata)
        publication_validated = True
        if backup_root is not None:
            _cleanup_or_report(backup_root, "backup")
            backup_root = None
    except BaseException:
        if publication_validated:
            raise
        if published and os.path.lexists(output_root):
            try:
                _safe_remove_tree(output_root)
            except (OSError, ValueError) as error:
                if backup_root is not None:
                    raise RuntimeError(
                        f"retained backup after rollback cleanup failure: {backup_root}"
                    ) from error
                raise
        if backup_root is not None and os.path.lexists(backup_root):
            os.replace(backup_root, output_root)
            backup_root = None
        if failed_root is not None and os.path.lexists(failed_root):
            _cleanup_or_report(failed_root, "failed output")
            failed_root = None
        if os.path.lexists(stage_root):
            _cleanup_or_report(stage_root, "stage")
        raise
    finally:
        if os.path.lexists(stage_root):
            _cleanup_or_report(stage_root, "stage")

    sorted_train = sorted(record.stem for record in train_records)
    sorted_val = sorted(record.stem for record in val_records)
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
        reviewed_core_stems=metadata.reviewed_core_stems,
        added_val_stems=frozenset(sorted_val) - metadata.reviewed_core_stems,
        duplicate_group_pairs=metadata.duplicate_group_pairs,
    )


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
    *,
    reviewed_archive: Path | None = None,
    duplicate_groups_csv: Path | None = None,
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
    resolved_source = source_root.resolve()
    resolved_output = output_root.resolve()
    if (
        resolved_source == resolved_output
        or resolved_source.is_relative_to(resolved_output)
        or resolved_output.is_relative_to(resolved_source)
    ):
        raise ValueError(
            "source_root and output_root overlap: "
            f"source_root={resolved_source}, output_root={resolved_output}"
        )
    for input_path, name in (
        (reviewed_archive, "reviewed_archive"),
        (duplicate_groups_csv, "duplicate_groups_csv"),
    ):
        if input_path is None:
            continue
        resolved_input = Path(input_path).resolve()
        if resolved_input == resolved_output or resolved_input.is_relative_to(resolved_output):
            raise ValueError(f"{name} must be outside output_root: {input_path}")

    _validate_existing_output_root(output_root)
    _validate_output_tree_paths(output_root)
    _assert_no_reparse_points(output_root.parent)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_points(output_root.parent)

    temporary_root: Path | None = None
    reviewed = _ReviewedArchive(frozenset(), {}, "", 0)
    try:
        if reviewed_archive is not None:
            temporary_root = Path(
                mkdtemp(prefix=f".{output_root.name}.reviewed-", dir=output_root.parent)
            )
            reviewed = _load_reviewed_archive(
                source_root,
                Path(reviewed_archive),
                temporary_root,
            )
        audit = audit_dataset(source_root, label_overrides=reviewed.label_overrides)
        raw_source_groups = audit.source_groups
        duplicate_pairs = _read_duplicate_group_pairs(
            Path(duplicate_groups_csv) if duplicate_groups_csv is not None else None,
            {record.stem for record in audit.records},
        )
        audit = _merge_duplicate_groups(audit, duplicate_pairs)
        metadata = _PreparationMetadata(
            reviewed_core_stems=reviewed.core_stems,
            reviewed_archive_sha256=reviewed.sha256 or None,
            reviewed_target_count=reviewed.target_count,
            duplicate_group_pairs=duplicate_pairs,
            raw_source_groups=raw_source_groups,
        )
        return _prepare_audited_dataset(
            audit,
            output_root,
            val_ratio,
            seed,
            metadata,
        )
    finally:
        if temporary_root is not None and os.path.lexists(temporary_root):
            _cleanup_or_report(temporary_root, "reviewed archive extraction")
