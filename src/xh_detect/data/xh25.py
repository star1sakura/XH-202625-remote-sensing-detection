from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import stat
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from types import MappingProxyType
from uuid import uuid4

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
_TRANSACTION_MARKER_NAME = ".xh25-transaction"
_WINDOWS_DELETE_RETRIES = 3


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


@dataclass
class _ValidationSearchFrame:
    selected: frozenset[str]
    class_id: int
    candidates: tuple[str, ...]
    next_index: int = 0


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
    selected = frozenset[str]()

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

    val_groups = set(
        _select_validation_groups(
            class_groups,
            group_classes,
            val_ratio,
            seed,
        )
    )

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
    val_groups = set(
        optimize_validation_groups(
            audit.records,
            val_groups,
            class_groups,
            val_ratio,
            seed,
            _stable_rank,
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


def _metadata_paths(output_root: Path) -> tuple[Path, ...]:
    manifests_dir = output_root / "manifests"
    reports_dir = output_root / "reports"
    return (
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
) -> None:
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

    val_image_map = json.loads(
        (root / "manifests" / "val-image-map.json").read_text(encoding="utf-8")
    )
    sorted_val = sorted(expected["val"])
    expected_map = {stem: image_id for image_id, stem in enumerate(sorted_val, start=1)}
    if val_image_map != expected_map:
        raise ValueError("materialized validation image map is inconsistent")

    coco = json.loads((root / "reports" / "val-ground-truth.json").read_text(encoding="utf-8"))
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


def _materialize_locked_stage(
    audit: DatasetAudit,
    train_records: tuple[ImageRecord, ...],
    val_records: tuple[ImageRecord, ...],
    stage_root: Path,
    published_root: Path,
    val_ratio: float,
    seed: int,
    transaction_id: str,
) -> None:
    demo_samples = _demo_samples(val_records)
    val_image_map = {
        record.stem: image_id
        for image_id, record in enumerate(
            sorted(val_records, key=lambda item: item.stem),
            start=1,
        )
    }
    coco = _coco_ground_truth(val_records, val_image_map)
    link_mode_counts: Counter[str] = Counter()
    for split, records in (("train", train_records), ("val", val_records)):
        for record in records:
            image_destination = stage_root / "images" / split / record.image_path.name
            label_destination = stage_root / "labels" / split / record.label_path.name
            _assert_no_reparse_points(image_destination.parent)
            link_mode_counts[_link_or_copy(record.image_path, image_destination)] += 1
            _assert_no_reparse_points(label_destination.parent)
            link_mode_counts[_link_or_copy(record.label_path, label_destination)] += 1

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
    )
    manifests_dir = stage_root / "manifests"
    reports_dir = stage_root / "reports"
    _atomic_write_text(manifests_dir / "train.txt", train_manifest, stage_root)
    _atomic_write_text(manifests_dir / "val.txt", val_manifest, stage_root)
    _atomic_write_json(manifests_dir / "source-groups.json", source_groups, stage_root)
    _atomic_write_json(manifests_dir / "val-image-map.json", val_image_map, stage_root)
    _atomic_write_json(manifests_dir / "demo-samples.json", demo_samples, stage_root)
    _atomic_write_text(stage_root / "dataset.yaml", dataset_yaml, stage_root)
    _atomic_write_json(reports_dir / "dataset-analysis.json", analysis, stage_root)
    _atomic_write_text(
        reports_dir / "dataset-analysis.md",
        _analysis_markdown(analysis),
        stage_root,
    )
    _atomic_write_json(reports_dir / "val-ground-truth.json", coco, stage_root)
    _atomic_write_text(
        stage_root / _TRANSACTION_MARKER_NAME,
        transaction_id,
        stage_root,
    )
    _validate_materialized_dataset(stage_root, train_records, val_records)


def _materialize_into_stage(
    audit: DatasetAudit,
    train_records: tuple[ImageRecord, ...],
    val_records: tuple[ImageRecord, ...],
    stage_root: Path,
    published_root: Path,
    val_ratio: float,
    seed: int,
    transaction_id: str,
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
        )


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

    _validate_existing_output_root(output_root)
    _validate_output_tree_paths(output_root)
    audit = audit_dataset(source_root)
    train_records, val_records = _select_split(audit, val_ratio, seed)
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
        _validate_materialized_dataset(output_root, train_records, val_records)
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
    )
