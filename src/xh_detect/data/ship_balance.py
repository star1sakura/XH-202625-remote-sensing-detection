from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

QHS_CLASS_ID = 2
MS_CLASS_ID = 3
_CLASS_COUNT = 25


@dataclass(frozen=True)
class ShipBalanceResult:
    output_root: Path
    original_train_images: int
    balanced_train_images: int
    duplicated_train_images: int
    original_train_targets: dict[int, int]
    balanced_train_targets: dict[int, int]
    duplicated_by_class: dict[int, int]


def _positive_factor(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} factor must be a positive integer")
    return value


def _class_ids(label_path: Path) -> tuple[int, ...]:
    class_ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path} contains an invalid YOLO label line")
        class_ids.append(int(fields[0]))
    return tuple(class_ids)


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _validate_source_root(source_root: Path) -> dict[object, object]:
    dataset_yaml = source_root / "dataset.yaml"
    required = [
        dataset_yaml,
        source_root / "images" / "train",
        source_root / "images" / "val",
        source_root / "labels" / "train",
        source_root / "labels" / "val",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError("source dataset is incomplete: " + ", ".join(missing))
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("names"), dict):
        raise ValueError("source dataset.yaml must contain names mapping")
    return payload


def _validate_output_root(source_root: Path, output_root: Path) -> None:
    resolved_source = source_root.resolve()
    resolved_output = output_root.resolve()
    if (
        resolved_source == resolved_output
        or resolved_source.is_relative_to(resolved_output)
        or resolved_output.is_relative_to(resolved_source)
    ):
        raise ValueError(
            f"source_root and output_root overlap: {resolved_source} / {resolved_output}"
        )
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"output_root already exists and is not empty: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output_root already exists and is not empty: {output_root}")


def _target_counts(label_paths: list[Path]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for path in label_paths:
        counts.update(_class_ids(path))
    return {class_id: counts.get(class_id, 0) for class_id in range(_CLASS_COUNT)}


def _frequency(class_ids: tuple[int, ...], *, qhs_factor: int, ms_factor: int) -> int:
    frequency = 1
    if QHS_CLASS_ID in class_ids:
        frequency = max(frequency, qhs_factor)
    if MS_CLASS_ID in class_ids:
        frequency = max(frequency, ms_factor)
    return frequency


def _materialize_split_once(source_root: Path, output_root: Path, split: str) -> list[str]:
    image_stems = {path.stem for path in (source_root / "images" / split).glob("*.jpg")}
    label_stems = {path.stem for path in (source_root / "labels" / split).glob("*.txt")}
    orphan_labels = sorted(label_stems - image_stems)
    if orphan_labels:
        raise ValueError(f"missing image for label {orphan_labels[0]}.txt")
    stems: list[str] = []
    for image_path in sorted((source_root / "images" / split).glob("*.jpg")):
        label_path = source_root / "labels" / split / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"missing label for image {image_path.name}")
        _copy_or_link(image_path, output_root / "images" / split / image_path.name)
        _copy_or_link(label_path, output_root / "labels" / split / label_path.name)
        stems.append(image_path.stem)
    return stems


def _write_manifest(output_root: Path, split: str, stems: list[str]) -> None:
    manifest = output_root / "manifests" / f"{split}.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(f"images/{split}/{stem}.jpg\n" for stem in sorted(stems)),
        encoding="utf-8",
    )


def _write_reports(
    result: ShipBalanceResult,
    *,
    output_root: Path,
    qhs_factor: int,
    ms_factor: int,
) -> None:
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_root": str(output_root.resolve()),
        "policy": {"qhs_factor": qhs_factor, "ms_factor": ms_factor},
        "original_train_images": result.original_train_images,
        "balanced_train_images": result.balanced_train_images,
        "duplicated_train_images": result.duplicated_train_images,
        "original_train_targets": result.original_train_targets,
        "balanced_train_targets": result.balanced_train_targets,
        "duplicated_by_class": result.duplicated_by_class,
    }
    (reports_dir / "ship-balance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown = "\n".join(
        [
            "# Ship-Balanced Dataset",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Original Train Images | {result.original_train_images} |",
            f"| Balanced Train Images | {result.balanced_train_images} |",
            f"| Duplicated Train Images | {result.duplicated_train_images} |",
            "",
        ]
    )
    (reports_dir / "ship-balance.md").write_text(markdown, encoding="utf-8")


def build_ship_balanced_dataset(
    source_root: Path,
    output_root: Path,
    *,
    qhs_factor: int = 2,
    ms_factor: int = 2,
) -> ShipBalanceResult:
    qhs_factor = _positive_factor(qhs_factor, "qhs")
    ms_factor = _positive_factor(ms_factor, "ms")
    source_root = Path(source_root)
    output_root = Path(output_root)
    dataset_payload = _validate_source_root(source_root)
    _validate_output_root(source_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_stems: list[str] = []
    balanced_label_paths: list[Path] = []
    original_label_paths = sorted((source_root / "labels" / "train").glob("*.txt"))
    duplicated_by_class: Counter[int] = Counter()

    for label_path in original_label_paths:
        image_path = source_root / "images" / "train" / f"{label_path.stem}.jpg"
        if not image_path.is_file():
            raise ValueError(f"missing image for label {label_path.name}")
        class_ids = _class_ids(label_path)
        frequency = _frequency(class_ids, qhs_factor=qhs_factor, ms_factor=ms_factor)
        for copy_index in range(frequency):
            suffix = "" if copy_index == 0 else f"__shipbal{copy_index:02d}"
            stem = f"{label_path.stem}{suffix}"
            if copy_index > 0:
                duplicated_by_class.update(set(class_ids) & {QHS_CLASS_ID, MS_CLASS_ID})
            _copy_or_link(image_path, output_root / "images" / "train" / f"{stem}.jpg")
            _copy_or_link(label_path, output_root / "labels" / "train" / f"{stem}.txt")
            train_stems.append(stem)
            balanced_label_paths.append(output_root / "labels" / "train" / f"{stem}.txt")

    val_stems = _materialize_split_once(source_root, output_root, "val")
    _write_manifest(output_root, "train", train_stems)
    _write_manifest(output_root, "val", val_stems)
    dataset_yaml = dict(dataset_payload)
    dataset_yaml["path"] = str(output_root.resolve())
    dataset_yaml["train"] = "images/train"
    dataset_yaml["val"] = "images/val"
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    result = ShipBalanceResult(
        output_root=output_root,
        original_train_images=len(original_label_paths),
        balanced_train_images=len(train_stems),
        duplicated_train_images=len(train_stems) - len(original_label_paths),
        original_train_targets=_target_counts(original_label_paths),
        balanced_train_targets=_target_counts(balanced_label_paths),
        duplicated_by_class=dict(sorted(duplicated_by_class.items())),
    )
    _write_reports(result, output_root=output_root, qhs_factor=qhs_factor, ms_factor=ms_factor)
    return result
