from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from xh_detect.data.ship_balance import build_ship_balanced_dataset


def _write_sample(root: Path, split: str, stem: str, label: str) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{stem}.jpg").write_bytes(b"fake-jpeg")
    (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")


def _write_dataset(root: Path) -> None:
    names = {index: f"class-{index}" for index in range(25)}
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_sample(root, "train", "aircraft", "4 0.5 0.5 0.2 0.2\n")
    _write_sample(root, "train", "qhs", "2 0.5 0.5 0.2 0.2\n")
    _write_sample(root, "train", "ms", "3 0.5 0.5 0.2 0.2\n")
    _write_sample(
        root,
        "train",
        "qhs_ms",
        "2 0.4 0.4 0.2 0.2\n3 0.6 0.6 0.2 0.2\n",
    )
    _write_sample(root, "val", "val_ship", "3 0.5 0.5 0.2 0.2\n")


def test_build_ship_balanced_dataset_duplicates_qhs_and_ms_train_images(
    tmp_path: Path,
) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    _write_dataset(source)

    result = build_ship_balanced_dataset(source, output)

    train_images = sorted(path.name for path in (output / "images" / "train").glob("*.jpg"))
    train_labels = sorted(path.name for path in (output / "labels" / "train").glob("*.txt"))
    assert train_images == [
        "aircraft.jpg",
        "ms.jpg",
        "ms__shipbal01.jpg",
        "qhs.jpg",
        "qhs__shipbal01.jpg",
        "qhs_ms.jpg",
        "qhs_ms__shipbal01.jpg",
    ]
    assert [Path(name).stem for name in train_labels] == [Path(name).stem for name in train_images]
    assert result.original_train_images == 4
    assert result.balanced_train_images == 7
    assert result.duplicated_train_images == 3
    assert result.duplicated_by_class == {2: 2, 3: 2}


def test_build_ship_balanced_dataset_caps_mixed_qhs_ms_by_max_factor(tmp_path: Path) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    _write_dataset(source)

    build_ship_balanced_dataset(source, output, qhs_factor=3, ms_factor=2)

    train_images = sorted(path.name for path in (output / "images" / "train").glob("*.jpg"))
    assert train_images == [
        "aircraft.jpg",
        "ms.jpg",
        "ms__shipbal01.jpg",
        "qhs.jpg",
        "qhs__shipbal01.jpg",
        "qhs__shipbal02.jpg",
        "qhs_ms.jpg",
        "qhs_ms__shipbal01.jpg",
        "qhs_ms__shipbal02.jpg",
    ]


def test_build_ship_balanced_dataset_keeps_validation_once_and_writes_reports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    _write_dataset(source)

    build_ship_balanced_dataset(source, output)

    assert sorted(path.name for path in (output / "images" / "val").glob("*.jpg")) == [
        "val_ship.jpg"
    ]
    assert sorted(path.name for path in (output / "labels" / "val").glob("*.txt")) == [
        "val_ship.txt"
    ]
    dataset_yaml = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset_yaml["path"] == str(output.resolve())
    assert dataset_yaml["train"] == "images/train"
    assert dataset_yaml["val"] == "images/val"
    report = json.loads((output / "reports" / "ship-balance.json").read_text(encoding="utf-8"))
    assert report["policy"] == {"qhs_factor": 2, "ms_factor": 2}
    assert report["original_train_images"] == 4
    assert report["balanced_train_images"] == 7
    markdown = (output / "reports" / "ship-balance.md").read_text(encoding="utf-8")
    assert "| Original Train Images | 4 |" in markdown


def test_build_ship_balanced_dataset_rejects_orphan_validation_label(tmp_path: Path) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    _write_dataset(source)
    (source / "labels" / "val" / "orphan.txt").write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing image"):
        build_ship_balanced_dataset(source, output)


def test_build_ship_balanced_dataset_rejects_overlapping_output(tmp_path: Path) -> None:
    source = tmp_path / "xh25"
    _write_dataset(source)

    with pytest.raises(ValueError, match="overlap"):
        build_ship_balanced_dataset(source, source / "nested")


def test_build_ship_balanced_dataset_rejects_existing_nonempty_output(tmp_path: Path) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    _write_dataset(source)
    output.mkdir()
    (output / "existing.txt").write_text("busy", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        build_ship_balanced_dataset(source, output)


def test_build_ship_balanced_dataset_rejects_output_root_file(tmp_path: Path) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    _write_dataset(source)
    output.write_text("busy", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        build_ship_balanced_dataset(source, output)


@pytest.mark.parametrize("kwargs", [{"qhs_factor": 0}, {"ms_factor": -1}, {"qhs_factor": True}])
def test_build_ship_balanced_dataset_rejects_bad_factors(
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    _write_dataset(source)

    with pytest.raises(ValueError, match="factor"):
        build_ship_balanced_dataset(source, output, **kwargs)
