import os
from pathlib import Path

import pytest
import yaml
from PIL import Image
from shapely.errors import GEOSException

import xh_detect.data.dota as dota_module
from xh_detect.data.dota import ConversionStats, convert_split, parse_label_file, write_dataset_yaml


def test_parse_label_file_maps_target_classes_and_skips_headers(tmp_path: Path) -> None:
    label = tmp_path / "P0001.txt"
    label.write_text(
        "\ufeffimagesource:GoogleEarth\n"
        "gsd:0.5\n"
        "10 10 30 10 30 30 10 30 plane 0\n"
        "40 40 60 40 60 60 40 60 ship 1\n"
        "70 70 90 70 90 90 70 90 small-vehicle 0\n"
        "12 12 24 12 24 24 12 24 large-vehicle 0\n"
        "14 14 26 14 26 26 14 26 small vehicle 0\n"
        "15 15 35 15 35 35 15 35 large vehicle 0\n"
        "20 20 40 20 40 40 20 40 tennis-court 0\n",
        encoding="utf-8",
    )

    annotations = parse_label_file(label, image_id="P0001")

    assert [item.class_id for item in annotations] == [0, 1, 2, 2, 2, 2]
    assert [item.difficult for item in annotations] == [False, True, False, False, False, False]


def test_conversion_stats_targets_is_read_only_mapping() -> None:
    source_targets = {0: 1, 1: 2, 2: 3}

    stats = ConversionStats(images=4, targets=source_targets, invalid_lines=5, skipped_images=6)
    source_targets[0] = 99

    assert stats.targets == {0: 1, 1: 2, 2: 3}
    with pytest.raises(TypeError):
        stats.targets[0] = 7


def test_convert_split_creates_empty_negative_label_when_source_label_missing(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_root = tmp_path / "converted"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 50), color="black").save(images_dir / "P0001.png")

    stats = convert_split(images_dir, labels_dir, output_root, split="train")

    assert (output_root / "images" / "train" / "P0001.png").exists()
    assert (output_root / "labels" / "train" / "P0001.txt").read_text(encoding="utf-8") == ""
    assert stats == ConversionStats(
        images=1,
        targets={0: 0, 1: 0, 2: 0},
        invalid_lines=0,
        skipped_images=0,
    )


def test_convert_split_excludes_difficult_targets_from_labels_and_counts(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_root = tmp_path / "converted"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 100), color="black").save(images_dir / "P0002.png")
    (labels_dir / "P0002.txt").write_text(
        "10 10 30 10 30 30 10 30 plane 1\n40 40 60 40 60 60 40 60 ship 0\n",
        encoding="utf-8",
    )

    stats = convert_split(images_dir, labels_dir, output_root, split="train")

    assert (output_root / "labels" / "train" / "P0002.txt").read_text(encoding="utf-8") == (
        "1 0.40000000 0.40000000 0.60000000 0.40000000 "
        "0.60000000 0.60000000 0.40000000 0.60000000\n"
    )
    assert stats.targets == {0: 0, 1: 1, 2: 0}


def test_convert_split_counts_only_malformed_mapped_target_lines(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_root = tmp_path / "converted"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 100), color="black").save(images_dir / "P0003.png")
    (labels_dir / "P0003.txt").write_text(
        "imagesource:GoogleEarth\n"
        "gsd:0.5\n"
        "oops 10 30 10 30 30 10 30 plane 0\n"
        "oops 10 30 10 30 30 10 30 harbor 0\n",
        encoding="utf-8",
    )

    stats = convert_split(images_dir, labels_dir, output_root, split="train")

    assert stats.invalid_lines == 1
    assert (output_root / "labels" / "train" / "P0003.txt").read_text(encoding="utf-8") == ""


def test_convert_split_rejects_invalid_difficult_tokens_for_mapped_targets(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_root = tmp_path / "converted"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 100), color="black").save(images_dir / "P0003.png")
    (labels_dir / "P0003.txt").write_text(
        "10 10 30 10 30 30 10 30 plane 0\n"
        "10 10 30 10 30 30 10 30 plane x\n"
        "10 10 30 10 30 30 10 30 plane\n"
        "10 10 30 10 30 30 10 30 harbor x\n",
        encoding="utf-8",
    )

    stats = convert_split(images_dir, labels_dir, output_root, split="train")

    assert stats.invalid_lines == 2
    assert stats.targets == {0: 1, 1: 0, 2: 0}
    assert (output_root / "labels" / "train" / "P0003.txt").read_text(encoding="utf-8") == (
        "0 0.10000000 0.10000000 0.30000000 0.10000000 "
        "0.30000000 0.30000000 0.10000000 0.30000000\n"
    )


@pytest.mark.parametrize(
    ("line", "image_name"),
    [
        ("10 10 130 10 130 30 10 30 plane 0", "P0004"),
        ("nan 10 30 10 30 30 10 30 plane 0", "P0005"),
        ("10 10 30 30 10 30 30 10 plane 0", "P0006"),
        ("10 10 10 10 10 10 10 10 plane 0", "P0007"),
    ],
)
def test_convert_split_rejects_invalid_target_geometry(
    tmp_path: Path,
    line: str,
    image_name: str,
) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_root = tmp_path / "converted"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 100), color="black").save(images_dir / f"{image_name}.png")
    (labels_dir / f"{image_name}.txt").write_text(f"{line}\n", encoding="utf-8")

    stats = convert_split(images_dir, labels_dir, output_root, split="train")

    assert stats.invalid_lines == 1
    assert stats.targets == {0: 0, 1: 0, 2: 0}
    assert (output_root / "labels" / "train" / f"{image_name}.txt").read_text(
        encoding="utf-8"
    ) == ""


@pytest.mark.parametrize(
    "split",
    [
        "../../outside",
        "/outside",
        "",
        "train/evil",
        "train\\evil",
    ],
)
def test_convert_split_rejects_invalid_split_without_touching_paths(
    tmp_path: Path,
    split: str,
) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_root = tmp_path / "converted"
    outside_dir = tmp_path / "outside"
    probe_file = tmp_path / "probe.txt"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 100), color="black").save(images_dir / "P0011.png")
    probe_file.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="split"):
        convert_split(images_dir, labels_dir, output_root, split=split)

    assert not output_root.exists()
    assert not outside_dir.exists()
    assert probe_file.read_text(encoding="utf-8") == "keep"


def test_link_or_copy_prefers_symlink_when_hardlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "nested" / "destination.bin"
    source.write_bytes(b"symlink-fallback")
    symlink_calls: list[tuple[Path, Path]] = []

    def fake_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("hardlinks unavailable")

    def fake_symlink_to(self: Path, target: Path, target_is_directory: bool = False) -> None:
        assert not target_is_directory
        symlink_calls.append((self, Path(target)))
        self.write_bytes(Path(target).read_bytes())

    monkeypatch.setattr(os, "link", fake_link)
    monkeypatch.setattr(Path, "symlink_to", fake_symlink_to)

    dota_module._link_or_copy(source, destination)

    assert destination.read_bytes() == b"symlink-fallback"
    assert symlink_calls == [(destination, source.resolve())]


def test_link_or_copy_falls_back_to_copy2_when_link_and_symlink_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "nested" / "destination.bin"
    source.write_bytes(b"copy-fallback")

    def fake_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("hardlinks unavailable")

    def fake_symlink_to(*_args: object, **_kwargs: object) -> None:
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(os, "link", fake_link)
    monkeypatch.setattr(Path, "symlink_to", fake_symlink_to)

    dota_module._link_or_copy(source, destination)

    assert destination.read_bytes() == b"copy-fallback"


def test_convert_split_skips_corrupt_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_root = tmp_path / "converted"
    images_dir.mkdir()
    labels_dir.mkdir()
    (images_dir / "broken.png").write_bytes(b"not-a-real-png")

    stats = convert_split(images_dir, labels_dir, output_root, split="train")

    assert stats == ConversionStats(
        images=0,
        targets={0: 0, 1: 0, 2: 0},
        invalid_lines=0,
        skipped_images=1,
    )
    assert not (output_root / "images" / "train" / "broken.png").exists()


def test_write_dataset_yaml_writes_expected_contents(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"

    dataset_yaml = write_dataset_yaml(dataset_root)

    assert dataset_yaml == dataset_root / "dataset.yaml"
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    assert payload == {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "aircraft", 1: "ship", 2: "vehicle"},
    }


def test_data_package_re_exports_dota_helpers() -> None:
    from xh_detect.data import convert_split as exported_convert_split
    from xh_detect.data import parse_label_file as exported_parse_label_file
    from xh_detect.data import write_dataset_yaml as exported_write_dataset_yaml

    assert exported_convert_split is convert_split
    assert exported_parse_label_file is parse_label_file
    assert exported_write_dataset_yaml is write_dataset_yaml


def test_convert_split_rerun_replaces_existing_output_hardlink(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_root = tmp_path / "converted"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (32, 32), color="black").save(images_dir / "P0008.png")

    first_stats = convert_split(images_dir, labels_dir, output_root, split="train")
    assert first_stats.images == 1

    destination_image = output_root / "images" / "train" / "P0008.png"
    sentinel = tmp_path / "sentinel.bin"
    sentinel.write_bytes(b"keep-me")
    destination_image.unlink()
    os.link(sentinel, destination_image)

    second_stats = convert_split(images_dir, labels_dir, output_root, split="train")

    assert second_stats.images == 1
    assert sentinel.read_bytes() == b"keep-me"
    assert destination_image.read_bytes() != b"keep-me"


def test_convert_split_treats_geos_failures_as_invalid_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenPolygon:
        @property
        def is_valid(self) -> bool:
            raise GEOSException("boom")

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_root = tmp_path / "converted"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 100), color="black").save(images_dir / "P0009.png")
    (labels_dir / "P0009.txt").write_text(
        "10 10 30 10 30 30 10 30 plane 0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dota_module, "Polygon", lambda *_args, **_kwargs: BrokenPolygon())

    stats = convert_split(images_dir, labels_dir, output_root, split="train")

    assert stats.invalid_lines == 1
    assert (output_root / "labels" / "train" / "P0009.txt").read_text(encoding="utf-8") == ""


def test_convert_split_keeps_images_with_only_non_target_labels(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_root = tmp_path / "converted"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (100, 100), color="black").save(images_dir / "P0010.png")
    (labels_dir / "P0010.txt").write_text(
        "imagesource:GoogleEarth\ngsd:0.5\n10 10 30 10 30 30 10 30 harbor 0\n",
        encoding="utf-8",
    )

    stats = convert_split(images_dir, labels_dir, output_root, split="train")

    assert (output_root / "images" / "train" / "P0010.png").exists()
    assert (output_root / "labels" / "train" / "P0010.txt").read_text(encoding="utf-8") == ""
    assert stats.targets == {0: 0, 1: 0, 2: 0}
