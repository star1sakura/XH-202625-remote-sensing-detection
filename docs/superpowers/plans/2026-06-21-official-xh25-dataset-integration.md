# Official XH25 Dataset Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the official 25-class YOLO HBB dataset into the existing training, tiled inference, competition evaluation, COCO export, and Gradio demo workflow.

**Architecture:** Keep the existing four-point internal box representation and add a shared taxonomy that maps 25 official classes to ship, aircraft, and vehicle. Materialize a deterministic group-stratified train/validation view without modifying `data/`, use an Ultralytics HBB detector for the official baseline, preserve fine-class IDs in JSON, and aggregate only inside competition evaluation and UI summaries.

**Tech Stack:** Python 3.12, PyTorch 2.5.1 CUDA 12.4, Ultralytics 8.4.71, OpenCV, Pillow, Shapely, PyYAML, Typer, Gradio, pytest, Ruff.

---

## File map

New focused modules:

- `src/xh_detect/taxonomy.py`: canonical fine-class names and coarse-class mappings.
- `src/xh_detect/data/xh25.py`: official YOLO HBB parsing, audit, grouping, splitting, linking, manifests, reports, and validation COCO ground truth.
- `tests/test_taxonomy.py`: taxonomy contract.
- `tests/test_xh25.py`: official-data adapter contract on synthetic fixtures.
- `configs/xh25-hbb.yaml`: official inference configuration.

Existing modules to modify:

- `src/xh_detect/config.py`: task and taxonomy-aware threshold validation.
- `src/xh_detect/detector.py`: HBB extraction while retaining OBB compatibility.
- `src/xh_detect/exporters.py`: configurable 25-class validation.
- `src/xh_detect/evaluator.py`: class-agnostic overall, coarse-class, and fine-class reports.
- `src/xh_detect/visualize.py`: taxonomy-aware labels, colors, and counts.
- `src/xh_detect/pipeline.py`: use the configured valid fine-class IDs.
- `src/xh_detect/app.py`: official HBB presentation and coarse/fine summaries.
- `src/xh_detect/cli.py`: `prepare-xh25`, generic detector construction, official evaluation, and richer training options.
- `src/xh_detect/training.py`: pass explicit reproducible Ultralytics training options.
- `src/xh_detect/data/__init__.py`: export official-data functions.
- `README.md`: official-data workflow and server commands.

Generated, ignored artifacts:

- `datasets/xh25/`
- `runs/train/xh25-baseline/`
- `outputs/xh25/`
- `outputs/xh25-demo/`

### Task 1: Add the canonical 25-class taxonomy

**Files:**
- Create: `src/xh_detect/taxonomy.py`
- Create: `tests/test_taxonomy.py`

- [ ] **Step 1: Write the failing taxonomy tests**

```python
from __future__ import annotations

import pytest

from xh_detect.taxonomy import get_taxonomy


def test_xh25_taxonomy_preserves_official_ids_and_names() -> None:
    taxonomy = get_taxonomy("xh25")

    assert taxonomy.valid_ids == frozenset(range(25))
    assert taxonomy.names[0] == "HM"
    assert taxonomy.names[8] == "A5_F-16"
    assert taxonomy.names[24] == "FSC"
    assert taxonomy.coarse_name(0) == "ship"
    assert taxonomy.coarse_name(4) == "aircraft"
    assert taxonomy.coarse_name(24) == "vehicle"


def test_legacy_taxonomy_keeps_existing_three_class_contract() -> None:
    taxonomy = get_taxonomy("legacy3")

    assert taxonomy.names == {0: "aircraft", 1: "ship", 2: "vehicle"}
    assert taxonomy.coarse_name(0) == "aircraft"
    assert taxonomy.coarse_name(1) == "ship"
    assert taxonomy.coarse_name(2) == "vehicle"


def test_unknown_taxonomy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown taxonomy"):
        get_taxonomy("unknown")
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_taxonomy.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'xh_detect.taxonomy'`.

- [ ] **Step 3: Implement the taxonomy module**

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Taxonomy:
    key: str
    names: Mapping[int, str]
    coarse_by_id: Mapping[int, str]

    def __post_init__(self) -> None:
        names = dict(self.names)
        coarse = dict(self.coarse_by_id)
        if not names or set(names) != set(coarse):
            raise ValueError("taxonomy names and coarse mapping must define identical IDs")
        if set(names) != set(range(len(names))):
            raise ValueError("taxonomy IDs must be contiguous from zero")
        if any(not name.strip() for name in names.values()):
            raise ValueError("taxonomy names must be non-empty")
        if any(group not in {"aircraft", "ship", "vehicle"} for group in coarse.values()):
            raise ValueError("taxonomy coarse classes must be aircraft, ship, or vehicle")
        object.__setattr__(self, "names", MappingProxyType(names))
        object.__setattr__(self, "coarse_by_id", MappingProxyType(coarse))

    @property
    def valid_ids(self) -> frozenset[int]:
        return frozenset(self.names)

    def coarse_name(self, class_id: int) -> str:
        try:
            return self.coarse_by_id[class_id]
        except KeyError as exc:
            raise ValueError(f"unknown class ID {class_id}") from exc


XH25_NAMES = {
    0: "HM",
    1: "LQS",
    2: "QHS",
    3: "MS",
    4: "A1_SU-35",
    5: "A2_C-130",
    6: "A3_C-17",
    7: "A4_C-5",
    8: "A5_F-16",
    9: "A6_TU-160",
    10: "A7_E-3",
    11: "A8_B-52",
    12: "A9_P-3C",
    13: "A10_B-1B",
    14: "A11_E-8",
    15: "A12_TU-22",
    16: "A13_F-15",
    17: "A14_KC-135",
    18: "A15_F-22",
    19: "A16_FA-18",
    20: "A17_TU-95",
    21: "A18_KC-10",
    22: "A19_SU-34",
    23: "A20_SU-24",
    24: "FSC",
}

TAXONOMIES = {
    "legacy3": Taxonomy(
        key="legacy3",
        names={0: "aircraft", 1: "ship", 2: "vehicle"},
        coarse_by_id={0: "aircraft", 1: "ship", 2: "vehicle"},
    ),
    "xh25": Taxonomy(
        key="xh25",
        names=XH25_NAMES,
        coarse_by_id={
            class_id: (
                "ship" if class_id <= 3 else "aircraft" if class_id <= 23 else "vehicle"
            )
            for class_id in XH25_NAMES
        },
    ),
}


def get_taxonomy(key: str) -> Taxonomy:
    try:
        return TAXONOMIES[key]
    except KeyError as exc:
        raise ValueError(f"unknown taxonomy {key!r}") from exc
```

- [ ] **Step 4: Run the taxonomy tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_taxonomy.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Run Ruff and commit**

```powershell
.\.venv\Scripts\python.exe -m ruff check src/xh_detect/taxonomy.py tests/test_taxonomy.py
git add src/xh_detect/taxonomy.py tests/test_taxonomy.py
git commit -m "feat: add official xh25 taxonomy"
```

### Task 2: Parse and audit official YOLO HBB data

**Files:**
- Create: `src/xh_detect/data/xh25.py`
- Create: `tests/test_xh25.py`
- Modify: `src/xh_detect/data/__init__.py`

- [ ] **Step 1: Write failing parser and audit tests**

```python
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from xh_detect.data.xh25 import audit_dataset, parse_yolo_hbb_label, source_group_id


def _write_image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def test_parse_yolo_hbb_label_returns_pixel_polygons(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("24 0.5 0.5 0.2 0.25\n", encoding="utf-8")

    annotations = parse_yolo_hbb_label(label, "sample", width=100, height=80)

    assert annotations[0].class_id == 24
    assert annotations[0].polygon == (
        (40.0, 30.0),
        (60.0, 30.0),
        (60.0, 50.0),
        (40.0, 50.0),
    )


@pytest.mark.parametrize(
    "line, message",
    [
        ("25 0.5 0.5 0.2 0.2", "class ID"),
        ("0 0.5 0.5 0.0 0.2", "width and height"),
        ("0 0.01 0.5 0.2 0.2", "outside image"),
        ("0 0.5 0.5 0.2", "five fields"),
    ],
)
def test_parse_yolo_hbb_label_rejects_invalid_lines(
    tmp_path: Path, line: str, message: str
) -> None:
    label = tmp_path / "sample.txt"
    label.write_text(f"{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        parse_yolo_hbb_label(label, "sample", width=100, height=80)


def test_source_group_removes_only_crop_suffix() -> None:
    assert source_group_id("scene_crop1") == "scene"
    assert source_group_id("scene_crop0002") == "scene"
    assert source_group_id("MAR20_1002") == "MAR20_1002"


def test_audit_dataset_counts_pairs_classes_and_image_modes(tmp_path: Path) -> None:
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    labels.mkdir(parents=True)
    _write_image(images / "a_crop1.jpg")
    _write_image(images / "a_crop2.jpg")
    (labels / "a_crop1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels / "a_crop2.txt").write_text("24 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = audit_dataset(tmp_path)

    assert report.images == 2
    assert report.labels == 2
    assert report.targets == {0: 1, 24: 1}
    assert report.source_groups == 1
    assert report.invalid_lines == 0


def test_audit_flags_same_visual_hash_across_different_groups(tmp_path: Path) -> None:
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    labels.mkdir(parents=True)
    _write_image(images / "first.jpg")
    _write_image(images / "second.jpg")
    (labels / "first.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels / "second.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = audit_dataset(tmp_path)

    assert report.near_duplicate_candidates == (("first", "second"),)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_xh25.py -q
```

Expected: collection fails because `xh_detect.data.xh25` does not exist.

- [ ] **Step 3: Implement strict parsing and audit records**

Implement these public types and functions in `src/xh_detect/data/xh25.py`:

```python
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


_CROP_SUFFIX = re.compile(r"_crop\d+$", re.IGNORECASE)


def source_group_id(stem: str) -> str:
    normalized = _CROP_SUFFIX.sub("", stem)
    return normalized or stem


def _average_hash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((8, 8))
    pixels = list(grayscale.getdata())
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if value >= mean else "0" for value in pixels)
    return f"{int(bits, 2):016x}"


def parse_yolo_hbb_label(
    path: Path,
    image_id: str,
    width: int,
    height: int,
) -> tuple[ObjectAnnotation, ...]:
    annotations: list[ObjectAnnotation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number} must contain five fields")
        try:
            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} contains a non-numeric value") from exc
        if class_id not in get_taxonomy("xh25").valid_ids:
            raise ValueError(f"{path}:{line_number} has invalid class ID {class_id}")
        values = (x_center, y_center, box_width, box_height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number} contains a non-finite coordinate")
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError(f"{path}:{line_number} width and height must be positive")
        xmin = (x_center - box_width / 2.0) * width
        ymin = (y_center - box_height / 2.0) * height
        xmax = (x_center + box_width / 2.0) * width
        ymax = (y_center + box_height / 2.0) * height
        if xmin < -1e-6 or ymin < -1e-6 or xmax > width + 1e-6 or ymax > height + 1e-6:
            raise ValueError(f"{path}:{line_number} box lies outside image")
        annotations.append(
            ObjectAnnotation(
                image_id=image_id,
                class_id=class_id,
                polygon=((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)),
            )
        )
    return tuple(annotations)
```

`audit_dataset(source_root)` must require paired `images/train/*.jpg` and
`labels/train/*.txt`, open every image with Pillow, parse every label, and return immutable
counts. It must raise one aggregated `ValueError` listing missing pairs or corrupt files instead
of silently skipping them. Group records by `perceptual_hash` and add sorted cross-source-group
pairs to `near_duplicate_candidates`; this is an audit warning and must not silently rewrite
source groups.

- [ ] **Step 4: Export the new API**

Add to `src/xh_detect/data/__init__.py`:

```python
from xh_detect.data.xh25 import (
    DatasetAudit,
    ImageRecord,
    audit_dataset,
    parse_yolo_hbb_label,
    source_group_id,
)

__all__ = [
    "DatasetAudit",
    "ImageRecord",
    "audit_dataset",
    "convert_split",
    "parse_label_file",
    "parse_yolo_hbb_label",
    "source_group_id",
    "write_dataset_yaml",
]
```

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_xh25.py -q
.\.venv\Scripts\python.exe -m ruff check src/xh_detect/data/xh25.py tests/test_xh25.py
git add src/xh_detect/data/xh25.py src/xh_detect/data/__init__.py tests/test_xh25.py
git commit -m "feat: audit official xh25 dataset"
```

Expected: all Task 2 tests pass.

### Task 3: Add deterministic group-stratified splitting and materialization

**Files:**
- Modify: `src/xh_detect/data/xh25.py`
- Modify: `tests/test_xh25.py`

- [ ] **Step 1: Write failing split tests**

Append:

```python
from xh_detect.data.xh25 import prepare_dataset


def test_prepare_dataset_is_reproducible_and_keeps_groups_together(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for class_id in range(25):
        for group_index in range(3):
            stem = f"class-{class_id}-group-{group_index}_crop1"
            image = source / "images" / "train" / f"{stem}.jpg"
            label = source / "labels" / "train" / f"{stem}.txt"
            _write_image(image)
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    paired_stem = "class-0-group-0_crop2"
    _write_image(source / "images" / "train" / f"{paired_stem}.jpg")
    (source / "labels" / "train" / f"{paired_stem}.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    first = prepare_dataset(source, tmp_path / "prepared-a", val_ratio=0.34, seed=42)
    second = prepare_dataset(source, tmp_path / "prepared-b", val_ratio=0.34, seed=42)

    assert first.train_stems == second.train_stems
    assert first.val_stems == second.val_stems
    assert {"class-0-group-0_crop1", "class-0-group-0_crop2"} <= first.train_stems or {
        "class-0-group-0_crop1",
        "class-0-group-0_crop2",
    } <= first.val_stems
    assert set(first.train_class_counts) == set(range(25))
    assert all(first.train_class_counts[class_id] > 0 for class_id in range(25))
    assert all(first.val_class_counts[class_id] > 0 for class_id in range(25))
    assert first.train_groups.isdisjoint(first.val_groups)


def test_prepare_dataset_rejects_class_with_only_one_source_group(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for suffix in ("crop1", "crop2"):
        image = source / "images" / "train" / f"only_{suffix}.jpg"
        label = source / "labels" / "train" / f"only_{suffix}.txt"
        _write_image(image)
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="class 0.*source groups"):
        prepare_dataset(source, tmp_path / "prepared", val_ratio=0.2, seed=42)
```

The first fixture includes three source groups for every official class and an extra crop in one
group. The returned count mappings must cover every official ID and contain non-zero train and val
counts.

- [ ] **Step 2: Run the split tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_xh25.py -q
```

Expected: import fails for `prepare_dataset`.

- [ ] **Step 3: Implement deterministic group selection**

Add:

```python
@dataclass(frozen=True)
class PreparedDataset:
    output_root: Path
    train_stems: frozenset[str]
    val_stems: frozenset[str]
    train_groups: frozenset[str]
    val_groups: frozenset[str]
    train_class_counts: Mapping[int, int]
    val_class_counts: Mapping[int, int]


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _select_validation_groups(
    records: tuple[ImageRecord, ...],
    val_ratio: float,
    seed: int,
) -> frozenset[str]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    groups_by_class: dict[int, set[str]] = defaultdict(set)
    for record in records:
        grouped[record.group_id].append(record)
        for class_id in {item.class_id for item in record.annotations}:
            groups_by_class[class_id].add(record.group_id)

    for class_id, groups in sorted(groups_by_class.items()):
        if len(groups) < 2:
            raise ValueError(
                f"class {class_id} appears in {len(groups)} source groups; "
                "train/val separation requires at least two"
            )

    selected: set[str] = set()
    for class_id in sorted(groups_by_class, key=lambda item: len(groups_by_class[item])):
        groups = groups_by_class[class_id]
        minimum_val_groups = 2 if len(groups) >= 3 else 1
        target = max(
            minimum_val_groups,
            min(len(groups) - 1, round(len(groups) * val_ratio)),
        )
        present = len(selected & groups)
        candidates = sorted(groups - selected, key=lambda item: _stable_rank(seed, item))
        selected.update(candidates[: max(0, target - present)])

    target_images = max(1, round(len(records) * val_ratio))
    remaining = sorted(set(grouped) - selected, key=lambda item: _stable_rank(seed, item))
    while sum(len(grouped[group]) for group in selected) < target_images and remaining:
        selected.add(remaining.pop(0))

    for class_id, groups in groups_by_class.items():
        if not selected & groups or not (groups - selected):
            raise ValueError(f"class {class_id} is missing from train or val after splitting")
    return frozenset(selected)
```

`prepare_dataset` must:

1. call `audit_dataset`;
2. choose validation groups;
3. create `images/{train,val}` and `labels/{train,val}`;
4. link or copy paired files using the existing DOTA fallback pattern;
5. write sorted `manifests/train.txt`, `manifests/val.txt`, and
   `manifests/source-groups.json`; train/val entries are relative POSIX paths such as
   `images/val/sample.jpg`, never machine-specific absolute paths;
6. write `manifests/val-image-map.json` with stable integer image IDs sorted by stem;
7. write `manifests/demo-samples.json` with one validation image containing ship, one containing
   aircraft, and one containing vehicle;
8. write `dataset.yaml` with absolute `path`, 25 names, train, and val;
9. generate `reports/dataset-analysis.json` and Markdown with all class counts and perceptual-hash
   duplicate candidates;
10. generate `reports/val-ground-truth.json` with `images`, `categories`, and `annotations`;
11. verify train/val stems and groups are disjoint before returning.

Use a temporary output sibling and `os.replace` for manifest/report files so interruption cannot
leave partial metadata.

- [ ] **Step 4: Run the data tests and inspect a real dry run**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_xh25.py -q
.\.venv\Scripts\python.exe -c "from pathlib import Path; from xh_detect.data.xh25 import audit_dataset; print(audit_dataset(Path('../../data')).images)"
```

Run the second command from `.worktrees/remote-sensing-demo`.

Expected: tests pass and real audit prints `4481`.

- [ ] **Step 5: Commit**

```powershell
git add src/xh_detect/data/xh25.py tests/test_xh25.py
git commit -m "feat: prepare reproducible xh25 splits"
```

### Task 4: Add `prepare-xh25` CLI and official configuration

**Files:**
- Modify: `src/xh_detect/cli.py`
- Modify: `tests/test_cli.py`
- Create: `configs/xh25-hbb.yaml`

- [ ] **Step 1: Write failing CLI tests**

Add a Typer runner test that patches `prepare_dataset`:

```python
@patch("xh_detect.cli.prepare_dataset")
def test_prepare_xh25_command_reports_output(
    prepare_dataset_mock: Mock, tmp_path: Path
) -> None:
    source = tmp_path / "data"
    source.mkdir()
    output = tmp_path / "xh25"
    prepare_dataset_mock.return_value = SimpleNamespace(
        output_root=output,
        train_stems=frozenset({"a", "b"}),
        val_stems=frozenset({"c"}),
        train_class_counts={class_id: 1 for class_id in range(25)},
        val_class_counts={class_id: 1 for class_id in range(25)},
    )

    result = runner.invoke(
        app,
        [
            "prepare-xh25",
            "--source-root",
            str(source),
            "--output-root",
            str(output),
            "--val-ratio",
            "0.15",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["train_images"] == 2
    assert json.loads(result.stdout)["val_images"] == 1
    prepare_dataset_mock.assert_called_once_with(source, output, val_ratio=0.15, seed=42)
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli.py::test_prepare_xh25_command_reports_output -q
```

Expected: command is not registered.

- [ ] **Step 3: Implement the command**

Add:

```python
@app.command("prepare-xh25")
def prepare_xh25(
    source_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Option()] = Path("datasets/xh25"),
    val_ratio: Annotated[float, typer.Option(min=0.05, max=0.4)] = 0.15,
    seed: Annotated[int, typer.Option(min=0)] = 42,
) -> None:
    prepared = prepare_dataset(source_root, output_root, val_ratio=val_ratio, seed=seed)
    typer.echo(
        json.dumps(
            {
                "output_root": str(prepared.output_root),
                "train_images": len(prepared.train_stems),
                "val_images": len(prepared.val_stems),
                "train_targets": dict(prepared.train_class_counts),
                "val_targets": dict(prepared.val_class_counts),
            },
            ensure_ascii=False,
        )
    )
```

Create `configs/xh25-hbb.yaml`:

```yaml
task: detect
taxonomy: xh25
model_path: runs/train/xh25-baseline/weights/best.pt
device: "0"
image_size: 1024
tile_size: 1024
overlap: 0.2
batch_size: 8
merge_iou: 0.3
edge_margin: 16
half: true
class_thresholds:
  0: 0.25
  1: 0.25
  2: 0.25
  3: 0.25
  4: 0.25
  5: 0.25
  6: 0.25
  7: 0.25
  8: 0.25
  9: 0.25
  10: 0.25
  11: 0.25
  12: 0.25
  13: 0.25
  14: 0.25
  15: 0.25
  16: 0.25
  17: 0.25
  18: 0.25
  19: 0.25
  20: 0.25
  21: 0.25
  22: 0.25
  23: 0.25
  24: 0.25
```

- [ ] **Step 4: Run CLI tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli.py -q
git add src/xh_detect/cli.py tests/test_cli.py configs/xh25-hbb.yaml
git commit -m "feat: add xh25 preparation command"
```

### Task 5: Make pipeline configuration task- and taxonomy-aware

**Files:**
- Modify: `src/xh_detect/config.py`
- Modify: `tests/test_config.py`
- Modify: `configs/baseline.yaml`
- Modify: `configs/tensorrt.yaml`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_xh25_detect_config_accepts_all_25_thresholds() -> None:
    config = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        model_path="best.pt",
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.valid_class_ids == frozenset(range(25))


def test_config_rejects_threshold_ids_not_matching_taxonomy() -> None:
    with pytest.raises(ValueError, match="class_thresholds"):
        PipelineConfig(
            task="detect",
            taxonomy="xh25",
            class_thresholds={0: 0.25, 1: 0.25, 2: 0.25},
        )


def test_config_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="task"):
        PipelineConfig(task="segment")  # type: ignore[arg-type]
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py -q
```

Expected: constructor rejects unknown `task` and `taxonomy` arguments.

- [ ] **Step 3: Implement configuration fields**

Add:

```python
task: str = "obb"
taxonomy: str = "legacy3"
```

In `__post_init__`:

```python
if self.task not in {"detect", "obb"}:
    raise ValueError("task must be 'detect' or 'obb'")
taxonomy = get_taxonomy(self.taxonomy)
if set(class_thresholds) != taxonomy.valid_ids:
    raise ValueError(
        f"class_thresholds must define exactly taxonomy IDs {sorted(taxonomy.valid_ids)}"
    )
```

Add:

```python
@property
def valid_class_ids(self) -> frozenset[int]:
    return get_taxonomy(self.taxonomy).valid_ids
```

Include `task` and `taxonomy` in YAML parsing and `to_dict`. Update legacy configs:

```yaml
task: obb
taxonomy: legacy3
```

- [ ] **Step 4: Run configuration and regression tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_build_config.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/xh_detect/config.py tests/test_config.py configs/baseline.yaml configs/tensorrt.yaml
git commit -m "feat: configure detection task and taxonomy"
```

### Task 6: Add Ultralytics HBB extraction

**Files:**
- Modify: `src/xh_detect/detector.py`
- Modify: `tests/test_detector.py`

- [ ] **Step 1: Write failing HBB extraction tests**

Use the existing fake tensor helpers and add:

```python
def test_extract_hbb_predictions_converts_xyxy_to_polygon() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=FakeTensor([[10.0, 20.0, 30.0, 40.0]]),
            cls=FakeTensor([24.0]),
            conf=FakeTensor([0.9]),
        )
    )

    predictions = _extract_predictions(result, result_index=0, task="detect")

    assert predictions == [
        BoxPrediction(
            class_id=24,
            score=0.9,
            polygon=((10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)),
        )
    ]


def test_extract_predictions_requires_requested_result_type() -> None:
    with pytest.raises(ValueError, match="missing HBB boxes"):
        _extract_predictions(SimpleNamespace(boxes=None), result_index=0, task="detect")
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_detector.py -q
```

Expected: `_extract_predictions` does not accept `task`.

- [ ] **Step 3: Implement generic extraction**

Refactor the current OBB extraction into `_extract_obb_predictions`. Add:

```python
def _extract_hbb_predictions(result: object, *, result_index: int) -> list[BoxPrediction]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise ValueError(f"result {result_index} is missing HBB boxes")
    coordinates = _to_numpy_array(
        boxes.xyxy.detach().cpu().numpy(),
        result_index=result_index,
        field_name="box",
    )
    classes = _to_numpy_array(
        boxes.cls.detach().cpu().numpy(),
        result_index=result_index,
        field_name="class",
    ).reshape(-1)
    scores = _to_numpy_array(
        boxes.conf.detach().cpu().numpy(),
        result_index=result_index,
        field_name="score",
    ).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError(
            f"result {result_index} has invalid HBB shape: expected (N, 4), "
            f"got {coordinates.shape}"
        )
    if len(coordinates) != len(classes) or len(coordinates) != len(scores):
        raise ValueError(f"result {result_index} has inconsistent HBB lengths")
    _ensure_finite(coordinates, result_index=result_index)
    validated_classes = _validate_class_ids(classes, result_index=result_index)
    validated_scores = _validate_scores(scores, result_index=result_index)
    return [
        BoxPrediction(
            class_id=class_id,
            score=score,
            polygon=((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)),
        )
        for (xmin, ymin, xmax, ymax), class_id, score in zip(
            coordinates.tolist(), validated_classes, validated_scores, strict=True
        )
    ]


def _extract_predictions(
    result: object,
    *,
    result_index: int,
    task: str = "obb",
) -> list[BoxPrediction]:
    if task == "detect":
        return _extract_hbb_predictions(result, result_index=result_index)
    if task == "obb":
        return _extract_obb_predictions(result, result_index=result_index)
    raise ValueError("task must be 'detect' or 'obb'")
```

Replace `UltralyticsOBBDetector` with `UltralyticsDetector`. Its constructor must use:

```python
def __init__(
    self,
    model_path: str,
    device: str,
    image_size: int,
    half: bool,
    task: str,
) -> None:
```

Validate `task` against `{"detect", "obb"}`, store it, and pass it to
`_extract_predictions`. Keep a compatibility subclass:

```python
class UltralyticsOBBDetector(UltralyticsDetector):
    def __init__(self, model_path: str, device: str, image_size: int, half: bool) -> None:
        super().__init__(model_path, device, image_size, half, task="obb")
```

- [ ] **Step 4: Run detector tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_detector.py tests/test_pipeline.py -q
git add src/xh_detect/detector.py tests/test_detector.py
git commit -m "feat: support ultralytics hbb inference"
```

### Task 7: Generalize filtering, visualization, and COCO export to 25 classes

**Files:**
- Modify: `src/xh_detect/exporters.py`
- Modify: `src/xh_detect/visualize.py`
- Modify: `src/xh_detect/pipeline.py`
- Modify: `tests/test_exporters.py`
- Modify: `tests/test_visualize.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing 25-class tests**

```python
def test_coco_validation_accepts_xh25_boundary_ids() -> None:
    records = [
        {"image_id": 1, "category_id": 0, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9},
        {"image_id": 1, "category_id": 24, "bbox": [5.0, 6.0, 7.0, 8.0], "score": 0.8},
    ]

    validate_coco_results(records, valid_class_ids=frozenset(range(25)))


def test_xh25_counts_include_coarse_and_fine_views() -> None:
    detections = [_detection(0), _detection(4), _detection(24)]

    counts = class_counts(detections, taxonomy=get_taxonomy("xh25"))

    assert counts["coarse"] == {"aircraft": 1, "ship": 1, "vehicle": 1}
    assert counts["fine"]["HM"] == 1
    assert counts["fine"]["A1_SU-35"] == 1
    assert counts["fine"]["FSC"] == 1


def test_pipeline_accepts_class_24_when_configured_for_xh25() -> None:
    config = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        device="cpu",
        half=False,
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )
    pipeline = InferencePipeline(
        RecordingDetector([[_prediction(class_id=24, score=0.9)]]),
        config,
        cache_root=None,
    )

    result = pipeline.run(np.zeros((64, 64, 3), dtype=np.uint8), "vehicle")

    assert result.detections[0].class_id == 24
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_exporters.py tests/test_visualize.py tests/test_pipeline.py -q
```

Expected: validation and visualization reject class 24.

- [ ] **Step 3: Implement taxonomy-aware behavior**

Change `validate_coco_results` to accept:

```python
def validate_coco_results(
    records: list[dict[str, object]],
    valid_class_ids: frozenset[int] = frozenset({0, 1, 2}),
) -> None:
```

Replace the hardcoded category check with:

```python
if category_id_int not in valid_class_ids:
    raise ValueError(f"category_id must be one of {sorted(valid_class_ids)}")
```

Change `export_coco_results` to accept:

```python
def export_coco_results(
    detections: Iterable[Detection],
    image_id_map: Mapping[str, int],
    destination: Path,
    valid_class_ids: frozenset[int] = frozenset({0, 1, 2}),
) -> Path:
```

Pass `valid_class_ids` to `validate_coco_results`. Replace `class_counts` with:

```python
def class_counts(
    detections: Iterable[Detection],
    taxonomy: Taxonomy = get_taxonomy("legacy3"),
) -> dict[str, dict[str, int]]:
    fine_counts: Counter[int] = Counter()
    coarse_counts: Counter[str] = Counter()
    for detection in detections:
        class_id = _validate_class_id(detection.class_id, taxonomy)
        fine_counts[class_id] += 1
        coarse_counts[taxonomy.coarse_name(class_id)] += 1
    return {
        "coarse": {
            name: coarse_counts[name] for name in ("aircraft", "ship", "vehicle")
        },
        "fine": {
            taxonomy.names[class_id]: fine_counts[class_id]
            for class_id in sorted(taxonomy.valid_ids)
        },
    }
```

Change the drawing signature to:

```python
def draw_detections(
    image: ImageArray,
    detections: Iterable[Detection],
    mode: str = "obb",
    taxonomy: Taxonomy = get_taxonomy("legacy3"),
) -> ImageArray:
```

Inside `draw_detections`, validate against `taxonomy.valid_ids`, call `_color(class_id)`, and use
`taxonomy.names[class_id]` in the label.

Generate stable colors:

```python
def _color(class_id: int) -> tuple[int, int, int]:
    return (
        64 + (class_id * 53) % 192,
        64 + (class_id * 97) % 192,
        64 + (class_id * 193) % 192,
    )
```

In `InferencePipeline`, reject detector outputs not present in `config.valid_class_ids` before
threshold filtering. Continue applying per-fine-class thresholds.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_exporters.py tests/test_visualize.py tests/test_pipeline.py -q
git add src/xh_detect/exporters.py src/xh_detect/visualize.py src/xh_detect/pipeline.py tests/test_exporters.py tests/test_visualize.py tests/test_pipeline.py
git commit -m "feat: support 25-class outputs"
```

### Task 8: Implement official overall, coarse, and fine evaluation

**Files:**
- Modify: `src/xh_detect/evaluator.py`
- Modify: `tests/test_evaluator.py`

- [ ] **Step 1: Write failing competition report tests**

```python
def test_xh25_overall_is_fine_class_agnostic_but_coarse_metrics_are_not() -> None:
    truth = [ObjectAnnotation("img", 0, GT)]
    predictions = [Detection("img", 4, 0.9, GT)]

    report = evaluate(predictions, truth, taxonomy=get_taxonomy("xh25"))

    assert report.overall_class_agnostic == Metrics(tp=1, fp=0, fn=0)
    assert report.by_coarse_class["ship"] == Metrics(tp=0, fp=0, fn=1)
    assert report.by_coarse_class["aircraft"] == Metrics(tp=0, fp=1, fn=0)
    assert report.by_fine_class[0] == Metrics(tp=0, fp=0, fn=1)
    assert report.by_fine_class[4] == Metrics(tp=0, fp=1, fn=0)


def test_xh25_same_coarse_different_fine_matches_coarse_only() -> None:
    truth = [ObjectAnnotation("img", 4, GT)]
    predictions = [Detection("img", 8, 0.9, GT)]

    report = evaluate(predictions, truth, taxonomy=get_taxonomy("xh25"))

    assert report.overall_class_agnostic.tp == 1
    assert report.by_coarse_class["aircraft"].tp == 1
    assert report.by_fine_class[4].fn == 1
    assert report.by_fine_class[8].fp == 1


def test_vehicle_truth_uses_point_35_threshold() -> None:
    truth = [ObjectAnnotation("img", 24, GT)]
    prediction = Detection("img", 24, 0.9, box_with_iou(0.35))

    report = evaluate([prediction], truth, taxonomy=get_taxonomy("xh25"))

    assert report.overall_class_agnostic.tp == 1
    assert report.by_coarse_class["vehicle"].tp == 1
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluator.py -q
```

Expected: `evaluate` has no taxonomy argument and report fields do not exist.

- [ ] **Step 3: Refactor matching around a reusable key function**

Introduce:

```python
@dataclass(frozen=True)
class CompetitionEvaluationReport:
    overall_class_agnostic: Metrics
    by_coarse_class: dict[str, Metrics]
    by_fine_class: dict[int, Metrics]
    by_image: dict[str, Metrics]


def _iou_threshold(truth_class_id: int, taxonomy: Taxonomy) -> float:
    return 0.35 if taxonomy.coarse_name(truth_class_id) == "vehicle" else 0.50


def _match(
    predictions: list[Detection],
    truth: list[ObjectAnnotation],
    taxonomy: Taxonomy,
    key: Callable[[int], Hashable],
) -> tuple[Metrics, dict[Hashable, Metrics], dict[str, Metrics]]:
    truth_by_key: dict[tuple[str, Hashable], list[ObjectAnnotation]] = defaultdict(list)
    for item in truth:
        if not item.difficult:
            truth_by_key[(item.image_id, key(item.class_id))].append(item)

    matched: dict[tuple[str, Hashable], set[int]] = defaultdict(set)
    keyed_counts: dict[Hashable, list[int]] = defaultdict(lambda: [0, 0, 0])
    image_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    indexed = list(enumerate(predictions))
    indexed.sort(key=lambda pair: (-pair[1].score, pair[0]))
    for _, prediction in indexed:
        metric_key = key(prediction.class_id)
        match_key = (prediction.image_id, metric_key)
        candidates = truth_by_key.get(match_key, [])
        prediction_hbb = obb_to_hbb(prediction.polygon)
        best_index = -1
        best_iou = -1.0
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in matched[match_key]:
                continue
            iou = hbb_iou(prediction_hbb, obb_to_hbb(candidate.polygon))
            if iou >= _iou_threshold(candidate.class_id, taxonomy) and iou > best_iou:
                best_iou = iou
                best_index = candidate_index
        if best_index >= 0:
            matched[match_key].add(best_index)
            keyed_counts[metric_key][0] += 1
            image_counts[prediction.image_id][0] += 1
        else:
            keyed_counts[metric_key][1] += 1
            image_counts[prediction.image_id][1] += 1

    for (image_id, metric_key), items in truth_by_key.items():
        missed = len(items) - len(matched[(image_id, metric_key)])
        keyed_counts[metric_key][2] += missed
        image_counts[image_id][2] += missed

    by_key = {
        metric_key: Metrics(tp=values[0], fp=values[1], fn=values[2])
        for metric_key, values in keyed_counts.items()
    }
    overall = Metrics(
        tp=sum(item.tp for item in by_key.values()),
        fp=sum(item.fp for item in by_key.values()),
        fn=sum(item.fn for item in by_key.values()),
    )
    by_image = {
        image_id: Metrics(tp=values[0], fp=values[1], fn=values[2])
        for image_id, values in sorted(image_counts.items())
    }
    return overall, by_key, by_image
```

`_match` must sort predictions by descending score, group candidates by `(image_id,
key(class_id))`, use the ground-truth class to choose IoU threshold, and enforce one prediction to
one truth matching.

Call `_match` three times:

```python
overall, _, by_image = _match(predictions, truth, taxonomy, key=lambda _: "all")
_, coarse, _ = _match(
    predictions,
    truth,
    taxonomy,
    key=lambda class_id: taxonomy.coarse_name(class_id),
)
_, fine, _ = _match(predictions, truth, taxonomy, key=lambda class_id: class_id)
```

For backward compatibility, default `taxonomy=get_taxonomy("legacy3")`. Use this exact
`report_to_dict` payload:

```python
return {
    "overall_class_agnostic": metrics_dict(report.overall_class_agnostic),
    "by_coarse_class": {
        name: metrics_dict(metrics)
        for name, metrics in sorted(report.by_coarse_class.items())
    },
    "by_fine_class": {
        str(class_id): metrics_dict(metrics)
        for class_id, metrics in sorted(report.by_fine_class.items())
    },
    "by_image": {
        image_id: metrics_dict(metrics)
        for image_id, metrics in sorted(report.by_image.items())
    },
}
```

Update loaders to accept a taxonomy and validate against `taxonomy.valid_ids`.

- [ ] **Step 4: Run evaluator tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluator.py -q
git add src/xh_detect/evaluator.py tests/test_evaluator.py
git commit -m "feat: evaluate xh25 competition metrics"
```

### Task 9: Wire generic detector, taxonomy, export, and evaluation through CLI

**Files:**
- Modify: `src/xh_detect/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing wiring tests**

```python
@patch("xh_detect.cli.UltralyticsDetector")
@patch("xh_detect.cli.PipelineConfig.from_yaml")
def test_infer_builds_detect_model_for_xh25(
    from_yaml: Mock,
    detector_class: Mock,
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.jpg"
    cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8))
    from_yaml.return_value = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        model_path="best.pt",
        device="cpu",
        half=False,
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )
    detector_class.return_value.predict.return_value = [[]]

    result = runner.invoke(app, ["infer", "--image-path", str(image)])

    assert result.exit_code == 0
    detector_class.assert_called_once_with(
        "best.pt",
        "cpu",
        1024,
        False,
        task="detect",
    )


@patch("xh_detect.cli.evaluate_detections")
def test_evaluate_command_uses_xh25_taxonomy(
    evaluate_mock: Mock, tmp_path: Path
) -> None:
    predictions = tmp_path / "predictions.json"
    ground_truth = tmp_path / "ground-truth.json"
    predictions.write_text("[]", encoding="utf-8")
    ground_truth.write_text('{"annotations":[]}', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--predictions-json",
            str(predictions),
            "--ground-truth-json",
            str(ground_truth),
            "--taxonomy",
            "xh25",
        ],
    )

    assert result.exit_code == 0


@patch("xh_detect.cli.InferencePipeline")
@patch("xh_detect.cli._build_detector")
@patch("xh_detect.cli.PipelineConfig.from_yaml")
def test_infer_dataset_exports_stable_image_ids(
    from_yaml: Mock,
    build_detector_mock: Mock,
    pipeline_class: Mock,
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "sample.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
    image_map = tmp_path / "image-map.json"
    image_map.write_text('{"sample": 7}', encoding="utf-8")
    output = tmp_path / "predictions.json"
    from_yaml.return_value = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        device="cpu",
        half=False,
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )
    pipeline_class.return_value.run.return_value = InferenceResult(
        detections=(Detection("sample", 24, 0.9, POLYGON),),
        timings=StageTimings(0.1, 0.2, 0.3),
    )

    result = runner.invoke(
        app,
        [
            "infer-dataset",
            "--images-dir",
            str(images),
            "--image-map-json",
            str(image_map),
            "--output-json",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))[0]["image_id"] == 7
    assert json.loads(output.read_text(encoding="utf-8"))[0]["category_id"] == 24
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli.py -q
```

Expected: generic detector and taxonomy option are unavailable.

- [ ] **Step 3: Implement wiring**

Create:

```python
def _build_detector(config: PipelineConfig) -> UltralyticsDetector:
    return UltralyticsDetector(
        config.model_path,
        config.device,
        config.image_size,
        config.half,
        task=config.task,
    )
```

Use it in `infer`, `benchmark`, and the Gradio factory. Pass
`get_taxonomy(config.taxonomy)` to drawing, export, and summaries. Add `--taxonomy` with
`legacy3` default to `evaluate` and `sweep-thresholds`, and pass it to loaders and evaluator.

Add an `infer-dataset` command with these exact options:

```python
@app.command("infer-dataset")
def infer_dataset(
    images_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    image_map_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    config_path: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = Path("configs/xh25-hbb.yaml"),
    output_json: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/val-predictions.json"
    ),
) -> None:
```

Load `image_map_json` as a `dict[str, int]`, reject duplicate or negative IDs, and iterate stems
in sorted order. For each stem, read `images_dir / f"{stem}.jpg"`, run the shared pipeline, and
append detections. Export once with `taxonomy.valid_ids`. Fail if an image-map stem has no image;
ignore image files not present in the map.

- [ ] **Step 4: Run CLI tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli.py tests/test_e2e.py -q
git add src/xh_detect/cli.py tests/test_cli.py
git commit -m "feat: wire xh25 through cli"
```

### Task 10: Update Gradio for official HBB demo

**Files:**
- Modify: `src/xh_detect/app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write failing app tests**

```python
def test_format_summary_contains_coarse_and_fine_counts() -> None:
    summary = format_summary(
        [
            Detection("image", 0, 0.9, POLYGON),
            Detection("image", 4, 0.8, POLYGON),
            Detection("image", 24, 0.7, POLYGON),
        ],
        StageTimings(0.1, 0.2, 0.3),
        taxonomy=get_taxonomy("xh25"),
    )

    assert summary["coarse_counts"] == {"aircraft": 1, "ship": 1, "vehicle": 1}
    assert summary["fine_counts"]["HM"] == 1
    assert summary["fine_counts"]["A1_SU-35"] == 1
    assert summary["fine_counts"]["FSC"] == 1


@patch("xh_detect.app.PipelineConfig.from_yaml")
@patch("xh_detect.app.UltralyticsDetector")
def test_xh25_app_uses_hbb_mode(
    detector_class: Mock,
    from_yaml: Mock,
    tmp_path: Path,
) -> None:
    from_yaml.return_value = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        device="cpu",
        half=False,
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )

    demo = build_app(tmp_path / "config.yaml")

    assert demo is not None
    detector_class.assert_called_once()
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_app.py -q
```

Expected: summary and detector construction are still legacy-only.

- [ ] **Step 3: Implement official presentation**

Load the taxonomy once in `build_app`. For `task == "detect"`:

- render HBB without showing the OBB/HBB radio;
- label the page “XH-202625 正式数据 25 类 HBB Demo”;
- return nested `coarse_counts` and non-zero `fine_counts`;
- export with `taxonomy.valid_ids`;
- evaluate uploaded truth with the configured taxonomy.

For `task == "obb"`, retain the current radio and legacy behavior. Build Gradio examples from
`datasets/xh25/manifests/demo-samples.json` only when the manifest exists. The generated manifest
contains exactly one ship, one aircraft, and one vehicle validation image. Do not commit official
images.

- [ ] **Step 4: Run app tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_app.py -q
git add src/xh_detect/app.py tests/test_app.py
git commit -m "feat: present official xh25 demo"
```

### Task 11: Expand reproducible training options

**Files:**
- Modify: `src/xh_detect/training.py`
- Modify: `src/xh_detect/cli.py`
- Modify: `tests/test_training.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing training tests**

```python
def test_train_model_passes_official_baseline_options(yolo_class: Mock) -> None:
    model = yolo_class.return_value

    train_model(
        "datasets/xh25/dataset.yaml",
        "yolo26s.pt",
        epochs=1,
        image_size=1024,
        device="0",
        batch=8,
        workers=4,
        amp=False,
        project="runs/train",
        name="xh25-baseline",
        resume=False,
    )

    model.train.assert_called_once_with(
        data="datasets/xh25/dataset.yaml",
        epochs=1,
        imgsz=1024,
        device="0",
        batch=8,
        workers=4,
        amp=False,
        seed=42,
        deterministic=True,
        project="runs/train",
        name="xh25-baseline",
        exist_ok=True,
        resume=False,
    )
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_training.py -q
```

Expected: `train_model` rejects the new keyword arguments.

- [ ] **Step 3: Implement explicit options**

Extend `train_model` with validated `batch`, `workers`, `amp`, `project`, `name`, and `resume`.
Add matching Typer options to `xh-detect train`; change the CLI default model to `yolo26s.pt`,
default name to `xh25-baseline`, and keep callers able to select the old OBB model explicitly.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_training.py tests/test_cli.py -q
git add src/xh_detect/training.py src/xh_detect/cli.py tests/test_training.py tests/test_cli.py
git commit -m "feat: configure reproducible xh25 training"
```

### Task 12: Generate the real official split and verify local integration

**Files:**
- Generated only: `datasets/xh25/**`
- Generated only: `outputs/xh25/**`

- [ ] **Step 1: Run the official preparation command**

From `.worktrees/remote-sensing-demo`:

```powershell
.\.venv\Scripts\xh-detect.exe prepare-xh25 `
  --source-root ..\..\data `
  --output-root datasets\xh25 `
  --val-ratio 0.15 `
  --seed 42
```

Expected:

- 4481 total images;
- approximately 672 validation images;
- 25 classes present in both splits;
- no train/val source-group overlap;
- reports and ground-truth JSON written.

- [ ] **Step 2: Independently verify manifests and source data immutability**

Run:

```powershell
Get-FileHash ..\..\data\dataset.yaml
.\.venv\Scripts\python.exe -c "from pathlib import Path; from xh_detect.data.xh25 import audit_dataset; print(audit_dataset(Path('../../data')).images)"
.\.venv\Scripts\python.exe -m pytest tests/test_xh25.py -q
```

Expected: hash is recorded, audit prints `4481`, tests pass.

- [ ] **Step 3: Run a CPU loader smoke without training**

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; model=YOLO('yolo26s.pt'); model.val(data='datasets/xh25/dataset.yaml', imgsz=640, batch=1, device='cpu', plots=False)"
```

Stop after the first loader batch if a full CPU validation is too slow. The required evidence is
that Ultralytics recognizes 25 names and both train/val paths without label-cache errors.

- [ ] **Step 4: Run the full local verification suite**

```powershell
.\.venv\Scripts\python.exe -m ruff format .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit implementation documentation only**

Generated data remains ignored. Commit any final README command corrections together with Task
13 rather than adding `datasets/xh25`.

### Task 13: Document the official workflow

**Files:**
- Modify: `README.md`
- Create: `docs/xh25-data-analysis.md`

- [ ] **Step 1: Write exact operational documentation**

`docs/xh25-data-analysis.md` must record:

- 4481 images and 20933 boxes;
- 25 class counts from the generated JSON report;
- 2682 ship, 17849 aircraft, and 402 vehicle boxes;
- zero invalid labels and zero missing pairs;
- validation ratio, seed, actual split counts, and source-group overlap result;
- HM and LQS scarcity;
- the distinction between official overall, coarse, and fine diagnostic metrics.

README must provide executable commands for:

```bash
.venv/bin/xh-detect prepare-xh25 \
  --source-root data \
  --output-root datasets/xh25 \
  --val-ratio 0.15 \
  --seed 42

.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25/dataset.yaml \
  --model yolo26s.pt \
  --epochs 1 \
  --image-size 1024 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --name xh25-baseline \
  --device 0

.venv/bin/xh-detect serve \
  --config-path configs/xh25-hbb.yaml \
  --host 127.0.0.1 \
  --port 7860

.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-hbb.yaml \
  --output-json outputs/xh25/val-predictions.json
```

- [ ] **Step 2: Verify documentation commands match CLI help**

```powershell
.\.venv\Scripts\xh-detect.exe prepare-xh25 --help
.\.venv\Scripts\xh-detect.exe train --help
.\.venv\Scripts\xh-detect.exe serve --help
.\.venv\Scripts\xh-detect.exe infer-dataset --help
git diff --check
```

- [ ] **Step 3: Commit**

```powershell
git add README.md docs/xh25-data-analysis.md
git commit -m "docs: add official xh25 workflow"
```

### Task 14: Run 4090 server acceptance

**Files:**
- Generated only on server: `datasets/xh25/**`, `runs/train/xh25-baseline/**`,
  `outputs/xh25/**`
- Generated locally: `outputs/xh25-server-acceptance.md`

- [ ] **Step 1: Sync code and official data**

Create and upload a Git bundle for code. Transfer `data/` with `scp -r` or an archive while
preserving file names. On the server verify:

```bash
find data/images/train -type f | wc -l
find data/labels/train -type f | wc -l
du -sh data
```

Expected: 4481 images, 4481 labels, approximately 1.2 GB.

- [ ] **Step 2: Prepare the split on the server**

```bash
.venv/bin/xh-detect prepare-xh25 \
  --source-root data \
  --output-root datasets/xh25 \
  --val-ratio 0.15 \
  --seed 42
```

Compare SHA-256 hashes of `manifests/train.txt`, `manifests/val.txt`, and
`source-groups.json` with local hashes. They must match.

- [ ] **Step 3: Run one-epoch HBB training**

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25/dataset.yaml \
  --model yolo26s.pt \
  --epochs 1 \
  --image-size 1024 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --name xh25-baseline \
  --device 0
```

If Ultralytics AMP validation attempts an external model download, keep `--no-amp` for the smoke
run. Required evidence:

- training and validation complete;
- `best.pt` and `last.pt` exist;
- GPU memory usage is recorded;
- no class-count or label-format errors.

- [ ] **Step 4: Point the official config to the smoke weight**

Use a generated copy rather than modifying the tracked config:

```bash
python - <<'PY'
from pathlib import Path
import yaml

source = Path("configs/xh25-hbb.yaml")
target = Path("outputs/xh25/config-smoke.yaml")
payload = yaml.safe_load(source.read_text())
payload["model_path"] = "runs/train/xh25-baseline/weights/best.pt"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(payload, sort_keys=False))
PY
```

- [ ] **Step 5: Run official-image inference and JSON validation**

Select one validation image from each coarse class using the generated report. For the first
manifest image, run:

```bash
IMAGE_REL=$(head -n 1 datasets/xh25/manifests/val.txt)
IMAGE_PATH="datasets/xh25/$IMAGE_REL"
.venv/bin/xh-detect infer \
  --image-path "$IMAGE_PATH" \
  --config-path outputs/xh25/config-smoke.yaml \
  --output-dir outputs/xh25/infer
```

Validate every generated JSON with `valid_class_ids=frozenset(range(25))`. Confirm visual output
uses HBB and official names.

- [ ] **Step 6: Run competition evaluation**

Generate predictions for the complete validation split:

```bash
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path outputs/xh25/config-smoke.yaml \
  --output-json outputs/xh25/val-predictions.json

.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-path outputs/xh25/val-evaluation.json
```

The report must contain:

```text
overall_class_agnostic
by_coarse_class.aircraft
by_coarse_class.ship
by_coarse_class.vehicle
by_fine_class.0 through by_fine_class.24
```

One epoch is only an integration smoke; do not present its metrics as the competition baseline.

- [ ] **Step 7: Run Gradio and 10k benchmark**

```bash
.venv/bin/xh-detect serve \
  --config-path outputs/xh25/config-smoke.yaml \
  --host 127.0.0.1 \
  --port 7860

.venv/bin/xh-detect benchmark \
  --config-path outputs/xh25/config-smoke.yaml \
  --repeats 5
```

Verify the UI returns official fine labels, coarse counts, an image, and a downloadable JSON.
Record median and P95 10,000×10,000 timing.

- [ ] **Step 8: Run final server verification**

```bash
env -u GRADIO_ANALYTICS_ENABLED -u HF_HUB_DISABLE_TELEMETRY \
  timeout 240 .venv/bin/python -m pytest
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
git diff --check
```

Expected: all tests and checks pass.

- [ ] **Step 9: Write acceptance report**

Create `outputs/xh25-server-acceptance.md` containing:

- commit SHA;
- GPU, CUDA, PyTorch, Ultralytics versions;
- dataset and split hashes;
- one-epoch completion and artifact paths;
- three official-image inference results;
- Gradio result;
- 10k median and P95;
- explicit remaining work: full training, threshold tuning, long-tail optimization, and 3090
  confirmation.

Do not include SSH credentials or private server addresses.

### Task 15: Final regression review

**Files:**
- All modified source, tests, configs, and docs

- [ ] **Step 1: Run complete local verification from the final commit**

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
git diff --check
git status --short
```

- [ ] **Step 2: Confirm requirement coverage**

Verify:

- original `data/` hash unchanged;
- deterministic split manifests;
- no source-group overlap;
- all 25 classes in both splits;
- HBB training and inference;
- 25-class COCO output;
- class-agnostic overall and coarse/fine metrics;
- official Demo;
- server smoke;
- documentation.

- [ ] **Step 3: Preserve branch for user acceptance**

Do not merge to `main` and do not push a PR until the user accepts:

- data report;
- sample detections;
- one-epoch logs;
- UI;
- benchmark;
- server acceptance report.
