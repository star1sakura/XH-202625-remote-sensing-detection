# Task 2 Report: Add P2 SPH Model And Runtime Config

## Scope

Implemented only the Task 2 files from the brief:

- `configs/models/xh25-yolo26s-sph-p2.yaml`
- `configs/xh25-sph-p2.yaml`
- `tests/test_sph_configs.py`

No NAM or full-variant work was added.

## TDD Execution

1. Created `tests/test_sph_configs.py` exactly as specified in the brief.
2. Ran `python -m pytest tests/test_sph_configs.py -q`.
3. Observed the expected failing state:
   - `configs/models/xh25-yolo26s-sph-p2.yaml` missing
   - `configs/xh25-sph-p2.yaml` missing
4. Added the requested model YAML and runtime config YAML exactly as specified.
5. Re-ran `python -m pytest tests/test_sph_configs.py -q`.
6. Confirmed the focused test target passed.

## Files Added

### `configs/models/xh25-yolo26s-sph-p2.yaml`

- Added the XH25 YOLO26s-style SPH P2 model definition.
- Includes four-scale detect head:
  - `[[19, 22, 25, 28], 1, Detect, [nc]]`
- Uses only the built-in Ultralytics layers called for in the brief.

### `configs/xh25-sph-p2.yaml`

- Added the runtime pipeline config for the SPH P2 model.
- Set:
  - `model_path: runs/train/xh25-sph-p2/weights/best.pt`
  - `image_size: 1024`
  - `batch_size: 8`
  - class thresholds for all 25 classes at `0.25`

### `tests/test_sph_configs.py`

- Added coverage for:
  - P2 model YAML structure and four-scale detect head
  - P2 runtime config loading through `PipelineConfig`
  - smoke-loading the model through `YOLO(...)` after custom module registration

## Verification

### Red

Command:

```bash
python -m pytest tests/test_sph_configs.py -q
```

Result:

- Failed as expected because the new config files did not yet exist.

### Green

Command:

```bash
python -m pytest tests/test_sph_configs.py -q
```

Result:

```text
...                                                                      [100%]
```

## Commit

- `09cf0fa` — `config: add sph p2 model`

## Concerns

- None.

## Review Fix: Lock P2 Runtime Knobs

- Updated only `tests/test_sph_configs.py` in response to the review finding.
- Extended `test_sph_p2_pipeline_config_loads` to assert the loaded runtime config values for:
  - `device == "0"`
  - `tile_size == 1024`
  - `overlap == 0.2`
  - `merge_iou == 0.3`
  - `edge_margin == 16`
  - `half is True`
  - `class_thresholds` keys exactly `0..24`
  - every `class_thresholds` value exactly `0.25`

### Review Fix Verification

Command:

```bash
python -m pytest tests/test_sph_configs.py -q
```

Result:

```text
...                                                                      [100%]
```
