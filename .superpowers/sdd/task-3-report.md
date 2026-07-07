# Task 3 Report: Full MKSNet Experiment Configs

## Summary

Implemented the requested config surface for the full MKSNet v2 experiment:

- Added `configs/models/xh25-yolo-mksnet-v2-full.yaml`
- Added `configs/xh25-mksnet-v2-full.yaml`
- Extended `tests/test_mksnet_configs.py` with exact assertions from the brief

Per the task instructions, no model source was edited.

## TDD Log

### 1. Added failing tests first

Appended:

- `test_mksnet_v2_full_model_yaml_contains_mks_stages`
- `test_mksnet_v2_full_pipeline_config_loads`

to `tests/test_mksnet_configs.py`.

### 2. Verified RED

Command:

```powershell
python -m pytest tests/test_mksnet_configs.py -q
```

Result:

- Exit code: `1`
- Existing two tests passed
- New two tests failed with `FileNotFoundError`

Observed failures:

- `configs/models/xh25-yolo-mksnet-v2-full.yaml` missing
- `configs/xh25-mksnet-v2-full.yaml` missing

This matches the expected RED condition from the brief.

### 3. Added requested configs

Created the model YAML and pipeline YAML exactly as specified in the brief.

### 4. Verified GREEN

Command:

```powershell
python -m pytest tests/test_mksnet_configs.py -q
```

Result:

- Exit code: `0`
- Output: `4 passed`

## Smoke-load Verification

Command:

```powershell
python -c "from xh_detect.models.ultralytics import register_custom_modules; from ultralytics import YOLO; register_custom_modules(); model=YOLO('configs/models/xh25-yolo-mksnet-v2-full.yaml'); print(model.model.__class__.__name__)"
```

Result:

- Exit code: `1`
- Warning:

```text
WARNING no model scale passed. Assuming scale='n'.
```

- Failure:

```text
RuntimeError: Given groups=1, weight of size [8, 128, 1, 1], expected input[1, 64, 1, 1] to have 128 channels, but got 64 channels instead
```

Key stack location:

- `src/xh_detect/models/mksnet_v2.py`, inside channel attention called by `MKSStage`

## Interpretation

This is the exact kind of integration issue called out in the brief:

- Ultralytics emitted a scale warning and assumed `scale='n'`
- The instantiated tensor shape entering the first `MKSStage` did not match the stage's expected channel count

Per instruction, I did **not** invent a different architecture or broaden scope into model-source changes.

## Final Status

`DONE_WITH_CONCERNS`

Reason:

- Required config files and tests were added successfully
- Required config tests pass
- Smoke-load does **not** currently succeed due to a real integration/channel-scaling issue outside the allowed edit scope for this task

## Commit

Intended commit message:

```text
config: add mksnet v2 full experiment
```

---

## Task 3 Fix Follow-up

Status: `FIXED`

### Root Cause Confirmed

- `ultralytics.nn.tasks.yaml_model_load()` overwrote the YAML `scale: s` with `guess_model_scale(path)`.
- For `configs/models/xh25-yolo-mksnet-v2-full.yaml`, the filename guess returned an empty scale, so Ultralytics fell back to the first `scales` entry (`n`).
- That fallback reduced early channels and produced the observed `MKSStage` mismatch.
- Even forcing `scale='s'` would still be wrong with the prior `scales.s` entry because `max_channels=1024` clamps the later `1536 * 0.5` stage to `512`, while this config expects `768`.
- Replacing `scales:` with explicit `depth_multiple: 0.50` and `width_multiple: 0.50` preserves the intended architecture without depending on filename scale inference.

### RED Evidence

Added regression test:

- `test_mksnet_v2_full_model_smoke_loads_with_detection_model`

Command run before YAML fix:

```powershell
python -m pytest tests/test_mksnet_configs.py -q
```

Observed result:

- Exit code: `1`
- Output summary: `....F`
- Failure site: `tests/test_mksnet_configs.py::test_mksnet_v2_full_model_smoke_loads_with_detection_model`
- Captured warning: `WARNING no model scale passed. Assuming scale='n'.`
- Captured error: `RuntimeError: Given groups=1, weight of size [8, 128, 1, 1], expected input[1, 64, 1, 1] to have 128 channels, but got 64 channels instead`

### Fix Applied

Updated only `configs/models/xh25-yolo-mksnet-v2-full.yaml`:

- kept `scale: s` for readability/static assertions
- removed the `scales:` mapping
- added:
  - `depth_multiple: 0.50`
  - `width_multiple: 0.50`

### GREEN Evidence

Regression suite:

```powershell
python -m pytest tests/test_mksnet_configs.py -q
```

Result:

- Exit code: `0`
- Output summary: `..... [100%]`

Smoke-load command:

```powershell
python -c "from xh_detect.models.ultralytics import register_custom_modules; from ultralytics import YOLO; register_custom_modules(); model=YOLO('configs/models/xh25-yolo-mksnet-v2-full.yaml'); print(model.model.__class__.__name__)"
```

Result:

- Exit code: `0`
- Output: `DetectionModel`

### Commit

```text
fix: make mksnet v2 full yaml loadable
```
