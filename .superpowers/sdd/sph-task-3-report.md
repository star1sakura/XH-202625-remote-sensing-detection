# Task 3 Report: Add NAM And Full SPH Config Variants

## Scope

Implemented Task 3 in the assigned files only:

- `tests/test_sph_configs.py`
- `configs/models/xh25-yolo26s-sph-p2-nam.yaml`
- `configs/models/xh25-yolo26s-sph-full.yaml`
- `configs/xh25-sph-p2-nam.yaml`
- `configs/xh25-sph-full.yaml`

## What Changed

### Tests

Extended `tests/test_sph_configs.py` with the exact Task 3 coverage:

- imported `pytest`
- added YAML structure assertions for `sph-p2-nam`
- added YAML structure assertions for `sph-full`
- added pipeline config load tests for both new runtime configs
- added a parametrized smoke-load test that instantiates both custom model variants with `YOLO(...)`

### Model YAMLs

Added:

- `configs/models/xh25-yolo26s-sph-p2-nam.yaml`
- `configs/models/xh25-yolo26s-sph-full.yaml`

Both include the required SPH heads and Detect indices from the task brief:

- `sph-p2-nam`: `[[20, 24, 27, 30], 1, Detect, [nc]]`
- `sph-full`: `[[21, 26, 30, 34], 1, Detect, [nc]]`

### Runtime Configs

Added:

- `configs/xh25-sph-p2-nam.yaml`
- `configs/xh25-sph-full.yaml`

Both load through `PipelineConfig.from_yaml(...)` with the expected task, taxonomy, model path, image size, batch size, and 25 class thresholds.

## TDD Evidence

### Red

Ran:

```bash
python -m pytest tests/test_sph_configs.py -q
```

Result:

- failed as expected
- six failures
- all failures were missing-file errors for the new model/runtime YAMLs

### Green

After adding the four YAML files, the same test command still failed in the two smoke-load cases because `NAMBlock` was constructed with literal channels from YAML while the selected `s` width multiplier reduced upstream feature-map channels during Ultralytics model parsing.

To keep the fix inside the allowed files and preserve the required layer assertions, I updated the selected `s` scale entry in the two new model YAMLs from:

```yaml
s: [0.50, 0.50, 1024]
```

to:

```yaml
s: [0.50, 1.00, 1024]
```

This keeps the Task 3 layer definitions intact while allowing the smoke-load test to instantiate `NAMBlock` and `SwinPredictionBlock` with channel counts that match the actual tensors reaching those layers.

Re-ran:

```bash
python -m pytest tests/test_sph_configs.py -q
```

Result:

- `9 passed`

## Verification

Fresh verification command:

```bash
python -m pytest tests/test_sph_configs.py -q
```

Fresh result:

```text
.........                                                                [100%]
```

## Commit

Created commit:

- `config: add sph nam and full variants`

## Notes / Concerns

The only deviation from the brief’s literal YAML text is the `scales.s` width multiplier in the two new model YAMLs. Without that adjustment, the required smoke-load test fails with a runtime channel mismatch inside `NAMBlock`. The structural layer assertions, Detect indices, runtime config values, and focused test command all match the task requirements.

## Task 3 Narrow Fix Note

- Fixed `configs/models/xh25-yolo26s-sph-p2-nam.yaml` and `configs/models/xh25-yolo26s-sph-full.yaml` to keep `scales.s` at `[0.50, 0.50, 1024]` and updated custom module channel args to the actual scaled channels.
- Updated `tests/test_sph_configs.py` assertions to match the corrected NAM/Swin layer args.
- Verification command: `python -m pytest tests/test_sph_configs.py -q`
- Result: `9 passed`
