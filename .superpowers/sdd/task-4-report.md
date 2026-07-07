# Task 4 Report: Experiment Runbook And Local Verification

## Summary

- Created `docs/experiments/mksnet-v2-full-vehicle.md` with the exact experiment runbook content from the task brief.
- Kept scope limited to the single requested runbook file.
- Staged and committed the runbook as `6b0bce6` (`docs: add mksnet v2 full runbook`).

## Self-Review

- Confirmed the new runbook matches the required title, narrative, command blocks, result table, post-eval note, and keep criteria from the brief.
- Confirmed the Git diff is limited to the new runbook file.

## Verification Commands

### 1. Pytest

Command:

```powershell
python -m pytest tests/test_mksnet_v2.py tests/test_mksnet_lite.py tests/test_mksnet_configs.py -q
```

Result:

- Exit code: `0`
- Output ended with `.........................                                                [100%]`

### 2. Ruff format check

Command:

```powershell
python -m ruff format --check src/xh_detect/models tests/test_mksnet_v2.py tests/test_mksnet_configs.py
```

Result:

- Exit code: `1`
- Failure: `No module named ruff`

### 3. Ruff lint check

Command:

```powershell
python -m ruff check src/xh_detect/models tests/test_mksnet_v2.py tests/test_mksnet_configs.py
```

Result:

- Exit code: `1`
- Failure: `No module named ruff`

### 4. Git diff check

Command:

```powershell
git diff --check
```

Result:

- Exit code: `0`

## Commit

- `6b0bce6 docs: add mksnet v2 full runbook`

## Concerns

- Local verification did not fully meet the brief's expected all-green outcome because the interpreter used by `python -m ruff ...` does not have `ruff` installed in this environment.
- I did not broaden scope to modify environment setup, source, configs, tests, or other docs.

## Verification Fix Addendum

### Scope

- Fixed only Ruff import-order/formatting issues in:
  - `src/xh_detect/models/__init__.py`
  - `src/xh_detect/models/ultralytics.py`
  - `tests/test_mksnet_v2.py`
  - `tests/test_mksnet_configs.py`
- Changes were limited to import reordering and wrapping long list-literal assertions.

### Fix Commands

#### 1. Ruff fix

Command:

```powershell
python -m ruff check --fix src/xh_detect/models/__init__.py src/xh_detect/models/ultralytics.py tests/test_mksnet_v2.py tests/test_mksnet_configs.py
```

Result:

- Exit code: `1`
- Fixed the import-order issues automatically.
- Reported remaining `E501` long lines in `tests/test_mksnet_configs.py` on lines 50-53.

#### 2. Ruff format

Command:

```powershell
python -m ruff format src/xh_detect/models/__init__.py src/xh_detect/models/ultralytics.py tests/test_mksnet_v2.py tests/test_mksnet_configs.py
```

Result:

- Exit code: `0`
- Output: `1 file reformatted, 3 files left unchanged`

#### 3. Manual follow-up

- Wrapped the four long `assert custom_layers[...] == ...` statements in `tests/test_mksnet_configs.py` to satisfy `E501` without changing test behavior.

### Verification Commands

#### 1. Pytest

Command:

```powershell
python -m pytest tests/test_mksnet_v2.py tests/test_mksnet_lite.py tests/test_mksnet_configs.py -q
```

Result:

- Exit code: `0`
- Output ended with `.........................                                                [100%]`

#### 2. Ruff format check

Command:

```powershell
python -m ruff format --check src/xh_detect/models tests/test_mksnet_v2.py tests/test_mksnet_configs.py
```

Result:

- Exit code: `0`
- Output: `6 files already formatted`

#### 3. Ruff lint check

Command:

```powershell
python -m ruff check src/xh_detect/models tests/test_mksnet_v2.py tests/test_mksnet_configs.py
```

Result:

- Exit code: `0`
- Output: `All checks passed!`

#### 4. Git diff check

Command:

```powershell
git diff --check
```

Result:

- Exit code: `0`
