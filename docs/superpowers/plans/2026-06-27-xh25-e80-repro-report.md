# XH25 YOLO26s e80 Reproducibility Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a sanitized, auditable reproducibility record for the completed `xh25-yolo26s-e80` training run.

**Architecture:** Store one Markdown report plus a small machine-readable metrics package under `docs/experiments/`. Derive all numbers from preserved server artifacts, remove machine-specific paths and credentials, validate the report against CSV/JSON sources, and keep weights, raw logs, predictions, and competition imagery outside Git.

**Tech Stack:** Markdown, YAML, CSV, PowerShell, Python 3.12, Ultralytics 8.4.71, Git

---

### Task 1: Create the sanitized metrics package

**Files:**
- Create: `docs/experiments/assets/xh25-yolo26s-e80/training-args.yaml`
- Create: `docs/experiments/assets/xh25-yolo26s-e80/results.csv`
- Create: `docs/experiments/assets/xh25-yolo26s-e80/results.png`
- Create: `docs/experiments/assets/xh25-yolo26s-e80/box-f1-curve.png`
- Create: `docs/experiments/assets/xh25-yolo26s-e80/box-pr-curve.png`
- Create: `docs/experiments/assets/xh25-yolo26s-e80/confusion-matrix-normalized.png`
- Source: `outputs/xh25-server/repro-e80-source/`

- [x] **Step 1: Create the tracked asset directory**

Run:

```powershell
New-Item -ItemType Directory `
  -Path docs/experiments/assets/xh25-yolo26s-e80 `
  -Force
```

Expected: the directory exists under `docs/experiments/assets/`.

- [x] **Step 2: Copy safe metric files with stable names**

Run:

```powershell
$source = 'outputs/xh25-server/repro-e80-source'
$target = 'docs/experiments/assets/xh25-yolo26s-e80'
Copy-Item "$source/results.csv" "$target/results.csv"
Copy-Item "$source/results.png" "$target/results.png"
Copy-Item "$source/BoxF1_curve.png" "$target/box-f1-curve.png"
Copy-Item "$source/BoxPR_curve.png" "$target/box-pr-curve.png"
Copy-Item "$source/confusion_matrix_normalized.png" `
  "$target/confusion-matrix-normalized.png"
```

Expected: one CSV and four PNG files exist; no training batch, label, prediction, or source image is copied.

- [x] **Step 3: Write the sanitized training arguments**

Create `training-args.yaml` with:

```yaml
task: detect
model: yolo26s.pt
data: datasets/xh25/dataset.yaml
epochs: 80
patience: 100
batch: 8
imgsz: 1024
device: "0"
workers: 4
project: runs/train
name: xh25-yolo26s-e80
pretrained: true
optimizer: auto
seed: 42
deterministic: true
amp: false
resume: false
fraction: 1.0
close_mosaic: 10
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 7.5
cls: 0.5
dfl: 1.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
translate: 0.1
scale: 0.5
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
```

Expected: repository-relative paths only; no `project` or `save_dir` absolute server path.

- [x] **Step 4: Verify asset count and file types**

Run:

```powershell
$assets = Get-ChildItem docs/experiments/assets/xh25-yolo26s-e80 -File
if ($assets.Count -ne 6) { throw "expected 6 assets, got $($assets.Count)" }
$assets | Select-Object Name,Length
```

Expected: six non-empty files.

### Task 2: Write the reproducibility report

**Files:**
- Create: `docs/experiments/xh25-yolo26s-e80.md`
- Reference: `configs/xh25-yolo26s-e80.yaml`
- Reference: `docs/xh25-data-analysis.md`
- Reference: `src/xh_detect/taxonomy.py`

- [x] **Step 1: Record identity, environment, and data provenance**

Include these verified facts:

```markdown
- Training code commit: `3254a6aa8377a88124a73bfe627176cdd3502255`
- GPU: NVIDIA GeForce RTX 4090, 24 GiB
- Python: 3.12.3
- PyTorch: 2.12.1+cu130
- Ultralytics: 8.4.71
- Dataset: 4,481 image-label pairs
- Train/val: 3,807 / 674 images
- Validation instances: 3,225
- Train manifest SHA256: `8fc0f7144cb3d922d9a943d531b9172777a52d25fb8e8313e513dfed323f20fe`
- Val manifest SHA256: `2ba3156d492ef108a89e55703f341d93a959d17512ac77e72cbd98ab65fe395d`
- Source groups SHA256: `ab5ab5fa22f9eb3275f7c23eb111cf83d7db8d47748ba4da14cd32a4a1d7a3ee`
```

- [x] **Step 2: Record the exact training command and resolved optimizer**

Use:

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25/dataset.yaml \
  --model yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --device 0 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --project runs/train \
  --name xh25-yolo26s-e80 \
  --no-resume
```

Record that `optimizer=auto` resolved to AdamW with learning rate `0.000345`,
momentum `0.9`, and weight decay `0.0005`.

- [x] **Step 3: Record artifact hashes**

Use:

```markdown
- Base `yolo26s.pt`: `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`
- `best.pt`: `930cf7e1c698a8850523ce42d2565d1b2652e5ae01bf7f049a35d05778dd5424`
- `last.pt`: `584e40ad59adb32d2f363f81ce935fc644d8983f4374b2a37a68e8a66f47991d`
- `results.csv`: `0fe812216ae3959c6a564969e5e758f21f26c85f175164c402bd4d98fff4fe4e`
- `results.png`: `da1b09da99eb5b1e0fd7d9380c738dbfa12ae7f24cb911ab4eb742f701a0ce24`
```

Clarify that model files remain outside Git.

- [x] **Step 4: Record Ultralytics metrics and epoch trends**

Include:

```markdown
| View | Epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Best checkpoint validation | — | 0.932 | 0.918 | 0.948 | 0.766 |
| Highest precision in CSV | 71 | 0.95098 | — | — | — |
| Highest recall in CSV | 45 | — | 0.93472 | — | — |
| Highest mAP50 in CSV | 45 | — | — | 0.95486 | — |
| Highest mAP50-95 in CSV | 62 | — | — | — | 0.76610 |
| Final epoch | 80 | 0.94218 | 0.90883 | 0.93559 | 0.75819 |
```

Record 80 epochs in 1.790 hours, final-ten-epoch average mAP50-95 of `0.758815`,
and best-checkpoint speed of 0.3 ms preprocessing, 3.7 ms inference, and 0.1 ms
postprocessing per image.

- [x] **Step 5: Record custom evaluator and threshold sweep**

Include:

```markdown
| Group | TP | FP | FN | Recall | FDR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 3102 | 120 | 124 | 0.961562 | 0.037244 |
| Aircraft | 2716 | 44 | 30 | 0.989075 | 0.015942 |
| Ship | 331 | 62 | 71 | 0.823383 | 0.157761 |
| Vehicle | 55 | 14 | 23 | 0.705128 | 0.202899 |
```

Include representative thresholds:

```markdown
| Threshold | Precision | Recall | FDR | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.909538 | 0.975511 | 0.090462 | 0.941370 |
| 0.10 | 0.933274 | 0.971172 | 0.066726 | 0.951846 |
| 0.20 | 0.955173 | 0.964352 | 0.044827 | 0.959741 |
| 0.25 | 0.962756 | 0.961562 | 0.037244 | 0.962159 |
| 0.30 | 0.965927 | 0.957843 | 0.034073 | 0.961868 |
| 0.40 | 0.972655 | 0.948233 | 0.027345 | 0.960289 |
```

State that `0.25` is the best overall custom-evaluator F1 in the recorded sweep.

- [x] **Step 6: Record weaknesses and next experiments**

Include the lowest-recall classes:

```markdown
| Class | GT | Recall | FDR |
| --- | ---: | ---: | ---: |
| HM | 2 | 0.5000 | 0.0000 |
| QHS | 97 | 0.6907 | 0.2209 |
| FSC | 78 | 0.7051 | 0.2029 |
| A5_F-16 | 164 | 0.7988 | 0.0076 |
| MS | 298 | 0.8255 | 0.1827 |
| A18_KC-10 | 38 | 0.8421 | 0.0000 |
| A13_F-15 | 222 | 0.9144 | 0.1362 |
| A20_SU-24 | 134 | 0.9179 | 0.1214 |
```

Recommend error analysis and data-centric experiments for QHS, MS, and FSC,
then controlled 1280/1536 image-size and tiled-inference experiments.

- [x] **Step 7: Link safe plots and list exclusions**

Embed relative links to all four PNG files and link `results.csv` and
`training-args.yaml`. State that raw logs, predictions, weights, datasets, and
sample imagery are intentionally excluded.

### Task 3: Validate report integrity and sanitization

**Files:**
- Verify: `docs/experiments/xh25-yolo26s-e80.md`
- Verify: `docs/experiments/assets/xh25-yolo26s-e80/`

- [x] **Step 1: Validate CSV metrics**

Run:

```powershell
$csv = Import-Csv docs/experiments/assets/xh25-yolo26s-e80/results.csv
if ($csv.Count -ne 80) { throw "expected 80 epochs" }
$best = $csv |
  Sort-Object { [double]$_.'metrics/mAP50-95(B)' } -Descending |
  Select-Object -First 1
if ($best.epoch -ne '62' -or $best.'metrics/mAP50-95(B)' -ne '0.7661') {
  throw "unexpected best mAP50-95 row"
}
```

Expected: exit code 0.

- [x] **Step 2: Scan tracked text for sensitive content**

Run:

```powershell
$paths = @(
  'docs/experiments/xh25-yolo26s-e80.md',
  'docs/experiments/assets/xh25-yolo26s-e80/training-args.yaml',
  'docs/experiments/assets/xh25-yolo26s-e80/results.csv'
)
$matches = rg -n `
  'root@|connect\.|44592|autodl-tmp|BEGIN .*PRIVATE KEY|gho_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+' `
  $paths
if ($LASTEXITCODE -eq 0) {
  $matches
  throw 'sensitive or machine-specific text found'
}
```

Expected: no matches.

- [x] **Step 3: Validate report references and hashes**

Run a PowerShell check that:

- the report contains the code SHA, three manifest hashes, base-model hash, and
  both weight hashes;
- every relative asset link resolves to an existing file;
- `Get-FileHash` for the tracked CSV and results PNG matches the recorded
  server hashes.

Expected: all checks exit 0.

- [x] **Step 4: Inspect all images**

Open `results.png`, `box-f1-curve.png`, `box-pr-curve.png`, and
`confusion-matrix-normalized.png`.

Expected: readable plots with no original remote-sensing imagery, credentials,
server path, or clipping that prevents interpretation.

- [x] **Step 5: Run repository checks**

Run:

```powershell
git diff --check
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pytest
```

Expected: no whitespace, Ruff, or pytest failures.

### Task 4: Commit, merge, and publish

**Files:**
- Stage: `docs/experiments/`
- Stage: `docs/superpowers/specs/2026-06-27-xh25-e80-repro-report-design.md`
- Stage: `docs/superpowers/plans/2026-06-27-xh25-e80-repro-report.md`
- Exclude: `demo_assets/`
- Exclude: `outputs/`

- [x] **Step 1: Review exact Git scope**

Run:

```powershell
git status --short
git diff --stat
```

Expected: only the design, plan, report, and six sanitized assets are in scope;
`demo_assets/` remains untracked.

- [x] **Step 2: Commit the implementation**

Run:

```powershell
git add -- docs/experiments `
  docs/superpowers/plans/2026-06-27-xh25-e80-repro-report.md
git diff --cached --check
git commit -m "docs: add xh25 e80 reproducibility record"
```

Expected: one implementation commit after the design commit.

- [x] **Step 3: Merge into main and verify**

Run:

```powershell
git switch main
git pull --ff-only
git merge --ff-only codex/xh25-e80-repro-report
.venv\Scripts\python.exe -m pytest
```

Expected: fast-forward merge and no pytest failures.

- [x] **Step 4: Push and compare SHAs**

Run:

```powershell
git push origin main
$local = git rev-parse HEAD
$remote = (git ls-remote origin refs/heads/main).Split()[0]
if ($local -ne $remote) { throw "local/remote SHA mismatch" }
```

Expected: `origin/main` equals the local `main` SHA.
