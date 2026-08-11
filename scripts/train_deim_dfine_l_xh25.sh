#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEIM_ROOT="${DEIM_ROOT:-${PROJECT_ROOT}/.third_party/DEIM}"
DEIM_PYTHON="${DEIM_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
DEIM_CONFIG="${DEIM_CONFIG:-${PROJECT_ROOT}/configs/deim/deim_dfine_l_xh25_1024.yml}"
DEIM_DATA_ROOT="${DEIM_DATA_ROOT:-${PROJECT_ROOT}/datasets/xh25}"
DEIM_PRETRAINED="${DEIM_PRETRAINED:-${PROJECT_ROOT}/weights/deim/deim_dfine_l_coco.pth}"
DEIM_OUTPUT_DIR="${DEIM_OUTPUT_DIR:-${PROJECT_ROOT}/runs/train/deim-dfine-l-xh25-1024}"
DEIM_BATCH_SIZE="${DEIM_BATCH_SIZE:-8}"
DEIM_VAL_BATCH_SIZE="${DEIM_VAL_BATCH_SIZE:-8}"
DEIM_WORKERS="${DEIM_WORKERS:-6}"
DEIM_VAL_WORKERS="${DEIM_VAL_WORKERS:-4}"
DEIM_SEED="${DEIM_SEED:-42}"
DEIM_EPOCHS="${DEIM_EPOCHS:-}"
DEIM_NO_AUG_EPOCHS="${DEIM_NO_AUG_EPOCHS:-}"

for path in \
  "${DEIM_ROOT}/train.py" \
  "${DEIM_CONFIG}" \
  "${DEIM_DATA_ROOT}/reports/train-ground-truth.json" \
  "${DEIM_DATA_ROOT}/reports/val-ground-truth.json"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 1
  fi
done
if [[ ! -x "${DEIM_PYTHON}" ]]; then
  echo "Python environment not found: ${DEIM_PYTHON}" >&2
  exit 1
fi

if [[ -z "${DEIM_RESUME:-}" && ! -s "${DEIM_PRETRAINED}" ]]; then
  echo "Pretrained checkpoint not found: ${DEIM_PRETRAINED}" >&2
  echo "Run scripts/bootstrap_deim.sh first." >&2
  exit 1
fi

read -r DEIM_HEAD_LR DEIM_BACKBONE_LR < <(
  "${DEIM_PYTHON}" - "${DEIM_BATCH_SIZE}" <<'PY'
import sys

batch = int(sys.argv[1])
if batch <= 0:
    raise SystemExit("DEIM_BATCH_SIZE must be positive")
print(0.0005 * batch / 32, 0.000025 * batch / 32)
PY
)
DEIM_HEAD_LR="${DEIM_HEAD_LR_OVERRIDE:-${DEIM_HEAD_LR}}"
DEIM_BACKBONE_LR="${DEIM_BACKBONE_LR_OVERRIDE:-${DEIM_BACKBONE_LR}}"
DEFAULT_OPTIMIZER_PARAMS="[{params: '^(?=.*backbone)(?!.*norm|bn).*$', lr: ${DEIM_BACKBONE_LR}}, {params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$', weight_decay: 0.0}]"
DEIM_OPTIMIZER_PARAMS="${DEIM_OPTIMIZER_PARAMS_OVERRIDE:-${DEFAULT_OPTIMIZER_PARAMS}}"

mkdir -p "${DEIM_OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/src:${DEIM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMAND=(
  "${DEIM_PYTHON}" "${DEIM_ROOT}/train.py"
  --config "${DEIM_CONFIG}"
  --device cuda
  --seed "${DEIM_SEED}"
  --use-amp
  --output-dir "${DEIM_OUTPUT_DIR}"
)
if [[ -n "${DEIM_RESUME:-}" ]]; then
  COMMAND+=(--resume "${DEIM_RESUME}")
else
  COMMAND+=(--tuning "${DEIM_PRETRAINED}")
fi
COMMAND+=(
  --update
  "train_dataloader.dataset.img_folder=${DEIM_DATA_ROOT}"
  "train_dataloader.dataset.ann_file=${DEIM_DATA_ROOT}/reports/train-ground-truth.json"
  "train_dataloader.total_batch_size=${DEIM_BATCH_SIZE}"
  "train_dataloader.num_workers=${DEIM_WORKERS}"
  "val_dataloader.dataset.img_folder=${DEIM_DATA_ROOT}"
  "val_dataloader.dataset.ann_file=${DEIM_DATA_ROOT}/reports/val-ground-truth.json"
  "val_dataloader.total_batch_size=${DEIM_VAL_BATCH_SIZE}"
  "val_dataloader.num_workers=${DEIM_VAL_WORKERS}"
  "optimizer.lr=${DEIM_HEAD_LR}"
  "optimizer.params=${DEIM_OPTIMIZER_PARAMS}"
)
if [[ -n "${DEIM_EPOCHS}" ]]; then
  COMMAND+=("epoches=${DEIM_EPOCHS}")
fi
if [[ -n "${DEIM_NO_AUG_EPOCHS}" ]]; then
  COMMAND+=("no_aug_epoch=${DEIM_NO_AUG_EPOCHS}")
fi

printf 'Launching:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
cd "${PROJECT_ROOT}"
exec "${COMMAND[@]}"
