#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEIM_ROOT="${DEIM_ROOT:-${PROJECT_ROOT}/.third_party/DEIM}"
DEIM_REPOSITORY="${DEIM_REPOSITORY:-https://github.com/ShihuaHuang95/DEIM.git}"
DEIM_COMMIT="${DEIM_COMMIT:-09d35d53d39ee3145a1e61e3a989b28b9468d1dd}"
DEIM_PYTHON="${DEIM_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
UV_BIN="${UV_BIN:-${PROJECT_ROOT}/../.tools/bin/uv}"
PRETRAINED="${DEIM_PRETRAINED:-${PROJECT_ROOT}/weights/deim/deim_dfine_l_coco.pth}"
PATCH_FILES=(
  "${PROJECT_ROOT}/patches/deim-preserve-epoch-checkpoints.patch"
  "${PROJECT_ROOT}/patches/deim-torchvision-v2-compat.patch"
  "${PROJECT_ROOT}/patches/deim-selective-class-row-finetune.patch"
  "${PROJECT_ROOT}/patches/deim-bhcl.patch"
)

if [[ ! -x "${DEIM_PYTHON}" ]]; then
  echo "Python environment not found: ${DEIM_PYTHON}" >&2
  echo "Set DEIM_PYTHON to the CUDA-enabled xh25 environment." >&2
  exit 1
fi

"${DEIM_PYTHON}" - <<'PY'
import torch

print({"torch": torch.__version__, "cuda": torch.version.cuda})
if not torch.cuda.is_available():
    raise SystemExit("DEIM bootstrap requires a CUDA-enabled PyTorch environment")
print({"gpu": torch.cuda.get_device_name(0)})
PY

mkdir -p "$(dirname "${DEIM_ROOT}")"
if [[ ! -d "${DEIM_ROOT}/.git" ]]; then
  git clone --filter=blob:none "${DEIM_REPOSITORY}" "${DEIM_ROOT}"
fi

CURRENT_COMMIT="$(git -C "${DEIM_ROOT}" rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${DEIM_COMMIT}" ]]; then
  if [[ -n "$(git -C "${DEIM_ROOT}" status --porcelain)" ]]; then
    echo "Refusing to switch a modified DEIM checkout at ${DEIM_ROOT}" >&2
    exit 1
  fi
  git -C "${DEIM_ROOT}" fetch origin "${DEIM_COMMIT}"
  git -C "${DEIM_ROOT}" switch --detach "${DEIM_COMMIT}"
fi

for patch_file in "${PATCH_FILES[@]}"; do
  if git -C "${DEIM_ROOT}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    echo "DEIM patch is already applied: $(basename "${patch_file}")"
  elif git -C "${DEIM_ROOT}" apply --check "${patch_file}"; then
    git -C "${DEIM_ROOT}" apply "${patch_file}"
  else
    echo "DEIM patch does not apply cleanly at ${DEIM_COMMIT}: ${patch_file}" >&2
    exit 1
  fi
done

if [[ -x "${UV_BIN}" ]]; then
  "${UV_BIN}" pip install --python "${DEIM_PYTHON}" \
    --index-url "${PIP_INDEX_URL}" \
    "faster-coco-eval>=1.6.5,<2" \
    "tensorboard>=2.16,<3" \
    "calflops>=0.3,<1" \
    "transformers>=4.40,<6" \
    "gdown>=5,<6"
  "${UV_BIN}" pip install --python "${DEIM_PYTHON}" \
    --no-deps --editable "${PROJECT_ROOT}"
else
  if ! "${DEIM_PYTHON}" -m pip --version >/dev/null 2>&1; then
    "${DEIM_PYTHON}" -m ensurepip --upgrade
  fi
  "${DEIM_PYTHON}" -m pip install --disable-pip-version-check \
    --index-url "${PIP_INDEX_URL}" \
    "faster-coco-eval>=1.6.5,<2" \
    "tensorboard>=2.16,<3" \
    "calflops>=0.3,<1" \
    "transformers>=4.40,<6" \
    "gdown>=5,<6"
  "${DEIM_PYTHON}" -m pip install --disable-pip-version-check \
    --no-deps --editable "${PROJECT_ROOT}"
fi

mkdir -p "$(dirname "${PRETRAINED}")"
if [[ ! -s "${PRETRAINED}" ]]; then
  "${DEIM_PYTHON}" -m gdown \
    "1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr" \
    --output "${PRETRAINED}"
fi

DEIM_PRETRAINED_PATH="${PRETRAINED}" "${DEIM_PYTHON}" - <<'PY'
import hashlib
import os
from pathlib import Path

import torch

path = Path(os.environ["DEIM_PRETRAINED_PATH"])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
if not isinstance(checkpoint, dict) or not ({"ema", "model"} & checkpoint.keys()):
    raise SystemExit(f"Unexpected DEIM checkpoint structure: {path}")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print({"checkpoint": str(path), "sha256": digest, "bytes": path.stat().st_size})
PY
