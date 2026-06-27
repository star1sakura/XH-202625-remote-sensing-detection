#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-.venv}"
UV_VERSION="${UV_VERSION:-0.11.23}"

"${PYTHON_BIN}" - <<'PY'
import torch

print(f"base torch={torch.__version__} cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the base Python environment")
print(f"base gpu={torch.cuda.get_device_name(0)}")
PY

"${PYTHON_BIN}" -m pip install --disable-pip-version-check "uv==${UV_VERSION}"
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  echo "Reusing existing virtual environment: ${VENV_DIR}"
else
  uv venv --python "$(command -v "${PYTHON_BIN}")" --system-site-packages "${VENV_DIR}"
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

uv pip install --python "${VENV_DIR}/bin/python" \
  "gradio==6.19.0" \
  "numpy>=1.26,<2" \
  "opencv-python>=4.10,<5" \
  "shapely>=2,<3" \
  "typer>=0.16,<1" \
  "rich>=13,<15" \
  "pytest>=8,<10" \
  "pytest-cov>=5,<8" \
  "ruff>=0.11,<1" \
  "editables>=0.3,<0.4" \
  "scipy>=1.13,<1.15" \
  py-cpuinfo \
  polars

# Keep the CUDA-enabled torch/torchvision supplied by the rented GPU image.
uv pip install --python "${VENV_DIR}/bin/python" ultralytics-thop --no-deps
uv pip install --python "${VENV_DIR}/bin/python" ultralytics==8.4.71 --no-deps
uv pip install --python "${VENV_DIR}/bin/python" -e . --no-deps

"${VENV_DIR}/bin/python" - <<'PY'
import cv2
import gradio
import torch
import ultralytics

assert torch.cuda.is_available()
device = torch.zeros(1, device="cuda").device
print(
    {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_tensor_device": str(device),
        "ultralytics": ultralytics.__version__,
        "gradio": gradio.__version__,
        "opencv": cv2.__version__,
    }
)
PY
