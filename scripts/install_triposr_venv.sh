#!/usr/bin/env bash
set -euo pipefail

TRIPOSR_DIR="${1:-TripoSR}"
VENV_DIR="${TRIPOSR_DIR}/.venv_triposr"

if [ ! -d "$TRIPOSR_DIR" ]; then
  echo "TripoSR directory not found: $TRIPOSR_DIR" >&2
  exit 1
fi

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install onnxruntime-gpu

python -m pip install git+https://github.com/tatsy/torchmcubes.git

python -m pip install -r "$TRIPOSR_DIR/requirements.txt"

python -m pip install \
  "transformers>=4.36,<4.38" \
  "tokenizers>=0.15,<0.16" \
  "huggingface_hub>=0.34,<1.0" \
  "numpy<2"
