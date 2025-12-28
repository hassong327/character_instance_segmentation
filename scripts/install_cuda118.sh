#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

pip install mmdet==3.3.0 mmengine==0.10.5

pip install "numpy<2" "transformers<4.38"

pip install -r requirements.txt
