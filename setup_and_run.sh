#!/usr/bin/env bash
set -euo pipefail

# ===== 사용자 설정 =====
VENV_DIR=".venv"
IMG_PATH="${1:-ani.webp}"  # 인자 없으면 ani.webp
REQ_FILE="requirements.txt"

echo "== [1] 기존 venv 삭제 =="
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo " - 현재 venv 활성화 상태라 deactivate 시도"
  deactivate || true
fi
rm -rf "$VENV_DIR"

echo "== [2] 새 venv 생성 =="
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "== [3] CUDA 11.8용 PyTorch 설치 =="
python -m pip install --no-cache-dir \
  torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118

echo "== [4] mmcv (cu118/torch2.1 전용 휠) 설치 =="
python -m pip install --no-cache-dir \
  mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

echo "== [5] mmdet/mmengine 설치 =="
python -m pip install --no-cache-dir mmdet==3.3.0 mmengine==0.10.5

echo "== [6] 나머지 의존성 설치 =="
python -m pip install --no-cache-dir "numpy<2" "transformers<4.38"
python -m pip install --no-cache-dir -r "$REQ_FILE"
python -m pip install --no-cache-dir "numpy<2"

echo "== [검증] mmcv.ops 로드 테스트 =="
python - << 'PY'
import mmcv
print("mmcv:", mmcv.__version__)
from mmcv.ops import roi_align
print("mmcv.ops OK (roi_align import success)")
PY

echo "== [7] 실행 =="
python extract.py --img "$IMG_PATH"
