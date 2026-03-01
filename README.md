# Character Instance Segmentation

애니메이션 캐릭터 이미지를 입력받아 배경이 제거된 `RGBA PNG`를 생성하는 프로젝트입니다.  
현재 레포는 **segmentation 전용**으로 정리되어 있으며, TripoSR/VRM 파이프라인은 포함하지 않습니다.

## 1. 환경 설치 (CUDA 11.8 권장)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118

pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

pip install mmdet==3.3.0 mmengine==0.10.5
pip install "numpy<2" "transformers<4.38"
pip install -r requirements.txt
```

원클릭 설치 스크립트:

```bash
chmod +x scripts/install_cuda118.sh
./scripts/install_cuda118.sh
```

## 2. 모델 가중치 준비

```bash
git lfs install
git clone https://huggingface.co/dreMaz/AnimeInstanceSegmentation models/AnimeInstanceSegmentation
```

- 기본 detector ckpt: `models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt`
- 기본 refine 모델: `models/AnimeInstanceSegmentation/refine_last.ckpt`
- `--refine animeseg` 사용 시: `models/anime-seg/isnetis.ckpt` 필요

## 3. 실행

```bash
python extract.py --img ./input.png
```

기본 출력 파일은 입력과 같은 폴더의 `*_cutout.png` 입니다.

### 주요 옵션

- `--out`: 출력 경로 직접 지정
- `--device`: `cuda` 또는 `cpu`
- `--det-size`: detector 입력 크기 (기본 `640`)
- `--score-thr`: detection 임계값 (기본 `0.3`)
- `--refine`: `refinenet_isnet`, `animeseg`, `none`

예시:

```bash
python extract.py --img ./input.png --out ./result.png --det-size 512 --refine none
```

## 4. 참고

- 모델 가중치: https://huggingface.co/dreMaz/AnimeInstanceSegmentation
- 원본 프로젝트: https://github.com/CartoonSegmentation/CartoonSegmentation

## 5. 원본 대비 차이점 (Diff Summary)

이 레포는 `CartoonSegmentation` 원본을 그대로 복제한 것이 아니라,  
**캐릭터 segmentation 추론 파이프라인만 남긴 경량화 포크**입니다.

- 범위 축소:
  - 원본의 3D Ken Burns / Style Editing / Web UI 경로는 제외
  - 본 레포는 `extract.py` 중심의 segmentation 추론에 집중
- 레포 구조 단순화:
  - 원본의 `configs/`, `depth_modules/`, `anime_3dkenburns/`, `examples/`, `Web_UI/` 등 부가 구성은 미포함
- 핵심 엔진 유지 + 안정성 보강:
  - `animeinsseg` 코어 로직은 유지
  - 텐서/넘파이 변환, bbox 리사이즈 축, 예외 처리(OOM fallback) 등 안정성 관련 코드 개선
- 실행 동선 단순화:
  - 단일 CLI로 `이미지 -> RGBA cutout PNG` 생성
  - 예: `python extract.py --img ./input.png`
