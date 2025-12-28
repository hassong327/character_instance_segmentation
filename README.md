# CartoonSegmentation 로컬 추론 가이드 (Windows RTX 3060)

## 환경 설정 가이드 (CUDA 11.8 권장)

- 가상환경 구성 후 순서대로 설치 추천 (충돌 방지: PyTorch → mmcv → mmdet/mmengine → 나머지)
- 예시 명령어:
  - `python -m venv .venv`
  - `.venv\Scripts\activate`
  - `python -m pip install --upgrade pip`
  - `pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118`
  - `pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html`
  - `pip install mmdet==3.3.0 mmengine==0.10.5 opencv-python`
  - `pip install -r requirements.txt`
- bash 설치 스크립트:
  - `chmod +x scripts/install_cuda118.sh`
  - `./scripts/install_cuda118.sh`
- CUDA 12.x 사용 시 `mmcv` 다운로드 URL을 `cu121/torch2.1`로 변경 권장.
- `requirements.txt` 안의 `git+https://...` 패키지는 빌드 도구가 필요할 수 있으니, Windows에서는 “Desktop development with C++” 설치 여부 확인 권장.

## 모델 가중치 다운로드 위치

- 레포 루트에서 실행:
  - `git lfs install`
  - `git clone https://huggingface.co/dreMaz/AnimeInstanceSegmentation models/AnimeInstanceSegmentation`
- 기본 체크포인트 경로: `models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt`
- 리파인 모델(기본값): `models/AnimeInstanceSegmentation/refine_last.ckpt`
- `--refine animeseg` 사용 시 별도 `models/anime-seg/isnetis.ckpt` 필요.

## 참고 레포

- 모델 가중치 레포: `https://huggingface.co/dreMaz/AnimeInstanceSegmentation`
- 원본 코드 레포: `https://github.com/CartoonSegmentation/CartoonSegmentation`

## 사용 예시

- CLI 실행:
  - `python extract.py --img "내_캐릭터.jpg"`
- 출력 파일 기본값: 입력 파일과 같은 폴더에 `*_cutout.png`
원하시면 `--refine none`/`--det-size 512` 옵션으로 속도/메모리 튜닝 가이드도 더 정리해 드릴게요.

---

## 🎨 Anime Character Segmentation for Desktop Pet

**Project Goal**: 데스크톱 펫 애플리케이션 개발을 위한 2D 애니메이션 캐릭터 자동 배경 제거 및 리깅 소스 추출 파이프라인 구축

### 🖥️ Development Environment

- OS: Windows 10/11
- GPU: NVIDIA GeForce RTX 3060 (VRAM 12GB)
- Target Platform: Electron (Node.js) + Python Backend

### 🚀 Development History & Model Comparison

캐릭터의 외곽선을 깔끔하게 추출(Instance Segmentation)하여 3D 리깅용 텍스처로 변환하기 위해 다양한 AI 모델을 테스트하고 비교했습니다.

#### 1. Attempt #1: SAM (Segment Anything Model)

Meta에서 공개한 범용 이미지 분할 모델 (ViT-H, ViT-L).

- 접근 방식: segment-anything 라이브러리를 사용, 포인트 프롬프트(Point Prompt) 방식 시도
- 결과: ❌ 채택 보류
- 원인 분석:
  - 과도한 리소스: ViT-H 모델은 로컬 런타임용으로 사용하기에 너무 무거움 (VRAM 점유율 높음, 추론 속도 느림)
  - 도메인 불일치: 실사 이미지(Photo) 위주로 학습되어 애니메이션 특유의 선화(Lineart)나 단색 채색 영역을 제대로 인식하지 못하고 캐릭터가 조각나는 현상 발생

#### 2. Attempt #2: MobileSAM / FastSAM

SAM의 경량화 버전. 실시간 추론을 목표로 테스트.

- 접근 방식: Ultralytics 및 MobileSAM 리포지토리 활용, Electron 앱 내 실시간 구동 테스트
- 결과: ❌ 실패 (품질 미달)
- 원인 분석:
  - 디테일 손실: 속도는 빨라졌으나, 리깅에 필요한 머리카락 끝/옷자락 등의 미세 디테일이 뭉개짐
  - 정확도 한계: 범용 데이터셋 기반이라 애니메이션 캐릭터와 배경을 명확히 분리하지 못함

#### 3. Attempt #3: SAM Fine-tuning (LoRA)

애니메이션 데이터셋으로 SAM 재학습 시도 고려.

- 접근 방식: Roboflow 데이터셋 + LoRA(Low-Rank Adaptation) 파인튜닝 기법 검토
- 결과: ⚠️ 중단 (비효율적)
- 원인 분석:
  - 데이터 준비 비용: 양질의 세그멘테이션 데이터(Polygon Labeling)를 직접 구축하는 데 시간 소요
  - 하드웨어 제약: RTX 3060으로 ViT-L급 모델 학습 시 VRAM 한계 명확

#### 🏆 Final Solution: CartoonSegmentation

애니메이션 및 만화 도메인에 특화된 세그멘테이션 모델.

- Repository: CartoonSegmentation
- 선정 이유:
  - 도메인 특화: 별도 파인튜닝 없이도 애니메이션 캐릭터 외곽선을 깔끔하게 인식
  - Semantic Understanding: 단순 객체 인식을 넘어, 만화적 표현(이펙트, 선화 등)을 이해하고 처리
  - 적절한 퍼포먼스: RTX 3060 환경에서 충분히 구동 가능한 가중치 크기와 추론 속도 제공
  - 구현: Python 스크립트로 추론 엔진을 구축하고, Electron에서 `child_process`로 호출해 PNG 추출 자동화 완료

### 📊 Summary Table

| Model | Type | Inference Speed | Edge Quality (Anime) | Verdict |
| --- | --- | --- | --- | --- |
| SAM (ViT-H) | General | Slow | Low (Fragmented) | ❌ Discarded |
| MobileSAM | Lightweight | Very Fast | Very Low (Blurry) | ❌ Discarded |
| Fine-tuned SAM | Custom | Slow | High (Expected) | ⚠️ Too Costly |
| CartoonSegmentation | Anime-Specific | Fast | High (Crisp) | ✅ Adopted |
