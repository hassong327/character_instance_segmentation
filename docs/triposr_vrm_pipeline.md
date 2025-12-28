# TripoSR → VRM Pipeline

## 1) TripoSR FastAPI Wrapper

`POST /generate`로 PNG를 업로드하면 GLB를 반환합니다. 응답은 `glb_path` 또는 `glb_base64` 중 하나입니다.

```bash
pip install fastapi uvicorn requests
```

```bash
export TRIPOSR_REPO=/path/to/TripoSR
export TRIPOSR_CMD="python run.py {image} --output-dir {output_dir} --model-save-format glb"
python tools/triposr_server.py
```

```bash
curl -X POST "http://localhost:8000/generate?response_type=path" \
  -F "file=@./sample.png"
```

### Colab 실행 셀 (예시)

```python
!pip -q install fastapi uvicorn requests
!git clone https://github.com/VAST-AI/TripoSR.git /content/TripoSR

%env TRIPOSR_REPO=/content/TripoSR
%env TRIPOSR_CMD=python run.py {image} --output-dir {output_dir} --model-save-format glb

!python /content/character_instance_segmentation/tools/triposr_server.py
```

필요하면 `TRIPOSR_CMD`를 실제 TripoSR 실행 커맨드에 맞게 수정하세요.

## 2) GLB → VRM 변환 (Blender Headless)

VRM Addon이 설치된 Blender가 필요합니다.

```bash
blender --background --python tools/blender_glb_to_vrm.py -- \
  --glb ./outputs/triposr/sample.glb --vrm ./outputs/vrm/sample.vrm
```

실패 시 `sample.vrm.log`에 `bone_missing`, `texture_missing`, `scale_not_unity` 같은 원인이 기록됩니다.

## 3) MCP Tool: png_to_vrm

TripoSR 래퍼 호출 → Blender 변환을 MCP Tool로 묶었습니다.

```bash
pip install mcp requests
python tools/mcp_png_to_vrm.py --host 0.0.0.0 --port 9333
```

환경 변수로 경로를 설정할 수 있습니다.

- `TRIPOSR_URL`: TripoSR 래퍼 URL (`http://localhost:8000/generate`)
- `TRIPOSR_RESPONSE_TYPE`: `path` 또는 `base64`
- `BLENDER_BIN`: Blender 실행 경로
- `BLENDER_VRM_SCRIPT`: Blender 변환 스크립트 경로
- `VRM_OUTPUT_DIR`: VRM 출력 폴더

Tool 입력: `png_path`

Tool 출력: `vrm_path`
