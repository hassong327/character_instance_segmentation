from __future__ import annotations

import base64
import os
import shlex
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

app = FastAPI(title="TripoSR Wrapper")


class GenerateResponse(BaseModel):
    glb_path: str | None = None
    glb_base64: str | None = None


def build_triposr_command(png_path: Path, glb_path: Path) -> tuple[list[str], str | None]:
    template = os.getenv("TRIPOSR_CMD")
    if template:
        parts = [
            token.format(image=str(png_path), output=str(glb_path))
            for token in shlex.split(template)
        ]
        return parts, os.getenv("TRIPOSR_CMD_CWD")

    python_bin = os.getenv("TRIPOSR_PYTHON", "python")
    repo_dir = os.getenv("TRIPOSR_REPO", "./TripoSR")
    return (
        [python_bin, "run.py", "--image", str(png_path), "--output", str(glb_path)],
        repo_dir,
    )


def run_triposr(png_path: Path, glb_path: Path) -> None:
    command, cwd = build_triposr_command(png_path, glb_path)
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "TripoSR execution failed",
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
            },
        )
    if not glb_path.exists():
        raise HTTPException(
            status_code=500,
            detail="TripoSR completed but GLB output not found",
        )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_glb(
    file: UploadFile = File(...),
    response_type: str = Query("path", pattern="^(path|base64)$"),
) -> GenerateResponse:
    if file.content_type and file.content_type != "image/png":
        raise HTTPException(status_code=400, detail="Only PNG uploads are supported")

    request_id = uuid.uuid4().hex
    output_dir = Path(os.getenv("TRIPOSR_OUTPUT_DIR", "./outputs/triposr")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{request_id}.png"
    glb_path = output_dir / f"{request_id}.glb"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")
    png_path.write_bytes(content)

    try:
        run_triposr(png_path, glb_path)
    finally:
        if png_path.exists():
            png_path.unlink()

    if response_type == "base64":
        encoded = base64.b64encode(glb_path.read_bytes()).decode("utf-8")
        return GenerateResponse(glb_base64=encoded)

    return GenerateResponse(glb_path=str(glb_path))


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("TRIPOSR_HOST", "0.0.0.0")
    port = int(os.getenv("TRIPOSR_PORT", "8000"))
    uvicorn.run("tools.triposr_server:app", host=host, port=port, reload=False)
