from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
from pathlib import Path

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("png_to_vrm")


def request_glb(png_path: Path) -> Path:
    tripo_url = os.getenv("TRIPOSR_URL", "http://localhost:8000/generate")
    response_type = os.getenv("TRIPOSR_RESPONSE_TYPE", "path")
    with png_path.open("rb") as handle:
        response = requests.post(
            tripo_url,
            params={"response_type": response_type},
            files={"file": (png_path.name, handle, "image/png")},
            timeout=300,
        )
    response.raise_for_status()
    payload = response.json()
    if response_type == "base64":
        output_dir = Path(os.getenv("TRIPOSR_OUTPUT_DIR", "./outputs/triposr"))
        output_dir.mkdir(parents=True, exist_ok=True)
        glb_path = output_dir / f"{png_path.stem}.glb"
        glb_path.write_bytes(base64.b64decode(payload["glb_base64"]))
        return glb_path

    if "glb_path" not in payload:
        raise ValueError("TripoSR response missing glb_path")
    return Path(payload["glb_path"]).resolve()


def run_blender(glb_path: Path, vrm_path: Path) -> None:
    blender_bin = os.getenv("BLENDER_BIN", "blender")
    script_path = Path(os.getenv("BLENDER_VRM_SCRIPT", "tools/blender_glb_to_vrm.py")).resolve()
    command = [
        blender_bin,
        "--background",
        "--python",
        str(script_path),
        "--",
        "--glb",
        str(glb_path),
        "--vrm",
        str(vrm_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = {
            "message": "Blender conversion failed",
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:],
        }
        raise RuntimeError(json.dumps(message, ensure_ascii=False))


@mcp.tool()
def png_to_vrm(png_path: str) -> str:
    input_path = Path(png_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"PNG not found: {input_path}")

    output_dir = Path(os.getenv("VRM_OUTPUT_DIR", "./outputs/vrm"))
    output_dir.mkdir(parents=True, exist_ok=True)
    vrm_path = output_dir / f"{input_path.stem}.vrm"

    glb_path = request_glb(input_path)
    run_blender(glb_path, vrm_path)
    return str(vrm_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP png_to_vrm tool server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9333)
    args = parser.parse_args()
    mcp.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
