from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Convert GLB to VRM")
    parser.add_argument("--glb", required=True)
    parser.add_argument("--vrm", required=True)
    parser.add_argument("--log")
    return parser.parse_args(argv)


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def analyze_scene() -> list[str]:
    issues: list[str] = []
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        issues.append("bone_missing")
    elif all(len(armature.data.bones) == 0 for armature in armatures):
        issues.append("bone_empty")

    images = [img for img in bpy.data.images if img.name not in {"Render Result", "Viewer Node"}]
    if not images:
        issues.append("texture_missing")
    else:
        missing_files = [
            img for img in images if img.filepath and not os.path.exists(bpy.path.abspath(img.filepath))
        ]
        if missing_files:
            issues.append("texture_file_missing")

    scales = [
        obj.scale[:] for obj in bpy.data.objects if obj.type in {"MESH", "ARMATURE"}
    ]
    if scales:
        max_scale = max(max(abs(value) for value in scale) for scale in scales)
        min_scale = min(min(abs(value) for value in scale) for scale in scales)
        if max_scale > 10 or min_scale < 0.1:
            issues.append("scale_out_of_range")
        elif any(any(abs(value - 1) > 0.1 for value in scale) for scale in scales):
            issues.append("scale_not_unity")

    return issues


def main() -> None:
    args = parse_args()
    glb_path = Path(args.glb).resolve()
    vrm_path = Path(args.vrm).resolve()
    log_path = Path(args.log) if args.log else vrm_path.with_suffix(vrm_path.suffix + ".log")
    configure_logging(log_path)

    if not glb_path.exists():
        logging.error("glb_not_found: %s", glb_path)
        sys.exit(1)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.gltf(filepath=str(glb_path))
    except Exception as exc:
        logging.exception("import_failed: %s", exc)
        sys.exit(1)

    issues = analyze_scene()
    if issues:
        logging.error("preflight_failed: %s", ", ".join(issues))
        sys.exit(2)

    try:
        result = bpy.ops.export_scene.vrm(filepath=str(vrm_path))
    except AttributeError:
        logging.error("vrm_export_operator_missing")
        sys.exit(3)
    except Exception as exc:
        logging.exception("export_failed: %s", exc)
        sys.exit(3)

    if "FINISHED" not in result:
        logging.error("export_failed: blender returned %s", result)
        sys.exit(3)

    logging.info("export_success: %s", vrm_path)


if __name__ == "__main__":
    main()
