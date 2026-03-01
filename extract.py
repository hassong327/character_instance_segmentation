import argparse
import os
import os.path as osp
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from animeinsseg import AnimeInsSeg
from utils.constants import DEFAULT_DETECTOR_CKPT

VALID_REFINE_METHODS = ("refinenet_isnet", "animeseg", "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract character segmentation mask and RGBA cutout")
    parser.add_argument("--img", required=True, help="Input image path (jpg/png)")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--ckpt", default=DEFAULT_DETECTOR_CKPT, help="Detector checkpoint path")
    parser.add_argument("--det-size", type=int, default=640, help="Detector input size")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Detection score threshold")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Inference device")
    parser.add_argument("--refine", default="refinenet_isnet", choices=VALID_REFINE_METHODS, help="Mask refine method")
    return parser.parse_args()


def resolve_checkpoint_path(ckpt_path: str) -> str:
    abs_ckpt_path = osp.abspath(ckpt_path) if not osp.isabs(ckpt_path) else ckpt_path
    if not osp.isfile(abs_ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {abs_ckpt_path}")
    return abs_ckpt_path


def resolve_output_path(image_path: str, output_path: Optional[str]) -> str:
    if output_path:
        return output_path
    stem, _ = osp.splitext(osp.basename(image_path))
    return osp.join(osp.dirname(image_path), f"{stem}_cutout.png")


def compose_instance_mask(instances, image_shape: tuple[int, int]) -> np.ndarray:
    if instances.is_empty:
        return np.zeros(image_shape, dtype=np.uint8)

    masks = instances.masks
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    masks = masks.astype(np.uint8)

    if masks.ndim == 2:
        merged = masks
    else:
        merged = np.any(masks > 0, axis=0).astype(np.uint8)
    return merged


def save_cutout_png(bgr: np.ndarray, mask: np.ndarray, output_path: str) -> None:
    if mask.shape[:2] != bgr.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    alpha = (mask * 255).astype(np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack((rgb, alpha))

    os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(output_path)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check GPU drivers/CUDA install.")

    ckpt_path = resolve_checkpoint_path(args.ckpt)
    output_path = resolve_output_path(args.img, args.out)

    bgr = cv2.imread(args.img, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to load image: {args.img}")

    refine_kwargs = {"refine_method": args.refine}
    try:
        model = AnimeInsSeg(ckpt=ckpt_path, device=args.device, refine_kwargs=refine_kwargs)
        instances = model.infer(bgr, pred_score_thr=args.score_thr, det_size=args.det_size, output_type="tensor")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError("CUDA out of memory. Try --det-size 512 or --refine none.") from exc
        raise

    mask = compose_instance_mask(instances, (bgr.shape[0], bgr.shape[1]))
    save_cutout_png(bgr, mask, output_path)


if __name__ == "__main__":
    main()
