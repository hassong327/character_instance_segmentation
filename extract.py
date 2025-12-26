import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image

from animeinsseg import AnimeInsSeg
from utils.constants import DEFAULT_DETECTOR_CKPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract anime character with alpha background")
    parser.add_argument("--img", required=True, help="Input image path (jpg/png)")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--ckpt", default=DEFAULT_DETECTOR_CKPT, help="Detector checkpoint path")
    parser.add_argument("--det-size", type=int, default=640, help="Detector input size")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Detection score threshold")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--refine", default="refinenet_isnet", help="refinenet_isnet/animeseg/none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not osp.isabs(args.ckpt):
        args.ckpt = osp.abspath(args.ckpt)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check GPU drivers/CUDA install.")
    if not osp.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    output_path = args.out
    if output_path is None:
        stem, _ = osp.splitext(osp.basename(args.img))
        output_path = osp.join(osp.dirname(args.img), f"{stem}_cutout.png")

    bgr = cv2.imread(args.img, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to load image: {args.img}")

    refine_method = args.refine if args.refine != "none" else None
    refine_kwargs = {"refine_method": refine_method} if refine_method else {"refine_method": "none"}

    try:
        model = AnimeInsSeg(ckpt=args.ckpt, device=args.device, refine_kwargs=refine_kwargs)
        instances = model.infer(bgr, pred_score_thr=args.score_thr, det_size=args.det_size, output_type="tensor")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError("CUDA out of memory. Try --det-size 512 or --refine none.") from exc
        raise

    if instances.is_empty:
        mask = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.uint8)
    else:
        masks = instances.masks
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        masks = masks.astype(np.uint8)
        if masks.ndim == 2:
            mask = masks
        else:
            mask = np.any(masks > 0, axis=0).astype(np.uint8)

    if mask.shape[:2] != (bgr.shape[0], bgr.shape[1]):
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    alpha = (mask * 255).astype(np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, alpha])

    os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(output_path)


if __name__ == "__main__":
    main()
