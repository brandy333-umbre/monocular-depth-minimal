#!/usr/bin/env python3
"""
Minimal monocular depth inference using MiDaS.

Supports:
- single image
- directory of images
- video file
- webcam (input=0)

Outputs:
- raw depth (16-bit PNG)  -> *_depth16.png
- colorized preview (8-bit PNG) -> *_depth_color.png
Optionally writes video outputs if input is a video/webcam.

Notes:
- MiDaS outputs "relative depth" (no absolute metric scale).
- Depth values are normalized per-frame for visualization.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union, List

import cv2
import numpy as np
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def list_images_in_dir(d: Path) -> List[Path]:
    files = [p for p in sorted(d.iterdir()) if p.is_file() and is_image_file(p)]
    return files


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_midas(model_type: str, device: str):
    """
    model_type: one of {"DPT_Large", "DPT_Hybrid", "MiDaS_small"}
    """
    # MiDaS repo via torch.hub
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    midas.to(device)
    return midas, transform


@torch.inference_mode()
def predict_depth(
    midas,
    transform,
    frame_bgr: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    Returns depth as float32 HxW in "relative depth" units.
    """
    # MiDaS expects RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    prediction = midas(input_batch)

    # Resize to original resolution
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame_bgr.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)

    depth = prediction.squeeze().detach().float().cpu().numpy()
    return depth


def depth_to_uint16(depth: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Convert depth float32 to 16-bit for saving.
    This uses per-frame min/max normalization (relative).
    """
    d = depth.astype(np.float32)
    d_min, d_max = float(np.min(d)), float(np.max(d))
    if abs(d_max - d_min) < eps:
        return np.zeros_like(d, dtype=np.uint16)
    d_norm = (d - d_min) / (d_max - d_min)
    d16 = (d_norm * 65535.0).clip(0, 65535).astype(np.uint16)
    return d16


def depth_to_color(depth: np.ndarray, cmap: int = cv2.COLORMAP_TURBO, eps: float = 1e-6) -> np.ndarray:
    """
    Create a colorized preview image from depth.
    Uses per-frame min/max normalization.
    """
    d = depth.astype(np.float32)
    d_min, d_max = float(np.min(d)), float(np.max(d))
    if abs(d_max - d_min) < eps:
        gray8 = np.zeros_like(d, dtype=np.uint8)
    else:
        d_norm = (d - d_min) / (d_max - d_min)
        gray8 = (d_norm * 255.0).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray8, cmap)
    return color


def save_depth_outputs(
    out_dir: Path,
    stem: str,
    depth: np.ndarray,
) -> Tuple[Path, Path]:
    """
    Saves:
    - <stem>_depth16.png (uint16)
    - <stem>_depth_color.png (uint8 BGR)
    Returns (path16, pathcolor).
    """
    ensure_dir(out_dir)
    depth16 = depth_to_uint16(depth)
    color = depth_to_color(depth)

    p16 = out_dir / f"{stem}_depth16.png"
    pc = out_dir / f"{stem}_depth_color.png"

    # cv2.imwrite supports 16-bit PNG if array is uint16
    cv2.imwrite(str(p16), depth16)
    cv2.imwrite(str(pc), color)
    return p16, pc


def open_video_source(input_arg: str) -> cv2.VideoCapture:
    """
    input_arg may be:
    - "0" (or any digit) for webcam index
    - path to a video file
    """
    if input_arg.isdigit():
        cap = cv2.VideoCapture(int(input_arg))
    else:
        cap = cv2.VideoCapture(input_arg)
    return cap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal monocular depth inference with MiDaS.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to image | directory | video, or webcam index (e.g., 0).",
    )
    parser.add_argument(
        "--out",
        default="sample_output",
        help="Output directory for depth results.",
    )
    parser.add_argument(
        "--model",
        default="MiDaS_small",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="MiDaS model type. MiDaS_small is fastest.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display live preview (for video/webcam) or image windows.",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="If input is video/webcam, also save a side-by-side video output.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Max frames to process for video/webcam. -1 = no limit.",
    )
    parser.add_argument(
        "--every_n",
        type=int,
        default=1,
        help="Process every Nth frame for video/webcam (speed control).",
    )
    return parser.parse_args()


def resolve_device(device_choice: str) -> str:
    if device_choice == "cpu":
        return "cpu"
    if device_choice == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def process_image_file(midas, transform, device: str, img_path: Path, out_dir: Path, display: bool) -> None:
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[WARN] Could not read image: {img_path}")
        return
    depth = predict_depth(midas, transform, frame, device)
    p16, pc = save_depth_outputs(out_dir, img_path.stem, depth)

    print(f"[OK] {img_path.name} -> {p16.name}, {pc.name}")

    if display:
        color = cv2.imread(str(pc))
        cv2.imshow("Input", frame)
        cv2.imshow("Depth (color)", color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_image_dir(midas, transform, device: str, dir_path: Path, out_dir: Path) -> None:
    images = list_images_in_dir(dir_path)
    if not images:
        print(f"[WARN] No images found in: {dir_path}")
        return
    for p in images:
        process_image_file(midas, transform, device, p, out_dir, display=False)


def process_video(
    midas,
    transform,
    device: str,
    input_arg: str,
    out_dir: Path,
    display: bool,
    save_video: bool,
    max_frames: int,
    every_n: int,
) -> None:
    cap = open_video_source(input_arg)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/webcam source: {input_arg}")

    ensure_dir(out_dir)

    writer = None
    out_video_path = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if save_video:
        out_video_path = out_dir / "depth_side_by_side.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, float(fps), (frame_w * 2, frame_h))

    frame_idx = 0
    processed = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if every_n > 1 and (frame_idx % every_n != 0):
                continue

            depth = predict_depth(midas, transform, frame, device)
            depth_color = depth_to_color(depth)

            # side-by-side preview
            sbs = np.concatenate([frame, depth_color], axis=1)

            if writer is not None:
                writer.write(sbs)

            if display:
                cv2.imshow("Depth (Input | Colorized)", sbs)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            if max_frames > 0 and processed >= max_frames:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    dt = time.time() - t0
    fps_eff = processed / dt if dt > 0 else 0.0
    print(f"[DONE] Processed {processed} frames in {dt:.2f}s ({fps_eff:.2f} FPS effective).")
    if out_video_path is not None:
        print(f"[OK] Saved video: {out_video_path}")


def main() -> int:
    args = parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)

    device = resolve_device(args.device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading MiDaS model: {args.model}")

    midas, transform = load_midas(args.model, device)

    # Decide input type
    if args.input.isdigit():
        # webcam
        process_video(
            midas, transform, device,
            input_arg=args.input,
            out_dir=out_dir,
            display=args.display,
            save_video=args.save_video,
            max_frames=args.max_frames,
            every_n=max(1, args.every_n),
        )
        return 0

    if in_path.exists() and in_path.is_file() and is_image_file(in_path):
        process_image_file(midas, transform, device, in_path, out_dir, display=args.display)
        return 0

    if in_path.exists() and in_path.is_dir():
        process_image_dir(midas, transform, device, in_path, out_dir)
        print(f"[OK] Saved outputs to: {out_dir}")
        return 0

    # assume video path
    process_video(
        midas, transform, device,
        input_arg=str(in_path),
        out_dir=out_dir,
        display=args.display,
        save_video=args.save_video,
        max_frames=args.max_frames,
        every_n=max(1, args.every_n),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
        raise SystemExit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
