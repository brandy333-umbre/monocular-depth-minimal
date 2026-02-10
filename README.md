# monocular-depth-minimal

This repo provides a **minimal, inspectable** monocular depth estimation pipeline using a **pretrained MiDaS** model.

✅ What it does
- Runs **depth inference** on:
  - a single image
  - a folder of images
  - a video file
  - a webcam feed
- Saves:
  - `*_depth16.png` (16-bit depth map, normalized per-frame)
  - `*_depth_color.png` (colorized visualization)
