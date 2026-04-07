"""Manually paint gripper mask with interactive drawing tool.

Captures one frame from the ZED server and opens an interactive OpenCV window
where you can paint/erase the gripper region. The resulting binary mask is
saved as gripper_mask.png next to this file.

Usage:
    python -m panda_express.perception.zed.paint_gripper_mask \
        [--server-url http://localhost:8765] \
        [--brush-size 20] \
        [--dilation-iters 8]
"""

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from scipy.ndimage import binary_dilation, binary_fill_holes

gripper_mask_path = Path(__file__).parent / "gripper_mask.png"


class MaskPainter:
    """Interactive mask painting tool using OpenCV."""

    def __init__(self, rgb: np.ndarray, brush_size: int = 20):
        self.rgb = rgb
        self.mask = np.zeros(rgb.shape[:2], dtype=bool)
        self.brush_size = brush_size
        self.drawing = False
        self.mode = "draw"  # 'draw' or 'erase'

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self._apply(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._apply(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def _apply(self, x: int, y: int):
        h, w = self.mask.shape
        r = self.brush_size // 2
        y1, y2 = max(0, y - r), min(h, y + r)
        x1, x2 = max(0, x - r), min(w, x + r)
        yy, xx = np.ogrid[y1:y2, x1:x2]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
        self.mask[y1:y2, x1:x2][circle] = (self.mode == "draw")

    def get_overlay(self) -> np.ndarray:
        overlay = self.rgb.copy()
        overlay[self.mask] = (overlay[self.mask] * 0.7 + np.array([255, 0, 0]) * 0.3).astype(np.uint8)
        return overlay


def paint_gripper_mask(server_url: str = "http://localhost:8765", brush_size: int = 20, dilation_iters: int = 8):
    # Grab one frame from the ZED server
    print(f"Fetching frame from {server_url}/capture ...")
    resp = requests.get(f"{server_url}/capture", timeout=60)
    resp.raise_for_status()
    data = np.load(io.BytesIO(resp.content))
    rgb = data["rgb"]
    print(f"Got frame: {rgb.shape}")

    painter = MaskPainter(rgb, brush_size=brush_size)

    # Load existing mask if available
    if gripper_mask_path.exists():
        existing = np.array(Image.open(gripper_mask_path)).astype(bool)
        if existing.shape == painter.mask.shape:
            painter.mask = existing
            print("Loaded existing mask.")
        else:
            print(f"Existing mask shape {existing.shape} != image {painter.mask.shape}, starting fresh.")

    window = "Paint Gripper Mask"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, painter.mouse_callback)

    print("Controls:")
    print("  Left Click+Drag: Paint/Erase")
    print("  e: Toggle draw/erase")
    print("  +/-: Brush size")
    print("  f: Fill holes  |  d: Dilate  |  c: Clear")
    print("  y: Save & exit  |  n/q: Exit without saving")

    save = False
    while True:
        overlay = painter.get_overlay()
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        mode_text = "DRAW" if painter.mode == "draw" else "ERASE"
        color = (0, 255, 0) if painter.mode == "draw" else (0, 0, 255)
        cv2.putText(overlay_bgr, f"Mode: {mode_text}  Brush: {painter.brush_size}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(overlay_bgr, "y:save  n:cancel  e:toggle  f:fill  d:dilate  c:clear",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(window, overlay_bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("y"):
            save = True
            break
        elif key in (ord("n"), ord("q")):
            break
        elif key == ord("e"):
            painter.mode = "erase" if painter.mode == "draw" else "draw"
        elif key in (ord("+"), ord("=")):
            painter.brush_size = min(100, painter.brush_size + 5)
        elif key in (ord("-"), ord("_")):
            painter.brush_size = max(5, painter.brush_size - 5)
        elif key == ord("f"):
            painter.mask = binary_fill_holes(painter.mask)
        elif key == ord("d"):
            painter.mask = binary_dilation(painter.mask, iterations=dilation_iters)
        elif key == ord("c"):
            painter.mask = np.zeros_like(painter.mask)

    cv2.destroyAllWindows()

    if save:
        mask_image = Image.fromarray((painter.mask.astype(np.uint8) * 255))
        mask_image.save(gripper_mask_path)
        print(f"Saved gripper mask to {gripper_mask_path}")
    else:
        print("Mask not saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paint gripper mask for ZED camera")
    parser.add_argument("--server-url", default="http://localhost:8765")
    parser.add_argument("--brush-size", type=int, default=20)
    parser.add_argument("--dilation-iters", type=int, default=8)
    args = parser.parse_args()
    paint_gripper_mask(args.server_url, args.brush_size, args.dilation_iters)
