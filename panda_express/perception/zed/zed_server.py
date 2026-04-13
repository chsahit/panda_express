"""ZED camera server.

Keeps the camera continuously open and captures frames in a background thread.
Uses dedicated sl.Mat buffers and copies data immediately after retrieval to
avoid tearing (see video_tool.py reference).

Exposes a FastAPI server that:
  - GET /capture  — runs FoundationStereo on the latest stereo pair and returns
                    an .npz blob with keys 'rgb' (H,W,3 uint8) and
                    'depth' (H,W float32, meters).
  - GET /intrinsics — returns left camera intrinsics as JSON.

Usage:
    python -m panda_express.perception.zed.zed_server \
        [--serial SERIAL] \
        [--foundation-stereo-url URL] \
        [--host HOST] \
        [--port PORT]
"""

import argparse
import io
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image

try:
    import pyzed.sl as sl
    _ZED_AVAILABLE = True
except ImportError:
    print("pyzed is not installed. Install the ZED SDK Python API.")
    sl = None
    _ZED_AVAILABLE = False

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    _SAM3_AVAILABLE = True
except ImportError:
    _SAM3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared frame state
# ---------------------------------------------------------------------------

@dataclass
class FrameState:
    rgb: Optional[np.ndarray] = None           # (H, W, 3) uint8
    left_bgr: Optional[np.ndarray] = None      # (H, W, 3) uint8
    right_bgr: Optional[np.ndarray] = None     # (H, W, 3) uint8
    timestamp: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


_state = FrameState()

# Camera intrinsics (populated once at startup)
_K: Optional[np.ndarray] = None
_baseline: Optional[float] = None
_image_size: Optional[tuple[int, int]] = None  # (H, W)
_foundation_stereo_url: Optional[str] = None
_gripper_mask: Optional[np.ndarray] = None  # (H, W) bool, True = gripper pixel


# ---------------------------------------------------------------------------
# Background capture thread
# ---------------------------------------------------------------------------

def _capture_loop(zed: "sl.Camera") -> None:
    """Continuously grab frames and update _state.

    Uses dedicated sl.Mat buffers to avoid tearing — data is copied out of
    the buffer immediately after retrieve, before the next grab() overwrites it.
    """
    print("Capture thread started.")
    runtime_params = sl.RuntimeParameters()
    left_buf = sl.Mat()
    right_buf = sl.Mat()

    while True:
        # grab() blocks until the next frame is ready from the sensor
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.01)
            continue

        zed.retrieve_image(left_buf, sl.VIEW.LEFT)
        zed.retrieve_image(right_buf, sl.VIEW.RIGHT)

        # Copy data out of the SDK buffers immediately to prevent tearing
        left_bgra = left_buf.get_data().copy()
        right_bgra = right_buf.get_data().copy()

        left_bgr = cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2BGR)
        right_bgr = cv2.cvtColor(right_bgra, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2RGB)

        with _state.lock:
            _state.rgb = rgb
            _state.left_bgr = left_bgr
            _state.right_bgr = right_bgr
            _state.timestamp = time.time()


# ---------------------------------------------------------------------------
# FoundationStereo call
# ---------------------------------------------------------------------------

def _call_foundation_stereo(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """Send stereo pair to FoundationStereo server; return depth (meters)."""
    _, left_bytes = cv2.imencode(".png", left_bgr)
    _, right_bytes = cv2.imencode(".png", right_bgr)

    files = {
        "left_image": ("left.png", left_bytes.tobytes(), "image/png"),
        "right_image": ("right.png", right_bytes.tobytes(), "image/png"),
    }
    data = {
        "fx": float(_K[0, 0]),
        "fy": float(_K[1, 1]),
        "cx": float(_K[0, 2]),
        "cy": float(_K[1, 2]),
        "baseline": float(_baseline),
        "scale": 1.0,
        "hiera": 0,
        "valid_iters": 32,
    }

    response = requests.post(
        f"{_foundation_stereo_url}/infer", files=files, data=data, timeout=60
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"FoundationStereo server error: {response.status_code} - {response.text}"
        )
    buf = io.BytesIO(response.content)
    return np.load(buf)["depth"]  # (H, W) float32, meters


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ZED Camera Server")


@app.get("/capture")
def capture() -> Response:
    """Return latest RGB + FoundationStereo depth as a numpy .npz binary response.

    The .npz contains:
      - 'rgb':   (H, W, 3) uint8
      - 'depth': (H, W) float32, metres
    """
    with _state.lock:
        if _state.rgb is None:
            raise HTTPException(status_code=503, detail="No frames captured yet")
        rgb = _state.rgb.copy()
        left_bgr = _state.left_bgr.copy()
        right_bgr = _state.right_bgr.copy()
        capture_time = _state.timestamp

    depth = _call_foundation_stereo(left_bgr, right_bgr)

    # Mask out gripper pixels in depth
    if _gripper_mask is not None:
        depth[_gripper_mask] = 0.0

    buf = io.BytesIO()
    np.savez_compressed(buf, rgb=rgb, depth=depth,
                        capture_time=np.float64(capture_time))
    buf.seek(0)
    return Response(content=buf.read(), media_type="application/octet-stream")


@app.get("/intrinsics")
def intrinsics() -> dict:
    """Return left camera intrinsics as JSON."""
    return {
        "fx": float(_K[0, 0]),
        "fy": float(_K[1, 1]),
        "cx": float(_K[0, 2]),
        "cy": float(_K[1, 2]),
        "width": _image_size[1],
        "height": _image_size[0],
        "K": _K.tolist(),
    }


@app.get("/health")
def health() -> dict:
    with _state.lock:
        age = time.time() - _state.timestamp if _state.timestamp > 0 else None
    return {"status": "ok", "last_frame_age_s": age}


# ---------------------------------------------------------------------------
# Startup / main
# ---------------------------------------------------------------------------

def _load_gripper_mask(mask_path: Path) -> Optional[np.ndarray]:
    """Load gripper mask from a PNG file. Returns (H, W) bool array or None."""
    if not mask_path.exists():
        return None
    mask = np.array(Image.open(mask_path)).astype(bool)
    print(f"Loaded gripper mask from {mask_path} (shape {mask.shape})")
    return mask


def _build_sam3_processor() -> "Sam3Processor":
    """Build the SAM3 model and processor (expensive, do once)."""
    import os
    import sam3 as sam3_mod

    modules_path = os.path.dirname(os.path.dirname(sam3_mod.__file__))
    bpe_path = os.path.join(modules_path, "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    print(f"Building SAM3 model (bpe_path={bpe_path})...")
    sam3_model = build_sam3_image_model(bpe_path=bpe_path)
    return Sam3Processor(sam3_model, confidence_threshold=0.1)


def _bbox_from_mask(mask: np.ndarray, padding: float = 0.15) -> list[float]:
    """Derive a normalized [cx, cy, w, h] bounding box from a bool mask with padding."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    h_img, w_img = mask.shape
    # Add padding
    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding
    x_min = max(0, x_min - pad_x)
    x_max = min(w_img - 1, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h_img - 1, y_max + pad_y)

    # Normalize to [0, 1]
    cx = (x_min + x_max) / 2.0 / w_img
    cy = (y_min + y_max) / 2.0 / h_img
    w = (x_max - x_min) / w_img
    h = (y_max - y_min) / h_img
    return [cx, cy, w, h]


def _run_sam3_mask(processor: "Sam3Processor", rgb: np.ndarray,
                   box_hint: Optional[list[float]] = None) -> np.ndarray:
    """Run SAM3 inference on an RGB frame. Returns (H, W) bool mask.

    Args:
        box_hint: optional [cx, cy, w, h] normalized bounding box to guide segmentation.
    """
    pil_image = Image.fromarray(rgb)
    inference_state = processor.set_image(pil_image)

    # Set text prompt first, then refine with box
    inference_state = processor.set_text_prompt(
        state=inference_state, prompt="black machine"
    )
    if box_hint is not None:
        inference_state = processor.add_geometric_prompt(
            box=box_hint, label=True, state=inference_state
        )

    scores = inference_state.get("scores")
    masks = inference_state.get("masks")  # [N, 1, H, W] bool

    if masks is not None and masks.numel() > 0:
        if scores is not None:
            import torch
            best_idx = scores.argmax().item()
            combined = masks[best_idx, 0].cpu().numpy()
        else:
            combined = masks.squeeze(1).any(dim=0).cpu().numpy()
    else:
        combined = np.zeros(rgb.shape[:2], dtype=bool)

    return combined


def _sam3_mask_loop(processor: "Sam3Processor", interval: float,
                    box_hint: Optional[list[float]] = None) -> None:
    """Background thread: periodically re-segment the gripper and update _gripper_mask."""
    global _gripper_mask
    while True:
        time.sleep(interval)
        with _state.lock:
            rgb = _state.rgb
        if rgb is None:
            continue
        try:
            _gripper_mask = _run_sam3_mask(processor, rgb.copy(), box_hint=box_hint)
        except Exception as e:
            print(f"SAM3 mask update failed: {e}")


def _start_camera(serial_number: Optional[int], host: str, port: int,
                   gripper_mask_file: Optional[str] = None,
                   use_sam3_mask: bool = False,
                   sam3_mask_interval: float = 5.0) -> None:
    global _K, _baseline, _image_size, _gripper_mask

    if not _ZED_AVAILABLE:
        raise ImportError("pyzed (ZED SDK Python API) not installed.")

    init_params = sl.InitParameters(
        depth_minimum_distance=0.1,
        camera_resolution=sl.RESOLUTION.HD1080,
        depth_stabilization=False,
        camera_fps=30,
        camera_image_flip=sl.FLIP_MODE.OFF,
    )

    import os
    init_params.optional_settings_path = os.path.expanduser("~/.stereolabs/settings/")

    if serial_number is not None:
        init_params.set_from_serial_number(serial_number)
        print(f"Targeting ZED camera: {serial_number}")

    if serial_number is not None:
        zed = sl.Camera(serial_number)
    else:
        zed = sl.Camera()

    print("Opening ZED camera...")
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED: {status}")

    cam_info = zed.get_camera_information()
    actual_serial = cam_info.serial_number
    print(f"Connected: ZED (serial {actual_serial})")

    if serial_number is not None:
        assert actual_serial == serial_number, (
            f"Expected serial {serial_number}, got {actual_serial}"
        )

    # Extract intrinsics and baseline
    calib = cam_info.camera_configuration.calibration_parameters
    left_cam = calib.left_cam
    _baseline = calib.get_camera_baseline() / 1000.0  # mm -> meters

    _K = np.array([
        [left_cam.fx, 0, left_cam.cx],
        [0, left_cam.fy, left_cam.cy],
        [0, 0, 1],
    ], dtype=np.float64)

    resolution = cam_info.camera_configuration.resolution
    _image_size = (resolution.height, resolution.width)

    # Start background capture thread
    t = threading.Thread(target=_capture_loop, args=(zed,), daemon=True)
    t.start()

    # Wait for at least one frame before serving
    print("Waiting for first frame...")
    deadline = time.time() + 10.0
    while time.time() < deadline:
        with _state.lock:
            ready = _state.rgb is not None
        if ready:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("Timed out waiting for first frame from camera.")

    # Load gripper mask
    if use_sam3_mask:
        if not _SAM3_AVAILABLE:
            raise ImportError("sam3 is required for --use-sam3-mask but is not installed.")
        processor = _build_sam3_processor()
        # Derive box hint from static mask if available
        box_hint = None
        reference_mask_path = Path(__file__).parent / "open_only_gripper_mask.png"
        if gripper_mask_file is not None:
            reference_mask_path = Path(gripper_mask_file)
        if reference_mask_path.exists():
            ref_mask = np.array(Image.open(reference_mask_path)).astype(bool)
            if ref_mask.any():
                box_hint = _bbox_from_mask(ref_mask)
                print(f"SAM3 box hint from {reference_mask_path.name}: {box_hint}")
        with _state.lock:
            rgb_for_mask = _state.rgb.copy()
        _gripper_mask = _run_sam3_mask(processor, rgb_for_mask, box_hint=box_hint)
        # Start background thread to keep the mask up-to-date
        mask_thread = threading.Thread(
            target=_sam3_mask_loop,
            args=(processor, sam3_mask_interval),
            kwargs={"box_hint": box_hint},
            daemon=True,
        )
        mask_thread.start()
        print(f"SAM3 mask refresh thread started (interval={sam3_mask_interval}s)")
    elif gripper_mask_file is not None:
        _gripper_mask = _load_gripper_mask(Path(gripper_mask_file))
    else:
        # Default: look for gripper_mask.png next to this file
        default_path = Path(__file__).parent / "gripper_mask.png"
        _gripper_mask = _load_gripper_mask(default_path)

    print(f"Camera ready. Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED camera server")
    parser.add_argument("--serial", type=int, default=None, help="Camera serial number")
    parser.add_argument(
        "--foundation-stereo-url", default="http://localhost:1234",
        help="FoundationStereo server URL",
    )
    parser.add_argument("--gripper-mask", default=None,
                        help="Path to gripper mask PNG (default: gripper_mask.png next to this file)")
    parser.add_argument("--use-sam3-mask", action="store_true",
                        help="Use SAM3 to dynamically segment the gripper "
                             "instead of loading a static PNG mask")
    parser.add_argument("--sam3-mask-interval", type=float, default=5.0,
                        help="Seconds between SAM3 mask refreshes (default: 5)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    _foundation_stereo_url = args.foundation_stereo_url
    _start_camera(args.serial, args.host, args.port, args.gripper_mask,
                   use_sam3_mask=args.use_sam3_mask,
                   sam3_mask_interval=args.sam3_mask_interval)
