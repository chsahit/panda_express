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
from typing import Optional

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

try:
    import pyzed.sl as sl
    _ZED_AVAILABLE = True
except ImportError:
    print("pyzed is not installed. Install the ZED SDK Python API.")
    sl = None
    _ZED_AVAILABLE = False


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

def _start_camera(serial_number: Optional[int], host: str, port: int) -> None:
    global _K, _baseline, _image_size

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

    print(f"Camera ready. Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED camera server")
    parser.add_argument("--serial", type=int, default=None, help="Camera serial number")
    parser.add_argument(
        "--foundation-stereo-url", default="http://localhost:1234",
        help="FoundationStereo server URL",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    _foundation_stereo_url = args.foundation_stereo_url
    _start_camera(args.serial, args.host, args.port)
