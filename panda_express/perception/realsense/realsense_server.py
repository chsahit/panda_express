"""RealSense camera server.

Keeps the camera continuously open and captures frames in a background thread.
Exposes a FastAPI server that:
  - GET /capture  — runs FoundationStereo on the latest IR pair and returns
                    an .npz blob with keys 'rgb' (H,W,3 uint8) and
                    'depth' (H,W float32, meters).
  - GET /intrinsics — returns color camera intrinsics as JSON.

Usage:
    python -m panda_express.perception.realsense.realsense_server \
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
    import pyrealsense2 as rs
    _REALSENSE_AVAILABLE = True
except ImportError:
    print("pyrealsense2 is not installed. Install with: pip install pyrealsense2")
    rs = None
    _REALSENSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared frame state
# ---------------------------------------------------------------------------

@dataclass
class FrameState:
    rgb: Optional[np.ndarray] = None          # (H, W, 3) uint8
    ir_left: Optional[np.ndarray] = None      # (H, W) uint8
    ir_right: Optional[np.ndarray] = None     # (H, W) uint8
    timestamp: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


_state = FrameState()

# Camera intrinsics (populated once at startup)
_K_ir: Optional[np.ndarray] = None
_K_color: Optional[np.ndarray] = None
_baseline: Optional[float] = None
_T_color_from_ir: Optional[np.ndarray] = None
_color_size: Optional[tuple[int, int]] = None  # (H, W)
_foundation_stereo_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Background capture thread
# ---------------------------------------------------------------------------

def _capture_loop(pipeline: "rs.pipeline", align: "rs.align") -> None:
    """Continuously pull frames from the pipeline and update _state."""
    print("Capture thread started.")
    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
        except Exception as e:
            print(f"[capture_loop] wait_for_frames error: {e}")
            time.sleep(0.1)
            continue

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        ir_left_frame = frames.get_infrared_frame(1)
        ir_right_frame = frames.get_infrared_frame(2)

        if not color_frame or not ir_left_frame or not ir_right_frame:
            continue

        bgr = np.asanyarray(color_frame.get_data())
        rgb = bgr[:, :, ::-1].copy()
        ir_l = np.asanyarray(ir_left_frame.get_data()).copy()
        ir_r = np.asanyarray(ir_right_frame.get_data()).copy()

        with _state.lock:
            _state.rgb = rgb
            _state.ir_left = ir_l
            _state.ir_right = ir_r
            _state.timestamp = time.time()


# ---------------------------------------------------------------------------
# FoundationStereo call + depth warp (same logic as RealSenseCamera)
# ---------------------------------------------------------------------------

def _call_foundation_stereo(ir_left: np.ndarray, ir_right: np.ndarray) -> np.ndarray:
    """Send IR stereo pair to FoundationStereo server; return depth in IR frame (meters)."""
    ir_left_rgb = np.stack([ir_left, ir_left, ir_left], axis=-1)
    ir_right_rgb = np.stack([ir_right, ir_right, ir_right], axis=-1)

    _, left_bytes = cv2.imencode(".png", ir_left_rgb)
    _, right_bytes = cv2.imencode(".png", ir_right_rgb)

    files = {
        "left_image": ("left.png", left_bytes.tobytes(), "image/png"),
        "right_image": ("right.png", right_bytes.tobytes(), "image/png"),
    }
    data = {
        "fx": float(_K_ir[0, 0]),
        "fy": float(_K_ir[1, 1]),
        "cx": float(_K_ir[0, 2]),
        "cy": float(_K_ir[1, 2]),
        "baseline": float(_baseline),
        "scale": 1.0,
        "hiera": 0,
        "valid_iters": 32,
    }

    response = requests.post(f"{_foundation_stereo_url}/infer", files=files, data=data, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"FoundationStereo server error: {response.status_code} - {response.text}"
        )
    buf = io.BytesIO(response.content)
    return np.load(buf)["depth"]  # (H, W) float32, meters in IR frame


def _depth_ir_to_color(depth_ir: np.ndarray) -> np.ndarray:
    """Warp IR-frame depth onto color pixel grid via forward splatting (z-buffer)."""
    Hc, Wc = _color_size
    Hi, Wi = depth_ir.shape

    fx_i, fy_i = float(_K_ir[0, 0]), float(_K_ir[1, 1])
    cx_i, cy_i = float(_K_ir[0, 2]), float(_K_ir[1, 2])
    fx_c, fy_c = float(_K_color[0, 0]), float(_K_color[1, 1])
    cx_c, cy_c = float(_K_color[0, 2]), float(_K_color[1, 2])

    u, v = np.meshgrid(np.arange(Wi, dtype=np.float32), np.arange(Hi, dtype=np.float32))
    z = depth_ir.astype(np.float32)
    valid = (z > 0.0) & np.isfinite(z)
    if not np.any(valid):
        return np.zeros((Hc, Wc), dtype=np.float32)

    x_i = (u[valid] - cx_i) / fx_i * z[valid]
    y_i = (v[valid] - cy_i) / fy_i * z[valid]
    pts_ir = np.stack([x_i, y_i, z[valid]], axis=0)

    R = _T_color_from_ir[:3, :3].astype(np.float32)
    t = _T_color_from_ir[:3, 3].astype(np.float32).reshape(3, 1)
    pts_c = R @ pts_ir + t
    Xc, Yc, Zc = pts_c[0], pts_c[1], pts_c[2]
    valid_c = Zc > 1e-6
    if not np.any(valid_c):
        return np.zeros((Hc, Wc), dtype=np.float32)
    Xc, Yc, Zc = Xc[valid_c], Yc[valid_c], Zc[valid_c]

    uc_f = fx_c * (Xc / Zc) + cx_c
    vc_f = fy_c * (Yc / Zc) + cy_c
    x0 = np.floor(uc_f).astype(np.int32)
    y0 = np.floor(vc_f).astype(np.int32)

    depth_color = np.full((Hc, Wc), np.inf, dtype=np.float32)

    def splat(ix, iy, zvals):
        inb = (ix >= 0) & (ix < Wc) & (iy >= 0) & (iy < Hc)
        if np.any(inb):
            np.minimum.at(depth_color, (iy[inb], ix[inb]), zvals[inb])

    splat(x0, y0, Zc)
    splat(x0 + 1, y0, Zc)
    splat(x0, y0 + 1, Zc)
    splat(x0 + 1, y0 + 1, Zc)

    holes = np.isinf(depth_color)
    if np.any(holes):
        depth_color[holes] = 0.0
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(5):
            holes_mask = depth_color <= 0.0
            if not np.any(holes_mask):
                break
            sentinel = np.where(depth_color > 0.0, depth_color, 65535.0).astype(np.float32)
            min_neigh = cv2.erode(sentinel, kernel)
            newly_filled = holes_mask & (min_neigh < 65000.0)
            depth_color[newly_filled] = min_neigh[newly_filled]
        depth_color[depth_color > 65000.0] = 0.0

    return depth_color


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RealSense Camera Server")


@app.get("/capture")
def capture() -> Response:
    """Return latest RGB + FoundationStereo depth as a numpy .npz binary response.

    The .npz contains:
      - 'rgb':   (H, W, 3) uint8
      - 'depth': (H, W) float32, metres, aligned to color frame
    """
    with _state.lock:
        if _state.rgb is None:
            raise HTTPException(status_code=503, detail="No frames captured yet")
        rgb = _state.rgb.copy()
        ir_left = _state.ir_left.copy()
        ir_right = _state.ir_right.copy()
        capture_time = _state.timestamp

    depth_ir = _call_foundation_stereo(ir_left, ir_right)
    depth_color = _depth_ir_to_color(depth_ir)

    buf = io.BytesIO()
    np.savez_compressed(buf, rgb=rgb, depth=depth_color,
                        capture_time=np.float64(capture_time))
    buf.seek(0)
    return Response(content=buf.read(), media_type="application/octet-stream")


@app.get("/intrinsics")
def intrinsics() -> dict:
    """Return color camera intrinsics as JSON."""
    return {
        "fx": float(_K_color[0, 0]),
        "fy": float(_K_color[1, 1]),
        "cx": float(_K_color[0, 2]),
        "cy": float(_K_color[1, 2]),
        "width": _color_size[1],
        "height": _color_size[0],
        "K": _K_color.tolist(),
    }


@app.get("/health")
def health() -> dict:
    with _state.lock:
        age = time.time() - _state.timestamp if _state.timestamp > 0 else None
    return {"status": "ok", "last_frame_age_s": age}


# ---------------------------------------------------------------------------
# Startup / main
# ---------------------------------------------------------------------------

def _start_camera(serial_number: Optional[str], host: str, port: int, ir_exposure: int = 0) -> None:
    global _K_ir, _K_color, _baseline, _T_color_from_ir, _color_size

    if not _REALSENSE_AVAILABLE:
        raise ImportError("pyrealsense2 not installed.")

    width, height, fps = 1280, 720, 6

    pipeline = rs.pipeline()
    config = rs.config()

    if serial_number is not None:
        config.enable_device(str(serial_number))
        print(f"Targeting RealSense camera: {serial_number}")

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)

    print(f"Starting RealSense camera ({width}x{height} @ {fps}fps)...")
    profile = pipeline.start(config)

    device = profile.get_device()
    actual_serial = device.get_info(rs.camera_info.serial_number)
    print(f"Connected: {device.get_info(rs.camera_info.name)} (serial {actual_serial})")

    # Configure IR sensor exposure
    stereo_sensor = device.first_depth_sensor()
    if ir_exposure > 0:
        stereo_sensor.set_option(rs.option.enable_auto_exposure, 0)
        stereo_sensor.set_option(rs.option.exposure, ir_exposure)
        print(f"IR exposure set to {ir_exposure} µs (auto-exposure off)")
    else:
        print(f"IR auto-exposure enabled (current: {stereo_sensor.get_option(rs.option.exposure)} µs)")

    color_stream = profile.get_stream(rs.stream.color)
    color_intr = color_stream.as_video_stream_profile().get_intrinsics()

    ir_left_profile = profile.get_stream(rs.stream.infrared, 1)
    ir_right_profile = profile.get_stream(rs.stream.infrared, 2)
    ir_intr = ir_left_profile.as_video_stream_profile().get_intrinsics()

    _K_ir = np.array([
        [ir_intr.fx, 0, ir_intr.ppx],
        [0, ir_intr.fy, ir_intr.ppy],
        [0, 0, 1],
    ], dtype=np.float32)

    extr = ir_left_profile.get_extrinsics_to(ir_right_profile)
    _baseline = np.linalg.norm(extr.translation)

    extr_color = ir_left_profile.get_extrinsics_to(color_stream)
    _T_color_from_ir = np.eye(4, dtype=np.float32)
    _T_color_from_ir[:3, :3] = np.array(extr_color.rotation).reshape(3, 3).T
    _T_color_from_ir[:3, 3] = np.array(extr_color.translation)

    _K_color = np.array([
        [color_intr.fx, 0, color_intr.ppx],
        [0, color_intr.fy, color_intr.ppy],
        [0, 0, 1],
    ], dtype=np.float32)

    _color_size = (height, width)

    align = rs.align(rs.stream.color)

    # Warm up
    print("Warming up camera (10 frames)...")
    for _ in range(10):
        pipeline.wait_for_frames()

    # Start background capture thread
    t = threading.Thread(target=_capture_loop, args=(pipeline, align), daemon=True)
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
    parser = argparse.ArgumentParser(description="RealSense camera server")
    parser.add_argument("--serial", default="231122071284", help="Camera serial number")
    parser.add_argument(
        "--foundation-stereo-url", default="http://localhost:1234",
        help="FoundationStereo server URL"
    )
    parser.add_argument("--ir-exposure", type=int, default=0,
                        help="IR sensor exposure in microseconds (lower = less motion blur). "
                             "Set to 0 to keep auto-exposure.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    _foundation_stereo_url = args.foundation_stereo_url
    _start_camera(args.serial, args.host, args.port, args.ir_exposure)
