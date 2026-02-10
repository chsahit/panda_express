"""RealSense camera wrapper compatible with ZedCamera interface."""
import io

import cv2
import numpy as np
import requests

try:
    import pyrealsense2 as rs
    _REALSENSE_AVAILABLE = True
except ImportError:
    print('pyrealsense2 is not installed. Install with: pip install pyrealsense2')
    rs = None
    _REALSENSE_AVAILABLE = False


class RealSenseCamera:
    """RealSense camera wrapper compatible with ZedCamera interface for calibration."""

    def __init__(self, serial_number: str | None = None, foundation_stereo_url: str | None = None):
        """
        Initialize RealSense camera.

        Args:
            serial_number: Camera serial number (None for first available)
            foundation_stereo_url: URL for FoundationStereo depth inference server
        """
        if not _REALSENSE_AVAILABLE:
            raise ImportError("RealSense SDK not available. Install pyrealsense2.")

        self.serial_number = serial_number
        self._foundation_stereo_url = foundation_stereo_url
        self.width = 1280
        self.height = 720
        self.fps = 6  # D435 max fps at 1280x720 for depth is 6fps

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()

        if serial_number is not None:
            config.enable_device(str(serial_number))
            print(f"Targeting RealSense camera: {serial_number}")

        # Configure streams at 1280x720 @ 6fps (max supported by D435 at this resolution)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
        config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)

        # Start streaming
        print(f"Starting RealSense camera ({self.width}x{self.height} @ {self.fps}fps)...")
        profile = self.pipeline.start(config)

        # Get device info
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        self.actual_serial = device.get_info(rs.camera_info.serial_number)
        print(f'Connected to RealSense: {device_name} (Serial: {self.actual_serial})')

        # Get color intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # Get IR intrinsics and stereo baseline
        ir_left_profile = profile.get_stream(rs.stream.infrared, 1)
        ir_right_profile = profile.get_stream(rs.stream.infrared, 2)
        ir_intr = ir_left_profile.as_video_stream_profile().get_intrinsics()
        self._K_ir = np.array([
            [ir_intr.fx, 0, ir_intr.ppx],
            [0, ir_intr.fy, ir_intr.ppy],
            [0, 0, 1],
        ], dtype=np.float32)

        extr = ir_left_profile.get_extrinsics_to(ir_right_profile)
        self._baseline = np.linalg.norm(extr.translation)

        # IR-left to color extrinsics (for warping depth to color frame)
        extr_color = ir_left_profile.get_extrinsics_to(color_stream)
        self._T_color_from_ir = np.eye(4, dtype=np.float32)
        self._T_color_from_ir[:3, :3] = np.array(extr_color.rotation).reshape(3, 3).T
        self._T_color_from_ir[:3, 3] = np.array(extr_color.translation)

        self._K_color = np.array([
            [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
            [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
            [0, 0, 1],
        ], dtype=np.float32)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Create persistent buffers
        self._image_buffer = None
        self._depth_buffer = None
        self._runtime_params = None

        # Warm up camera
        for _ in range(10):
            self.pipeline.wait_for_frames()

        # Initial frame
        self.get_bgra_frame()

    def get_bgra_frame(self) -> np.ndarray:
        """Get BGRA frame (compatible with ZedCamera interface)."""
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if not color_frame:
            raise RuntimeError("Failed to get color frame")

        # Get BGR image
        bgr_image = np.asanyarray(color_frame.get_data())

        # Convert BGR to BGRA (add alpha channel with full opacity)
        bgra_image = np.dstack([bgr_image, np.ones((self.height, self.width), dtype=np.uint8) * 255])

        self._image_buffer = bgra_image
        return self._image_buffer

    def get_depth_frame(self) -> np.ndarray:
        """Get depth frame in meters."""
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()

        if not depth_frame:
            raise RuntimeError("Failed to get depth frame")

        # Get depth in uint16 millimeters, convert to float meters
        depth_mm = np.asanyarray(depth_frame.get_data())
        depth_m = depth_mm.astype(np.float32) / 1000.0

        self._depth_buffer = depth_m
        return self._depth_buffer

    def get_foundation_depth_frame(self) -> np.ndarray:
        """Get depth using FoundationStereo server instead of RealSense's built-in depth.

        Uses IR stereo pair for inference, then warps depth to the color frame.
        Returns depth in meters, aligned to the color image.
        """
        if self._foundation_stereo_url is None:
            raise RuntimeError("foundation_stereo_url was not provided at init")

        frames = self.pipeline.wait_for_frames(timeout_ms=1000)

        # Get IR stereo pair (single-channel Y8)
        ir_left_frame = frames.get_infrared_frame(1)
        ir_right_frame = frames.get_infrared_frame(2)
        if not ir_left_frame or not ir_right_frame:
            raise RuntimeError("Failed to get IR stereo frames")

        ir_left = np.asanyarray(ir_left_frame.get_data())
        ir_right = np.asanyarray(ir_right_frame.get_data())

        # Convert single-channel IR to 3-channel for FoundationStereo
        ir_left_rgb = np.stack([ir_left, ir_left, ir_left], axis=-1)
        ir_right_rgb = np.stack([ir_right, ir_right, ir_right], axis=-1)

        # Encode as PNG
        _, left_bytes = cv2.imencode('.png', ir_left_rgb)
        _, right_bytes = cv2.imencode('.png', ir_right_rgb)

        files = {
            'left_image': ('left.png', left_bytes.tobytes(), 'image/png'),
            'right_image': ('right.png', right_bytes.tobytes(), 'image/png'),
        }

        data = {
            'fx': float(self._K_ir[0, 0]),
            'fy': float(self._K_ir[1, 1]),
            'cx': float(self._K_ir[0, 2]),
            'cy': float(self._K_ir[1, 2]),
            'baseline': self._baseline,
            'scale': 1.0,
            'hiera': 0,
            'valid_iters': 32,
        }

        response = requests.post(
            f"{self._foundation_stereo_url}/infer", files=files, data=data
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"FoundationStereo server error: {response.status_code} - {response.text}"
            )

        buffer = io.BytesIO(response.content)
        depth_ir = np.load(buffer)['depth']  # Depth in IR-left frame, meters

        # Warp from IR frame to color frame
        depth_color = _depth_ir_to_color(
            depth_ir,
            self._K_ir,
            self._T_color_from_ir,
            self._K_color,
            color_size=(self.height, self.width),
        )
        return depth_color

    def get_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns camera matrix and distortion coefficients."""
        intr = self.color_intrinsics

        camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float64)

        # RealSense distortion: [k1, k2, p1, p2, k3]
        dist_coeffs = np.array([
            intr.coeffs[0], intr.coeffs[1],
            intr.coeffs[2], intr.coeffs[3],
            intr.coeffs[4]
        ], dtype=np.float64)

        return camera_matrix, dist_coeffs

    def close(self):
        """Close the camera."""
        if hasattr(self, 'pipeline'):
            try:
                self.pipeline.stop()
                print(f'Closed RealSense camera: {self.actual_serial}')
            except Exception as e:
                print(f"Error closing RealSense: {e}")


def _depth_ir_to_color(
    depth_ir: np.ndarray,
    K_ir: np.ndarray,
    T_color_from_ir: np.ndarray,
    K_color: np.ndarray,
    color_size: tuple[int, int],
) -> np.ndarray:
    """Warp IR-frame depth (meters) onto color pixel grid via forward splatting with z-buffer."""
    Hc, Wc = color_size
    Hi, Wi = depth_ir.shape

    fx_i, fy_i = float(K_ir[0, 0]), float(K_ir[1, 1])
    cx_i, cy_i = float(K_ir[0, 2]), float(K_ir[1, 2])
    fx_c, fy_c = float(K_color[0, 0]), float(K_color[1, 1])
    cx_c, cy_c = float(K_color[0, 2]), float(K_color[1, 2])

    u, v = np.meshgrid(np.arange(Wi, dtype=np.float32), np.arange(Hi, dtype=np.float32))
    z = depth_ir.astype(np.float32)
    valid = (z > 0.0) & np.isfinite(z)
    if not np.any(valid):
        return np.zeros((Hc, Wc), dtype=np.float32)

    # Unproject IR pixels to 3D
    x_i = (u[valid] - cx_i) / fx_i * z[valid]
    y_i = (v[valid] - cy_i) / fy_i * z[valid]
    pts_ir = np.stack([x_i, y_i, z[valid]], axis=0)

    # Transform to color frame
    R = T_color_from_ir[:3, :3].astype(np.float32)
    t = T_color_from_ir[:3, 3].astype(np.float32).reshape(3, 1)
    pts_c = R @ pts_ir + t
    Xc, Yc, Zc = pts_c[0], pts_c[1], pts_c[2]
    valid_c = Zc > 1e-6
    if not np.any(valid_c):
        return np.zeros((Hc, Wc), dtype=np.float32)
    Xc, Yc, Zc = Xc[valid_c], Yc[valid_c], Zc[valid_c]

    # Project to color image
    uc_f = fx_c * (Xc / Zc) + cx_c
    vc_f = fy_c * (Yc / Zc) + cy_c
    x0 = np.floor(uc_f).astype(np.int32)
    y0 = np.floor(vc_f).astype(np.int32)

    depth_color = np.full((Hc, Wc), np.inf, dtype=np.float32)

    def splat(ix, iy, zvals):
        inb = (ix >= 0) & (ix < Wc) & (iy >= 0) & (iy < Hc)
        if np.any(inb):
            np.minimum.at(depth_color, (iy[inb], ix[inb]), zvals[inb])

    # Splat to 4 neighbors to reduce gaps
    splat(x0, y0, Zc)
    splat(x0 + 1, y0, Zc)
    splat(x0, y0 + 1, Zc)
    splat(x0 + 1, y0 + 1, Zc)

    # Fill small holes via iterative min-filter
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
