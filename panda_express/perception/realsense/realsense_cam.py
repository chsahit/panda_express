"""RealSense camera wrapper compatible with ZedCamera interface."""
import numpy as np

try:
    import pyrealsense2 as rs
    _REALSENSE_AVAILABLE = True
except ImportError:
    print('pyrealsense2 is not installed. Install with: pip install pyrealsense2')
    rs = None
    _REALSENSE_AVAILABLE = False


class RealSenseCamera:
    """RealSense camera wrapper compatible with ZedCamera interface for calibration."""

    def __init__(self, serial_number: str | None = None):
        """
        Initialize RealSense camera.

        Args:
            serial_number: Camera serial number (None for first available)
        """
        if not _REALSENSE_AVAILABLE:
            raise ImportError("RealSense SDK not available. Install pyrealsense2.")

        self.serial_number = serial_number
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

        # Start streaming
        print(f"Starting RealSense camera ({self.width}x{self.height} @ {self.fps}fps)...")
        profile = self.pipeline.start(config)

        # Get device info
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        self.actual_serial = device.get_info(rs.camera_info.serial_number)
        print(f'Connected to RealSense: {device_name} (Serial: {self.actual_serial})')

        # Get intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

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
