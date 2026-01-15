import pyzed.sl as sl
import numpy as np
import requests
import io
import cv2


class ZedCamera:
    def __init__(self, serial_number: int | None = None):
        self.serial_number = serial_number
        standard_params = dict(
            depth_minimum_distance=0.1,
            camera_resolution=sl.RESOLUTION.HD720,
            depth_stabilization=False,
            camera_fps=60,
            camera_image_flip=sl.FLIP_MODE.OFF,
        )
        init_params = sl.InitParameters(**standard_params)
        if serial_number is None:
            print("opening camera")
            self.zed = sl.Camera()
        else:
            print(f"opening camera {serial_number}")
            self.zed = sl.Camera(serial_number)
            init_params.set_from_serial_number(serial_number)
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED: {status}")
        if serial_number is not None:
            assert self.zed.get_camera_information().serial_number == serial_number

        # Create persistent buffers for this camera instance
        self._image_buffer = sl.Mat()
        self._right_image_buffer = sl.Mat()
        self._depth_buffer = sl.Mat()
        self._runtime_params = sl.RuntimeParameters()

        # FoundationStereo server URL
        self._foundation_stereo_url = "http://localhost:1234"

        self.get_bgra_frame()
        self.get_depth_frame()

    def get_bgra_frame(self) -> np.ndarray:
        if self.zed.grab(self._runtime_params) <= sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self._image_buffer, sl.VIEW.LEFT)
        return self._image_buffer.get_data()

    def get_depth_frame(self) -> np.ndarray:
        if self.zed.grab(self._runtime_params) <= sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self._depth_buffer, sl.MEASURE.DEPTH)
        depth_mm = self._depth_buffer.get_data()
        return depth_mm / 1000.0

    def get_foundation_depth_frame(self) -> np.ndarray:
        """Get depth using FoundationStereo server instead of ZED's built-in depth."""
        # Grab frame and retrieve left/right images
        if self.zed.grab(self._runtime_params) <= sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self._image_buffer, sl.VIEW.LEFT)
            self.zed.retrieve_image(self._right_image_buffer, sl.VIEW.RIGHT)

        left_bgra = self._image_buffer.get_data()
        right_bgra = self._right_image_buffer.get_data()

        # Convert BGRA to BGR for encoding
        left_bgr = cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2BGR)
        right_bgr = cv2.cvtColor(right_bgra, cv2.COLOR_BGRA2BGR)

        # Encode images as PNG
        _, left_bytes = cv2.imencode('.png', left_bgr)
        _, right_bytes = cv2.imencode('.png', right_bgr)

        # Get camera intrinsics and baseline
        camera_info = self.zed.get_camera_information()
        calib_params = camera_info.camera_configuration.calibration_parameters
        left_cam = calib_params.left_cam
        baseline = calib_params.get_camera_baseline() / 1000.0  # Convert mm to meters

        files = {
            'left_image': ('left.png', left_bytes.tobytes(), 'image/png'),
            'right_image': ('right.png', right_bytes.tobytes(), 'image/png'),
        }

        data = {
            'fx': left_cam.fx,
            'fy': left_cam.fy,
            'cx': left_cam.cx,
            'cy': left_cam.cy,
            'baseline': baseline,
            'scale': 1.0,
            'hiera': 0,
            'valid_iters': 32
        }

        response = requests.post(f"{self._foundation_stereo_url}/infer", files=files, data=data)

        if response.status_code == 200:
            buffer = io.BytesIO(response.content)
            depth = np.load(buffer)['depth']
            return depth
        else:
            raise RuntimeError(f"FoundationStereo server error: {response.status_code} - {response.text}")

    def get_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns camera matrix and distortion coefficients."""
        camera_info = self.zed.get_camera_information()
        calibration_params = camera_info.camera_configuration.calibration_parameters.left_cam
        camera_matrix = np.array([[calibration_params.fx, 0, calibration_params.cx],
                                  [0, calibration_params.fy, calibration_params.cy],
                                  [0, 0, 1]])
        dist_coeffs = np.array(calibration_params.disto)
        return camera_matrix, dist_coeffs

    def close(self):
        self.zed.close()
