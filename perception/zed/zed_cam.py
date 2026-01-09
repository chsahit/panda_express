import pyzed.sl as sl
import numpy as np


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
        self._depth_buffer = sl.Mat()
        self._runtime_params = sl.RuntimeParameters()

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
