"""Modified from tiptop/DROID: https://github.com/NishanthJKumar/tiptop-robot/blob/main/tiptop/scripts/calibrate_wrist_cam.py"""

import logging
import time
from collections import defaultdict

import cv2
import numpy as np
from cv2 import aruco
from scipy.spatial.transform import Rotation as R
from skills.go_to_conf import goto_hand_position
from bamboo.client import BambooFrankaClient

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = SQUARES_Y = 9
CHARUCOBOARD_COLCOUNT = SQUARES_X = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.015
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

# Create Board #
CHARUCO_BOARD = aruco.CharucoBoard(
    size=(SQUARES_X, SQUARES_Y),
    squareLength=CHARUCOBOARD_CHECKER_SIZE,
    markerLength=CHARUCOBOARD_MARKER_SIZE,
    dictionary=ARUCO_DICT,
)

# Detector Params
detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_FOCAL_LENGTH

charuco_params = aruco.CharucoParameters()
charuco_params.tryRefineMarkers = True


_log = logging.getLogger(__name__)


def angle_diff(target, source, degrees=False):
    target_rot = R.from_euler("xyz", target, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    result = target_rot * source_rot.inv()
    return result.as_euler("xyz")


def pose_diff(target, source, degrees=False):
    lin_diff = np.array(target[:3]) - np.array(source[:3])
    rot_diff = angle_diff(target[3:6], source[3:6], degrees=degrees)
    result = np.concatenate([lin_diff, rot_diff])
    return result


def rmat_to_euler(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_rmat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_matrix()


def change_pose_frame(pose, frame, degrees=False):
    R_frame = euler_to_rmat(frame[3:6], degrees=degrees)
    R_pose = euler_to_rmat(pose[3:6], degrees=degrees)
    t_frame, t_pose = frame[:3], pose[:3]
    euler_new = rmat_to_euler(R_frame @ R_pose, degrees=degrees)
    t_new = R_frame @ t_pose + t_frame
    result = np.concatenate([t_new, euler_new])
    return result


def calibration_traj(t, pos_scale=0.1, angle_scale=0.2, hand_camera=False):
    x = -np.abs(np.sin(3 * t)) * pos_scale
    y = -0.8 * np.sin(2 * t) * pos_scale
    z = 0.5 * np.sin(4 * t) * pos_scale
    a = -np.sin(4 * t) * angle_scale
    b = np.sin(3 * t) * angle_scale
    c = np.sin(2 * t) * angle_scale
    if hand_camera:
        value = np.array([z, y, -x, c / 1.5, b / 1.5, -a / 1.5])
    else:
        value = np.array([x, y, z, a, b, c])
    return value


class CharucoDetector:
    def __init__(
        self,
        cameraMatrix,
        distCoeffs,
        inlier_error_threshold=3.0,
        reprojection_error_threshold=3.0,
        num_img_threshold=10,
        num_corner_threshold=10,
    ):
        # Set Parameters
        self.inlier_error_threshold = inlier_error_threshold
        self.reprojection_error_threshold = reprojection_error_threshold
        self.num_img_threshold = num_img_threshold
        self.num_corner_threshold = num_corner_threshold
        self.intrinsic_params = {}
        self._readings_dict = defaultdict(list)
        self._pose_dict = defaultdict(list)
        self._curr_cam_id = None
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

    def process_image(self, image):
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError
        img_size = image.shape[:2]

        # Find Aruco Markers In Image #
        detector = aruco.ArucoDetector(ARUCO_DICT, detectorParams=detector_params)

        # corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=ARUCO_DICT, parameters=detector_params)
        corners, ids, rejected = detector.detectMarkers(image=gray)

        corners, ids, _, _ = detector.refineDetectedMarkers(
            gray,
            CHARUCO_BOARD,
            corners,
            ids,
            rejected,
            self.cameraMatrix,
            self.distCoeffs,
        )

        # Find Charuco Corners #
        if len(corners) == 0:
            return None

        charuco_detector = aruco.CharucoDetector(CHARUCO_BOARD, charuco_params, detector_params)
        # num_corners_found, charuco_corners, charuco_ids = detector.interpolateCornersCharuco(
        #     markerCorners=corners, markerIds=ids, image=gray, board=CHARUCO_BOARD, **self.intrinsic_params
        # )
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

        num_corners_found = len(charuco_corners) if charuco_corners is not None else 0
        if num_corners_found < self.num_corner_threshold:
            return None

        return corners, charuco_corners, charuco_ids, img_size

    def add_sample(self, cam_id, image, pose):
        readings = self.process_image(image)
        if readings is None:
            return
        self._readings_dict[cam_id].append(readings)
        self._pose_dict[cam_id].append(pose)

    def calculate_target_to_cam(self, readings, train=True):
        init_corners_all = []  # Corners discovered in all images processed
        init_ids_all = []  # Aruco ids corresponding to corners discovered
        fixed_image_size = readings[0][3]

        # Proccess Readings #
        init_successes = []
        for i in range(len(readings)):
            corners, charuco_corners, charuco_ids, img_size = readings[i]
            assert img_size == fixed_image_size
            init_corners_all.append(charuco_corners)
            init_ids_all.append(charuco_ids)
            init_successes.append(i)

        # First Pass: Find Outliers #
        threshold = self.num_img_threshold if train else 5
        if len(init_successes) < threshold:
            return None
        # print('Not enough points round 1')
        # print('Num Points: ', len(init_successes))
        # return None

        obj_points_all, img_points_all = [], []
        for corners, ids in zip(init_corners_all, init_ids_all):
            objPoints, imgPoints = CHARUCO_BOARD.matchImagePoints(corners, ids)
            obj_points_all.append(objPoints)
            img_points_all.append(imgPoints)

        obj_points_flat = np.vstack(obj_points_all)
        img_points_flat = np.vstack(img_points_all)
        assert obj_points_flat.shape[1] == img_points_flat.shape[1] == 1

        calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs, stdIntrinsics, stdExtrinsics, perViewErrors = (
            cv2.calibrateCameraExtended(
                objectPoints=obj_points_all,
                imagePoints=img_points_all,
                imageSize=fixed_image_size,
                cameraMatrix=self.cameraMatrix,
                distCoeffs=self.distCoeffs,
                flags=calib_flags,
            )
        )

        # Remove Outliers #
        threshold = self.num_img_threshold if train else 5
        final_corners_all = [
            init_corners_all[i] for i in range(len(perViewErrors)) if perViewErrors[i] <= self.inlier_error_threshold
        ]
        final_ids_all = [
            init_ids_all[i] for i in range(len(perViewErrors)) if perViewErrors[i] <= self.inlier_error_threshold
        ]
        final_successes = [
            init_successes[i] for i in range(len(perViewErrors)) if perViewErrors[i] <= self.inlier_error_threshold
        ]
        if len(final_successes) < threshold:
            return None
        # print('Not enough points round 2')
        # print('Num Points: ', len(final_successes))
        # print('Error Mean: ', perViewErrors.mean())
        # print('Error Std: ', perViewErrors.std())
        # return None

        # Second Pass: Calculate Finalized Extrinsics #
        final_obj_points_all, final_img_points_all = [], []
        for corners, ids in zip(final_corners_all, final_ids_all):
            objPoints, imgPoints = CHARUCO_BOARD.matchImagePoints(corners, ids)
            final_obj_points_all.append(objPoints)
            final_img_points_all.append(imgPoints)

        calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=final_obj_points_all,
            imagePoints=final_img_points_all,
            imageSize=fixed_image_size,
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeffs,
            flags=calib_flags,
        )

        # Return Transformation #
        if calibration_error > self.reprojection_error_threshold:
            return None
        # print('Failed Calibration Threshold')
        # print('Calibration Error: ', calibration_error)
        # return None

        rmats = [R.from_rotvec(rvec.flatten()).as_matrix() for rvec in rvecs]
        tvecs = [tvec.flatten() for tvec in tvecs]

        return rmats, tvecs, final_successes

    def augment_image(self, cam_id, image, visualize=False, visual_type=["markers", "axes"]):
        if type(visual_type) != list:
            visual_type = [visual_type]
        assert all([t in ["markers", "charuco", "axes"] for t in visual_type])
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        self._curr_cam_id = cam_id

        image = np.copy(image)
        readings = self.process_image(image)

        if readings is None:
            if visualize:
                cv2.imshow("Charuco board: {0}".format(cam_id), image)
                cv2.waitKey(20)
            return image

        corners, charuco_corners, charuco_ids, image_size = readings

        # Outline the aruco markers found in our query image
        if "markers" in visual_type:
            image = aruco.drawDetectedMarkers(image=image, corners=corners)

        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        if "charuco" in visual_type:
            image = aruco.drawDetectedCornersCharuco(
                image=image, charucoCorners=charuco_corners, charucoIds=charuco_ids
            )

        if "axes" in visual_type:
            objPoints, imgPoints = CHARUCO_BOARD.matchImagePoints(charuco_corners, charuco_ids)
            # cameraMatrix = self._intrinsics_dict[self._curr_cam_id]["cameraMatrix"]
            # distCoeffs = self._intrinsics_dict[self._curr_cam_id]["distCoeffs"]
            cameraMatrix = self.cameraMatrix
            distCoeffs = self.distCoeffs
            validPose, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
            cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
            # calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            #     charucoCorners=[charuco_corners],
            #     charucoIds=[charuco_ids],
            #     board=CHARUCO_BOARD,
            #     imageSize=image_size,
            #     flags=calib_flags,
            #     **self._intrinsics_dict[self._curr_cam_id],
            # )
            # cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvecs[0], tvecs[0], 0.1)

        # Visualize
        if visualize:
            cv2.imshow("Charuco board: {0}".format(cam_id), image)
            cv2.waitKey(20)

        return image


class HandCameraCalibrator(CharucoDetector):
    def __init__(self, intrinsics, distCoeffs, lin_error_threshold=1e-3, rot_error_threshold=1e-2, train_percentage=0.7, **kwargs):
        self.lin_error_threshold = lin_error_threshold
        self.rot_error_threshold = rot_error_threshold
        self.train_percentage = train_percentage
        super().__init__(intrinsics, distCoeffs, **kwargs)

    def calibrate(self, cam_id):
        return self._calibrate_cam_to_gripper(cam_id=cam_id)

    def _calibrate_cam_to_gripper(self, cam_id=None, readings=None, gripper_poses=None, target2cam_results=None):
        # Get Calibration Data #
        if cam_id is not None:
            readings, gripper_poses = self._readings_dict[cam_id], self._pose_dict[cam_id]
            self._curr_cam_id = cam_id

        # Get Target2Cam Transformation #
        if target2cam_results is None:
            target2cam_results = self.calculate_target_to_cam(readings)
        if target2cam_results is None:
            return None

        R_target2cam, t_target2cam, successes = target2cam_results
        gripper_poses = np.array(gripper_poses)[successes]

        # Calculate Appropriate Transformations #
        t_gripper2base = [np.array(pose[:3]) for pose in gripper_poses]
        R_gripper2base = [R.from_euler("xyz", pose[3:6]).as_matrix() for pose in gripper_poses]

        # Perform Calibration #
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=4,
        )

        # Return Pose #
        pos = pos.flatten()
        angle = R.from_matrix(rmat).as_euler("xyz")
        pose = np.concatenate([pos, angle])

        X_extrin = np.eye(4)
        X_extrin[:3, :3] = rmat
        X_extrin[:3, 3] = pos
        np.save("camera_to_gripper_extrinsics.npy", X_extrin)

        return pose

    def _calibrate_base_to_target(self, cam_id=None, readings=None, gripper_poses=None, target2cam_results=None):
        # Get Calibration Data #
        if cam_id is not None:
            readings, gripper_poses = self._readings_dict[cam_id], self._pose_dict[cam_id]
            self._curr_cam_id = cam_id

        # Get Target2Cam Transformation #
        if target2cam_results is None:
            target2cam_results = self.calculate_target_to_cam(readings)
        if target2cam_results is None:
            return None

        R_target2cam, t_target2cam, successes = target2cam_results
        gripper_poses = np.array(gripper_poses)[successes]

        # Calculate Appropriate Transformations #
        t_gripper2base = [np.array(pose[:3]) for pose in gripper_poses]
        R_gripper2base = [R.from_euler("xyz", pose[3:6]).as_matrix() for pose in gripper_poses]

        # Perform Calibration #
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_target2cam,
            t_gripper2base=t_target2cam,
            R_target2cam=R_gripper2base,
            t_target2cam=t_gripper2base,
            method=4,
        )

        # Return Pose #
        pos = pos.flatten()
        angle = R.from_matrix(rmat).as_euler("xyz")
        pose = np.concatenate([pos, angle])

        return pose

    def _calculate_gripper_to_base(self, train_readings, train_gripper_poses, eval_readings=None):
        if eval_readings is None:
            eval_readings = train_readings

        # Get Eval Target2Cam Transformations #
        eval_results = self.calculate_target_to_cam(eval_readings, train=False)
        if eval_results is None:
            return None
        eval_R_target2cam, eval_t_target2cam, eval_successes = eval_results
        rmats, tvecs = [], []

        # Get Train Target2Cam Transformations #
        train_results = self.calculate_target_to_cam(train_readings)
        if train_results is None:
            return None

        # Use Training Data For Calibrations #
        base2target = self._calibrate_base_to_target(
            gripper_poses=train_gripper_poses, target2cam_results=train_results
        )
        R_base2target = R.from_euler("xyz", base2target[3:]).as_matrix()
        t_base2target = np.array(base2target[:3])

        cam2gripper = self._calibrate_cam_to_gripper(
            gripper_poses=train_gripper_poses, target2cam_results=train_results
        )
        R_cam2gripper = R.from_euler("xyz", cam2gripper[3:]).as_matrix()
        t_cam2gripper = np.array(cam2gripper[:3])

        # Calculate Gripper2Base #
        for i in range(len(eval_R_target2cam)):
            R_base2cam = eval_R_target2cam[i] @ R_base2target
            t_base2cam = eval_R_target2cam[i] @ t_base2target + eval_t_target2cam[i]

            R_base2gripper = R_cam2gripper @ R_base2cam
            t_base2gripper = R_cam2gripper @ t_base2cam + t_cam2gripper

            R_gripper2base = R.from_matrix(R_base2gripper).inv().as_matrix()
            t_gripper2base = -R_gripper2base @ t_base2gripper

            rmats.append(R_gripper2base)
            tvecs.append(t_gripper2base)

        # Return Poses #
        eulers = np.array([R.from_matrix(rmat).as_euler("xyz") for rmat in rmats])
        eval_poses = np.concatenate([np.array(tvecs), eulers], axis=1)

        return eval_poses, eval_successes

    def is_calibration_accurate(self, cam_id):
        # Set Camera #
        self._curr_cam_id = cam_id

        # Split Into Train / Test #
        readings = self._readings_dict[cam_id]
        if len(readings) == 0:
            return False
        poses = np.array(self._pose_dict[cam_id])
        ind = np.random.choice(len(readings), size=len(readings), replace=False)
        num_train = int(len(readings) * self.train_percentage)

        train_ind, test_ind = ind[:num_train], ind[num_train:]
        train_poses, test_poses = poses[train_ind], poses[test_ind]
        train_readings = [readings[i] for i in train_ind]
        test_readings = [readings[i] for i in test_ind]

        # Calculate Approximate Gripper2Base Transformations #
        results = self._calculate_gripper_to_base(train_readings, train_poses, eval_readings=test_readings)
        if results is None:
            return False
        approx_poses, successes = results
        test_poses = np.array(test_poses)[successes]

        # Calculate Per Dimension Error #
        pose_error = np.array([pose_diff(pose, approx_pose) for pose, approx_pose in zip(test_poses, approx_poses)])
        lin_error = np.linalg.norm(pose_error[:, :3], axis=0) ** 2 / pose_error.shape[0]
        rot_error = np.linalg.norm(pose_error[:, 3:6], axis=0) ** 2 / pose_error.shape[0]

        # Check Calibration Error #
        lin_success = np.all(lin_error < self.lin_error_threshold)
        rot_success = np.all(rot_error < self.rot_error_threshold)

        # print('Pose Std: ', poses.std(axis=0))
        # print('Lin Error: ', lin_error)
        # print('Rot Error: ', rot_error)

        return lin_success and rot_success


def calibrate_wrist_camera(camera_type='zed', serial_number=16779706, server_ip="128.30.224.88"):
    """Calibrates the wrist (hand) camera.

    Args:
        camera_type: 'zed' or 'realsense'
        serial_number: Camera serial number (int for ZED, str for RealSense)
        server_ip: Bamboo/Franka server IP address
    """
    if camera_type == 'zed':
        from perception.zed.zed_cam import ZedCamera
        serial = int(serial_number) if isinstance(serial_number, str) else serial_number
        cam = ZedCamera(serial_number=serial)
        cam_id = serial
    elif camera_type == 'realsense':
        from perception.realsense.realsense_cam import RealSenseCamera
        serial = str(serial_number) if isinstance(serial_number, int) else serial_number
        cam = RealSenseCamera(serial_number=serial)
        cam_id = serial
    else:
        raise ValueError(f"Unknown camera type: {camera_type}. Choose 'zed' or 'realsense'.")

    intrinsics, distortion = cam.get_intrinsics()
    client = BambooFrankaClient(server_ip=server_ip, enable_gripper=False)
    # client = BambooFrankaClient(server_ip=server_ip)
    calibrator = HandCameraCalibrator(intrinsics, distortion)

    # Setup motion planner. Hard-code the time_dilation_factor as the movements are small
    _log.info("Setting up motion planner...")

    # Visualize the camera feed
    while True:
        _ = cam.get_bgra_frame()
        viz_img = cam.get_bgra_frame()
        viz_img = calibrator.augment_image(cam_id=cam_id, image=viz_img)
        viz_img = cv2.putText(
            viz_img,
            "Move robot s.t. calibration board is visible.",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        viz_img = cv2.putText(
            viz_img, "Press 'y' to continue, 'n' to exit", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Calibration View", viz_img)
        key = cv2.waitKey(1)
        if key == ord("y"):
            break
        elif key == ord("n"):
            return

    def get_q_curr():
        _ = client.get_joint_states()['ee_pose']
        q_curr = np.array(client.get_joint_positions())
        return q_curr

    def get_mat4x4():
        _ = client.get_joint_states()['ee_pose']
        return np.array(client.get_joint_states()['ee_pose'])

    # Bad hack for now, flush out the communication channel
    _log.debug("Attempting to flush out the buffer")
    for _ in range(100):
        get_q_curr()
    _log.debug("Flushed out the buffer (I hope)")

    pose_origin_mat4x4 = get_mat4x4()
    pose_origin = np.zeros(6)
    pose_origin[:3] = pose_origin_mat4x4[:3, 3]
    pose_origin[3:] = rmat_to_euler(pose_origin_mat4x4[:3, :3])
    i = 0

    step_size = 0.15
    while True:
        calib_pose = calibration_traj(i * step_size, hand_camera=True)
        desired_pose = change_pose_frame(calib_pose, pose_origin)
        desired_pose_mat4x4 = np.eye(4)
        desired_pose_mat4x4[:3, 3] = desired_pose[:3]
        desired_pose_mat4x4[:3, :3] = euler_to_rmat(desired_pose[3:])
        exec_failure = goto_hand_position(client, desired_pose_mat4x4, 2.0)

        if exec_failure:
            raise RuntimeError(f"Could not move robot!")

        # env.update_robot(action, action_space="cartesian_position", blocking=False)
        time.sleep(0.4)  # wait for robot to stabilize
        pose_origin_mat4x4 = get_mat4x4()
        # pose_origin_mat4x4 = np.array(state["ee_pose"])
        pose = np.zeros(6)
        pose[:3] = pose_origin_mat4x4[:3, 3]
        pose[3:] = rmat_to_euler(pose_origin_mat4x4[:3, :3])

        # Add Sample + Augment Images #
        _ = cam.get_bgra_frame()
        image = cam.get_bgra_frame()

        cycle_complete = (i * step_size) >= (2 * np.pi)
        cycle_prop_complete = 100 * (i * step_size) / (2 * np.pi)
        _log.debug(f"{cycle_prop_complete:.2f}% calibration complete")

        calibrator.add_sample(cam_id=cam_id, image=np.copy(image), pose=pose)
        augmented_image = calibrator.augment_image(cam_id=cam_id, image=image)
        augmented_image = cv2.putText(
            augmented_image,
            f"Calibration {cycle_prop_complete:.2f}% complete...",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Calibration View", augmented_image)
        cv2.waitKey(1)

        # Check if cycle is complete
        if cycle_complete:
            break
        i += 1

    success = calibrator.is_calibration_accurate(cam_id)
    if not success:
        raise RuntimeError(f"Calibration failed as it wasn't accurate enough")

    # Save the calibration
    client.close()
    cam.close()
    transformation = calibrator.calibrate(cam_id)
    print(f"{transformation=}")
    # update_calibration_info(cam_id, transformation)
    _log.info(f"Updated calibration info for {cam_id}. Transformation: {transformation}")


def calibrate_wrist_camera_entrypoint():
    import argparse

    parser = argparse.ArgumentParser(description='Calibrate wrist-mounted camera (ZED or RealSense)')
    parser.add_argument('--camera-type', type=str, default='zed', choices=['zed', 'realsense'],
                        help='Camera type: zed or realsense (default: zed)')
    parser.add_argument('--serial', type=str, default='16779706',
                        help='Camera serial number (default: 16779706 for ZED)')
    parser.add_argument('--server-ip', type=str, default='128.30.224.88',
                        help='Bamboo/Franka server IP address (default: 128.30.224.88)')

    args = parser.parse_args()

    calibrate_wrist_camera(
        camera_type=args.camera_type,
        serial_number=args.serial,
        server_ip=args.server_ip
    )


if __name__ == "__main__":
    calibrate_wrist_camera_entrypoint()

