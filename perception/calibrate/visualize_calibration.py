#!/usr/bin/env python3
"""
Visualize hand-eye calibration results using rerun.
"""
import numpy as np
import rerun as rr
import glob
import os
import cv2
from scipy.spatial.transform import Rotation as R

# ChArUco Board Configuration (must match calibration script)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
CHARUCO_BOARD = cv2.aruco.CharucoBoard((14, 9), 0.020, 0.015, ARUCO_DICT)

detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

charuco_params = cv2.aruco.CharucoParameters()
charuco_params.tryRefineMarkers = True

calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_FOCAL_LENGTH


def load_camera_intrinsics():
    """Load camera intrinsics from ZED camera."""
    from perception.zed.zed_cam import ZedCamera
    camera = ZedCamera(serial_number=16779706)
    camera_matrix, dist_coeffs = camera.get_intrinsics()
    camera.close()
    return camera_matrix, dist_coeffs


def detect_board_in_images(calibration_dir, camera_matrix, dist_coeffs):
    """Detect board poses from saved images."""
    image_files = sorted(glob.glob(os.path.join(calibration_dir, "capture_*.png")))

    all_corners = []
    all_ids = []
    valid_indices = []

    for i, image_file in enumerate(image_files):
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        charuco_detector = cv2.aruco.CharucoDetector(CHARUCO_BOARD, charuco_params, detector_params)
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        if charuco_corners is not None and len(charuco_corners) > 50:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            valid_indices.append(i)

    if len(all_corners) == 0:
        print("WARNING: No valid board detections found!")
        return None, None

    # Prepare object and image points
    obj_points_all, img_points_all = [], []
    for corners, ids in zip(all_corners, all_ids):
        objPoints, imgPoints = CHARUCO_BOARD.matchImagePoints(corners, ids)
        obj_points_all.append(objPoints)
        img_points_all.append(imgPoints)

    # Run calibration to get poses
    img_size = cv2.imread(image_files[0]).shape[:2]
    calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=obj_points_all,
        imagePoints=img_points_all,
        imageSize=img_size,
        cameraMatrix=camera_matrix.copy(),
        distCoeffs=dist_coeffs.copy(),
        flags=calib_flags,
    )

    # Convert rotation vectors to matrices
    R_target2cam = [R.from_rotvec(rvec.flatten()).as_matrix() for rvec in rvecs]
    t_target2cam = [tvec.flatten() for tvec in tvecs]

    return R_target2cam, t_target2cam, valid_indices


def visualize_calibration(calibration_dir):
    """Visualize calibration data from a calibration directory."""

    # Load extrinsics
    extrinsics_path = os.path.join(calibration_dir, "gripper_to_camera_extrinsics.npy")
    if not os.path.exists(extrinsics_path):
        print(f"ERROR: Extrinsics file not found: {extrinsics_path}")
        return

    X_GC = np.load(extrinsics_path)
    print(f"Loaded gripper-to-camera transform:")
    print(X_GC)

    # Load camera intrinsics
    print("\nLoading camera intrinsics...")
    camera_matrix, dist_coeffs = load_camera_intrinsics()

    # Detect boards in images
    print("Detecting boards in images...")
    board_data = detect_board_in_images(calibration_dir, camera_matrix, dist_coeffs)

    # Load all poses
    pose_files = sorted(glob.glob(os.path.join(calibration_dir, "*_pose.npy")))
    print(f"\nFound {len(pose_files)} pose files")

    if len(pose_files) == 0:
        print("ERROR: No pose files found!")
        return

    # Initialize rerun
    rr.init("hand_eye_calibration", spawn=True)

    # Draw world frame axes as arrows
    axis_length = 0.2
    rr.log("world/x_axis", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[axis_length, 0, 0]], colors=[[255, 0, 0]]))
    rr.log("world/y_axis", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[0, axis_length, 0]], colors=[[0, 255, 0]]))
    rr.log("world/z_axis", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[0, 0, axis_length]], colors=[[0, 0, 255]]))

    # Prepare board data if available
    board_positions = []
    if board_data[0] is not None:
        R_target2cam, t_target2cam, valid_indices = board_data
    else:
        R_target2cam = None

    # Process each pose
    for i, pose_file in enumerate(pose_files):
        X_WG = np.load(pose_file)  # World to gripper (ee_pose)

        # Draw gripper axes
        gripper_pos = X_WG[:3, 3]
        gripper_rot = X_WG[:3, :3]
        axis_len = 0.05
        rr.log(f"trajectory/grippers/gripper_{i:03d}/x", rr.Arrows3D(
            origins=[gripper_pos], vectors=[gripper_rot @ [axis_len, 0, 0]], colors=[[255, 0, 0]]))
        rr.log(f"trajectory/grippers/gripper_{i:03d}/y", rr.Arrows3D(
            origins=[gripper_pos], vectors=[gripper_rot @ [0, axis_len, 0]], colors=[[0, 255, 0]]))
        rr.log(f"trajectory/grippers/gripper_{i:03d}/z", rr.Arrows3D(
            origins=[gripper_pos], vectors=[gripper_rot @ [0, 0, axis_len]], colors=[[0, 0, 255]]))

        # Compute and log camera pose: X_WC = X_WG @ X_GC
        X_WC = X_WG @ X_GC
        camera_pos = X_WC[:3, 3]
        camera_rot = X_WC[:3, :3]
        axis_len = 0.08
        rr.log(f"trajectory/cameras/camera_{i:03d}/x", rr.Arrows3D(
            origins=[camera_pos], vectors=[camera_rot @ [axis_len, 0, 0]], colors=[[255, 0, 0]]))
        rr.log(f"trajectory/cameras/camera_{i:03d}/y", rr.Arrows3D(
            origins=[camera_pos], vectors=[camera_rot @ [0, axis_len, 0]], colors=[[0, 255, 0]]))
        rr.log(f"trajectory/cameras/camera_{i:03d}/z", rr.Arrows3D(
            origins=[camera_pos], vectors=[camera_rot @ [0, 0, axis_len]], colors=[[0, 0, 255]]))

        # Compute board pose in world frame if we have board detections
        if R_target2cam is not None and i in valid_indices:
            idx = valid_indices.index(i)

            # Build target-to-camera transform
            X_TC = np.eye(4)
            X_TC[:3, :3] = R_target2cam[idx]
            X_TC[:3, 3] = t_target2cam[idx]

            # Compute board in world frame: X_WB = X_WG @ X_GC @ X_CT
            X_CT = np.linalg.inv(X_TC)
            X_WB = X_WG @ X_GC @ X_CT

            board_positions.append(X_WB[:3, 3])

            # Draw board axes
            board_pos = X_WB[:3, 3]
            board_rot = X_WB[:3, :3]
            axis_len = 0.15
            rr.log(f"boards/board_{i:03d}/x", rr.Arrows3D(
                origins=[board_pos], vectors=[board_rot @ [axis_len, 0, 0]], colors=[[255, 0, 0]]))
            rr.log(f"boards/board_{i:03d}/y", rr.Arrows3D(
                origins=[board_pos], vectors=[board_rot @ [0, axis_len, 0]], colors=[[0, 255, 0]]))
            rr.log(f"boards/board_{i:03d}/z", rr.Arrows3D(
                origins=[board_pos], vectors=[board_rot @ [0, 0, axis_len]], colors=[[0, 0, 255]]))

    # If we have board positions, compute and visualize statistics
    if len(board_positions) > 0:
        board_positions = np.array(board_positions)
        mean_pos = np.mean(board_positions, axis=0)
        std_pos = np.std(board_positions, axis=0)

        print(f"\n{'='*60}")
        print("Board Position Statistics (World Frame)")
        print(f"{'='*60}")
        print(f"Mean position: {mean_pos}")
        print(f"Std dev: {std_pos * 1000} mm")
        print(f"Overall std: {np.std(board_positions) * 1000:.2f} mm")

        # Draw mean board position axes
        axis_len = 0.25
        rr.log("boards/mean_board/x", rr.Arrows3D(
            origins=[mean_pos], vectors=[[axis_len, 0, 0]], colors=[[255, 100, 100]]))
        rr.log("boards/mean_board/y", rr.Arrows3D(
            origins=[mean_pos], vectors=[[0, axis_len, 0]], colors=[[100, 255, 100]]))
        rr.log("boards/mean_board/z", rr.Arrows3D(
            origins=[mean_pos], vectors=[[0, 0, axis_len]], colors=[[100, 100, 255]]))

        # Log all board positions as points
        rr.log("boards/all_positions", rr.Points3D(board_positions, radii=0.01))

    print(f"\nVisualization complete!")
    print(f"Red = X, Green = Y, Blue = Z")
    print(f"Hierarchy: trajectory/grippers (small), trajectory/cameras (medium), boards (large)")

    # Keep the script running so rerun viewer stays open
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        calibration_dir = sys.argv[1]
    else:
        # Find the most recent calibration directory (search from repo root)
        search_paths = [
            "calibration_data/calibration_data_*",
            "../../calibration_data/calibration_data_*",
            "../calibration_data/calibration_data_*"
        ]
        calibration_dirs = []
        for pattern in search_paths:
            calibration_dirs.extend(glob.glob(pattern))
        calibration_dirs = sorted(calibration_dirs)

        if not calibration_dirs:
            print("ERROR: No calibration directories found!")
            print("Usage: python visualize_calibration.py [calibration_dir]")
            sys.exit(1)
        calibration_dir = calibration_dirs[-1]

    print(f"Loading calibration data from: {calibration_dir}")
    visualize_calibration(calibration_dir)
