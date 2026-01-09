#!/usr/bin/env python3
"""
Diagnose frame convention issues in hand-eye calibration.
Tests both normal and inverted ee_pose to see which gives better results.
"""
import numpy as np
import glob
import os
import cv2
from scipy.spatial.transform import Rotation as R

# ChArUco Board Configuration
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


def test_calibration(calibration_dir, invert_poses=False):
    """Test hand-eye calibration with optional pose inversion."""

    print(f"\n{'='*70}")
    print(f"Testing with invert_poses={invert_poses}")
    print(f"{'='*70}\n")

    # Load camera intrinsics
    camera_matrix, dist_coeffs = load_camera_intrinsics()

    # Detect boards
    R_target2cam, t_target2cam, valid_indices = detect_board_in_images(
        calibration_dir, camera_matrix, dist_coeffs)

    # Load poses
    pose_files = sorted(glob.glob(os.path.join(calibration_dir, "*_pose.npy")))

    # Filter to only valid indices
    all_poses = []
    for i in valid_indices:
        pose_file = pose_files[i]
        X_WG = np.load(pose_file)

        if invert_poses:
            # Test if ee_pose is actually gripper-to-world instead of world-to-gripper
            X_WG = np.linalg.inv(X_WG)

        all_poses.append(X_WG)

    # Prepare data for hand-eye calibration
    R_gripper2base = []
    t_gripper2base = []

    for pose_matrix in all_poses:
        R_gripper2base.append(pose_matrix[:3, :3])
        t_gripper2base.append(pose_matrix[:3, 3].reshape(3, 1))

    # Prepare target-to-cam data
    t_target2cam_reshaped = [t.reshape(3, 1) for t in t_target2cam]

    # Solve hand-eye calibration
    print("Solving hand-eye calibration...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam_reshaped,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # Build final transform matrix
    X_GC = np.eye(4)
    X_GC[:3, :3] = R_cam2gripper
    X_GC[:3, 3] = t_cam2gripper.flatten()

    print("Gripper-to-Camera Transform:")
    print(X_GC)
    print(f"\nTranslation: {t_cam2gripper.flatten()}")
    print(f"Translation magnitude: {np.linalg.norm(t_cam2gripper):.4f} m")

    # Compute board positions in world frame
    board_positions_world = []

    for i, (R_t2c, t_t2c, X_WG) in enumerate(zip(R_target2cam, t_target2cam, all_poses)):
        # Build target-to-camera transform
        X_TC = np.eye(4)
        X_TC[:3, :3] = R_t2c
        X_TC[:3, 3] = t_t2c

        # Compute board in world frame: X_WB = X_WG @ X_GC @ X_CT
        X_CT = np.linalg.inv(X_TC)
        X_WB = X_WG @ X_GC @ X_CT

        board_positions_world.append(X_WB[:3, 3])

    board_positions_world = np.array(board_positions_world)

    # Calculate statistics
    mean_pos = np.mean(board_positions_world, axis=0)
    std_pos = np.std(board_positions_world, axis=0)
    overall_std = np.std(board_positions_world)

    print(f"\nBoard Position Statistics (World Frame):")
    print(f"Mean position: {mean_pos}")
    print(f"Std dev: [{std_pos[0]*1000:.2f}, {std_pos[1]*1000:.2f}, {std_pos[2]*1000:.2f}] mm")
    print(f"Overall std: {overall_std * 1000:.2f} mm")

    if overall_std * 1000 < 10:
        print("\n✓✓✓ EXCELLENT! This is the correct configuration! ✓✓✓")
    elif overall_std * 1000 < 50:
        print("\n✓ GOOD - within acceptable range")
    else:
        print("\n✗ POOR - calibration has significant errors")

    return overall_std


if __name__ == "__main__":
    # Find most recent calibration directory
    calibration_dirs = sorted(glob.glob("../../calibration_data/calibration_data_*"))
    if not calibration_dirs:
        print("ERROR: No calibration directories found!")
        exit(1)

    calibration_dir = calibration_dirs[-1]
    print(f"Loading calibration data from: {calibration_dir}")

    # Test both configurations
    std_normal = test_calibration(calibration_dir, invert_poses=False)
    std_inverted = test_calibration(calibration_dir, invert_poses=True)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Normal poses:   {std_normal * 1000:.2f} mm std dev")
    print(f"Inverted poses: {std_inverted * 1000:.2f} mm std dev")

    if std_inverted < std_normal * 0.5:
        print("\n*** FIX: Your ee_pose needs to be INVERTED! ***")
        print("Add this to your calibration script before using poses:")
        print("    X_WG = np.linalg.inv(actual_ee_pose)")
    elif std_normal < std_inverted * 0.5:
        print("\n*** ee_pose convention is CORRECT (no inversion needed) ***")
    else:
        print("\n*** WARNING: Neither configuration gives good results! ***")
        print("The problem is likely NOT the pose convention.")
