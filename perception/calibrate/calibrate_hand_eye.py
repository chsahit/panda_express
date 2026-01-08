import cv2
import numpy as np
import os
from datetime import datetime
from skills.go_to_conf import goto_hand_position
from bamboo.client import BambooFrankaClient
from scipy.spatial.transform import Rotation as R
from perception.zed.zed_cam import ZedCamera

# ============================================================================
# ChArUco Board Configuration
# ============================================================================
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
CHARUCO_BOARD = cv2.aruco.CharucoBoard((14, 9), 0.020, 0.015, ARUCO_DICT)

# Detector parameters
detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

charuco_params = cv2.aruco.CharucoParameters()
charuco_params.tryRefineMarkers = True

# Calibration flags - use camera intrinsics as initial guess but allow refinement
# This gives best results by trusting the camera's calibration but allowing minor corrections
calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_FOCAL_LENGTH

# ============================================================================
# Calibration Thresholds
# ============================================================================
INLIER_ERROR_THRESHOLD = 3.0  # Pixel error threshold for outlier removal (first pass)
REPROJECTION_ERROR_THRESHOLD = 3.0  # Final reprojection error threshold
NUM_CORNER_THRESHOLD = 50  # Minimum corners to accept a detection
NUM_IMG_THRESHOLD = 10  # Minimum successful images needed for calibration


# ============================================================================
# Helper Functions
# ============================================================================

def generate_calibration_poses(num_poses=30):
    """Generate varied calibration poses for the robot."""
    poses = []
    center = np.array([0.475, 0.0, 0.4])  # meters, adjust to your setup
    grasp_orientation = np.array([[1.0, 0.0, 0.0], [0.0, -1, 0], [-0.0, 0, -1.0]])

    for i in range(num_poses):
        # Vary position slightly
        pos = center + np.random.uniform(-0.025, 0.025, 3)

        # Vary orientation
        rx = np.random.uniform(-np.pi/10, np.pi/10)
        ry = np.random.uniform(-np.pi/10, np.pi/10)
        rz = np.random.uniform(-np.pi/6, np.pi/6)
        noise = R.from_euler('xyz', [rx, ry, rz], degrees=False)

        # Create transform matrix
        X_WG = np.eye(4)
        X_WG[:3, 3] = pos
        X_WG[:3, :3] = grasp_orientation @ noise.as_matrix()
        poses.append(X_WG)

    return poses


def process_image(image, camera_matrix, dist_coeffs):
    """
    Process image to detect ChArUco board.
    Returns (corners, charuco_corners, charuco_ids, img_size) or None if detection fails.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
    else:
        gray = image

    img_size = gray.shape[:2]

    # Detect ArUco markers
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, detectorParams=detector_params)
    corners, ids, rejected = detector.detectMarkers(image=gray)

    if ids is None or len(ids) < 4:
        return None

    # Refine detected markers
    corners, ids, _, _ = detector.refineDetectedMarkers(
        gray, CHARUCO_BOARD, corners, ids, rejected,
        camera_matrix, dist_coeffs
    )

    if len(corners) == 0:
        return None

    # Detect ChArUco board
    charuco_detector = cv2.aruco.CharucoDetector(CHARUCO_BOARD, charuco_params, detector_params)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    num_corners_found = len(charuco_corners) if charuco_corners is not None else 0
    if num_corners_found < NUM_CORNER_THRESHOLD:
        return None

    return corners, charuco_corners, charuco_ids, img_size


def calculate_target_to_cam(readings, camera_matrix, dist_coeffs):
    """
    Calculate target-to-camera transformations with outlier removal.

    Returns: (R_target2cam_list, t_target2cam_list, success_indices) or None
    """
    if len(readings) < NUM_IMG_THRESHOLD:
        print(f"Not enough successful readings: {len(readings)} < {NUM_IMG_THRESHOLD}")
        return None

    # First pass: collect all corners
    init_corners_all = []
    init_ids_all = []
    fixed_image_size = readings[0][3]
    init_successes = []

    for i in range(len(readings)):
        corners, charuco_corners, charuco_ids, img_size = readings[i]
        assert img_size == fixed_image_size, "All images must have same size"
        init_corners_all.append(charuco_corners)
        init_ids_all.append(charuco_ids)
        init_successes.append(i)

    # Prepare object and image points
    obj_points_all, img_points_all = [], []
    for corners, ids in zip(init_corners_all, init_ids_all):
        objPoints, imgPoints = CHARUCO_BOARD.matchImagePoints(corners, ids)
        obj_points_all.append(objPoints)
        img_points_all.append(imgPoints)

    # First calibration pass - get per-view errors to identify outliers
    print(f"Running calibration on {len(obj_points_all)} images...")
    calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs, stdIntrinsics, stdExtrinsics, perViewErrors = (
        cv2.calibrateCameraExtended(
            objectPoints=obj_points_all,
            imagePoints=img_points_all,
            imageSize=fixed_image_size,
            cameraMatrix=camera_matrix.copy(),
            distCoeffs=dist_coeffs.copy(),
            flags=calib_flags,
        )
    )

    print(f"Initial calibration error: {calibration_error:.4f} pixels")
    print(f"Per-view error - mean: {perViewErrors.mean():.4f}, std: {perViewErrors.std():.4f}")

    # Remove outliers based on per-view error
    final_corners_all = [
        init_corners_all[i] for i in range(len(perViewErrors))
        if perViewErrors[i] <= INLIER_ERROR_THRESHOLD
    ]
    final_ids_all = [
        init_ids_all[i] for i in range(len(perViewErrors))
        if perViewErrors[i] <= INLIER_ERROR_THRESHOLD
    ]
    final_successes = [
        init_successes[i] for i in range(len(perViewErrors))
        if perViewErrors[i] <= INLIER_ERROR_THRESHOLD
    ]

    num_outliers = len(init_successes) - len(final_successes)
    print(f"Removed {num_outliers} outliers (error > {INLIER_ERROR_THRESHOLD} pixels)")

    if len(final_successes) < NUM_IMG_THRESHOLD:
        print(f"Not enough images after outlier removal: {len(final_successes)} < {NUM_IMG_THRESHOLD}")
        return None

    # Second calibration pass on filtered data
    final_obj_points_all, final_img_points_all = [], []
    for corners, ids in zip(final_corners_all, final_ids_all):
        objPoints, imgPoints = CHARUCO_BOARD.matchImagePoints(corners, ids)
        final_obj_points_all.append(objPoints)
        final_img_points_all.append(imgPoints)

    calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=final_obj_points_all,
        imagePoints=final_img_points_all,
        imageSize=fixed_image_size,
        cameraMatrix=camera_matrix.copy(),
        distCoeffs=dist_coeffs.copy(),
        flags=calib_flags,
    )

    print(f"Final calibration error after outlier removal: {calibration_error:.4f} pixels")

    if calibration_error > REPROJECTION_ERROR_THRESHOLD:
        print(f"Calibration error too high: {calibration_error:.4f} > {REPROJECTION_ERROR_THRESHOLD}")
        return None

    # Convert rotation vectors to matrices
    rmats = [R.from_rotvec(rvec.flatten()).as_matrix() for rvec in rvecs]
    tvecs = [tvec.flatten() for tvec in tvecs]

    return rmats, tvecs, final_successes


# ============================================================================
# Main Calibration Script
# ============================================================================

# Initialize robot and camera
client = BambooFrankaClient(server_ip="128.30.224.88")
camera = ZedCamera(serial_number=16779706)
camera_matrix, dist_coeffs = camera.get_intrinsics()

print("Camera intrinsics loaded:")
print(f"Camera matrix:\n{camera_matrix}")
print(f"Distortion coefficients: {dist_coeffs}")

# Create directory for saving calibration data
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"calibration_data/calibration_data_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
print(f"\nSaving calibration data to: {save_dir}")

# Data collection lists
all_readings = []  # Store (corners, charuco_corners, charuco_ids, img_size)
all_poses = []     # Store 4x4 transformation matrices

try:
    # ========================================================================
    # Data Collection Loop (from calibrate_hand_eye.py)
    # ========================================================================
    capture_idx = 0
    successful_captures = 0

    print(f"\n{'='*60}")
    print("Starting data collection...")
    print(f"{'='*60}\n")

    while successful_captures < 30:
        # Move robot to random pose
        X_WG = generate_calibration_poses(num_poses=5)[-1]
        result = goto_hand_position(client, X_WG, 2)

        if result != 0:
            print(f"ERROR: Trajectory execution failed. Stopping calibration.")
            raise RuntimeError("Trajectory execution failed")

        # Get actual hand pose
        actual_ee_pose = np.array(client.get_joint_states()['ee_pose'])

        # Capture image
        image = camera.get_bgra_frame()
        image2 = np.copy(image)

        # Process image to detect board
        readings = process_image(image, camera_matrix, dist_coeffs)

        if readings is not None:
            corners, charuco_corners, charuco_ids, img_size = readings
            num_corners = len(charuco_corners) if charuco_corners is not None else 0

            print(f"Capture {capture_idx}: Detected {num_corners} ChArUco corners - SUCCESS")

            successful_captures += 1

            # Save image and pose
            image_path = os.path.join(save_dir, f"capture_{capture_idx:03d}.png")
            pose_path = os.path.join(save_dir, f"capture_{capture_idx:03d}_pose.npy")
            cv2.imwrite(image_path, image2)
            np.save(pose_path, actual_ee_pose)

            # Store for calibration
            all_readings.append(readings)
            all_poses.append(actual_ee_pose)

            print(f"  → Saved capture {successful_captures}/30")
        else:
            print(f"Capture {capture_idx}: Board detection failed")

        capture_idx += 1

    print(f"\n{'='*60}")
    print("Data collection complete!")
    print(f"{'='*60}\n")

    # ========================================================================
    # Calibration with Outlier Removal 
    # ========================================================================
    print("Running calibration with outlier removal...\n")

    # Calculate target-to-camera transformations with outlier filtering
    target2cam_results = calculate_target_to_cam(all_readings, camera_matrix, dist_coeffs)

    if target2cam_results is None:
        raise RuntimeError("Failed to calculate target-to-camera transformations")

    R_target2cam, t_target2cam, successes = target2cam_results

    # Filter poses to match successful detections
    gripper_poses = np.array(all_poses)[successes]
    print(f"\nUsing {len(successes)}/{len(all_poses)} poses for hand-eye calibration")

    # Prepare data for hand-eye calibration
    # Note: Despite ee_pose being base-to-gripper, we pass it directly (no inversion)
    # This was determined empirically - inverting gives unrealistic results
    R_gripper2base = []
    t_gripper2base = []

    for pose_matrix in gripper_poses:
        R_gripper2base.append(pose_matrix[:3, :3])
        t_gripper2base.append(pose_matrix[:3, 3].reshape(3, 1))

    # Prepare target-to-cam data
    t_target2cam = [t.reshape(3, 1) for t in t_target2cam]

    # Solve hand-eye calibration
    print("\nSolving hand-eye calibration...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI 
    )

    # Build final transform matrix
    X_GC = np.eye(4)
    X_GC[:3, :3] = R_cam2gripper
    X_GC[:3, 3] = t_cam2gripper.flatten()

    # Save results
    extrinsics_path = os.path.join(save_dir, "gripper_to_camera_extrinsics.npy")
    np.save(extrinsics_path, X_GC)

    print(f"\n{'='*60}")
    print("Calibration Complete!")
    print(f"{'='*60}")
    print(f"\nGripper-to-Camera Transform:")
    print(X_GC)
    print(f"\nTranslation: {t_cam2gripper.flatten()}")
    print(f"Rotation (euler XYZ): {R.from_matrix(R_cam2gripper).as_euler('xyz', degrees=True)} degrees")
    print(f"\nSaved extrinsics to: {extrinsics_path}")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    raise
finally:
    client.close()
    camera.close()
    print("\nRobot and camera closed.")
