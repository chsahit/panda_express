import cv2
import numpy as np
import os
import time
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

# Detector parameters with subpix refinement
detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

charuco_params = cv2.aruco.CharucoParameters()
charuco_params.tryRefineMarkers = True

# Calibration flags - use camera intrinsics as initial guess but allow refinement
calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_FOCAL_LENGTH

# ============================================================================
# Calibration Thresholds
# ============================================================================
NUM_CORNER_THRESHOLD = 20  # Minimum corners to accept a detection
NUM_IMG_THRESHOLD = 10  # Minimum successful images needed for calibration


# ============================================================================
# Helper Functions
# ============================================================================

def calibration_traj(t, pos_scale=0.1, angle_scale=0.3, hand_camera=False):
    """Generate smooth calibration trajectory based on time parameter."""
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


def generate_calibration_pose(t, center_pos, center_orientation, pos_scale=0.1, angle_scale=0.2, hand_camera=False):
    """Generate calibration pose at time t, centered around given position and orientation."""
    # Get trajectory delta
    delta = calibration_traj(t, pos_scale=pos_scale, angle_scale=angle_scale, hand_camera=hand_camera)

    # Apply position offset
    pos = center_pos + delta[:3]

    # Apply orientation offset
    delta_rot = R.from_euler('xyz', delta[3:6], degrees=False)

    # Create transform matrix
    X_WG = np.eye(4)
    X_WG[:3, 3] = pos
    X_WG[:3, :3] = center_orientation @ delta_rot.as_matrix()

    return X_WG


def process_image(image, camera_matrix, dist_coeffs):
    """
    Process image to detect ChArUco board.
    Returns (charuco_corners, charuco_ids, marker_corners, marker_ids, img_size) or None if detection fails.
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

    # Detect ChArUco board
    charuco_detector = cv2.aruco.CharucoDetector(CHARUCO_BOARD, charuco_params, detector_params)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    num_corners_found = len(charuco_corners) if charuco_corners is not None else 0
    if num_corners_found < NUM_CORNER_THRESHOLD:
        return None

    return charuco_corners, charuco_ids, marker_corners, marker_ids, img_size


def calculate_target_to_cam(readings, camera_matrix, dist_coeffs):
    """
    Calculate target-to-camera transformations.

    Returns: (R_target2cam_list, t_target2cam_list) or None
    """
    if len(readings) < NUM_IMG_THRESHOLD:
        print(f"Not enough successful readings: {len(readings)} < {NUM_IMG_THRESHOLD}")
        return None

    # Collect all corners and IDs
    corners_all = []
    ids_all = []
    fixed_image_size = readings[0][4]  # Updated index for img_size

    for i in range(len(readings)):
        charuco_corners, charuco_ids, marker_corners, marker_ids, img_size = readings[i]
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)

    # Prepare object and image points
    obj_points_all, img_points_all = [], []
    for corners, ids in zip(corners_all, ids_all):
        objPoints, imgPoints = CHARUCO_BOARD.matchImagePoints(corners, ids)
        obj_points_all.append(objPoints)
        img_points_all.append(imgPoints)

    # Single calibration pass
    print(f"Running calibration on {len(obj_points_all)} images...")
    calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=obj_points_all,
        imagePoints=img_points_all,
        imageSize=fixed_image_size,
        cameraMatrix=camera_matrix.copy(),
        distCoeffs=dist_coeffs.copy(),
        flags=calib_flags,
    )

    print(f"Calibration error: {calibration_error:.4f} pixels")

    # Convert rotation vectors to matrices
    rmats = [R.from_rotvec(rvec.flatten()).as_matrix() for rvec in rvecs]
    tvecs = [tvec.flatten() for tvec in tvecs]

    return rmats, tvecs


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
all_readings = []  # Store (charuco_corners, charuco_ids, img_size)
all_poses = []     # Store 4x4 transformation matrices

# Define center position and orientation for trajectory
center_pos = np.array([0.45, 0.0, 0.35])  # meters, adjust to your setup
center_orientation = np.array([[1.0, 0.0, 0.0], [0.0, -1, 0], [-0.0, 0, -1.0]])

try:
    # ========================================================================
    # Data Collection Loop
    # ========================================================================
    capture_idx = 0
    successful_captures = 0
    num_captures = 30
    hand_camera = True

    print(f"\n{'='*60}")
    print("Starting data collection...")
    print(f"Center position: {center_pos}")
    print(f"{'='*60}\n")

    # Generate time values spanning trajectory
    t_values = np.linspace(0, 2 * np.pi, num_captures)

    for t in t_values:
        # Generate pose from trajectory
        X_WG = generate_calibration_pose(
            t,
            center_pos,
            center_orientation,
            pos_scale=0.1,
            angle_scale=0.2,
            hand_camera=hand_camera
        )
        result = goto_hand_position(client, X_WG, 2)
        time.sleep(0.5)  # let things settle

        if result != 0:
            print(f"ERROR: Trajectory execution failed. Stopping calibration.")
            raise RuntimeError("Trajectory execution failed")

        # Get actual hand pose
        actual_ee_pose = np.array(client.get_joint_states()['ee_pose'])

        # Compare commanded vs actual pose
        pos_error = np.linalg.norm(X_WG[:3, 3] - actual_ee_pose[:3, 3])
        R_commanded = X_WG[:3, :3]
        R_actual = actual_ee_pose[:3, :3]
        R_diff = R_commanded.T @ R_actual
        angle_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)) * 180 / np.pi

        print(f"Capture {capture_idx}: Pose error - position: {pos_error*1000:.2f} mm, rotation: {angle_error:.2f} deg")

        # Capture image
        image = camera.get_bgra_frame()
        image2 = np.copy(image)

        # Process image to detect board
        readings = process_image(image, camera_matrix, dist_coeffs)

        if readings is not None:
            charuco_corners, charuco_ids, marker_corners, marker_ids, img_size = readings
            num_corners = len(charuco_corners) if charuco_corners is not None else 0
            num_markers = len(marker_ids) if marker_ids is not None else 0

            print(f"  → Detected {num_markers} ArUco markers, {num_corners} ChArUco corners - SUCCESS")

            successful_captures += 1

            # Create debug visualization image
            debug_image = cv2.cvtColor(image2, cv2.COLOR_BGRA2BGR)

            # Draw detected ArUco markers
            if marker_corners is not None and marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(debug_image, marker_corners, marker_ids)

            # Draw detected ChArUco corners
            if charuco_corners is not None and charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(debug_image, charuco_corners, charuco_ids, (0, 255, 0))

            # Estimate board pose and draw axes
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) >= 4:
                # Get object points for the detected corners
                objPoints, imgPoints = CHARUCO_BOARD.matchImagePoints(charuco_corners, charuco_ids)

                # Solve PnP to get board pose
                success, rvec, tvec = cv2.solvePnP(
                    objPoints, imgPoints, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    # Draw coordinate axes (X=red, Y=green, Z=blue)
                    axis_length = 0.05  # 5cm axes
                    cv2.drawFrameAxes(debug_image, camera_matrix, dist_coeffs, rvec, tvec, axis_length, 3)

            # Save images and pose
            image_path = os.path.join(save_dir, f"capture_{capture_idx:03d}.png")
            debug_path = os.path.join(save_dir, f"capture_{capture_idx:03d}_debug.png")
            gray_path = os.path.join(save_dir, f"capture_{capture_idx:03d}_gray.png")
            pose_path = os.path.join(save_dir, f"capture_{capture_idx:03d}_pose.npy")
            cv2.imwrite(image_path, image2)
            cv2.imwrite(debug_path, debug_image)
            # Also save grayscale for verification
            gray_save = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            cv2.imwrite(gray_path, gray_save)
            np.save(pose_path, actual_ee_pose)

            # Store for calibration
            all_readings.append(readings)
            all_poses.append(actual_ee_pose)

            print(f"  → Saved capture {successful_captures}/{num_captures}")
        else:
            print(f"  → Board detection failed")

        capture_idx += 1

    print(f"\n{'='*60}")
    print("Data collection complete!")
    print(f"{'='*60}\n")

    # ========================================================================
    # Calibration
    # ========================================================================
    print("Running calibration...\n")

    # Calculate target-to-camera transformations
    target2cam_results = calculate_target_to_cam(all_readings, camera_matrix, dist_coeffs)

    if target2cam_results is None:
        raise RuntimeError("Failed to calculate target-to-camera transformations")

    R_target2cam, t_target2cam = target2cam_results

    # Prepare data for hand-eye calibration
    R_gripper2base = []
    t_gripper2base = []

    for pose_matrix in all_poses:
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

    # ========================================================================
    # Verification: Board Pose in World Frame
    # ========================================================================
    print(f"\n{'='*60}")
    print("Board Pose Verification (World Frame)")
    print(f"{'='*60}\n")

    # For each measurement, compute board pose in WORLD frame
    # X_WB = X_WG @ X_GC @ X_CT
    # where X_CT = inverse(X_TC)
    board_positions_world = []
    board_rotations_world = []

    for i, (R_t2c, t_t2c, X_WG) in enumerate(zip(R_target2cam, t_target2cam, all_poses)):
        # Build target-to-camera transform
        X_TC = np.eye(4)
        X_TC[:3, :3] = R_t2c
        X_TC[:3, 3] = t_t2c.flatten()

        # Invert to get camera-to-target
        X_CT = np.linalg.inv(X_TC)

        # Compute board in world frame: X_WB = X_WG @ X_GC @ X_CT
        X_WB = X_WG @ X_GC @ X_CT

        board_positions_world.append(X_WB[:3, 3])
        euler = R.from_matrix(X_WB[:3, :3]).as_euler('xyz', degrees=True)
        board_rotations_world.append(euler)

    board_positions_world = np.array(board_positions_world)  # Shape: (N, 3)
    board_rotations_world = np.array(board_rotations_world)  # Shape: (N, 3)

    # Calculate mean board position in world frame
    mean_position = np.mean(board_positions_world, axis=0)
    print(f"Mean board position (world frame): {mean_position} m")

    # Calculate variance of board positions (should be LOW for good calibration)
    position_std = np.std(board_positions_world, axis=0)
    print(f"Board position std dev (x, y, z): [{position_std[0]*1000:.4f}, {position_std[1]*1000:.4f}, {position_std[2]*1000:.4f}] mm")

    # Calculate variance of board rotations
    rotation_std = np.std(board_rotations_world, axis=0)
    print(f"Board rotation std dev (rx, ry, rz): [{rotation_std[0]:.4f}, {rotation_std[1]:.4f}, {rotation_std[2]:.4f}] deg")

    # Overall position spread
    position_spread = np.std(board_positions_world)
    print(f"\nOverall position spread (std dev): {position_spread * 1000:.4f} mm")
    print(f"Overall rotation spread (std dev): {np.std(board_rotations_world):.4f} deg")

    # Show range of positions
    position_range = np.max(board_positions_world, axis=0) - np.min(board_positions_world, axis=0)
    print(f"Position range (x, y, z): [{position_range[0]*1000:.4f}, {position_range[1]*1000:.4f}, {position_range[2]*1000:.4f}] mm")

    if position_spread > 0.01:  # 10mm
        print("\n⚠️  WARNING: Board position variance in world frame is high (>10mm)!")
        print("   This indicates calibration issues.")
    else:
        print("\n✓ Board position variance looks good (<10mm)")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    raise
finally:
    client.close()
    camera.close()
    print("\nRobot and camera closed.")
