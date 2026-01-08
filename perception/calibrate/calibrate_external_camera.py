"""
Calibrate external (fixed) camera relative to robot base.

Approach:
1. Keep robot fixed in one position
2. Manually move ChArUco board to ~20 different positions
3. Both wrist camera and external camera see the board
4. Compute transform from robot base to external camera

Math:
For each board position i:
  - X_base_to_target[i] = X_base_to_gripper @ X_gripper_to_wrist @ X_wrist_to_target[i]
  - X_base_to_external @ X_external_to_target[i] = X_base_to_target[i]
  - Therefore: X_base_to_external = X_base_to_target[i] @ X_target_to_external[i]

We compute this for all positions and average to get a robust estimate.
"""

import cv2
import numpy as np
import os
from datetime import datetime
from skills.go_to_conf import goto_hand_position
from bamboo.client import BambooFrankaClient
from scipy.spatial.transform import Rotation as R
from perception.zed.zed_cam import ZedCamera
from glob import glob

# ============================================================================
# Configuration
# ============================================================================
WRIST_CAMERA_SERIAL = 16779706
EXTERNAL_CAMERA_SERIAL = 35317039

NUM_SAMPLES = 20

# ChArUco Board Configuration (must match what you used for hand-eye calibration)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
CHARUCO_BOARD = cv2.aruco.CharucoBoard((14, 9), 0.020, 0.015, ARUCO_DICT)
detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

NUM_CORNER_THRESHOLD = 20  # Reduced from 50 for more forgiving detection

# ============================================================================
# Helper Functions
# ============================================================================

def detect_board_pose(image, camera_matrix, dist_coeffs):
    """
    Detect ChArUco board and return its pose relative to camera.
    Returns: (R_target2cam, t_target2cam) or None if detection fails
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detect markers
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, detectorParams=detector_params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        return None

    # Detect ChArUco corners
    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, CHARUCO_BOARD
    )

    if not ret or ret < NUM_CORNER_THRESHOLD:
        return None

    # Estimate pose
    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, CHARUCO_BOARD,
        camera_matrix, dist_coeffs, None, None
    )

    if not success:
        return None

    R_target2cam = cv2.Rodrigues(rvec)[0]
    t_target2cam = tvec.flatten()

    return R_target2cam, t_target2cam


def invert_transform(R, t):
    """Invert a rotation matrix and translation vector."""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def transform_to_matrix(R, t):
    """Convert rotation matrix and translation to 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def average_transforms(transforms):
    """
    Average multiple 4x4 transformation matrices.
    Uses simple mean for translation and SVD-based averaging for rotation.
    """
    # Extract rotations and translations
    rotations = [T[:3, :3] for T in transforms]
    translations = [T[:3, 3] for T in transforms]

    # Average translations (simple mean)
    t_avg = np.mean(translations, axis=0)

    # Average rotations using quaternion averaging
    quats = [R.from_matrix(rot).as_quat() for rot in rotations]

    # Ensure all quaternions are in the same hemisphere
    quats = np.array(quats)
    if np.dot(quats[0], quats[1]) < 0:
        quats[1] *= -1
    for i in range(2, len(quats)):
        if np.dot(quats[0], quats[i]) < 0:
            quats[i] *= -1

    # Average and normalize
    q_avg = np.mean(quats, axis=0)
    q_avg = q_avg / np.linalg.norm(q_avg)
    R_avg = R.from_quat(q_avg).as_matrix()

    # Build averaged transform
    T_avg = np.eye(4)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg

    return T_avg


# ============================================================================
# Main Calibration Script
# ============================================================================

print("="*80)
print("External Camera Calibration")
print("="*80)
print("\nThis script will calibrate the external camera relative to the robot base.")
print("Instructions:")
print("  1. Keep the robot in its current position (DO NOT MOVE IT)")
print("  2. Manually move the ChArUco board to different positions")
print("  3. Ensure both cameras can see the board at each position")
print("  4. Press SPACE to capture when ready, 'q' to finish early")
print("="*80)

# Load hand-eye calibration for wrist camera
print("\n\nLoading wrist camera hand-eye calibration...")
calib_dirs = sorted(glob("calibration_data/calibration_data_*"))
if not calib_dirs:
    raise FileNotFoundError("No hand-eye calibration found! Run calibrate_hand_eye_improved.py first")

latest_calib_dir = calib_dirs[-1]
wrist_extrinsics_path = os.path.join(latest_calib_dir, "gripper_to_camera_extrinsics.npy")

if not os.path.exists(wrist_extrinsics_path):
    raise FileNotFoundError(f"Wrist camera calibration not found at {wrist_extrinsics_path}")

X_gripper_to_wrist = np.load(wrist_extrinsics_path)
print(f"Loaded wrist camera calibration from: {wrist_extrinsics_path}")
print(f"X_gripper_to_wrist:\n{X_gripper_to_wrist}")

# Initialize cameras and robot
print("\nInitializing robot and cameras...")
client = BambooFrankaClient(server_ip="128.30.224.88")
X_WG = np.array([[1.0, 0.0, 0.0, 0.3], [0.0, -1, 0.0, 0.2], [0.0, 0.0, -1.0, 0.7], [0, 0, 0, 1.0]])
X_GG2 = np.eye(4)
X_GG2[:3, :3] = R.from_euler('xyz', [np.pi/6, 0, 0], degrees=False).as_matrix()
goto_hand_position(client, X_WG @ X_GG2, 3)
wrist_camera = ZedCamera(serial_number=WRIST_CAMERA_SERIAL)
wrist_cam_matrix, wrist_dist_coeffs = wrist_camera.get_intrinsics()

external_camera = ZedCamera(serial_number=EXTERNAL_CAMERA_SERIAL)
external_cam_matrix, external_dist_coeffs = external_camera.get_intrinsics()

# Get robot's current (fixed) pose
X_base_to_gripper = np.array(client.get_joint_states()['ee_pose'])

print(f"\nRobot gripper pose (KEEP FIXED):")
print(X_base_to_gripper)

# Compute base to wrist camera transform
X_base_to_wrist = X_base_to_gripper @ X_gripper_to_wrist
print(f"\nBase to wrist camera transform:")
print(X_base_to_wrist)

# Create save directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"calibration_data/external_camera_calib_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
print(f"\nSaving data to: {save_dir}")

# Data collection
all_base_to_target = []  # Target position in base frame (computed from wrist camera)
all_external_to_target = []  # Target position in external camera frame (detected)
sample_idx = 0

print("\n" + "="*80)
print("Ready to collect samples!")
print("Move the board to a new position, then press SPACE to capture")
print("Press 'q' when done (or after 20 samples)")
print("="*80 + "\n")

try:
    while sample_idx < NUM_SAMPLES:
        # Capture from both cameras
        wrist_image = wrist_camera.get_bgra_frame()
        external_image = external_camera.get_bgra_frame()

        # Detect board in wrist camera
        wrist_detection = detect_board_pose(wrist_image, wrist_cam_matrix, wrist_dist_coeffs)

        # Detect board in external camera
        external_detection = detect_board_pose(external_image, external_cam_matrix, external_dist_coeffs)

        # Show visualization
        wrist_viz = cv2.cvtColor(wrist_image, cv2.COLOR_BGRA2BGR)
        external_viz = cv2.cvtColor(external_image, cv2.COLOR_BGRA2BGR)

        # Add status text
        wrist_status = "Wrist: DETECTED" if wrist_detection else "Wrist: NOT DETECTED"
        external_status = "External: DETECTED" if external_detection else "External: NOT DETECTED"

        color = (0, 255, 0) if (wrist_detection and external_detection) else (0, 0, 255)
        cv2.putText(wrist_viz, wrist_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(external_viz, external_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(wrist_viz, f"Samples: {sample_idx}/{NUM_SAMPLES}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(external_viz, "Press SPACE to capture, 'q' to quit", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display
        cv2.imshow("Wrist Camera", wrist_viz)
        cv2.imshow("External Camera", external_viz)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nUser requested quit")
            break
        elif key == ord(' '):  # Spacebar
            if wrist_detection and external_detection:
                # Extract detections
                R_target2wrist, t_target2wrist = wrist_detection
                R_target2external, t_target2external = external_detection

                # Compute target position in base frame (using wrist camera)
                R_wrist2target, t_wrist2target = invert_transform(R_target2wrist, t_target2wrist)
                X_wrist2target = transform_to_matrix(R_wrist2target, t_wrist2target)
                X_base2target = X_base_to_wrist @ X_wrist2target

                # Store detections
                all_base_to_target.append(X_base2target)

                X_target2external = transform_to_matrix(R_target2external, t_target2external)
                all_external_to_target.append(X_target2external)

                # Save images
                cv2.imwrite(os.path.join(save_dir, f"wrist_{sample_idx:03d}.png"), wrist_image)
                cv2.imwrite(os.path.join(save_dir, f"external_{sample_idx:03d}.png"), external_image)

                sample_idx += 1
                print(f"Captured sample {sample_idx}/{NUM_SAMPLES}")
            else:
                print("Board not detected in both cameras! Move board and try again.")

    print("\n" + "="*80)
    print("Data collection complete!")
    print("="*80)

    if len(all_base_to_target) < 5:
        raise RuntimeError(f"Not enough samples: {len(all_base_to_target)} < 5")

    # Compute base to external camera transform for each sample
    print(f"\nComputing base-to-external transform from {len(all_base_to_target)} samples...")

    all_base_to_external = []
    for i in range(len(all_base_to_target)):
        X_base2target = all_base_to_target[i]
        X_target2external = all_external_to_target[i]

        # X_base2external @ X_external2target = X_base2target
        # Therefore: X_base2external = X_base2target @ X_target2external
        X_base2external = X_base2target @ X_target2external
        all_base_to_external.append(X_base2external)

    # Average all transforms
    X_base_to_external_avg = average_transforms(all_base_to_external)

    # Compute statistics
    translations = [T[:3, 3] for T in all_base_to_external]
    t_std = np.std(translations, axis=0)
    print(f"\nTranslation std dev: {t_std} (should be < 0.01 for good calibration)")

    # Save result
    result_path = os.path.join(save_dir, "base_to_external_camera.npy")
    np.save(result_path, X_base_to_external_avg)

    print("\n" + "="*80)
    print("Calibration Complete!")
    print("="*80)
    print(f"\nBase-to-External Camera Transform:")
    print(X_base_to_external_avg)
    print(f"\nTranslation: {X_base_to_external_avg[:3, 3]}")
    print(f"Rotation (euler XYZ): {R.from_matrix(X_base_to_external_avg[:3, :3]).as_euler('xyz', degrees=True)} degrees")
    print(f"\nSaved to: {result_path}")

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    cv2.destroyAllWindows()
    client.close()
    wrist_camera.close()
    if EXTERNAL_CAMERA_SERIAL is not None:
        external_camera.close()
    print("\nCameras and robot closed.")
