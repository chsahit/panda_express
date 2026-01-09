"""
Simple test to verify if the hand-eye calibration makes sense.

We'll:
1. Move the robot to a known position
2. Detect the board with the wrist camera
3. Use hand-eye calibration to compute where the board is in base frame
4. Check if this position makes physical sense
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from glob import glob
import os

from bamboo.client import BambooFrankaClient
from perception.zed.zed_cam import ZedCamera

# Load hand-eye calibration
hand_eye_dirs = sorted(glob("calibration_data/calibration_data_*"))
if not hand_eye_dirs:
    print("No hand-eye calibration found!")
    exit(1)

latest_calib = hand_eye_dirs[-1]
X_gripper_to_wrist = np.load(os.path.join(latest_calib, "gripper_to_camera_extrinsics.npy"))

print("="*80)
print("Hand-Eye Calibration Test")
print("="*80)
print("\nLoaded calibration:")
print(X_gripper_to_wrist)
print(f"\nTranslation: {X_gripper_to_wrist[:3, 3]}")
print(f"Translation magnitude: {np.linalg.norm(X_gripper_to_wrist[:3, 3]):.4f} m")
print(f"Rotation (euler): {R.from_matrix(X_gripper_to_wrist[:3, :3]).as_euler('xyz', degrees=True)}")

# Initialize robot and camera
client = BambooFrankaClient(server_ip="128.30.224.88")
camera = ZedCamera(serial_number=16779706)
camera_matrix, dist_coeffs = camera.get_intrinsics()

# Setup ChArUco board
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
CHARUCO_BOARD = cv2.aruco.CharucoBoard((14, 9), 0.020, 0.015, ARUCO_DICT)
detector_params = cv2.aruco.DetectorParameters()

try:
    # Get current robot pose
    X_base_to_gripper = np.array(client.get_joint_states()['ee_pose'])

    print("\n" + "="*80)
    print("Current robot pose (base to gripper):")
    print("="*80)
    print(X_base_to_gripper)
    print(f"\nGripper position in base frame: {X_base_to_gripper[:3, 3]}")

    # Compute base to wrist camera transform
    X_base_to_wrist = X_base_to_gripper @ X_gripper_to_wrist
    print("\n" + "="*80)
    print("Computed wrist camera pose (base to wrist):")
    print("="*80)
    print(X_base_to_wrist)
    print(f"\nWrist camera position in base frame: {X_base_to_wrist[:3, 3]}")
    print("\nDoes this position make sense?")
    print(f"  - Should be close to gripper position ({X_base_to_gripper[:3, 3]})")
    print(f"  - Difference: {X_base_to_wrist[:3, 3] - X_base_to_gripper[:3, 3]}")
    print(f"  - Magnitude of difference: {np.linalg.norm(X_base_to_wrist[:3, 3] - X_base_to_gripper[:3, 3]):.4f} m")

    # Capture image and detect board
    print("\n" + "="*80)
    print("Detecting calibration board...")
    print("="*80)
    print("Place the board in view of the wrist camera and press Enter...")
    input()

    image = camera.get_bgra_frame()
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, detectorParams=detector_params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        print("✗ No markers detected!")
        exit(1)

    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, CHARUCO_BOARD
    )

    if not ret or ret < 20:
        print(f"✗ Insufficient corners: {ret}")
        exit(1)

    print(f"✓ Detected {ret} corners")

    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, CHARUCO_BOARD,
        camera_matrix, dist_coeffs, None, None
    )

    if not success:
        print("✗ Pose estimation failed!")
        exit(1)

    R_target2wrist = cv2.Rodrigues(rvec)[0]
    t_target2wrist = tvec.flatten()

    print(f"\nBoard position relative to wrist camera: {t_target2wrist}")
    print(f"Distance from wrist camera: {np.linalg.norm(t_target2wrist):.4f} m")

    # Compute board position in base frame
    R_wrist2target = R_target2wrist.T
    t_wrist2target = -R_wrist2target @ t_target2wrist

    X_wrist2target = np.eye(4)
    X_wrist2target[:3, :3] = R_wrist2target
    X_wrist2target[:3, 3] = t_wrist2target

    X_base2target = X_base_to_wrist @ X_wrist2target

    print("\n" + "="*80)
    print("Board position in BASE frame (computed via hand-eye calibration):")
    print("="*80)
    print(f"Position: {X_base2target[:3, 3]}")
    print(f"\nDoes this make sense?")
    print(f"  - Should be somewhere in the robot workspace (typically 0.2-0.8m from base)")
    print(f"  - Distance from base: {np.linalg.norm(X_base2target[:3, 3]):.4f} m")
    print(f"  - X coordinate: {X_base2target[0, 3]:.4f} m (forward/backward from base)")
    print(f"  - Y coordinate: {X_base2target[1, 3]:.4f} m (left/right from base)")
    print(f"  - Z coordinate: {X_base2target[2, 3]:.4f} m (height from base)")

    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("If the board position in BASE frame seems wrong (e.g., outside the workspace,")
    print("or in a physically impossible location), then the hand-eye calibration is incorrect.")
    print("\nThis would explain why the external camera calibration has 10cm errors -")
    print("the errors compound when using an incorrect hand-eye calibration.")

finally:
    client.close()
    camera.close()
