import cv2
import numpy as np
import os
from datetime import datetime
from skills.go_to_conf import goto_hand_position
from bamboo.client import BambooFrankaClient
from scipy.spatial.transform import Rotation as R
from perception.zed.zed_cam import ZedCamera

# Setup ChArUco board
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((14, 9), 0.020, 0.015, dictionary)  
detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
successful_captures = 0

# Data collection
R_gripper2base = []  # Rotations from base to gripper
t_gripper2base = []  # Translations from base to gripper
R_target2cam = []    # Rotations from camera to board
t_target2cam = []    # Translations from camera to board

# Generate calibration poses (example)
def generate_calibration_poses(num_poses=30):
    poses = []
    # Generate poses with varied orientations around a central point
    # This is just an example - customize for your workspace
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
        # You'll need to convert rx, ry, rz to rotation matrix
        # Using your robot's convention (e.g., XYZ Euler, quaternions, etc.)
        X_WG[:3, 3] = pos
        X_WG[:3, :3] = grasp_orientation @ noise.as_matrix()
        poses.append(X_WG)
    
    return poses

# init robot and camera
client = BambooFrankaClient(server_ip="128.30.224.88")
camera = ZedCamera(serial_number = 16779706)
camera_matrix, dist_coeffs = camera.get_intrinsics()

# Create directory for saving calibration data
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"calibration_data/calibration_data_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
print(f"Saving calibration data to: {save_dir}")

try:
    # Data collection loop
    capture_idx = 0
    while successful_captures < 30:
        # Move robot
        X_WG = generate_calibration_poses(num_poses=5)[-1]
        result = goto_hand_position(client, X_WG, 2)

        # Check if trajectory execution was successful
        if result != 0:
            print(f"ERROR: Trajectory execution failed. Stopping calibration.")
            raise RuntimeError("Trajectory execution failed")

        # Get actual hand pose (not commanded pose)
        actual_ee_pose = np.array(client.get_joint_states()['ee_pose'])

        # Capture image from wrist camera
        image = camera.get_bgra_frame()
        image2 = np.copy(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)


        # Detect ChArUco board
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)

        if ids is not None and len(ids) > 4:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )

            print(f"Capture {capture_idx}: Detected {ret if ret else 0} ChArUco corners")

            if ret > 50:  # Need at least 4 corners
                successful_captures += 1
                # Save BGRA image and actual pose
                image_path = os.path.join(save_dir, f"capture_{capture_idx:03d}.png")
                pose_path = os.path.join(save_dir, f"capture_{capture_idx:03d}_pose.npy")
                cv2.imwrite(image_path, image2)
                np.save(pose_path, actual_ee_pose)

                # Get pose of board w.r.t camera
                ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, board,
                    camera_matrix, dist_coeffs, None, None
                )
                
                if ret:
                    # Convert to rotation matrix
                    R_cam2board, _ = cv2.Rodrigues(rvec)
                    
                    # We want board-to-camera, so invert
                    R_board2cam = R_cam2board.T
                    t_board2cam = -R_board2cam @ tvec.flatten()

                    # Store gripper-to-base transform (invert actual pose which is base-to-gripper)
                    R_base2gripper = actual_ee_pose[:3, :3]
                    t_base2gripper = actual_ee_pose[:3, 3]

                    # Invert to get gripper-to-base
                    R_gripper2base_inv = R_base2gripper.T
                    t_gripper2base_inv = -R_gripper2base_inv @ t_base2gripper

                    R_gripper2base.append(R_gripper2base_inv)
                    t_gripper2base.append(t_gripper2base_inv.reshape(3, 1))
                    R_target2cam.append(R_board2cam)
                    t_target2cam.append(t_board2cam.reshape(3, 1))

                    print(f"Collected pose {len(R_gripper2base)}")
            else:
                print(f"Capture {capture_idx}: Insufficient corners detected ({ret if ret else 0} < 50)")
        else:
            num_markers = len(ids) if ids is not None else 0
            print(f"Capture {capture_idx}: Insufficient ArUco markers detected ({num_markers})")

        capture_idx += 1

    # Solve hand-eye calibration
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI  # or PARK, HORAUD, ANDREFF, DANIILIDIS
    )

    # Build transform matrix
    X_GC = np.eye(4)
    X_GC[:3, :3] = R_cam2gripper
    X_GC[:3, 3] = t_cam2gripper.flatten()

    # Save extrinsics to calibration folder
    extrinsics_path = os.path.join(save_dir, "gripper_to_camera_extrinsics.npy")
    np.save(extrinsics_path, X_GC)
    print(f"Saved extrinsics to: {extrinsics_path}")

    print("Gripper-to-Camera Transform:")
    print(X_GC)
except Exception as e:
    print(f"error {e}")
    raise
finally:
    client.close()
    camera.close()
