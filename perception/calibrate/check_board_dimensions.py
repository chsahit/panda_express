"""
Quick check to see if board dimensions might be wrong.

We'll look at the detected board distances and see if they're consistent
with the expected physical size.
"""

import cv2
import numpy as np
from glob import glob
import os

calib_dirs = sorted(glob("calibration_data/calibration_data_*"))
latest_dir = calib_dirs[-1]
image_files = sorted(glob(os.path.join(latest_dir, "*.png")))

from perception.zed.zed_cam import ZedCamera
camera = ZedCamera(serial_number=16779706)
camera_matrix, dist_coeffs = camera.get_intrinsics()
camera.close()

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
CHARUCO_BOARD = cv2.aruco.CharucoBoard((14, 9), 0.020, 0.015, ARUCO_DICT)
detector_params = cv2.aruco.DetectorParameters()

print("="*80)
print("Board Dimension Check")
print("="*80)
print("\nConfigured board parameters:")
print("  Square size: 20mm")
print("  Marker size: 15mm")
print("  Board size: 14 x 9 squares")
print("\nPhysically measure your board:")
print("  - Measure 10 squares edge-to-edge")
print("  - Should be: 10 * 20mm = 200mm")
print("  - If it's different, update the CHARUCO_BOARD dimensions")
print("\n" + "="*80)
print("Detected board distances from camera:")
print("="*80)

distances = []

for img_file in image_files[:10]:  # Check first 10
    image = cv2.imread(img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, detectorParams=detector_params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 4:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, CHARUCO_BOARD
        )

        if ret and ret > 50:
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, CHARUCO_BOARD,
                camera_matrix, dist_coeffs, None, None
            )

            if success:
                dist = np.linalg.norm(tvec)
                distances.append(dist)
                print(f"  {os.path.basename(img_file)}: {dist:.4f} m")

if distances:
    print(f"\nDistance statistics:")
    print(f"  Mean: {np.mean(distances):.4f} m")
    print(f"  Std: {np.std(distances):.4f} m")
    print(f"  Range: {np.min(distances):.4f} - {np.max(distances):.4f} m")

    if np.std(distances) < 0.02:  # < 2cm variation
        print("\n✓ Distances are consistent - board detections look good")
    else:
        print("\n⚠ Large variation in detected distances - might indicate:")
        print("    - Wrong board dimensions")
        print("    - Poor detections")
        print("    - Board not flat")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("1. Physically measure your ChArUco board")
print("2. Verify the square size is exactly 20mm")
print("3. Verify marker size is exactly 15mm")
print("4. If different, update CHARUCOBOARD_CHECKER_SIZE and CHARUCOBOARD_MARKER_SIZE")
print("   in all calibration scripts and re-run calibration")
