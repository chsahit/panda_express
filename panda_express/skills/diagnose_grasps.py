#!/usr/bin/env python3
"""
Diagnostic script to analyze Contact-GraspNet predictions and their reachability.

This helps debug why grasps might be failing IK.
"""

import numpy as np
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation

# Add skills directory to path
sys.path.insert(0, str(Path(__file__).parent))

from contact_grasp_utils import (
    load_calibration_from_npy,
    load_grasp_predictions,
    transform_grasps_to_base_frame,
)


def analyze_grasps():
    """Analyze grasp predictions and their properties."""
    print("=" * 70)
    print("GRASP PREDICTIONS DIAGNOSTIC")
    print("=" * 70)

    # File paths
    predictions_file = "/home/aditya/policies/infra/grasping/contact_graspnet/results/predictions_zed_capture.npz"
    calib_npy_path = "/home/aditya/policies/infra/calibration/dual_zed_calib_20260106_202824/T_base_external.npy"

    # Load data
    print("\n[1] Loading calibration and predictions...")
    base_T_camera = load_calibration_from_npy(calib_npy_path, invert=False)
    grasps_cam, scores, contact_pts = load_grasp_predictions(predictions_file)
    grasps_base = transform_grasps_to_base_frame(grasps_cam, base_T_camera)

    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    print("\n[2] Analyzing top 5 grasps:")
    print("=" * 70)

    for i in range(min(5, len(grasps_base))):
        idx = sorted_indices[i]
        grasp = grasps_base[idx]
        score = scores[idx]

        # Extract components
        position = grasp[:3, 3]
        R = grasp[:3, :3]

        # Convert rotation to euler angles for interpretation
        rot = Rotation.from_matrix(R)
        euler_deg = rot.as_euler('xyz', degrees=True)

        print(f"\nGrasp #{idx} (Rank {i+1}):")
        print(f"  Score: {score:.4f}")
        print(f"  Position (base frame):")
        print(f"    X: {position[0]:7.4f} m")
        print(f"    Y: {position[1]:7.4f} m")
        print(f"    Z: {position[2]:7.4f} m")
        print(f"  Orientation (Euler XYZ, degrees):")
        print(f"    Roll:  {euler_deg[0]:7.2f}°")
        print(f"    Pitch: {euler_deg[1]:7.2f}°")
        print(f"    Yaw:   {euler_deg[2]:7.2f}°")

        # Check if rotation matrix is valid
        det = np.linalg.det(R)
        print(f"  Rotation matrix det: {det:.6f} (should be ~1.0)")

        # Compute approach direction (Z-axis of grasp frame)
        approach = R[:, 2]
        print(f"  Approach direction: [{approach[0]:6.3f}, {approach[1]:6.3f}, {approach[2]:6.3f}]")

        # Distance from robot base
        dist = np.linalg.norm(position)
        print(f"  Distance from base: {dist:.4f} m")

    print("\n" + "=" * 70)
    print("[3] Calibration Info:")
    print("=" * 70)
    print(f"Translation (camera to base): {base_T_camera[:3, 3]}")
    rot_calib = Rotation.from_matrix(base_T_camera[:3, :3])
    euler_calib = rot_calib.as_euler('xyz', degrees=True)
    print(f"Rotation (Euler XYZ, degrees): [{euler_calib[0]:.2f}°, {euler_calib[1]:.2f}°, {euler_calib[2]:.2f}°]")

    print("\n" + "=" * 70)
    print("[4] Workspace Analysis:")
    print("=" * 70)
    print("Typical Franka robot workspace:")
    print("  Radius: ~0.85m from base")
    print("  Height: -0.2m to +0.9m from base")
    print("\nCurrent grasp positions:")
    all_positions = np.array([grasps_base[i][:3, 3] for i in range(len(grasps_base))])
    print(f"  X range: [{all_positions[:, 0].min():.3f}, {all_positions[:, 0].max():.3f}] m")
    print(f"  Y range: [{all_positions[:, 1].min():.3f}, {all_positions[:, 1].max():.3f}] m")
    print(f"  Z range: [{all_positions[:, 2].min():.3f}, {all_positions[:, 2].max():.3f}] m")

    distances = np.linalg.norm(all_positions, axis=1)
    print(f"  Distance range: [{distances.min():.3f}, {distances.max():.3f}] m")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nSuggestions:")
    print("1. Check if grasp positions are within robot workspace")
    print("2. Verify calibration is correct (visualize in RViz or similar)")
    print("3. Consider grasp orientations - some may require joint limits exceeded")
    print("4. Try adjusting pre-grasp offset or approach direction")


if __name__ == "__main__":
    analyze_grasps()
