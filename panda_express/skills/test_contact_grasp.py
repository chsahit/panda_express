#!/usr/bin/env python3
"""
Test script for Contact-GraspNet integration (without robot).

This script tests the data loading and transformation pipeline without
requiring a robot connection.
"""

import numpy as np
import sys
from pathlib import Path

# Add skills directory to path
sys.path.insert(0, str(Path(__file__).parent))

from contact_grasp_utils import (
    load_calibration_from_npy,
    load_grasp_predictions,
    transform_grasps_to_base_frame,
    select_best_grasp,
    compute_pregrasp_pose,
    validate_grasp_predictions
)


def test_integration():
    """Test the full integration pipeline without robot execution."""
    print("=" * 70)
    print("TESTING CONTACT-GRASPNET INTEGRATION")
    print("=" * 70)

    # File paths
    predictions_file = "/home/aditya/policies/infra/grasping/contact_graspnet/results/predictions_zed_capture.npz"
    calib_npy_path = "/home/aditya/policies/infra/calibration/dual_zed_calib_20260106_202824/T_base_external.npy"

    try:
        # Test 1: Load calibration
        print("\n[TEST 1] Loading calibration...")
        base_T_camera = load_calibration_from_npy(calib_npy_path, invert=False)
        print(f"  Using npy calibration: {calib_npy_path}")
        print(f"  ✓ Calibration loaded successfully")
        print(f"  Shape: {base_T_camera.shape}")
        print(f"  Translation: {base_T_camera[:3, 3]}")
        print(f"  Rotation det: {np.linalg.det(base_T_camera[:3, :3]):.6f}")

        # Test 2: Load predictions
        print("\n[TEST 2] Loading predictions...")
        grasps_cam, scores, contact_pts = load_grasp_predictions(predictions_file)
        print(f"  ✓ Predictions loaded successfully")
        print(f"  Number of grasps: {len(grasps_cam)}")
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        # Test 3: Validate predictions
        print("\n[TEST 3] Validating predictions...")
        validate_grasp_predictions(grasps_cam, scores)
        print(f"  ✓ Validation passed")

        # Test 4: Transform to base frame
        print("\n[TEST 4] Transforming to base frame...")
        grasps_base = transform_grasps_to_base_frame(grasps_cam, base_T_camera)
        print(f"  ✓ Transformation complete")
        print(f"  Grasps in base frame shape: {grasps_base.shape}")

        # Test 5: Select best grasp
        print("\n[TEST 5] Selecting best grasp...")
        best_grasp, best_idx, best_score = select_best_grasp(grasps_base, scores, min_score=0.0)
        print(f"  ✓ Best grasp selected")
        print(f"  Index: {best_idx}")
        print(f"  Score: {best_score:.4f}")
        print(f"  Position (base frame): {best_grasp[:3, 3]}")

        # Test 6: Compute pre-grasp
        print("\n[TEST 6] Computing pre-grasp pose...")
        pregrasp_pose = compute_pregrasp_pose(best_grasp, offset_z=0.2)
        print(f"  ✓ Pre-grasp computed")
        print(f"  Pre-grasp position: {pregrasp_pose[:3, 3]}")
        print(f"  Z-offset: {pregrasp_pose[2, 3] - best_grasp[2, 3]:.3f}m")

        # Summary
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  - Loaded {len(grasps_cam)} grasp predictions")
        print(f"  - Best grasp: #{best_idx} with score {best_score:.4f}")
        print(f"  - Grasp position: [{best_grasp[0,3]:.3f}, {best_grasp[1,3]:.3f}, {best_grasp[2,3]:.3f}]")
        print(f"  - Pre-grasp position: [{pregrasp_pose[0,3]:.3f}, {pregrasp_pose[1,3]:.3f}, {pregrasp_pose[2,3]:.3f}]")
        print(f"\nReady for robot execution!")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
