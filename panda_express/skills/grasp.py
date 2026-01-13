import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, Any

from bamboo.client import BambooFrankaClient
from skills.skills_config import intrinsics, X_GC
from skills.go_to_conf import goto_hand_position
from skills.contact_grasp_utils import (
    load_calibration_from_npy,
    load_grasp_predictions,
    transform_grasps_to_base_frame,
    select_best_grasp,
    compute_pregrasp_pose,
    validate_grasp_predictions
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def grasp_at_pixel(robot: BambooFrankaClient, rgbd: np.ndarray, pixel: np.ndarray):
    # get xyz position p_WP by using pixel (x, y) to look up a depth coordinate in rgbd
    # and project to world coordinates

    p_WP = None
    p_WP = np.array([0.4, 0.0, 0.3])
    # compute hand position relative to xyz position (assume top-down grasp)
    grasp_orientation = np.array([[1.0, 0.0, 0.0], [0.0, -1, 0], [-0.0, 0, -1.0]])

    X_WPregrasp = np.eye(4)
    X_WPregrasp[:3, :3] = grasp_orientation
    X_WPregrasp[:3, 3] = np.copy(p_WP) + np.array([0, 0, 0.2])

    X_WGrasp = np.eye(4)
    X_WGrasp[:3, :3] = grasp_orientation
    X_WGrasp[:3, 3] = np.copy(p_WP) + np.array([0, 0, 0.1])

    # move robot to hand position
    goto_hand_position(robot, X_WPregrasp, 5.0)
    goto_hand_position(robot, X_WGrasp, 2.0)

    # close fingers
    robot.close_gripper()


def grasp_from_contact_graspnet(
    robot: BambooFrankaClient,
    predictions_npz_path: str,
    calib_npy_path: str = "/home/aditya/policies/infra/calibration/dual_zed_calib_20260106_202824/T_base_external.npy",
    calib_npy_invert: bool = False,
    min_score: float = 0.0,
    pregrasp_offset_z: float = 0.2,
    pregrasp_time: float = 5.0,
    grasp_time: float = 2.0,
    use_pregrasp: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Execute grasp using Contact-GraspNet predictions.

    Loads grasp predictions from Contact-GraspNet, transforms them from camera frame
    to robot base frame using calibration, selects the best grasp, and executes it.

    Args:
        robot: BambooFrankaClient instance for robot control
        predictions_npz_path: Path to Contact-GraspNet predictions .npz file
        calib_json_path: Path to calibration_info.json file
        camera_id: Camera ID in calibration file (default: "231122071283" for ZED)
        min_score: Minimum confidence score threshold (0-1 range)
        pregrasp_offset_z: Pre-grasp height offset above grasp in meters
        pregrasp_time: Time to reach pre-grasp pose in seconds
        grasp_time: Time to reach grasp pose in seconds
        use_pregrasp: If True, move to pre-grasp first; if False, go directly to grasp
        verbose: Print execution details

    Returns:
        Dict with execution status:
        - 'success': bool indicating if grasp executed successfully
        - 'grasp_pose': 4x4 transformation matrix of executed grasp
        - 'pregrasp_pose': 4x4 transformation matrix of pre-grasp
        - 'score': confidence score of executed grasp
        - 'index': index of grasp in predictions
        - 'error': error message if failed (None if successful)

    Raises:
        FileNotFoundError: If predictions or calibration files not found
        ValueError: If prediction data is invalid
        RuntimeError: If robot execution fails
    """
    result = {
        'success': False,
        'grasp_pose': None,
        'pregrasp_pose': None,
        'score': None,
        'index': None,
        'error': None
    }

    try:
        # Step 1: Load calibration (camera -> base extrinsics)
        if verbose:
            print("=" * 70)
            print("CONTACT-GRASPNET GRASP EXECUTION")
            print("=" * 70)
            print(f"Loading calibration from npy: {calib_npy_path} (invert={calib_npy_invert})")

        base_T_camera = load_calibration_from_npy(calib_npy_path, invert=calib_npy_invert)

        # Step 2: Load predictions
        if verbose:
            print(f"Loading predictions from: {predictions_npz_path}")

        grasps_cam, scores, contact_pts = load_grasp_predictions(predictions_npz_path)

        # Step 3: Validate predictions
        validate_grasp_predictions(grasps_cam, scores)

        # Step 4: Transform to base frame
        if verbose:
            print("Transforming grasps to robot base frame...")

        grasps_base = transform_grasps_to_base_frame(grasps_cam, base_T_camera)

        # Step 5: Try top grasps until one succeeds (fallback mechanism)
        if verbose:
            print(f"Selecting best grasps (min_score={min_score:.2f})...")

        # Sort grasps by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        max_attempts = min(3, len(sorted_indices))  # Try up to 3 best grasps

        grasp_executed = False
        for attempt in range(max_attempts):
            best_idx = sorted_indices[attempt]
            best_score = scores[best_idx]
            best_grasp = grasps_base[best_idx]

            if best_score < min_score:
                if verbose:
                    print(f"\nGrasp #{best_idx} score {best_score:.4f} below threshold {min_score:.2f}")
                continue

            if verbose:
                print(f"\nAttempt {attempt + 1}/{max_attempts}: Grasp #{best_idx}")
                print(f"  Score: {best_score:.4f}")
                print(f"  Position (base frame): {best_grasp[:3, 3]}")

            # Step 6: Compute pre-grasp pose
            pregrasp_pose = compute_pregrasp_pose(best_grasp, pregrasp_offset_z)

            try:
                # Step 7: Execute motion to pre-grasp (optional)
                if use_pregrasp:
                    if verbose:
                        print(f"  Executing pre-grasp (offset: {pregrasp_offset_z:.2f}m, time: {pregrasp_time:.1f}s)...")
                    goto_hand_position(robot, pregrasp_pose, pregrasp_time)
                else:
                    if verbose:
                        print(f"  Skipping pre-grasp, going directly to grasp pose...")

                # Step 8: Execute motion to grasp
                if verbose:
                    print(f"  Executing grasp (time: {grasp_time:.1f}s)...")

                goto_hand_position(robot, best_grasp, grasp_time)

                # If we got here, grasp succeeded
                grasp_executed = True
                if verbose:
                    print(f"  ✓ Grasp executed successfully!")
                break

            except RuntimeError as e:
                if "IK solution" in str(e):
                    if verbose:
                        print(f"  ✗ IK failed for grasp #{best_idx}, trying next best...")
                    if attempt == max_attempts - 1:
                        raise ValueError(f"All {max_attempts} top grasps failed IK. Best score was {scores[sorted_indices[0]]:.4f}")
                else:
                    raise

        if not grasp_executed:
            raise ValueError(f"No grasps could be executed (tried {max_attempts} grasps)")

        # Step 9: Close gripper
        if verbose:
            print("Closing gripper...")

        robot.close_gripper()

        # Success!
        result['success'] = True
        result['grasp_pose'] = best_grasp
        result['pregrasp_pose'] = pregrasp_pose
        result['score'] = best_score
        result['index'] = best_idx

        if verbose:
            print("\n" + "=" * 70)
            print("GRASP EXECUTION SUCCESSFUL")
            print("=" * 70)

        return result

    except FileNotFoundError as e:
        error_msg = f"File not found: {e}"
        logger.error(error_msg)
        result['error'] = error_msg
        if verbose:
            print(f"\nERROR: {error_msg}")
        raise

    except ValueError as e:
        error_msg = f"Invalid data: {e}"
        logger.error(error_msg)
        result['error'] = error_msg
        if verbose:
            print(f"\nERROR: {error_msg}")
        raise

    except Exception as e:
        error_msg = f"Execution failed: {e}"
        logger.error(error_msg)
        result['error'] = error_msg
        if verbose:
            print(f"\nERROR: {error_msg}")
        raise RuntimeError(error_msg) from e


if __name__ == "__main__":
    # Example usage: Execute grasp from Contact-GraspNet predictions
    predictions_file = "/home/aditya/policies/infra/grasping/contact_graspnet/results/predictions_zed_capture.npz"

    with BambooFrankaClient(server_ip="128.30.224.88") as rob:
        # Use Contact-GraspNet predictions
        result = grasp_from_contact_graspnet(
            robot=rob,
            predictions_npz_path=predictions_file,
            use_pregrasp=False,  # Skip pre-grasp, go directly to grasp
            verbose=True
        )

        if result['success']:
            print(f"\nGrasp executed successfully!")
            print(f"Used grasp #{result['index']} with score {result['score']:.4f}")

        # Legacy pixel-based grasp (commented out)
        # grasp_at_pixel(rob, None, None)