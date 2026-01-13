"""
Utility functions for integrating Contact-GraspNet predictions with robot control.

This module provides utilities for:
- Loading and converting camera calibration from JSON
- Loading grasp predictions from NPZ files
- Transforming grasps between coordinate frames
- Selecting and validating grasp candidates
"""

import json
import logging
import numpy as np
import runpy
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from scipy.spatial.transform import Rotation


# Configure logging
logger = logging.getLogger(__name__)

def _validate_homogeneous_transform(T: np.ndarray, *, source: str) -> None:
    if T.shape != (4, 4):
        raise ValueError(f"Invalid transform shape from {source}: {T.shape}, expected (4,4)")

    if not np.allclose(T[3, :], np.array([0, 0, 0, 1], dtype=float), atol=1e-6):
        raise ValueError(
            f"Invalid homogeneous transform from {source}: last row must be [0,0,0,1], got {T[3, :]}"
        )

    det = np.linalg.det(T[:3, :3])
    if not np.isfinite(det) or not np.isclose(det, 1.0, atol=0.1):
        raise ValueError(
            f"Invalid rotation in transform from {source}: det(R)={det:.6f}, expected ≈1.0"
        )


def _coerce_to_4x4_matrix(value: Any) -> Optional[np.ndarray]:
    """
    Best-effort coercion of a candidate value into a (4,4) float numpy array.
    Returns None if it cannot be coerced.
    """
    try:
        arr = np.array(value, dtype=float)
    except Exception:
        return None
    if arr.shape != (4, 4):
        return None
    return arr


def load_calibration_from_py(
    calib_py_path: str,
    matrix_key: Optional[str] = None,
    invert: bool = False,
) -> np.ndarray:
    """
    Load camera extrinsics from a Python file and return a 4x4 base_T_camera matrix.

    Supports calibration files that define one or more numpy arrays / lists at module scope,
    e.g. `T_base_external`, `base_T_camera`, `T_base_camera`.

    Args:
        calib_py_path: Path to a .py file containing the extrinsics matrix.
        matrix_key: Optional variable name to use (recommended). If None, first (4,4) found is used.
        invert: If True, return inverse of the loaded matrix (use if file stores camera_T_base).

    Returns:
        base_T_camera: (4,4) homogeneous transform from camera frame to robot base frame.
    """
    path = Path(calib_py_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration python file not found: {calib_py_path}\n"
            f"Please ensure the file exists and is readable."
        )

    # Execute file in isolated namespace
    ns = runpy.run_path(str(path))

    candidate = None
    chosen_key = matrix_key
    if matrix_key is not None:
        if matrix_key not in ns:
            raise KeyError(
                f"Matrix key '{matrix_key}' not found in calibration file: {calib_py_path}\n"
                f"Available keys: {[k for k in ns.keys() if not k.startswith('__')]}"
            )
        candidate = _coerce_to_4x4_matrix(ns[matrix_key])
        if candidate is None:
            raise KeyError(
                f"Key '{matrix_key}' exists but is not coercible to a (4,4) matrix."
            )
    else:
        # Heuristic: pick first coercible (4,4) matrix
        for k, v in ns.items():
            if k.startswith("__"):
                continue
            m = _coerce_to_4x4_matrix(v)
            if m is not None:
                candidate = m
                chosen_key = k
                break

        if candidate is None:
            raise ValueError(
                f"No (4,4) matrix found in calibration file: {calib_py_path}\n"
                f"Define a 4x4 numpy array at module scope (e.g., T_base_external)."
            )

    base_T_camera = candidate

    _validate_homogeneous_transform(base_T_camera, source=f"py:{calib_py_path} (key='{chosen_key}')")

    if invert:
        base_T_camera = np.linalg.inv(base_T_camera)
        _validate_homogeneous_transform(base_T_camera, source=f"inv(py:{calib_py_path} (key='{chosen_key}'))")

    logger.info(f"Loaded calibration from python file: {calib_py_path} (key='{chosen_key}', invert={invert})")
    logger.debug(f"  Translation: {base_T_camera[:3, 3]}")
    logger.debug(f"  det(R): {np.linalg.det(base_T_camera[:3, :3]):.6f}")
    return base_T_camera


def load_calibration_from_npy(calib_npy_path: str, invert: bool = False) -> np.ndarray:
    """
    Load camera extrinsics from a .npy file and return a 4x4 base_T_camera matrix.

    Args:
        calib_npy_path: Path to a .npy file containing a (4,4) matrix.
        invert: If True, return inverse of the loaded matrix (use if file stores camera_T_base).

    Returns:
        base_T_camera: (4,4) homogeneous transform from camera frame to robot base frame.
    """
    path = Path(calib_npy_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration npy file not found: {calib_npy_path}\n"
            f"Please ensure the file exists and is readable."
        )

    T = np.load(str(path))
    T = np.array(T, dtype=float)
    _validate_homogeneous_transform(T, source=f"npy:{calib_npy_path}")

    if invert:
        T = np.linalg.inv(T)
        _validate_homogeneous_transform(T, source=f"inv(npy:{calib_npy_path})")

    logger.info(f"Loaded calibration from npy file: {calib_npy_path} (invert={invert})")
    logger.debug(f"  Translation: {T[:3, 3]}")
    logger.debug(f"  det(R): {np.linalg.det(T[:3, :3]):.6f}")
    return T


def load_calibration_from_json(calib_json_path: str, camera_id: str) -> np.ndarray:
    """
    Load camera calibration from JSON and convert to 4x4 transformation matrix.

    Args:
        calib_json_path: Path to calibration_info.json file
        camera_id: Camera ID key in the JSON (e.g., "231122071283")

    Returns:
        base_T_camera: 4x4 homogeneous transformation matrix from camera to base frame

    Raises:
        FileNotFoundError: If calibration file doesn't exist
        KeyError: If camera_id not found in calibration file
        ValueError: If calibration data is malformed
    """
    calib_path = Path(calib_json_path)

    if not calib_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calib_json_path}\n"
            f"Please ensure the calibration file exists at the specified path."
        )

    # Load calibration JSON
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)

    # Extract camera calibration
    if camera_id not in calib_data:
        available_ids = list(calib_data.keys())
        raise KeyError(
            f"Camera ID '{camera_id}' not found in calibration file.\n"
            f"Available camera IDs: {available_ids}"
        )

    camera_calib = calib_data[camera_id]
    pose_6d = camera_calib.get('pose')

    if pose_6d is None or len(pose_6d) != 6:
        raise ValueError(
            f"Invalid calibration format for camera '{camera_id}'.\n"
            f"Expected 6D pose [x, y, z, rx, ry, rz], got: {pose_6d}"
        )

    # Extract position and rotation
    position = np.array(pose_6d[:3])  # [x, y, z] in meters
    euler_angles = np.array(pose_6d[3:])  # [rx, ry, rz] in radians

    # Convert Euler angles to rotation matrix (XYZ convention)
    rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()

    # Build 4x4 homogeneous transformation matrix
    base_T_camera = np.eye(4)
    base_T_camera[:3, :3] = rotation_matrix
    base_T_camera[:3, 3] = position

    logger.info(f"Loaded calibration for camera '{camera_id}'")
    logger.debug(f"  Position: {position}")
    logger.debug(f"  Euler angles (XYZ, rad): {euler_angles}")

    return base_T_camera


def load_grasp_predictions(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Contact-GraspNet predictions from NPZ file.

    Args:
        npz_path: Path to predictions .npz file

    Returns:
        grasps_cam: (N, 4, 4) grasp poses in camera frame
        scores: (N,) confidence scores for each grasp (0-1 range)
        contact_pts: (N, 3) contact points in camera frame

    Raises:
        FileNotFoundError: If NPZ file doesn't exist
        ValueError: If prediction data is malformed or missing required keys
    """
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {npz_path}\n"
            f"Please ensure Contact-GraspNet predictions have been generated."
        )

    # Load NPZ file
    results = np.load(npz_path, allow_pickle=True)

    # Check for required keys
    required_keys = ['pred_grasps_cam', 'scores', 'contact_pts']
    missing_keys = [key for key in required_keys if key not in results]
    if missing_keys:
        raise ValueError(
            f"Predictions file missing required keys: {missing_keys}\n"
            f"Available keys: {list(results.keys())}"
        )

    grasps = results['pred_grasps_cam']
    scores = results['scores']
    contacts = results['contact_pts']

    # Handle 0-d object arrays (extract from wrapper)
    if isinstance(grasps, np.ndarray) and grasps.shape == ():
        grasps = grasps.item()
    if isinstance(scores, np.ndarray) and scores.shape == ():
        scores = scores.item()
    if isinstance(contacts, np.ndarray) and contacts.shape == ():
        contacts = contacts.item()

    # Handle dictionary format (Contact-GraspNet sometimes stores as dict)
    if isinstance(grasps, dict):
        grasps = list(grasps.values())[0]
    if isinstance(scores, dict):
        scores = list(scores.values())[0]
    if isinstance(contacts, dict):
        contacts = list(contacts.values())[0]

    # Convert to numpy arrays if needed
    grasps = np.array(grasps) if not isinstance(grasps, np.ndarray) else grasps
    scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores
    contacts = np.array(contacts) if not isinstance(contacts, np.ndarray) else contacts

    # Validate shapes
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise ValueError(
            f"Invalid grasp shape: {grasps.shape}\n"
            f"Expected (N, 4, 4) for N grasp poses"
        )

    n_grasps = grasps.shape[0]
    if scores.shape != (n_grasps,):
        raise ValueError(
            f"Score shape mismatch: grasps={n_grasps}, scores={scores.shape}"
        )

    logger.info(f"Loaded predictions from: {npz_path}")
    logger.info(f"  Number of grasps: {n_grasps}")
    logger.info(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    logger.debug(f"  Grasps shape: {grasps.shape}")
    logger.debug(f"  Contacts shape: {contacts.shape}")

    return grasps, scores, contacts


def transform_grasps_to_base_frame(
    grasps_cam: np.ndarray,
    base_T_camera: np.ndarray
) -> np.ndarray:
    """
    Transform grasp poses from camera frame to robot base frame.

    Args:
        grasps_cam: (N, 4, 4) grasp poses in camera frame
        base_T_camera: (4, 4) transformation from camera to base frame

    Returns:
        grasps_base: (N, 4, 4) grasp poses in base frame

    Raises:
        ValueError: If input shapes are invalid
    """
    # Validate inputs
    if base_T_camera.shape != (4, 4):
        raise ValueError(
            f"Invalid calibration shape: {base_T_camera.shape}\n"
            f"Expected (4, 4) transformation matrix"
        )

    if grasps_cam.ndim != 3 or grasps_cam.shape[1:] != (4, 4):
        raise ValueError(
            f"Invalid grasps shape: {grasps_cam.shape}\n"
            f"Expected (N, 4, 4) for N grasp poses"
        )

    # Transform each grasp: grasps_base = base_T_camera @ grasps_cam
    grasps_base = np.array([base_T_camera @ grasp for grasp in grasps_cam])

    logger.debug(f"Transformed {len(grasps_cam)} grasps to base frame")

    return grasps_base


def select_best_grasp(
    grasps_base: np.ndarray,
    scores: np.ndarray,
    min_score: float = 0.0
) -> Tuple[np.ndarray, int, float]:
    """
    Select the best grasp based on confidence scores.

    Args:
        grasps_base: (N, 4, 4) grasp poses in base frame
        scores: (N,) confidence scores
        min_score: Minimum acceptable score (0-1 range)

    Returns:
        best_grasp: (4, 4) best grasp pose
        best_idx: Index of best grasp
        best_score: Confidence score of best grasp

    Raises:
        ValueError: If no grasps meet minimum score threshold or inputs are empty
    """
    if len(grasps_base) == 0 or len(scores) == 0:
        raise ValueError("Cannot select from empty grasp set")

    if len(grasps_base) != len(scores):
        raise ValueError(
            f"Grasp and score count mismatch: {len(grasps_base)} vs {len(scores)}"
        )

    # Filter by minimum score
    valid_mask = scores >= min_score
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        raise ValueError(
            f"No grasps meet minimum score threshold {min_score:.4f}\n"
            f"Best available score: {scores.max():.4f}"
        )

    # Select grasp with highest score
    valid_scores = scores[valid_mask]
    best_among_valid = np.argmax(valid_scores)
    best_idx = valid_indices[best_among_valid]
    best_score = scores[best_idx]
    best_grasp = grasps_base[best_idx]

    logger.info(f"Selected grasp {best_idx} with score {best_score:.4f}")
    logger.debug(f"  Position: {best_grasp[:3, 3]}")

    return best_grasp, best_idx, best_score


def compute_pregrasp_pose(grasp_pose: np.ndarray, offset_z: float = 0.2) -> np.ndarray:
    """
    Compute pre-grasp pose by adding Z-offset in world frame.

    Args:
        grasp_pose: (4, 4) grasp pose in world/base frame
        offset_z: Offset distance in meters (positive = above grasp)

    Returns:
        pregrasp_pose: (4, 4) pre-grasp pose

    Raises:
        ValueError: If grasp_pose has invalid shape
    """
    if grasp_pose.shape != (4, 4):
        raise ValueError(
            f"Invalid grasp pose shape: {grasp_pose.shape}\n"
            f"Expected (4, 4) transformation matrix"
        )

    # Copy grasp pose and offset position in Z (world frame)
    pregrasp_pose = grasp_pose.copy()
    pregrasp_pose[:3, 3] += np.array([0, 0, offset_z])

    logger.debug(f"Computed pre-grasp with Z-offset: {offset_z:.3f}m")
    logger.debug(f"  Grasp position: {grasp_pose[:3, 3]}")
    logger.debug(f"  Pre-grasp position: {pregrasp_pose[:3, 3]}")

    return pregrasp_pose


def validate_grasp_predictions(grasps: np.ndarray, scores: np.ndarray) -> bool:
    """
    Validate grasp prediction data integrity.

    Args:
        grasps: (N, 4, 4) grasp poses
        scores: (N,) confidence scores

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails with detailed error message
    """
    # Check shapes
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise ValueError(
            f"Invalid grasps shape: {grasps.shape}, expected (N, 4, 4)"
        )

    n_grasps = grasps.shape[0]
    if n_grasps == 0:
        raise ValueError("No grasps in prediction data")

    if scores.shape != (n_grasps,):
        raise ValueError(
            f"Shape mismatch: {n_grasps} grasps but {scores.shape} scores"
        )

    # Check score ranges
    if not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError(
            f"Scores out of range [0, 1]: min={scores.min():.4f}, max={scores.max():.4f}"
        )

    # Check rotation matrices are valid (determinant ≈ 1)
    for i, grasp in enumerate(grasps):
        R = grasp[:3, :3]
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=0.1):
            raise ValueError(
                f"Invalid rotation matrix at grasp {i}: det(R)={det:.4f}, expected ≈1.0"
            )

    logger.debug(f"Validation passed for {n_grasps} grasps")

    return True
