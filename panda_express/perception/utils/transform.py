import numpy as np
from typing import Tuple

def pixel_to_world_xyz(
    u: int,
    v: int,
    depth_m: np.ndarray,
    K_zed: np.ndarray,
    T_world_zed: np.ndarray,
) -> np.ndarray:
    """Back-project a zed pixel (u, v) to world frame using depth, intrinsics, and T_world_zed."""
    H, W = depth_m.shape
    if v < 0 or v >= H or u < 0 or u >= W:
        raise ValueError("Pixel out of bounds")

    z = float(depth_m[v, u])
    if not np.isfinite(z) or z <= 0:
        # Small neighborhood fallback
        win = 3
        v0, v1 = max(0, v - win), min(H, v + win + 1)
        u0, u1 = max(0, u - win), min(W, u + win + 1)
        patch = depth_m[v0:v1, u0:u1]
        vals = patch[np.isfinite(patch) & (patch > 0)]
        if vals.size == 0:
            raise RuntimeError("No valid depth near pixel")
        z = float(np.median(vals))

    fx, fy = K_zed[0, 0], K_zed[1, 1]
    cx, cy = K_zed[0, 2], K_zed[1, 2]

    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    p_cam_h = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)

    p_body = (T_world_zed @ p_cam_h)[:3]
    return p_body


def depth_to_colored_pcd(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """takes a depth image from a camera with intrinsics K and extrinsics T_world_cam and returns an Nx3 pointcloud and Nx3 colors"""
    H, W = depth.shape

    # Extract intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten arrays
    u_flat = u.flatten()
    v_flat = v.flatten()
    z_flat = depth.flatten()
    rgb_flat = rgb.reshape(-1, 3)

    # Filter valid depth values
    valid_mask = np.isfinite(z_flat) & (z_flat > 0)
    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]
    z_valid = z_flat[valid_mask]
    colors_valid = rgb_flat[valid_mask]

    # Back-project to camera frame
    x_cam = (u_valid - cx) / fx * z_valid
    y_cam = (v_valid - cy) / fy * z_valid

    # Create homogeneous coordinates (Nx4)
    points_cam_h = np.stack([x_cam, y_cam, z_valid, np.ones_like(z_valid)], axis=1)

    # Transform to world frame
    points_world_h = (T_world_cam @ points_cam_h.T).T

    # Return xyz and rgb colors
    return points_world_h[:, :3], colors_valid
