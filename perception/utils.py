import numpy as np

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

