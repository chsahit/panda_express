#!/usr/bin/env python3
"""
Visualize a ZED RGB-D capture and Contact-GraspNet grasp proposals.

Inputs:
- ZED capture saved as .npy dict with keys: depth (H,W), rgb (H,W,3), K (3,3), segmap (H,W) [optional]
- Contact-GraspNet predictions .npz with keys: pred_grasps_cam, scores, contact_pts
- Extrinsics base_T_camera as .npy (4,4), used to visualize in base/world frame

This script reconstructs a pointcloud from depth + intrinsics (K), optionally transforms
the pointcloud and grasps into the robot base frame, and renders them with Open3D.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from contact_grasp_utils import load_grasp_predictions, load_calibration_from_npy


def load_zed_capture_npy(path: str) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=True)
    if not (isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object):
        raise ValueError(
            f"Expected {path} to be a 0-d object numpy array containing a dict, got shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}"
        )
    data = arr.item()
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a dict, got {type(data)}")
    for k in ("depth", "rgb", "K"):
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}. Available keys: {list(data.keys())}")
    return data


def reconstruct_pointcloud_from_depth(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    *,
    stride: int = 2,
    depth_min: float = 0.05,
    depth_max: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct XYZ (camera frame) + RGB (0..1) from depth + intrinsics.

    depth: (H,W) meters
    rgb:   (H,W,3) uint8
    K:     (3,3)
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H,W), got {depth.shape}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be (H,W,3), got {rgb.shape}")
    if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
        raise ValueError(f"rgb/depth shape mismatch: rgb={rgb.shape}, depth={depth.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if fx == 0 or fy == 0:
        raise ValueError(f"Invalid intrinsics: fx={fx}, fy={fy}")

    v = np.arange(0, H, stride)
    u = np.arange(0, W, stride)
    uu, vv = np.meshgrid(u, v)

    z = depth[vv, uu].astype(np.float32)
    valid = np.isfinite(z) & (z > depth_min) & (z < depth_max)
    uu = uu[valid].astype(np.float32)
    vv = vv[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    cols = rgb[vv.astype(np.int32), uu.astype(np.int32), :].astype(np.float32) / 255.0
    return pts, cols


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if T.shape != (4, 4):
        raise ValueError(f"T must be (4,4), got {T.shape}")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts must be (N,3), got {pts.shape}")
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    out = (T @ pts_h.T).T
    return out[:, :3]


def make_grasp_frame_mesh(T: np.ndarray, size: float) -> "o3d.geometry.TriangleMesh":
    # Lazy import so --no_gui can run without Open3D installed.
    import open3d as o3d  # type: ignore

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    mesh.transform(T)
    return mesh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zed_capture", type=str, default="/home/aditya/policies/infra/grasping/contact_graspnet/test_data/zed_capture.npy")
    ap.add_argument("--predictions", type=str, default="/home/aditya/policies/infra/grasping/contact_graspnet/results/predictions_zed_capture.npz")
    ap.add_argument("--calib_npy", type=str, default="/home/aditya/policies/infra/calibration/dual_zed_calib_20260106_202824/T_base_external.npy")
    ap.add_argument("--calib_invert", action="store_true", help="Invert calib matrix if it is camera_T_base instead of base_T_camera.")

    ap.add_argument("--frame", choices=["camera", "base"], default="base", help="Frame to visualize in.")
    ap.add_argument("--stride", type=int, default=2, help="Pixel stride when reconstructing pointcloud (higher = fewer points).")
    ap.add_argument("--depth_min", type=float, default=0.05)
    ap.add_argument("--depth_max", type=float, default=2.0)
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size in meters (0 = disabled).")

    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--min_score", type=float, default=0.0)
    ap.add_argument("--grasp_axis_size", type=float, default=0.06)
    ap.add_argument("--show_contact_pts", action="store_true")

    ap.add_argument("--no_gui", action="store_true", help="Do not open Open3D GUI; just print stats.")
    ap.add_argument("--write_ply", type=str, default=None, help="Write reconstructed pointcloud to a .ply (in chosen frame).")

    # Frames / origin visualization
    ap.add_argument("--show_origin_frame", action="store_true", default=True, help="Show a coordinate frame at the visualization origin.")
    ap.add_argument("--origin_frame_size", type=float, default=0.15, help="Size of the origin coordinate frame.")
    ap.add_argument("--show_origin_marker", action="store_true", help="Show a small sphere at the origin.")
    ap.add_argument("--origin_marker_radius", type=float, default=0.012)
    ap.add_argument("--show_base_frame", default=True, help="Show robot base frame axes (in the chosen visualization frame).")
    ap.add_argument("--base_frame_size", type=float, default=0.12)
    ap.add_argument("--show_camera_frame", default=True, help="Show camera frame axes (in the chosen visualization frame).")
    ap.add_argument("--camera_frame_size", type=float, default=0.10)
    args = ap.parse_args()

    zed_path = Path(args.zed_capture)
    pred_path = Path(args.predictions)
    calib_path = Path(args.calib_npy)
    if not zed_path.exists():
        raise FileNotFoundError(f"zed_capture not found: {zed_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions not found: {pred_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"calib_npy not found: {calib_path}")

    zed = load_zed_capture_npy(str(zed_path))
    depth = np.array(zed["depth"], dtype=np.float32)
    rgb = np.array(zed["rgb"], dtype=np.uint8)
    K = np.array(zed["K"], dtype=float)

    pts_cam, cols = reconstruct_pointcloud_from_depth(
        depth, rgb, K, stride=args.stride, depth_min=args.depth_min, depth_max=args.depth_max
    )

    base_T_camera = load_calibration_from_npy(str(calib_path), invert=args.calib_invert)

    if args.frame == "base":
        pts_vis = transform_points(base_T_camera, pts_cam)
    else:
        pts_vis = pts_cam

    grasps_cam, scores, contact_pts = load_grasp_predictions(str(pred_path))

    valid = scores >= float(args.min_score)
    idxs = np.where(valid)[0]
    idxs = idxs[np.argsort(scores[idxs])[::-1]]
    idxs = idxs[: max(0, int(args.top_k))]

    if args.no_gui and args.write_ply is None:
        print(f"Pointcloud: {pts_vis.shape[0]} points (frame={args.frame})")
        print(f"Grasps: showing {len(idxs)} / {len(scores)} (min_score={args.min_score})")
        if len(idxs):
            print("Top grasp indices/scores:")
            for i in idxs[: min(10, len(idxs))]:
                print(f"  idx={int(i):3d} score={float(scores[i]):.4f}")
        return 0

    # Open3D is only required if we are writing PLY or opening the GUI.
    try:
        import open3d as o3d  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "open3d is required for visualization/PLY export. Install it in your environment, e.g.:\n"
            "  pip install open3d"
        ) from e

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_vis.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

    if args.voxel and args.voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel))

    if args.write_ply:
        out = Path(args.write_ply)
        out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out), pcd)
        if args.no_gui:
            print(f"Wrote pointcloud to: {out}")
        return 0

    geoms = []

    # Origin frame / marker
    if args.show_origin_frame:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.origin_frame_size)))
    if args.show_origin_marker:
        s0 = o3d.geometry.TriangleMesh.create_sphere(radius=float(args.origin_marker_radius))
        s0.paint_uniform_color([1.0, 1.0, 1.0])
        geoms.append(s0)

    # Show base and camera frames for debugging extrinsics.
    # If visualizing in base frame:
    #   - base frame is identity
    #   - camera frame is base_T_camera
    # If visualizing in camera frame:
    #   - camera frame is identity
    #   - base frame is inv(base_T_camera)
    if args.frame == "base":
        vis_T_base = np.eye(4, dtype=float)
        vis_T_cam = base_T_camera.astype(np.float64)
    else:
        vis_T_base = np.linalg.inv(base_T_camera).astype(np.float64)
        vis_T_cam = np.eye(4, dtype=float)

    if args.show_base_frame:
        base_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.base_frame_size))
        base_axes.transform(vis_T_base)
        geoms.append(base_axes)

    if args.show_camera_frame:
        cam_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.camera_frame_size))
        cam_axes.transform(vis_T_cam)
        geoms.append(cam_axes)

    geoms.append(pcd)

    # Grasp axes
    for i in idxs:
        T = grasps_cam[i].astype(np.float64)
        if args.frame == "base":
            T = (base_T_camera @ T).astype(np.float64)
        geoms.append(make_grasp_frame_mesh(T, size=float(args.grasp_axis_size)))

    # Contact points (as small spheres)
    if args.show_contact_pts and len(idxs):
        pts = contact_pts[idxs].astype(np.float64)
        if args.frame == "base":
            pts = transform_points(base_T_camera, pts)
        for p in pts:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
            s.translate(p)
            s.paint_uniform_color([1.0, 0.2, 0.2])
            geoms.append(s)

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Contact-GraspNet Visualization ({args.frame} frame)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

