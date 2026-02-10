#!/usr/bin/env python3
"""Standalone script to test grasp visualization without robot hardware.

Usage:
    # Test with live camera and M2T2 server
    python test_grasp_visualization.py

    # Test with saved point cloud (if you have one)
    python test_grasp_visualization.py --pcd-file my_pcd.npy
"""
import argparse
import numpy as np
import cv2
from PIL import Image

from panda_express.perception.zed.zed_cam import ZedCamera
from panda_express.perception.utils.transform import pixel_to_world_xyz, depth_to_colored_pcd
from panda_express.skills.grasp_vlm import visualize_m2t2_grasps


def test_with_camera(server_url: str = "http://0.0.0.0:8123", click_mode: bool = True):
    """Test visualization with live camera feed.

    Args:
        server_url: URL of M2T2 server
        click_mode: If True, click on image to select point. If False, use center.
    """
    print("Capturing from ZED camera...")
    cam = ZedCamera(serial_number=35317039)
    bgra = cam.get_bgra_frame()
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    depth = cam.get_depth_frame()
    K = cam.get_intrinsics()[0]
    cam.close()

    extrinsics = np.load("panda_express/perception/zed/X_WE.npy")

    if click_mode:
        print("\nClick on the object you want to grasp, then press any key...")
        selected_pixel = [None]

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_pixel[0] = (x, y)
                # Draw circle at clicked point
                display_img = rgb.copy()
                cv2.circle(display_img, (x, y), 5, (255, 0, 0), -1)
                cv2.imshow("Click to select grasp point", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))

        cv2.namedWindow("Click to select grasp point")
        cv2.setMouseCallback("Click to select grasp point", mouse_callback)
        cv2.imshow("Click to select grasp point", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if selected_pixel[0] is None:
            print("No point selected, using center")
            pixel = (rgb.shape[1] // 2, rgb.shape[0] // 2)
        else:
            pixel = selected_pixel[0]
    else:
        # Use center of image
        pixel = (rgb.shape[1] // 2, rgb.shape[0] // 2)

    print(f"Selected pixel: {pixel}")

    # Convert to 3D point
    pixel_xyz = pixel_to_world_xyz(pixel[0], pixel[1], depth, K, extrinsics)
    print(f"3D point in world frame: {pixel_xyz}")

    # Generate point cloud
    pcd, pcd_colors = depth_to_colored_pcd(rgb, depth, K, extrinsics)
    print(f"Point cloud size: {pcd.shape[0]} points")

    # Visualize grasps
    print(f"\nQuerying M2T2 server at {server_url}...")
    visualize_m2t2_grasps(pixel_xyz, pcd, pcd_colors, server_url, num_top_grasps=5)


def test_with_saved_data(pcd_file: str, point: np.ndarray, server_url: str = "http://0.0.0.0:8123"):
    """Test visualization with saved point cloud data.

    Args:
        pcd_file: Path to .npy file containing point cloud (N, 6) with xyz+rgb
        point: (3,) array with target point
        server_url: URL of M2T2 server
    """
    print(f"Loading point cloud from {pcd_file}...")
    data = np.load(pcd_file)

    if data.shape[1] == 6:
        pcd = data[:, :3]
        pcd_colors = data[:, 3:6]
    elif data.shape[1] == 3:
        pcd = data
        pcd_colors = np.ones_like(pcd) * 0.5  # Gray
    else:
        raise ValueError(f"Expected (N, 3) or (N, 6) array, got shape {data.shape}")

    print(f"Point cloud size: {pcd.shape[0]} points")
    print(f"Target point: {point}")

    # Visualize grasps
    visualize_m2t2_grasps(point, pcd, pcd_colors, server_url, num_top_grasps=5)


def test_with_dummy_data(server_url: str = "http://0.0.0.0:8123"):
    """Test visualization with synthetic data (no camera needed).

    Args:
        server_url: URL of M2T2 server
    """
    print("Generating dummy point cloud...")

    # Create a simple table + object scene
    np.random.seed(42)

    # Table surface
    table_x = np.random.uniform(0.3, 0.7, 2000)
    table_y = np.random.uniform(-0.3, 0.3, 2000)
    table_z = np.ones(2000) * 0.05
    table_pcd = np.stack([table_x, table_y, table_z], axis=1)
    table_colors = np.ones_like(table_pcd) * np.array([0.6, 0.4, 0.2])  # Brown

    # Object (cylinder)
    n_obj = 500
    theta = np.random.uniform(0, 2*np.pi, n_obj)
    r = np.random.uniform(0, 0.04, n_obj)
    obj_x = 0.5 + r * np.cos(theta)
    obj_y = 0.0 + r * np.sin(theta)
    obj_z = np.random.uniform(0.05, 0.15, n_obj)
    obj_pcd = np.stack([obj_x, obj_y, obj_z], axis=1)
    obj_colors = np.ones_like(obj_pcd) * np.array([0.8, 0.2, 0.2])  # Red

    # Combine
    pcd = np.vstack([table_pcd, obj_pcd])
    pcd_colors = np.vstack([table_colors, obj_colors])

    # Target point on object
    target_point = np.array([0.5, 0.0, 0.1])

    print(f"Point cloud size: {pcd.shape[0]} points")
    print(f"Target point: {target_point}")

    # Visualize grasps
    visualize_m2t2_grasps(target_point, pcd, pcd_colors, server_url, num_top_grasps=5)


def main():
    parser = argparse.ArgumentParser(description="Test grasp visualization")
    parser.add_argument("--mode", choices=["camera", "file", "dummy"], default="camera",
                       help="Data source: camera (live ZED), file (saved .npy), or dummy (synthetic)")
    parser.add_argument("--pcd-file", type=str, help="Path to point cloud .npy file (for 'file' mode)")
    parser.add_argument("--point", type=float, nargs=3, default=[0.5, 0.0, 0.1],
                       help="Target point x y z (for 'file' or 'dummy' mode)")
    parser.add_argument("--server-url", type=str, default="http://0.0.0.0:8123",
                       help="M2T2 server URL")
    parser.add_argument("--no-click", action="store_true",
                       help="Don't use click mode, use center of image (for 'camera' mode)")

    args = parser.parse_args()

    print("="*70)
    print("GRASP VISUALIZATION TEST")
    print("="*70)

    if args.mode == "camera":
        test_with_camera(args.server_url, click_mode=not args.no_click)
    elif args.mode == "file":
        if args.pcd_file is None:
            parser.error("--pcd-file required for 'file' mode")
        test_with_saved_data(args.pcd_file, np.array(args.point), args.server_url)
    elif args.mode == "dummy":
        test_with_dummy_data(args.server_url)

    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)


if __name__ == "__main__":
    main()
