"""Visualization utilities for debugging gripper conventions and grasp poses."""
import numpy as np
import open3d as o3d
from typing import List, Optional


def create_coordinate_frame(pose: np.ndarray, scale: float = 0.1, label: str = "") -> List[o3d.geometry.LineSet]:
    """Create a coordinate frame visualization from a 4x4 pose matrix.

    Args:
        pose: 4x4 transformation matrix
        scale: Length of axis arrows
        label: Optional label for the frame

    Returns:
        List of LineSet objects for X (red), Y (green), Z (blue) axes
    """
    origin = pose[:3, 3]
    x_axis = pose[:3, 0]
    y_axis = pose[:3, 1]
    z_axis = pose[:3, 2]

    # Create arrows for each axis
    frames = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue
    axes = [x_axis, y_axis, z_axis]
    axis_names = ['X', 'Y', 'Z']

    for axis, color, name in zip(axes, colors, axis_names):
        points = [origin, origin + scale * axis]
        lines = [[0, 1]]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color])
        frames.append(line_set)

    return frames


def create_gripper_mesh(pose: np.ndarray, width: float = 0.08, depth: float = 0.1034) -> o3d.geometry.TriangleMesh:
    """Create a simple gripper mesh visualization.

    Args:
        pose: 4x4 transformation matrix (Panda convention: Y=opening, Z=approach)
        width: Gripper opening width
        depth: Gripper depth (distance from TCP to fingers)

    Returns:
        TriangleMesh representing the gripper
    """
    # Create simple box representing gripper body
    gripper = o3d.geometry.TriangleMesh.create_box(width=0.02, height=width, depth=depth)
    gripper.translate([-0.01, -width/2, -depth])

    # Create two finger boxes
    finger_width = 0.01
    finger_length = 0.04

    # Left finger
    left_finger = o3d.geometry.TriangleMesh.create_box(
        width=finger_width, height=finger_width, depth=finger_length
    )
    left_finger.translate([-finger_width/2, -width/2, -finger_length])

    # Right finger
    right_finger = o3d.geometry.TriangleMesh.create_box(
        width=finger_width, height=finger_width, depth=finger_length
    )
    right_finger.translate([-finger_width/2, width/2 - finger_width, -finger_length])

    # Combine all parts
    gripper = gripper + left_finger + right_finger
    gripper.paint_uniform_color([0.7, 0.7, 0.7])
    gripper.compute_vertex_normals()

    # Transform to pose
    gripper.transform(pose)

    return gripper


def visualize_grasp_comparison(
    m2t2_grasp: np.ndarray,
    panda_grasp: np.ndarray,
    pcd: Optional[np.ndarray] = None,
    pcd_colors: Optional[np.ndarray] = None,
    window_name: str = "Grasp Comparison"
) -> None:
    """Visualize M2T2 and Panda grasp conventions side-by-side.

    Args:
        m2t2_grasp: 4x4 grasp pose in M2T2 convention
        panda_grasp: 4x4 grasp pose in Panda convention
        pcd: Optional (N, 3) point cloud
        pcd_colors: Optional (N, 3) RGB colors in [0, 1] range
        window_name: Window title
    """
    geometries = []

    # Add point cloud if provided
    if pcd is not None:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)
        if pcd_colors is not None:
            point_cloud.colors = o3d.utility.Vector3dVector(pcd_colors)
        geometries.append(point_cloud)

    # M2T2 coordinate frame (larger scale, thicker)
    m2t2_frames = create_coordinate_frame(m2t2_grasp, scale=0.15, label="M2T2")
    geometries.extend(m2t2_frames)

    # Panda coordinate frame (slightly smaller to distinguish)
    panda_frames = create_coordinate_frame(panda_grasp, scale=0.12, label="Panda")
    geometries.extend(panda_frames)

    # Add gripper mesh for Panda convention
    gripper_mesh = create_gripper_mesh(panda_grasp)
    geometries.append(gripper_mesh)

    # Add text labels using spheres at the origins
    m2t2_origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    m2t2_origin.translate(m2t2_grasp[:3, 3])
    m2t2_origin.paint_uniform_color([1, 0, 1])  # Magenta for M2T2
    geometries.append(m2t2_origin)

    panda_origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    panda_origin.translate(panda_grasp[:3, 3])
    panda_origin.paint_uniform_color([0, 1, 1])  # Cyan for Panda
    geometries.append(panda_origin)

    # Add world frame at origin
    world_frame = create_coordinate_frame(np.eye(4), scale=0.2, label="World")
    geometries.extend(world_frame)

    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1200,
        height=800,
        left=50,
        top=50
    )


def print_grasp_info(m2t2_grasp: np.ndarray, panda_grasp: np.ndarray) -> None:
    """Print detailed information about grasp poses for debugging.

    Args:
        m2t2_grasp: 4x4 grasp pose in M2T2 convention
        panda_grasp: 4x4 grasp pose in Panda convention
    """
    print("\n" + "="*70)
    print("GRASP POSE COMPARISON")
    print("="*70)

    print("\nM2T2 Convention (X=opening, Y=cross, Z=approach):")
    print("-" * 70)
    print(f"Position: {m2t2_grasp[:3, 3]}")
    print(f"X-axis (opening):     {m2t2_grasp[:3, 0]}")
    print(f"Y-axis (cross):       {m2t2_grasp[:3, 1]}")
    print(f"Z-axis (approach):    {m2t2_grasp[:3, 2]}")

    print("\nPanda Convention (X=cross, Y=opening, Z=approach):")
    print("-" * 70)
    print(f"Position: {panda_grasp[:3, 3]}")
    print(f"X-axis (cross):       {panda_grasp[:3, 0]}")
    print(f"Y-axis (opening):     {panda_grasp[:3, 1]}")
    print(f"Z-axis (approach):    {panda_grasp[:3, 2]}")

    # Compute angles between corresponding axes
    print("\nAxis Alignment Check:")
    print("-" * 70)

    # M2T2 X (opening) should align with Panda Y (opening)
    opening_dot = np.dot(m2t2_grasp[:3, 0], panda_grasp[:3, 1])
    opening_angle = np.degrees(np.arccos(np.clip(opening_dot, -1, 1)))
    print(f"M2T2 X vs Panda Y (opening):  dot={opening_dot:.3f}, angle={opening_angle:.1f}°")

    # M2T2 Z (approach) should align with Panda Z (approach)
    approach_dot = np.dot(m2t2_grasp[:3, 2], panda_grasp[:3, 2])
    approach_angle = np.degrees(np.arccos(np.clip(approach_dot, -1, 1)))
    print(f"M2T2 Z vs Panda Z (approach): dot={approach_dot:.3f}, angle={approach_angle:.1f}°")

    # M2T2 Y should align with Panda -X
    cross_dot = np.dot(m2t2_grasp[:3, 1], -panda_grasp[:3, 0])
    cross_angle = np.degrees(np.arccos(np.clip(cross_dot, -1, 1)))
    print(f"M2T2 Y vs Panda -X (cross):   dot={cross_dot:.3f}, angle={cross_angle:.1f}°")

    print("\n" + "="*70)
    print("Legend:")
    print("  - Angles close to 0° indicate good alignment")
    print("  - Angles close to 180° indicate opposite directions")
    print("  - Angles close to 90° indicate perpendicular axes (ERROR!)")
    print("="*70 + "\n")


def visualize_multiple_grasps(
    m2t2_grasps: List[np.ndarray],
    panda_grasps: List[np.ndarray],
    pcd: Optional[np.ndarray] = None,
    pcd_colors: Optional[np.ndarray] = None,
    max_grasps: int = 10,
    window_name: str = "Multiple Grasps"
) -> None:
    """Visualize multiple grasp poses on the same scene.

    Args:
        m2t2_grasps: List of 4x4 grasp poses in M2T2 convention
        panda_grasps: List of 4x4 grasp poses in Panda convention
        pcd: Optional (N, 3) point cloud
        pcd_colors: Optional (N, 3) RGB colors in [0, 1] range
        max_grasps: Maximum number of grasps to visualize
        window_name: Window title
    """
    geometries = []

    # Add point cloud if provided
    if pcd is not None:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)
        if pcd_colors is not None:
            point_cloud.colors = o3d.utility.Vector3dVector(pcd_colors)
        geometries.append(point_cloud)

    # Limit number of grasps
    num_grasps = min(len(m2t2_grasps), max_grasps)

    # Visualize each grasp
    for i in range(num_grasps):
        # Panda frames (what will be sent to IK)
        panda_frames = create_coordinate_frame(panda_grasps[i], scale=0.08)
        geometries.extend(panda_frames)

        # Add small gripper mesh
        gripper = create_gripper_mesh(panda_grasps[i], width=0.06, depth=0.08)
        geometries.append(gripper)

    # Add world frame
    world_frame = create_coordinate_frame(np.eye(4), scale=0.15)
    geometries.extend(world_frame)

    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{window_name} ({num_grasps} grasps)",
        width=1200,
        height=800,
        left=50,
        top=50
    )


if __name__ == "__main__":
    # Example usage with dummy data
    from scipy.spatial.transform import Rotation as R

    # Create a sample M2T2 grasp
    m2t2_grasp = np.eye(4)
    m2t2_grasp[:3, :3] = R.from_euler('xyz', [0, 0, 45], degrees=True).as_matrix()
    m2t2_grasp[:3, 3] = [0.5, 0.0, 0.3]

    # Convert to Panda convention
    from grasp_vlm import m2t2_to_panda
    panda_grasp = m2t2_to_panda(m2t2_grasp)

    # Print info
    print_grasp_info(m2t2_grasp, panda_grasp)

    # Create dummy point cloud
    pcd = np.random.rand(1000, 3) * 0.5 + np.array([0.3, -0.2, 0.1])
    pcd_colors = np.random.rand(1000, 3)

    # Visualize
    visualize_grasp_comparison(m2t2_grasp, panda_grasp, pcd, pcd_colors)
