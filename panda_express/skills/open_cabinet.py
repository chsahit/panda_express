from typing import List, Optional, Tuple

import cv2
import json
import numpy as np
import rerun as rr
from PIL import Image
from scipy.spatial.transform import Rotation as R

from bamboo.client import BambooFrankaClient
from panda_express.perception.utils.pretrained_model_interface import GoogleGeminiVLM
from panda_express.perception.utils.transform import pixel_to_world_xyz
from panda_express.perception.zed.zed_cam import ZedCamera
from panda_express.skills.go_to_conf import goto_hand_position
from panda_express.skills.grasp_vlm import _get_pixel_from_gemini

prompt_get_handle_pixel = """
    Point to the CENTER (MIDDLE) of the TOP ORANGE handle of the drawer.
    The answer should follow the json format: [{"point": , "label": }, ...]. The points are in [y, x] format normalized to 0-1000.
    """
prompt_get_drawer_surface_pixel = """
    Point to 10 points that are evenly spread across the flat front face of the TOP drawer.
    IMPORTANT: Avoid the ORANGE handle, any other handles, edges, and corners of the drawer.
    Only select points on the flat front panel surface.
    The answer should follow the json format: [{"point": , "label": }, ...]. The points are in [y, x] format normalized to 0-1000.
    """
prompt_get_objects_inside_drawer = """
    Give me a descriptive list of objects inside this drawer.
    The answer should follow the format: ["object1", "object2", ...].
    """
prompt_handle = """
    Does the drawer contained in this image have a ORANGE handle? Answer with only "Yes" or "No".
    """

CABINET_GRASPING_ROT = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1.0, 0]])

def add_rotation_noise(rotation_matrix: np.ndarray, noise_deg: float) -> np.ndarray:
    """Add random noise to roll, pitch, yaw of a rotation matrix."""
    # Convert rotation matrix to euler angles
    r = R.from_matrix(rotation_matrix)
    euler = r.as_euler('xyz', degrees=True)

    # Add random noise to each angle
    noise = np.random.uniform(-noise_deg, noise_deg, size=3)
    noisy_euler = euler + noise

    # Convert back to rotation matrix
    noisy_r = R.from_euler('xyz', noisy_euler, degrees=True)
    return noisy_r.as_matrix()


def goto_with_retries(robot: BambooFrankaClient, X_W: np.ndarray, duration: float,
                      base_rotation: np.ndarray, max_attempts: int = 10, noise_deg: float = 10.0):
    """Try to goto a pose, retrying with rotation noise if IK fails."""
    for attempt in range(max_attempts):
        try:
            goto_hand_position(robot, X_W, duration)
            return  # Success!
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying with rotation noise...")
                # Add noise to the rotation
                noisy_rotation = add_rotation_noise(base_rotation, noise_deg)
                X_W[:3, :3] = noisy_rotation
            else:
                print(f"All {max_attempts} attempts failed.")
                raise


def draw_colored_pixels(image_pil: Image, pixels: list[Tuple[int, int]], path: str, color: str):
    pixels_obj = image_pil.load()
    for pixel in pixels:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                px = min(max(pixel[0] + dx, 0), image_pil.width - 1)
                py = min(max(pixel[1] + dy, 0), image_pil.height - 1)
                pixels_obj[px, py] = (255, 0, 0) if color == "red" else (0, 0, 255)
    image_pil.save(path)


def get_multiple_pixels_from_gemini(
    vlm_query_str: str, pil_image: Image, num_pixels: int = 15
) -> List[Tuple[int, int]]:
    """Get multiple pixel coordinates from Gemini VLM.

    Args:
        vlm_query_str: The prompt to send to Gemini
        pil_image: PIL image to query
        num_pixels: Number of pixels to extract

    Returns:
        List of (x, y) pixel coordinates
    """
    vlm = GoogleGeminiVLM("gemini-2.0-flash")

    def parse_json_output(json_output_str: str) -> str:
        lines = json_output_str.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                json_output_str = "\n".join(lines[i + 1:])
                json_output_str = json_output_str.split("```")[0]
                break
        return json_output_str.strip()

    vlm_output_list = vlm.sample_completions(
        prompt=vlm_query_str,
        imgs=[pil_image],
        temperature=0.0,
        seed=42,
        num_completions=1,
    )
    vlm_output_str = vlm_output_list[0]

    json_string_to_parse = parse_json_output(vlm_output_str)
    parsed_data = json.loads(json_string_to_parse)

    if not isinstance(parsed_data, list) or not parsed_data:
        raise ValueError("Parsed JSON is not a non-empty list.")
    if len(parsed_data) < num_pixels:
        raise ValueError(f"Parsed JSON has less than {num_pixels} points.")

    pixels = []
    for point_obj in parsed_data[:num_pixels]:
        if (
            "point" not in point_obj
            or not isinstance(point_obj["point"], list)
            or len(point_obj["point"]) != 2
        ):
            raise ValueError(
                "Some element in JSON does not contain a valid 'point' list [y, x]."
            )
        y_norm, x_norm = point_obj["point"]
        if not isinstance(y_norm, (int, float)) or not isinstance(x_norm, (int, float)):
            raise ValueError("Normalized coordinates are not numbers.")

        # Denormalize from 0-1000 range to image pixel coordinates
        img_height = pil_image.height
        img_width = pil_image.width
        y = int(y_norm * img_height / 1000.0)
        x = int(x_norm * img_width / 1000.0)

        # Clamp coordinates to be within image bounds
        y = max(0, min(y, img_height - 1))
        x = max(0, min(x, img_width - 1))
        pixels.append((x, y))

    return pixels


def pixels_to_world_points(
    pixels: List[Tuple[int, int]],
    depth: np.ndarray,
    K: np.ndarray,
    extrinsics: np.ndarray,
    max_depth: float = 3.0,
) -> np.ndarray:
    """Convert multiple 2D pixels to 3D world coordinates.

    Args:
        pixels: List of (x, y) pixel coordinates
        depth: Depth image (H, W) in meters
        K: Camera intrinsics (3, 3)
        extrinsics: Camera extrinsics X_WC (4, 4)
        max_depth: Maximum valid depth (meters)

    Returns:
        Nx3 array of 3D points in world coordinates
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    points_world = []
    for (u, v) in pixels:
        if v < 0 or v >= depth.shape[0] or u < 0 or u >= depth.shape[1]:
            print(f"Skipping pixel ({u}, {v}): out of image bounds")
            continue

        z = float(depth[v, u])
        if z <= 0 or z > max_depth:
            print(f"Skipping pixel ({u}, {v}): invalid depth {z:.3f} m")
            continue

        # Unproject to camera coordinates
        x_cam = (float(u) - cx) / fx * z
        y_cam = (float(v) - cy) / fy * z
        point_cam = np.array([x_cam, y_cam, z])

        # Transform to world coordinates
        R_WC = extrinsics[:3, :3]
        t_WC = extrinsics[:3, 3]
        point_world = R_WC @ point_cam + t_WC
        points_world.append(point_world)

    if not points_world:
        raise ValueError("No valid depth points for provided pixels")

    return np.array(points_world, dtype=np.float32)


def fit_plane_to_points(points: np.ndarray, camera_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a plane to 3D points and return its centroid and normal vector.

    Args:
        points: Nx3 array of 3D points
        camera_position: (3,) array, camera position in world coordinates

    Returns:
        centroid: (3,) array, mean position of points
        normal: (3,) array, unit normal vector of best-fit plane (pointing toward camera)
    """
    assert points.shape[1] == 3, "Points must be Nx3"

    # Compute centroid
    centroid = np.mean(points, axis=0)

    # Subtract centroid
    Q = points - centroid

    # Compute SVD - the normal is the eigenvector with smallest singular value
    _, _, vh = np.linalg.svd(Q)
    normal = vh[-1, :]  # last row of V^T (smallest singular value)

    # Normalize
    normal = normal / np.linalg.norm(normal)

    # Ensure normal points toward the camera
    direction_to_camera = camera_position - centroid
    if np.dot(normal, direction_to_camera) < 0:
        normal = -normal

    return centroid, normal


def fit_plane_ransac(
    points: np.ndarray,
    camera_position: np.ndarray,
    n_iterations: int = 100,
    distance_threshold: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a plane to 3D points using RANSAC for robustness to outliers.

    Args:
        points: Nx3 array of 3D points
        camera_position: (3,) array, camera position in world coordinates
        n_iterations: Number of RANSAC iterations
        distance_threshold: Max distance (meters) for a point to be considered an inlier

    Returns:
        centroid: (3,) array, mean position of inlier points
        normal: (3,) array, unit normal vector of best-fit plane (pointing toward camera)
        inlier_mask: (N,) boolean array indicating which points are inliers
    """
    assert points.shape[1] == 3, "Points must be Nx3"
    n_points = points.shape[0]

    if n_points < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    best_inlier_count = 0
    best_normal = None
    best_centroid = None
    best_inlier_mask = None

    for _ in range(n_iterations):
        # Randomly sample 3 points
        indices = np.random.choice(n_points, size=3, replace=False)
        p1, p2, p3 = points[indices]

        # Compute plane normal from 3 points
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_length = np.linalg.norm(normal)

        if norm_length < 1e-10:
            # Degenerate case: points are collinear
            continue

        normal = normal / norm_length
        plane_point = p1

        # Compute distances from all points to the plane
        distances = np.abs(np.dot(points - plane_point, normal))

        # Count inliers
        inlier_mask = distances < distance_threshold
        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal
            best_centroid = plane_point
            best_inlier_mask = inlier_mask

    if best_normal is None:
        raise ValueError("RANSAC failed to find a valid plane")

    # Refit plane using all inliers for better estimate
    inlier_points = points[best_inlier_mask]
    centroid = np.mean(inlier_points, axis=0)
    Q = inlier_points - centroid
    _, _, vh = np.linalg.svd(Q)
    normal = vh[-1, :]
    normal = normal / np.linalg.norm(normal)

    # Ensure normal points toward the camera
    direction_to_camera = camera_position - centroid
    if np.dot(normal, direction_to_camera) < 0:
        normal = -normal

    print(f"RANSAC: {best_inlier_count}/{n_points} inliers")

    return centroid, normal, best_inlier_mask


def generate_point_cloud_from_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    extrinsics: np.ndarray,
    downsample_factor: int = 4,
    max_depth: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate point cloud from RGB-D image.

    Args:
        rgb: RGB image (H, W, 3)
        depth: Depth image (H, W) in meters
        K: Camera intrinsics (3, 3)
        extrinsics: Camera extrinsics X_WC (4, 4)
        downsample_factor: Factor to downsample for efficiency
        max_depth: Maximum depth to include (meters)

    Returns:
        points: (N, 3) world coordinates
        colors: (N, 3) RGB colors normalized to 0-1
    """
    h, w = depth.shape

    # Downsample for efficiency
    rgb_ds = rgb[::downsample_factor, ::downsample_factor]
    depth_ds = depth[::downsample_factor, ::downsample_factor]
    h_ds, w_ds = depth_ds.shape

    # Create pixel grid
    u = np.arange(0, w, downsample_factor)
    v = np.arange(0, h, downsample_factor)
    u, v = np.meshgrid(u, v)

    # Filter valid depth
    valid_mask = (depth_ds > 0) & (depth_ds < max_depth)

    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = depth_ds[valid_mask]

    # Unproject to camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (u_valid - cx) * z_valid / fx
    y_cam = (v_valid - cy) * z_valid / fy
    z_cam = z_valid

    # Stack into (N, 3) points in camera frame
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    # Transform to world coordinates
    R_WC = extrinsics[:3, :3]
    t_WC = extrinsics[:3, 3]
    points_world = (R_WC @ points_cam.T).T + t_WC

    # Get colors (normalized to 0-1)
    colors = rgb_ds[valid_mask].astype(np.float32) / 255.0

    return points_world, colors


def visualize_pixel_rerun(
    rgb_image: np.ndarray,
    pixel: Tuple[int, int],
    depth: np.ndarray,
    K: np.ndarray,
    extrinsics: np.ndarray,
    pixel_xyz: np.ndarray,
    label: str = "handle",
    surface_points: Optional[np.ndarray] = None,
    surface_centroid: Optional[np.ndarray] = None,
    surface_normal: Optional[np.ndarray] = None,
):
    """Visualize pixel on image, point cloud, 3D handle point, and surface normal in rerun."""
    rr.init("open_cabinet", spawn=True)

    # Draw pixel on image
    annotated_image = rgb_image.copy()
    x, y = pixel[0], pixel[1]

    # Draw a circle at the pixel location
    cv2.circle(annotated_image, (x, y), 10, (255, 0, 0), -1)  # Filled red circle
    cv2.circle(annotated_image, (x, y), 12, (255, 255, 255), 2)  # White outline
    cv2.putText(annotated_image, label, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Log the annotated RGB image
    rr.log("camera/rgb_annotated", rr.Image(annotated_image))
    rr.log("camera/rgb_original", rr.Image(rgb_image))

    # Generate and visualize point cloud
    points, colors = generate_point_cloud_from_rgbd(rgb_image, depth, K, extrinsics)
    rr.log(
        "world/point_cloud",
        rr.Points3D(positions=points, colors=colors, radii=0.005),
        static=True,
    )

    # Visualize the 3D point corresponding to the detected pixel
    rr.log(
        "world/handle_point",
        rr.Points3D(positions=[pixel_xyz], colors=[[1.0, 0.0, 0.0]], radii=0.02),
        static=True,
    )

    # Visualize surface points if provided
    if surface_points is not None:
        rr.log(
            "world/surface_points",
            rr.Points3D(
                positions=surface_points,
                colors=[[0.0, 0.0, 1.0]] * len(surface_points),  # Blue
                radii=0.015,
            ),
            static=True,
        )

    # Visualize surface normal as arrow if provided
    if surface_centroid is not None and surface_normal is not None:
        arrow_length = 0.15  # 15cm arrow
        rr.log(
            "world/surface_normal",
            rr.Arrows3D(
                origins=[surface_centroid],
                vectors=[surface_normal * arrow_length],
                colors=[[0.0, 1.0, 0.0]],  # Green
                radii=0.008,
            ),
            static=True,
        )
        print(f"Surface normal: {surface_normal}")
        print(f"Surface centroid: {surface_centroid}")

    print(f"Visualized pixel: x={x}, y={y}")
    print(f"Visualized 3D point: {pixel_xyz}")


def open_drawer(robot: BambooFrankaClient):
    robot.open_gripper()

    cam = ZedCamera(serial_number=35317039)
    bgra = cam.get_bgra_frame()
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    depth = cam.get_foundation_depth_frame()
    K = cam.get_intrinsics()[0]
    cam.close()
    extrinsics = np.load("panda_express/perception/zed/X_WE.npy")
    depth_pil = Image.fromarray(depth)
    image_pil = Image.fromarray(rgb)

    # Ask Gemini if this drawer has a handle; if not, we can't interact with it, so return None
    vlm = GoogleGeminiVLM("gemini-2.5-pro")
    vlm_output_list = vlm.sample_completions(
        prompt=prompt_handle,
        imgs=[image_pil],
        temperature=0.0,  # Low temp for deterministic output
        seed=42,
        num_completions=1,
    )
    vlm_output_str = vlm_output_list[0]
    if vlm_output_str == "No":
        return None

    # Get a 2D pixel on the handle, and convert to 3D point
    handle_pixel = _get_pixel_from_gemini(prompt_get_handle_pixel, image_pil)
    draw_colored_pixels(image_pil, [handle_pixel], "image_logs/annotated_hand_camera_output.jpg", "red")
    pixel_xyz = pixel_to_world_xyz(handle_pixel[0], handle_pixel[1], depth, K, extrinsics)

    # Get pixels on surface of drawer via Gemini and compute surface normal
    surface_pixels = get_multiple_pixels_from_gemini(prompt_get_drawer_surface_pixel, image_pil, num_pixels=10)
    draw_colored_pixels(image_pil, surface_pixels, "image_logs/annotated_hand_camera_output.jpg", "blue")

    # Convert surface pixels to 3D world points
    surface_points_3d = pixels_to_world_points(surface_pixels, depth, K, extrinsics)

    # Fit plane using RANSAC and get normal vector (pointing toward camera)
    camera_position = extrinsics[:3, 3]
    surface_centroid, surface_normal, inlier_mask = fit_plane_ransac(surface_points_3d, camera_position)
    print(f"Computed surface normal: {surface_normal}")

    # Visualize in rerun (image, point cloud, handle point, surface points, and normal)
    # Only visualize inlier points for cleaner visualization
    inlier_points = surface_points_3d[inlier_mask]
    visualize_pixel_rerun(
        rgb, handle_pixel, depth, K, extrinsics, pixel_xyz,
        label="handle",
        surface_points=inlier_points,
        surface_centroid=surface_centroid,
        surface_normal=surface_normal,
    )

    pregrasp_xyz = pixel_xyz - np.array([0.25, 0.0, 0.0])
    X_WPregrasp = np.eye(4)
    X_WPregrasp[:3, :3] = CABINET_GRASPING_ROT
    X_WPregrasp[:3, 3] = pregrasp_xyz


    grasp_xyz = pixel_xyz - np.array([0.16, 0.0, 0.0])
    X_WGrasp = np.eye(4)
    X_WGrasp[:3, :3] = CABINET_GRASPING_ROT
    X_WGrasp[:3, 3] = grasp_xyz

    print(f"{X_WPregrasp=}")
    goto_with_retries(robot, X_WPregrasp, 6.0, CABINET_GRASPING_ROT)
    goto_with_retries(robot, X_WGrasp, 3.0, CABINET_GRASPING_ROT)
    robot.close_gripper()
    goto_with_retries(robot, X_WPregrasp, 3.0, CABINET_GRASPING_ROT)


if __name__ == "__main__":
    with BambooFrankaClient(server_ip="128.30.224.88") as rob:
        open_drawer(rob)
