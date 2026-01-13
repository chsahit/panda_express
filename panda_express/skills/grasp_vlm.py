from typing import Tuple
from PIL import Image
import cv2
import json
from datetime import datetime
import numpy as np
import os
import requests
from scipy.spatial.transform import Rotation as R

from bamboo.client import BambooFrankaClient
from panda_express.perception.zed.zed_cam import ZedCamera
from panda_express.perception.utils.transform import pixel_to_world_xyz, depth_to_colored_pcd
from panda_express.perception.utils.pretrained_model_interface import GoogleGeminiVLM
from panda_express.skills.go_to_conf import goto_hand_position, TOP_DOWN_GRASP_ROT


def get_closest_m2t2_grasp(gemini_pt: np.ndarray, pcd: np.ndarray, pcd_colors: np.ndarray, server_url) -> np.ndarray:
    """Get the closest high-confidence grasp from M2T2 server.

    Args:
        gemini_pt: (3,) array with (x, y, z) point from Gemini VLM
        pcd: (N, 3) point cloud array
        pcd_colors: (N, 3) RGB colors in [0, 1] range
        server_url: URL of the M2T2 server

    Returns:
        (4, 4) numpy array representing the best grasp pose
    """
    payload = {
        "pointcloud": {
            "points": pcd.tolist(),
            "rgb": pcd_colors.tolist()
        },
        "num_points": 16384,
        "num_runs": 5,
        "mask_thresh": 0.2,
        "apply_bounds": True
    }
    try:
        response = requests.post(
            f"{server_url}/predict",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}")
        raise

    result = response.json()
    grasps_list = result.get("grasps", [])
    confidences_list = result.get("grasp_confidence", [])

    # Flatten grasps and confidences from all objects into single lists
    all_grasps = []
    all_confidences = []

    for obj_grasps, obj_confs in zip(grasps_list, confidences_list):
        # Convert to numpy arrays
        obj_grasps_np = np.array(obj_grasps, dtype=np.float32)
        obj_confs_np = np.array(obj_confs, dtype=np.float32)
        all_grasps.append(obj_grasps_np)
        all_confidences.append(obj_confs_np)

    # Concatenate all objects' grasps
    if len(all_grasps) == 0:
        raise ValueError("No grasps returned from M2T2 server")

    all_grasps = np.concatenate(all_grasps, axis=0)  # (total_N, 4, 4)
    all_confidences = np.concatenate(all_confidences, axis=0)  # (total_N,)

    # Extract positions (translation component) from each grasp
    # Grasp poses are 4x4 transformation matrices, position is in [:3, 3]
    grasp_positions = all_grasps[:, :3, 3]  # (total_N, 3)

    # Compute distances from gemini_pt to each grasp position
    distances = np.linalg.norm(grasp_positions - gemini_pt[np.newaxis, :], axis=1)  # (total_N,)

    # Find indices of 5 closest grasps
    num_closest = min(5, len(distances))
    closest_indices = np.argpartition(distances, num_closest - 1)[:num_closest]

    # Among the 5 closest, find the one with highest confidence
    closest_confidences = all_confidences[closest_indices]
    print(f"{closest_confidences=}")
    best_idx_among_closest = np.argmax(closest_confidences)
    best_grasp_idx = closest_indices[best_idx_among_closest]

    best_grasp = all_grasps[best_grasp_idx]
    best_conf = all_confidences[best_grasp_idx]
    best_dist = distances[best_grasp_idx]

    print(f"Selected grasp with confidence {best_conf:.3f} at distance {best_dist:.3f}m from Gemini point")

    return best_grasp


def _m2t2_to_panda_guess1(m2t2_grasp: np.ndarray) -> np.ndarray:
    """4x4 transform to take M2T2 grasp poses to the convention expected by the goto skill."""
    base_to_tcp = np.eye(4)
    base_to_tcp[2, 3] = 0.06

    # To panda frame with z-up
    to_panda_frame = np.eye(4)
    to_panda_frame[:3, :3] = R.from_euler("xyz", np.array([np.pi, 0, -np.pi / 2])).as_matrix() 
    m2t2_to_panda_frame = base_to_tcp @ to_panda_frame
    return m2t2_grasp @ m2t2_to_panda_frame


def _m2t2_to_panda_guess2(X_grasp: np.ndarray) -> np.ndarray:
    correction = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    correction2 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    finger_adjustment = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]]
    )
    return X_grasp @ correction @ finger_adjustment @ correction2


def m2t2_to_panda(m2t2_grasp: np.ndarray) -> np.ndarray:
    """Convert M2T2 grasp convention to Panda link8 convention.

    M2T2 convention:
        X-axis: contact_dir (gripper opening direction)
        Y-axis: cross(approach, contact)
        Z-axis: approach_dir (gripper approach)

    Panda link8 convention (Robotiq 2F-85):
        X-axis: perpendicular to opening and approach
        Y-axis: gripper opening direction
        Z-axis: gripper approach direction

    Transformation: Rotate -90° around Z to swap X and Y axes.
    """
    # Rotation matrix for -90° around Z-axis (swaps X→Y, Y→-X)
    transform = np.array([
        [ 0, -1,  0,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  1,  -0.1034],
        [ 0,  0,  0,  1]
    ])
    return m2t2_grasp @ transform


def _get_pixel_from_gemini(vlm_query_str: str, pil_image: Image.Image) -> Tuple[int, int]:
    """Query Gemini VLM to get a single pixel [y, x] normalized to 0-1000, then
    denormalize to image pixel coordinates.

    This mirrors the usage pattern used in the wipe skill: construct a
    ``GoogleGeminiVLM`` instance and call ``sample_completions`` directly.
    """

    vlm = GoogleGeminiVLM("gemini-2.5-pro")
    print(f"Using Gemini VLM model: {vlm.get_id()}")
    print(f"Querying Gemini VLM with prompt: {vlm_query_str}")

    def _strip_markdown_fence(json_output_str: str) -> str:
        """Remove ```json fences if present and return the inner JSON string."""
        lines = json_output_str.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                json_output_str = "\n".join(lines[i + 1 :])
                json_output_str = json_output_str.split("```")[0]
                break
        return json_output_str.strip()

    # 1) Query the VLM
    vlm_output_list = vlm.sample_completions(
        prompt=vlm_query_str,
        imgs=[pil_image],
        temperature=0.0,
        seed=42,
        num_completions=1,
    )
    vlm_output_str = vlm_output_list[0]

    # 2) Parse JSON output
    json_string_to_parse = _strip_markdown_fence(vlm_output_str)
    parsed_data = json.loads(json_string_to_parse)

    if not isinstance(parsed_data, list) or not parsed_data:
        raise ValueError("Parsed JSON is not a non-empty list.")

    first_point_obj = parsed_data[0]
    if (
        "point" not in first_point_obj
        or not isinstance(first_point_obj["point"], list)
        or len(first_point_obj["point"]) != 2
    ):
        raise ValueError(
            "First element in JSON does not contain a valid 'point' list [y, x]."
        )

    y_norm, x_norm = first_point_obj["point"]
    if not isinstance(y_norm, (int, float)) or not isinstance(x_norm, (int, float)):
        raise ValueError("Normalized coordinates are not numbers.")

    # 3) Denormalize from 0–1000 range to image pixel coordinates
    img_height = pil_image.height
    img_width = pil_image.width
    y = int(y_norm * img_height / 1000.0)
    x = int(x_norm * img_width / 1000.0)

    # Clamp to image bounds
    y = max(0, min(y, img_height - 1))
    x = max(0, min(x, img_width - 1))

    # Return as (x, y) pixel coordinate
    return (x, y)


def visualize_m2t2_grasps(
    gemini_pt: np.ndarray,
    pcd: np.ndarray,
    pcd_colors: np.ndarray,
    server_url: str,
    num_top_grasps: int = 5
) -> None:
    """Visualize M2T2 grasps and their conversion to Panda convention.

    Args:
        gemini_pt: (3,) array with (x, y, z) point from Gemini VLM
        pcd: (N, 3) point cloud array
        pcd_colors: (N, 3) RGB colors in [0, 1] range
        server_url: URL of the M2T2 server
        num_top_grasps: Number of top grasps to visualize
    """
    from panda_express.skills.visualize_grasps import (
        visualize_grasp_comparison,
        visualize_multiple_grasps,
        print_grasp_info
    )

    # Get all grasps from M2T2
    payload = {
        "pointcloud": {
            "points": pcd.tolist(),
            "rgb": pcd_colors.tolist()
        },
        "num_points": 16384,
        "num_runs": 5,
        "mask_thresh": 0.2,
        "apply_bounds": True
    }

    response = requests.post(f"{server_url}/predict", json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()

    grasps_list = result.get("grasps", [])
    confidences_list = result.get("grasp_confidence", [])

    # Flatten grasps
    all_grasps = []
    all_confidences = []
    for obj_grasps, obj_confs in zip(grasps_list, confidences_list):
        all_grasps.append(np.array(obj_grasps, dtype=np.float32))
        all_confidences.append(np.array(obj_confs, dtype=np.float32))

    all_grasps = np.concatenate(all_grasps, axis=0)
    all_confidences = np.concatenate(all_confidences, axis=0)

    # Get top grasps near the gemini point
    grasp_positions = all_grasps[:, :3, 3]
    distances = np.linalg.norm(grasp_positions - gemini_pt[np.newaxis, :], axis=1)

    # Get closest grasps
    num_closest = min(num_top_grasps * 2, len(distances))
    closest_indices = np.argpartition(distances, num_closest - 1)[:num_closest]

    # Sort by confidence
    sorted_by_conf = closest_indices[np.argsort(-all_confidences[closest_indices])]
    top_indices = sorted_by_conf[:num_top_grasps]

    m2t2_top_grasps = [all_grasps[i] for i in top_indices]
    panda_top_grasps = [m2t2_to_panda(g) for g in m2t2_top_grasps]

    # Visualize the best grasp in detail
    best_m2t2 = m2t2_top_grasps[0]
    best_panda = panda_top_grasps[0]

    print(f"\nVisualization 1: Best grasp comparison (M2T2 vs Panda)")
    print(f"Confidence: {all_confidences[top_indices[0]]:.3f}")
    print(f"Distance from Gemini point: {distances[top_indices[0]]:.3f}m")

    print_grasp_info(best_m2t2, best_panda)
    visualize_grasp_comparison(best_m2t2, best_panda, pcd, pcd_colors,
                               window_name="Best Grasp: M2T2 vs Panda")

    # Visualize multiple top grasps
    print(f"\nVisualization 2: Top {num_top_grasps} grasps in Panda convention")
    visualize_multiple_grasps(m2t2_top_grasps, panda_top_grasps, pcd, pcd_colors,
                             max_grasps=num_top_grasps,
                             window_name="Top Grasps (Panda Convention)")


def grasp_with_vlm(
    robot: BambooFrankaClient,
    text_prompt: str | None,
    use_m2t2: bool = True,
    visualize: bool = False
) -> None:
    """High-level grasp skill that uses a VLM to choose a pixel from a prompt.

    Args:
        robot: Robot client
        text_prompt: Text description of object to grasp
        use_m2t2: Whether to use M2T2 for grasp selection
        visualize: Whether to visualize grasps before executing
    """
    print(f"Calling grasping with VLM for text prompt: {text_prompt}")

    cam = ZedCamera(serial_number=35317039)
    bgra = cam.get_bgra_frame()
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    depth = cam.get_depth_frame()
    K = cam.get_intrinsics()[0]
    cam.close()
    extrinsics = np.load("panda_express/perception/zed/X_WE.npy")

    # Call Gemini to point to the object described by the prompt.
    vlm_query_template = f"""
    Point to the {text_prompt}. If you cannot see the {text_prompt} fully, point to the best guess.
    The answer should follow the json format: [{{"point": , "label": }}, ...]. The points are in [y, x] format normalized to 0-1000.
    """
    image_pil = Image.fromarray(rgb)
    pixel = _get_pixel_from_gemini(vlm_query_template, image_pil)

    # Draw pixel on the image for logging.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.circle(bgr, pixel, 5, (0, 0, 255), -1)
    rgb_annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folderpath = "image_logs/grasping"
    os.makedirs(save_folderpath, exist_ok=True)
    Image.fromarray(rgb_annotated).save(os.path.join(save_folderpath, f"annotated_rgb_{timestamp}.png"))

    if pixel is None:
        raise RuntimeError("Grasp failed: VLM did not return a valid pixel.")

    pixel_xyz = pixel_to_world_xyz(pixel[0], pixel[1], depth, K, extrinsics)
    pcd, pcd_colors = depth_to_colored_pcd(rgb, depth, K, extrinsics)

    # Visualize grasps if requested
    if visualize and use_m2t2:
        print("\n" + "="*70)
        print("VISUALIZING GRASPS (close windows to continue)")
        print("="*70)
        visualize_m2t2_grasps(pixel_xyz, pcd, pcd_colors, "http://0.0.0.0:8123", num_top_grasps=5)
        input("\nPress Enter to continue with grasp execution...")

    # go to pre-grasp pose
    print("go to pre-grasp")
    X_WPregrasp = np.eye(4)
    X_WPregrasp[:3, :3] = TOP_DOWN_GRASP_ROT
    X_WPregrasp[:3, 3] = pixel_xyz + np.array([0.0, 0.0, 0.25])
    goto_hand_position(robot, X_WPregrasp, 5.0)

    # go to grasp pose

    if use_m2t2:
        print("querying M2T2 for grasp...")
        m2t2_best_grasp = get_closest_m2t2_grasp(pixel_xyz, pcd, pcd_colors, "http://0.0.0.0:8123")
        X_WGoal = m2t2_to_panda(m2t2_best_grasp)
    else:
        X_WGoal = np.eye(4)
        X_WGoal[:3, :3] = TOP_DOWN_GRASP_ROT
        X_WGoal[:3, 3] = pixel_xyz + np.array([0.0, 0.0, 0.15])
    print(f"{X_WGoal=}")
    goto_hand_position(robot, X_WGoal, 3.0)
    robot.close_gripper()


if __name__ == "__main__":
    with BambooFrankaClient(server_ip="128.30.224.88") as rob:
        grasp_with_vlm(rob, "blue mug")
