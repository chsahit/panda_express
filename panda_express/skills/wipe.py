import os
import json
import numpy as np
from datetime import datetime
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R
import rerun as rr

from bamboo.client import BambooFrankaClient
from panda_express.perception.zed.zed_cam import ZedCamera
from panda_express.perception.utils.transform import pixel_to_world_xyz
from panda_express.perception.utils.pretrained_model_interface import GoogleGeminiVLM
from panda_express.skills.go_to_conf import goto_hand_position, TOP_DOWN_GRASP_ROT, goto_joint_angles
from importlib.resources import files


DEFAULT_WIPE_ONLINE_Z_OFFSET = 0.175
DEFAULT_WIPE_VLM_QUERY_TEMPLATE = (
    "You are given an image. Identify the word SPILL written on a whiteboard if present.\n"
    "Return a bounding box that tightly encloses the spill region.\n"
    "If there is no spill visible or it is ambiguous, return a bbox of null.\n\n"
    'Output format (return EXACTLY one JSON object and nothing else):\n'
    '{"bbox": [ymin, xmin, ymax, xmax] | null, "label": "spill"}\n'
    "The bbox coordinates MUST be normalized to 0-1000 and are in [ymin, xmin, ymax, xmax] order.\n"
)

def _add_offset(pose: np.ndarray, offset: np.ndarray) -> np.ndarray:
    new_pose = np.copy(pose)
    new_pose[:3, 3] += offset
    return new_pose


def get_bbox_from_gemini(
    vlm_query_str: str, pil_image: Image.Image
) -> list[int]:
    """
    Query Gemini VLM to get the bbox coordinates corresponding to the query.

    Args:
        vlm_query_str: Prompt asking Gemini to identify the spill
        pil_image: PIL Image to analyze

    Returns:
        List of [ymin, xmin, ymax, xmax] in pixel coordinates
    """
    # Ensure API key is set for Gemini
    # vlm = GoogleGeminiVLM("gemini-2.5-flash-preview-05-20")
    print(f'inside the function to get the bbox from gemini')
    # vlm = GoogleGeminiVLM("gemini-2.5-flash")
    # vlm = GoogleGeminiVLM("gemini-2.0-flash")
    vlm = GoogleGeminiVLM("gemini-2.5-pro")
    def _parse_bbox_list(raw: str) -> list[float]:
        """Parse a bbox dict {"bbox": [ymin, xmin, ymax, xmax]} from model output.
        Supports optional ```json fenced blocks. Returns raw numeric values
        (assumed normalized 0-1000) without scaling.
        """
        s = raw.strip()
        if "```" in s:
            parts = s.split("```")
            if len(parts) >= 2:
                block = parts[1]
                if block.startswith("json\n"):
                    block = "\n".join(block.splitlines()[1:])
                s = block.strip()
        # Load JSON object
        try:
            obj = json.loads(s)
        except Exception:
            l, r = s.find("{"), s.rfind("}")
            if l == -1 or r == -1 or r <= l:
                raise ValueError("Could not find JSON object in model response.")
            obj = json.loads(s[l:r + 1])

        if not isinstance(obj, dict) or "bbox" not in obj:
            raise ValueError("Expected a JSON object with key 'bbox'.")
        bbox = obj["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4):
            raise ValueError("'bbox' must be a list of 4 numbers [ymin, xmin, ymax, xmax].")
        return [float(v) for v in bbox]

    # Query the VLM
    print(f'vlm: {vlm}, the query string is: {vlm_query_str}')
    vlm_output_list = vlm.sample_completions(
        prompt=vlm_query_str,
        imgs=[pil_image],
        temperature=0.0,
        seed=42,
        num_completions=1,
    )
    vlm_output_str = vlm_output_list[0]
    print(f'vlm_output_str: {vlm_output_str}')

    # Parse bbox and convert from normalized [0-1000] to pixel coordinates
    ymin_n, xmin_n, ymax_n, xmax_n = _parse_bbox_list(vlm_output_str)
    img_height = pil_image.height
    img_width = pil_image.width
    ymin = int(round(ymin_n * img_height / 1000.0))
    xmin = int(round(xmin_n * img_width / 1000.0))
    ymax = int(round(ymax_n * img_height / 1000.0))
    xmax = int(round(xmax_n * img_width / 1000.0))

    # Clamp to image bounds
    ymin = max(0, min(ymin, img_height - 1))
    xmin = max(0, min(xmin, img_width - 1))
    ymax = max(0, min(ymax, img_height - 1))
    xmax = max(0, min(xmax, img_width - 1))

    bbox = [ymin, xmin, ymax, xmax]
    return bbox


def _bbox_zed_to_corners_world(bbox: list[int], depth_m, K_zed, T_world_zed):
    ymin, xmin, ymax, xmax = bbox
    p_br = (int(xmax), int(ymax))
    p_tr = (int(xmax), int(ymin))
    p_bl = (int(xmin), int(ymax))

    P_br = pixel_to_world_xyz(*p_br, depth_m, K_zed, T_world_zed)
    P_tr = pixel_to_world_xyz(*p_tr, depth_m, K_zed, T_world_zed)
    P_bl = pixel_to_world_xyz(*p_bl, depth_m, K_zed, T_world_zed)

    return P_br, P_tr, P_bl


def _bbox_world_to_corners_world(bbox: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a list of 3D points (world frame) to bounding box corners.

    Args:
        bbox: List of 3D points (n x 3) in [x, y, z] order (world frame)
              describing the wipe target region

    Returns:
        Tuple of (P_br, P_tr, P_bl) where each is a numpy array of shape (3,)
        representing bottom-right, top-right, and bottom-left corners
    """
    points = np.array(bbox)

    xmin, ymin = points[:, 0].min(), points[:, 1].min()
    xmax, ymax = points[:, 0].max(), points[:, 1].max()
    z_mean = points[:, 2].mean()

    P_br = np.array([xmax, ymin, z_mean])  # bottom right
    P_tr = np.array([xmax, ymax, z_mean])  # top right
    P_bl = np.array([xmin, ymin, z_mean])  # bottom left

    return P_br, P_tr, P_bl


def _compute_wipe_params_from_bbox_zed(
    bbox: list[int],
    depth_m: np.ndarray,
    K_zed: np.ndarray,
    T_world_zed: np.ndarray,
    clearance: float = 0.08,
    spacing_m: float = 0.05,
    max_stroke_len: float = 0.35,
):
    corners_world = _bbox_zed_to_corners_world(bbox, depth_m, K_zed, T_world_zed)
    return _compute_wipe_params_from_corners_world(corners_world, clearance=clearance, spacing_m=spacing_m, max_stroke_len=max_stroke_len)


def _compute_wipe_params_from_agent(
    bbox: list[list[float]],
    clearance: float = 0.08,
    spacing_m: float = 0.05,
    max_stroke_len: float = 0.35,
):
    corners_world = _bbox_world_to_corners_world(bbox)
    return _compute_wipe_params_from_corners_world(corners_world, clearance=clearance, spacing_m=spacing_m, max_stroke_len=max_stroke_len)


def _compute_wipe_params_from_corners_world(
    corners_world,
    clearance: float = 0.08,
    spacing_m: float = 0.05,
    max_stroke_len: float = 0.35,
):
    """Compute wipe parameters from bottom right, top right, bottom left of bbox in world frame"""
    P_br, P_tr, P_bl = corners_world
    wipe_start_rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1, 0], [-0.0, 0, -1.0]])
    wipe_start_pose = np.eye(4)
    wipe_start_pose[:3, :3] = TOP_DOWN_GRASP_ROT
    wipe_start_pose[:3, 3] = P_br + np.array([0, 0, clearance])

    # Stroke direction (up)
    up_vec = P_tr - P_br
    up_vec[2] = 0.0
    up_len = float(np.linalg.norm(up_vec[:2]))
    if up_len < 1e-6:
        up_len = 0.0
        up_dir = np.array([0.0, 0.0])
    else:
        up_dir = up_vec[:2] / up_len
    stroke_len = min(up_len, max_stroke_len)
    stroke_dx = float(up_dir[0] * stroke_len)
    stroke_dy = float(up_dir[1] * stroke_len)

    # Spacing across width (right -> left)
    side_vec = P_bl - P_br
    side_vec[2] = 0.0
    width_m = float(np.linalg.norm(side_vec[:2]))
    if width_m > 1e-6:
        side_dir = side_vec[:2] / width_m
    else:
        side_dir = np.array([0.0, 0.0])
    delta_x_y_z_between_strokes = (
        float(side_dir[0] * spacing_m),
        float(side_dir[1] * spacing_m),
        0.0
    )
    num_strokes = max(1, int(np.ceil(width_m / max(spacing_m, 1e-3))) + 1)

    end_look_pose = np.eye(4)
    end_look_pose[:3, :3] = TOP_DOWN_GRASP_ROT
    end_look_pose[:3, 3] = np.array([0.4, 0.0, 0.5])

    return (
        wipe_start_pose,
        stroke_dx,
        stroke_dy,
        delta_x_y_z_between_strokes,
        num_strokes,
        end_look_pose,
    )



def wipe_multiple_strokes(
    robot: BambooFrankaClient,
    wipe_start_pose: np.ndarray,
    end_look_pose: np.ndarray,
    stroke_dx: float,
    stroke_dy: float,
    delta_x_y_z_between_strokes: np.ndarray,
    num_strokes: int,
    duration_per_stroke: float,
    num_attempts_per_stroke: int,
):
    """
    Execute multiple wipe strokes. After each stroke (and attempts) the start pose
    is shifted by delta_x_y_z_between_strokes in BODY frame.
    """
    curr = wipe_start_pose
    for _ in range(num_strokes):
        for _ in range(num_attempts_per_stroke):
            goto_hand_position(robot, curr, 3.0)
            first_move_pose = _add_offset(curr, np.array([stroke_dx, stroke_dy, 0]))
            goto_hand_position(robot, first_move_pose, duration_per_stroke)
            goto_hand_position(robot, curr, duration_per_stroke)
        # Shift to next stroke start
        # ASSUME delta_x_y_between strokes is shape (3, )
        curr = _add_offset(curr, delta_x_y_z_between_strokes)
    # End look pose
    goto_hand_position(robot, end_look_pose, 5.0)


def wipe_online(
        robot: BambooFrankaClient,
        vlm_query_template: str = DEFAULT_WIPE_VLM_QUERY_TEMPLATE,
        z_offset: float = DEFAULT_WIPE_ONLINE_Z_OFFSET,
        expand_percentage: float = 0.0
):
    cam = ZedCamera(serial_number=35317039)
    bgra = cam.get_bgra_frame()
    rgb_image = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    depth_img = cam.get_foundation_depth_frame()
    intrinsics = cam.get_intrinsics()[0]
    cam.close()
    extrinsics_path = files("panda_express").joinpath("perception/zed/X_WE.npy")
    extrinsics = np.load(extrinsics_path)

    rgb_pil = Image.fromarray(rgb_image)

    # save rgb for logging
    save_folderpath = "wipe_spill_images"
    os.makedirs(save_folderpath, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ## save the rgb and the depth image to the disk
    rgb_pil.save(os.path.join(save_folderpath, f"rgb_{timestamp}.png"))

    depth_m = depth_img.astype(np.float32)
    # Run VLM on the full-resolution RGB image and get bbox in RGB pixel coordinates.
    gemini_structure = (
        "If the desired property is not visible or it is ambiguous, return a bbox of null.\n\n"
        "Output format (return EXACTLY one JSON object and nothing else):\n"
        '{"bbox": [ymin, xmin, ymax, xmax] | null, "label": "spill"}\n'
        "The bbox coordinates MUST be normalized to 0-1000 and are in [ymin, xmin, ymax, xmax] order.\n"
    )
    
    vlm_query_template = vlm_query_template + "\n" + gemini_structure
    print(f'vlm_query_template inside wipe_online: {vlm_query_template}')

    bbox = get_bbox_from_gemini(vlm_query_template, rgb_pil)
    print(f"The coordinates of the bounding box (RGB space) are: {bbox}")

    # Visualize bbox in rerun
    visualize_bbox_rerun(rgb_image, bbox, label="detection")

    # Optionally expand bbox in image space by a percentage along all directions
    if expand_percentage and expand_percentage > 0.0:
        ymin, xmin, ymax, xmax = bbox
        H, W = depth_img.shape[0], depth_img.shape[1]
        height_px = max(1, (ymax - ymin))
        width_px = max(1, (xmax - xmin))
        dy = int(round(0.5 * expand_percentage * height_px))
        dx = int(round(0.5 * expand_percentage * width_px))
        ymin_exp = max(0, ymin - dy)
        ymax_exp = min(H - 1, ymax + dy)
        xmin_exp = max(0, xmin - dx)
        xmax_exp = min(W - 1, xmax + dx)
        bbox = [ymin_exp, xmin_exp, ymax_exp, xmax_exp]
        print(f"Expanded bbox by {expand_percentage*100:.1f}% -> {bbox}")

    (
        wipe_start_pose,
        stroke_dx,
        stroke_dy,
        delta_x_y_z_between_strokes,
        num_strokes,
        end_look_pose,
    ) = _compute_wipe_params_from_bbox_zed(
        bbox,
        depth_m,
        intrinsics,
        extrinsics,
        clearance=z_offset,
        spacing_m=0.05,
        max_stroke_len=0.35,
    )
    q_neutral = np.array([-0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, -0.14])
    goto_joint_angles(robot, q_neutral, 5)
    wipe_multiple_strokes(
        robot=robot,
        wipe_start_pose=wipe_start_pose,
        end_look_pose=end_look_pose,
        stroke_dx=stroke_dx + 0.05,
        stroke_dy=stroke_dy,
        delta_x_y_z_between_strokes=delta_x_y_z_between_strokes,
        num_strokes=num_strokes,
        duration_per_stroke=1.5,
        num_attempts_per_stroke=1,
    )


def wipe_bbox(robot, bbox: list[int], z_offset: float = DEFAULT_WIPE_ONLINE_Z_OFFSET):
    (
        wipe_start_pose,
        stroke_dx,
        stroke_dy,
        delta_x_y_z_between_strokes,
        num_strokes,
        end_look_pose,
    ) = _compute_wipe_params_from_agent(
        bbox,
        clearance=z_offset,
        spacing_m=0.05,
        max_stroke_len=0.35,
    )
    q_neutral = np.array([-0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, -0.14])
    goto_joint_angles(robot, q_neutral, 5)
    wipe_multiple_strokes(
        robot=robot,
        wipe_start_pose=wipe_start_pose,
        end_look_pose=end_look_pose,
        stroke_dx=stroke_dx + 0.05,
        stroke_dy=stroke_dy,
        delta_x_y_z_between_strokes=delta_x_y_z_between_strokes,
        num_strokes=num_strokes,
        duration_per_stroke=1.5,
        num_attempts_per_stroke=1,
    )


def visualize_bbox_rerun(rgb_image: np.ndarray, bbox: list[int], label: str = "detection"):
    """Visualize bounding box on image in rerun."""
    rr.init("wipe_detection", spawn=True)

    # Draw bbox on image: bbox is [ymin, xmin, ymax, xmax]
    ymin, xmin, ymax, xmax = bbox
    annotated_image = rgb_image.copy()

    # Draw rectangle (color is RGB: red)
    cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)

    # Draw label background and text
    cv2.putText(annotated_image, label, (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Log the annotated RGB image
    rr.log("camera/rgb_annotated", rr.Image(annotated_image))

    # Also log the original image for comparison
    rr.log("camera/rgb_original", rr.Image(rgb_image))

    print(f"Visualized bbox: ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run wipe_online with rerun visualization")
    parser.add_argument("query", nargs="?", default="erase the citrus fruit on the whiteboard",
                        help="VLM query for detection")
    parser.add_argument("--robot-ip", default="128.30.224.88", help="Robot IP address")
    args = parser.parse_args()

    print(f"Running wipe_online with query: {args.query}")
    with BambooFrankaClient(server_ip=args.robot_ip) as rob:
        wipe_online(rob, vlm_query_template=args.query)


if __name__ == "__main__":
    main()

