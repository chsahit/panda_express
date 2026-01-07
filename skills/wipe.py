import json
import numpy as np
from PIL import Image

from bamboo.client import BambooFrankaClient
from scipy.spatial.transform import Rotation as R
from skills.go_to_conf import goto_hand_position
from skills.utils.pretrained_model_interface import GoogleGeminiVLM


DEFAULT_WIPE_ONLINE_Z_OFFSET = 0.08
DEFAULT_WIPE_VLM_QUERY_TEMPLATE = (
    "You are given an image. Identify the spill region (liquid/food spill/stain) if present.\n"
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

def _compute_wipe_params_from_bbox_zed(
    bbox: list[int],
    depth_m: np.ndarray,
    K_zed: np.ndarray,
    T_world_zed: np.ndarray,
    clearance: float = 0.08,
    spacing_m: float = 0.05,
    max_stroke_len: float = 0.35,
):
    """Compute wipe parameters from bbox"""
    ymin, xmin, ymax, xmax = bbox
    p_br = (int(xmax), int(ymax))
    p_tr = (int(xmax), int(ymin))
    p_bl = (int(xmin), int(ymax))

    P_br = pixel_to_world_xyz(*p_br, depth_m, K_zed, T_world_zed)
    P_tr = pixel_to_world_xyz(*p_tr, depth_m, K_zed, T_world_zed)
    P_bl = pixel_to_world_xyz(*p_bl, depth_m, K_zed, T_world_zed)


    # wipe_start_rotation = R.from_euler('y', np.pi/2 - 0.087)
    wipe_start_rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1, 0], [-0.0, 0, -1.0]])
    wipe_start_pose = np.eye(4)
    wipe_start_pose[:3, :3] = wipe_start_rotation
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
    end_look_pose[:3, :3] = wipe_start_rotation
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
        rgb_image: np.ndarray,
        depth_img: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        vlm_query_template: str = DEFAULT_WIPE_VLM_QUERY_TEMPLATE,
        z_offset: float = DEFAULT_WIPE_ONLINE_Z_OFFSET,
        expand_percentage: float = 0.0
):
    rgb_pil = Image.fromarray(rgb_image)
    depth_m = depth_img.astype(np.float32)
    # Run VLM on the full-resolution RGB image and get bbox in RGB pixel coordinates.
    bbox = get_bbox_from_gemini(vlm_query_template, rgb_pil)
    print(f"The coordinates of the bounding box (RGB space) are: {bbox}")

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


if __name__ == "__main__":
    with BambooFrankaClient(server_ip="128.30.224.88") as rob:
        stroke_dx = 0.15
        stroke_dy = 0.0
        delta_x_y_z_between_strokes = np.array([0, 0.05, 0.0])
        start_pose = np.array([[1.0, 0.0, 0.0, 0.4], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.6], [0, 0, 0, 1]])
        wipe_multiple_strokes(
            rob,
            start_pose,
            start_pose,
            stroke_dx,
            stroke_dy,
            delta_x_y_z_between_strokes,
            4,
            1.0,
            1
        )

