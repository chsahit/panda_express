from typing import Tuple
from PIL import Image
import cv2
import json
from datetime import datetime
import numpy as np
import os

from bamboo.client import BambooFrankaClient
from perception.zed.zed_cam import ZedCamera
from perception.utils.transform import pixel_to_world_xyz
from perception.utils.pretrained_model_interface import GoogleGeminiVLM
from skills.go_to_conf import goto_hand_position, TOP_DOWN_GRASP_ROT


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


def grasp_with_vlm(
    robot: BambooFrankaClient,
    text_prompt: str | None,
) -> None:
    """High-level grasp skill that uses a VLM to choose a pixel from a prompt.
    """
    print(f"Calling grasping with VLM for text prompt: {text_prompt}")

    cam = ZedCamera(serial_number=35317039)
    bgra = cam.get_bgra_frame()
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    depth = cam.get_depth_frame()
    K = cam.get_intrinsics()[0]
    cam.close()
    extrinsics = np.load("perception/zed/X_WE.npy")

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

    if pixel is not None:
        # Grasp at the pixel with a top-down grasp.
        pixel_xyz = pixel_to_world_xyz(pixel[0], pixel[1], depth, K, extrinsics)
        X_WGoal = np.eye(4)
        X_WGoal[:3, :3] = TOP_DOWN_GRASP_ROT
        X_WGoal[:3, 3] = pixel_xyz + np.array([0.0, 0.0, 0.15])
        goto_hand_position(robot, X_WGoal, 5.0)
        rob.close_gripper()
        return

    raise RuntimeError("Grasp failed: VLM did not return a valid pixel.")

if __name__ == "__main__":
    with BambooFrankaClient(server_ip="128.30.224.88") as rob:
        grasp_with_vlm(rob, "red mug")
