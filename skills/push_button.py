"""Skill to identify a button and push it.

1) VLM-based bounding box (Gemini) → center pixel
"""

from typing import Optional, Literal, Tuple, List

import json
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
from datetime import datetime
from bamboo.client import BambooFrankaClient
from perception.zed.zed_cam import ZedCamera
from perception.utils import pixel_to_world_xyz
from skills.go_to_conf import goto_hand_position, TOP_DOWN_GRASP_ROT
from skills.utils.pretrained_model_interface import GoogleGeminiVLM

def overlay_pixels_on_image(
    image: Image.Image,
    pixels: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
) -> Image.Image:
    """Draw small circles at the given (x, y) pixels on a copy of the image."""
    draw = ImageDraw.Draw(image)
    for x, y in pixels:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=2,
        )
    return image


def get_multiple_pixels_from_gemini(
    vlm_query_str: str, pil_image: Image.Image, num_pixels: int = 15
) -> List[Tuple[int, int]]:
    """Query Gemini VLM to return multiple pixels on the target object.

    Expects the model to respond with a JSON array like:
    [ {"point": [y, x]}, ... ] with coordinates normalized to [0, 1000].
    """
    vlm = GoogleGeminiVLM("gemini-2.5-pro")

    def parse_json_output(json_output_str: str) -> str:
        lines = json_output_str.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                json_output_str = "\n".join(lines[i + 1 :])
                json_output_str = json_output_str.split("```")[0]
                break
        json_output_str = json_output_str.strip()
        return json_output_str

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
    # if len(parsed_data) < num_pixels:
    #     raise ValueError(f"Parsed JSON has less than {num_pixels} points.")

    pixels: List[Tuple[int, int]] = []
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
        img_height = pil_image.height
        img_width = pil_image.width
        y = int(y_norm * img_height / 1000.0)
        x = int(x_norm * img_width / 1000.0)
        y = max(0, min(y, img_height - 1))
        x = max(0, min(x, img_width - 1))
        pixels.append((x, y))

    return pixels


def get_single_pixel_from_gemini(vlm_query_str: str, pil_image: Image.Image) -> Tuple[int, int]:
    """Query Gemini VLM to return a single (x, y) pixel on the target object."""
    pixels = get_multiple_pixels_from_gemini(vlm_query_str, pil_image, num_pixels=1)
    if not pixels:
        raise ValueError("Gemini returned no pixels for single-pixel query.")
    return pixels[0]


def push_button(
    robot: BambooFrankaClient,
    label: str = "button",
    z_clearance: float = 0.15,
    press_depth: float = 0.035,
    press_duration: float = 0.5,
) -> None:
    """Identify a button and push on it.

    Args:
        robot: bamboo client
        label: Target label to find (default: "button")
        z_clearance: Approach standoff distance in meters
        press_depth: Distance above the button to press the button (to account for spot finger length)
        press_duration: Duration for press motion (s)
    """

    cam = ZedCamera(serial_number=35317039)
    bgra = cam.get_bgra_frame()
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    depth = cam.get_depth_frame()
    K = cam.get_intrinsics()[0]
    cam.close()

    pil = Image.fromarray(rgb)

    save_folderpath = "push_button_images"
    os.makedirs(save_folderpath, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ## save the rgb and the depth image to the disk
    rgb_pil = Image.fromarray(rgb)
    annotated_rgb_pil = rgb_pil.copy()
    # depth_pil = Image.fromarray(depth)
    rgb_pil.save(os.path.join(save_folderpath, f"rgb_{timestamp}.png"))
    # depth_pil.save(os.path.join(save_folderpath, f"depth_{timestamp}.png"))


    # 2) Select a single center pixel via VLM (this will be used as the press target).
    vlm_query_center = f"""
    Point to the center of the {label} in the image. Return a JSON list with exactly one element like
    [{{"point": [y, x]}}] with coordinates normalized to 0-1000.
    """
    extrinsics = np.load("perception/zed/base_to_external_camera.npy")
    center_pixel = get_single_pixel_from_gemini(vlm_query_center, pil)
    annotated_rgb_pil = overlay_pixels_on_image(annotated_rgb_pil, [center_pixel], color=(0, 0, 255), radius=3)
    annotated_rgb_pil.save(os.path.join(save_folderpath, f"annotated_rgb_{timestamp}.png"))
    center_xyz = pixel_to_world_xyz(center_pixel[0], center_pixel[1], depth, K, extrinsics)

    X_Wsetpoint1 = np.eye(4)
    X_Wsetpoint1[:3, :3] = TOP_DOWN_GRASP_ROT
    X_Wsetpoint1[:3, 3] = center_xyz + np.array([0.0, 0.0, z_clearance])
    breakpoint()

    goto_hand_position(robot, X_Wsetpoint1, 5.0)


def main():
    with BambooFrankaClient(server_ip="128.30.224.88") as rob:
        push_button(rob)

if __name__ == "__main__":
    main()
