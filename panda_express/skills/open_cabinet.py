import cv2
import numpy as np
from typing import Tuple
from PIL import Image
from scipy.spatial.transform import Rotation as R

from bamboo.client import BambooFrankaClient
from panda_express.perception.zed.zed_cam import ZedCamera
from panda_express.perception.utils.transform import pixel_to_world_xyz
from panda_express.perception.utils.pretrained_model_interface import GoogleGeminiVLM
from panda_express.skills.go_to_conf import goto_hand_position, TOP_DOWN_GRASP_ROT
from panda_express.skills.grasp_vlm import _get_pixel_from_gemini

prompt_get_handle_pixel = """
    Point to the CENTER (MIDDLE) of the TOP ORANGE handle of the drawer.
    The answer should follow the json format: [{"point": , "label": }, ...]. The points are in [y, x] format normalized to 0-1000.
    """
prompt_get_drawer_surface_pixel = """
    Point to 15 points on the front face of the TOP drawer, but avoid the drawer handles (including the ORANGE handle) or the edges of the front face.
    Make sure the points are on the front face, not the side face.
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

    pregrasp_xyz = pixel_xyz - np.array([0.25, 0.0, 0.0])
    X_WPregrasp = np.eye(4)
    X_WPregrasp[:3, :3] = CABINET_GRASPING_ROT
    X_WPregrasp[:3, 3] = pregrasp_xyz


    grasp_xyz = pixel_xyz - np.array([0.125, 0.0, 0.0])
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
