import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bamboo.bamboo_client import BambooFrankaClient
from skills.skills_config import intrinsics, X_GC
from skills.go_to_conf import goto_hand_position


def grasp_at_pixel(robot: BambooFrankaClient, rgbd: np.ndarray, pixel: np.ndarray):
    # get xyz position p_WP by using pixel (x, y) to look up a depth coordinate in rgbd
    # and project to world coordinates

    p_WP = None
    p_WP = np.array([0.4, 0.0, 0.3])
    # compute hand position relative to xyz position (assume top-down grasp)
    grasp_orientation = np.array([[1.0, 0.0, 0.0], [0.0, -1, 0], [-0.0, 0, -1.0]])

    X_WPregrasp = np.eye(4)
    X_WPregrasp[:3, :3] = grasp_orientation
    X_WPregrasp[:3, 3] = np.copy(p_WP) + np.array([0, 0, 0.2])

    X_WGrasp = np.eye(4)
    X_WGrasp[:3, :3] = grasp_orientation
    X_WGrasp[:3, 3] = np.copy(p_WP) + np.array([0, 0, 0.1])

    # move robot to hand position
    goto_hand_position(robot, X_WPregrasp, 5.0)
    goto_hand_position(robot, X_WGrasp, 2.0)

    # close fingers
    robot.close_gripper()


if __name__ == "__main__":
    with BambooFrankaClient(server_ip="128.30.224.88") as rob:
        grasp_at_pixel(rob, None, None)
