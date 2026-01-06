import numpy as np

from bamboo.client import BambooFrankaClient
from skills.go_to_conf import goto_hand_position
from skills.utils.pretrained_model_interface import GoogleGeminiVLM


def _add_offset(pose: np.ndarray, offset: np.ndarray) -> np.ndarray:
    new_pose = np.copy(pose)
    new_pose[:3, 3] += offset
    return new_pose

def wipe_multiple_strokes(
    robot: BambooFrankaClient,
    wipe_start_pose: np.ndarray,
    end_look_pose: np.ndarray,
    stroke_dx: float,
    stroke_dy: float,
    delta_x_y_between_strokes: np.ndarray,
    num_strokes: int,
    duration_per_stroke: float,
    num_attempts_per_stroke: int,
):
    """
    Execute multiple wipe strokes. After each stroke (and attempts) the start pose
    is shifted by delta_x_y_between_strokes in BODY frame.
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
        curr = _add_offset(curr, delta_x_y_between_strokes)
    # End look pose
    goto_hand_position(robot, end_look_pose, 5.0)



if __name__ == "__main__":
    with BambooFrankaClient(server_ip="128.30.224.88") as rob:
        stroke_dx = 0.15
        stroke_dy = 0.0
        delta_x_y_between_strokes = np.array([0, 0.05, 0.0])
        start_pose = np.array([[1.0, 0.0, 0.0, 0.4], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.6], [0, 0, 0, 1]])
        wipe_multiple_strokes(
            rob,
            start_pose,
            start_pose,
            stroke_dx,
            stroke_dy,
            delta_x_y_between_strokes,
            4,
            1.0,
            1
        )

