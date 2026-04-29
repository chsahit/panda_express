import argparse
import logging
from functools import cache
import numpy as np

import torch
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from cutamp.config import TAMPConfiguration
from cutamp.robots.ur5 import get_ur5_ik_solver, ur5_curobo_cfg, ur5_home

from tiptop.ur5.ur5_client import UR5Client
from tiptop.workspace import ur5_workspace

# UR5 cuTAMP config — robot field tells downstream solvers which embodiment.
# Other fields stay at TAMPConfiguration defaults; revisit if/when we wire up
# full pick-and-place planning.
TAMP_CONFIG: TAMPConfiguration = TAMPConfiguration(robot="ur5")

tensor_args = TensorDeviceType()


@cache
def get_world_cfg() -> WorldConfig:
    """cuRobo WorldConfig containing UR5 workspace cuboids for collision checking."""
    return WorldConfig(cuboid=list(ur5_workspace()))


@cache
def get_ik_solver() -> IKSolver:
    """Module-level cuRobo IK solver for UR5e + Robotiq 2F-85 against the workspace.

    The configured ee_link is `grasp_frame` (see cutamp ur5e_robotiq_2f_85.yml);
    pass world-to-grasp_frame poses to rtb_IK / goto_hand_position.
    """
    return get_ur5_ik_solver(get_world_cfg())


# Time dilation factor passed to MotionGen plans (1.0 = max speed, lower = slower).
# 0.3 is a conservative default for first-time use; bump up once trusted.
DEFAULT_TIME_DILATION_FACTOR = 0.3


@cache
def get_motion_gen(warmup_iters: int = 16) -> MotionGen:
    """Module-level cuRobo MotionGen for UR5 against the workspace, warmed up on first call."""
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg=ur5_curobo_cfg(),
        world_model=get_world_cfg(),
        use_cuda_graph=True,
        collision_activation_distance=0.0,
        position_threshold=0.01,
        rotation_threshold=0.1,
    )
    motion_gen = MotionGen(motion_gen_cfg)
    for _ in range(warmup_iters):
        motion_gen.warmup()
    return motion_gen


def get_ee_pose(robot: UR5Client) -> np.ndarray:
    """Return the current world-to-grasp_frame pose as a 4x4 numpy matrix.

    Computed by cuRobo forward kinematics on the UR5's current joint
    positions. The frame returned is the same `grasp_frame` (gripper tip)
    that goto_hand_position / rtb_IK accept as their target, so reads and
    commands are mutually consistent.

    Stand-in for `np.array(client.get_joint_states()['ee_pose'])` from the
    Bamboo/panda API, which UR5Client doesn't expose.
    """
    q = np.asarray(robot.get_joint_positions(), dtype=np.float32)
    if q.shape != (6,):
        raise ValueError(f"Robot returned {q.shape[0]}-DOF state, expected 6")
    state = get_motion_gen().kinematics.get_state(tensor_args.to_device(q))
    return state.ee_pose.get_numpy_matrix()[0]


def contactgraspnet_to_panda(cg_grasp: np.ndarray) -> np.ndarray:
    """No-op for the UR5 grasp_frame.

    The panda variant of this function applies a -90° rotation (CGN→panda
    convention) plus a +45° rotation that compensates for the panda URDF's
    panda_link8 → panda_hand offset. Neither applies to the UR5: cuTAMP's
    UR5 ee_link is `grasp_frame`, which is already the gripper tip, with no
    link8 indirection. CGN poses pass through unchanged here.

    Kept under the panda-suffixed name for drop-in import compatibility.
    """
    return cg_grasp


def goto_joint_angles(
    robot: UR5Client,
    q: np.ndarray,
    time: float,
    time_dilation_factor: float = DEFAULT_TIME_DILATION_FACTOR,
) -> int:
    """Motion-plan to a 6-DOF UR5 joint config and execute on the robot.

    Same I/O contract as the panda goto_joint_angles: returns 0 on success,
    1 on failure. `time` is accepted for API parity but is unused — the
    trajectory length is set by cuRobo's MotionGen and `time_dilation_factor`.
    """
    del time  # accepted for API parity, not used

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        q_target = np.asarray(q, dtype=np.float32)
        if q_target.shape[0] >= 7:
            q_target = q_target[:6]
        if q_target.shape != (6,):
            raise ValueError(f"Expected 6-DOF q_target, got shape {q_target.shape}")

        q_start = np.asarray(robot.get_joint_positions(), dtype=np.float32)
        if q_start.shape != (6,):
            raise ValueError(f"Robot returned {q_start.shape[0]}-DOF state, expected 6")

        motion_gen = get_motion_gen()
        js_start = JointState.from_position(tensor_args.to_device(q_start)[None])
        js_target = JointState.from_position(tensor_args.to_device(q_target)[None])

        plan_config = MotionGenPlanConfig(time_dilation_factor=time_dilation_factor)
        torch.cuda.synchronize()
        result = motion_gen.plan_single_js(js_start, js_target, plan_config)
        torch.cuda.synchronize()

        if not result.success.all():
            print(f"✗ Motion plan failed: {result.status}")
            return 1

        print("\nSending trajectory to robot...")
        plan = result.interpolated_plan
        durations = [result.interpolation_dt] * plan.position.shape[0]
        exec_result = robot.execute_joint_impedance_path(
            joint_confs=plan.position.cpu().numpy(),
            joint_vels=plan.velocity.cpu().numpy(),
            durations=durations,
        )

        final_joints = np.asarray(robot.get_joint_positions())
        print(f"Final joint angles: {[f'{x:.4f}' for x in final_joints]}")
        position_error = np.linalg.norm(final_joints - q_target)
        print(f"Final position error: {position_error:.6f}")

        if exec_result.get('success'):
            print("✓ Trajectory executed successfully!")
        else:
            print(f"✗ Trajectory failed: {exec_result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def rtb_IK(X_WG: np.ndarray, q0: np.ndarray, gripper_type: str = "robotiq", clean: bool = True):
    """IK for the UR5 in the configured ee_link frame (`grasp_frame`).

    Same I/O contract as the panda rtb_IK: returns a numpy joint vector on
    success, or None on failure. The vector is 6-DOF here (UR5) where the
    panda version returns 7-DOF.

    Args:
        X_WG: 4x4 world-to-ee pose.
        q0: seed configuration; first 6 entries are used as the IK seed.
        gripper_type: ignored — the UR5 wrap supports Robotiq 2F-85 only.
        clean: orthonormalise the rotation block of X_WG before solving.
    """
    del gripper_type  # accepted for API parity, not used

    X_WG_clean = np.asarray(X_WG, dtype=np.float64, order='C')
    if X_WG_clean.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {X_WG_clean.shape}")

    if clean:
        R = X_WG_clean[:3, :3].copy()
        U, _, Vt = np.linalg.svd(R)
        R_clean = U @ Vt
        if np.linalg.det(R_clean) < 0:
            Vt[-1, :] *= -1
            R_clean = U @ Vt
        X_WG_clean = X_WG_clean.copy()
        X_WG_clean[:3, :3] = R_clean

    # cuRobo expects torch tensors on the IK solver's device.
    from curobo.types.math import Pose

    pose = Pose.from_matrix(X_WG_clean)
    seed = tensor_args.to_device(np.asarray(q0, dtype=np.float32)[:6]).view(1, 1, 6)
    result = get_ik_solver().solve_single(
        pose, seed_config=seed, retract_config=seed.view(1, 6)
    )
    if not bool(result.success.view(-1)[0].item()):
        return None
    return result.solution.view(-1, 6)[0].cpu().numpy()


def goto_hand_position(
    rob: UR5Client,
    X_WG: np.ndarray,
    time: float,
    gripper_type: str = "robotiq",
    n_ik_attempts: int = 5,
    max_joint_dist: float = 1.0,
    ik_solver: str = "ik_lm",
) -> int:
    """Solve IK for X_WG (in `grasp_frame`), then motion-plan and execute on the UR5.

    Same I/O contract as the panda goto_hand_position. `gripper_type`,
    `n_ik_attempts`, and `ik_solver` are accepted for API parity but are not
    used — cuRobo's IK solver handles seed restarts internally.
    """
    del gripper_type, n_ik_attempts, ik_solver  # accepted for API parity

    q_current = np.asarray(rob.get_joint_positions(), dtype=np.float32)
    if q_current.shape != (6,):
        raise ValueError(f"Robot returned {q_current.shape[0]}-DOF state, expected 6")

    ik_soln = rtb_IK(X_WG, q_current)
    if ik_soln is None:
        return 1

    dist = float(np.linalg.norm(ik_soln - q_current))
    if dist > max_joint_dist:
        raise RuntimeError(
            f"IK solution too far from current config: joint_dist={dist:.2f} rad "
            f"(max={max_joint_dist}). Skipping to avoid drastic motion."
        )

    print(f"IK solved (joint_dist={dist:.4f} rad)")
    return goto_joint_angles(rob, ik_soln, time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='move UR5 to home configuration')
    parser.add_argument('--ur5-ip', type=str, required=True,
                        help='UR5 IP address')
    args = parser.parse_args()
    q_neutral = np.asarray(ur5_home, dtype=np.float64)
    rob = UR5Client(args.ur5_ip)
    try:
        goto_joint_angles(rob, q_neutral, 5)
    finally:
        rob.close()
