# import logging
# import numpy as np
# from pathlib import Path
# from typing import Literal
# import roboticstoolbox as rtb
# from spatialmath import SE3
# from scipy.spatial.transform import Rotation

# from bamboo.client import BambooFrankaClient

# # URDF directory and mapping for different gripper types
# URDF_DIR = Path(__file__).parent / "urdfs"
# URDF_MAP = {
#     "robotiq": URDF_DIR / "fr3_robotiq_2f_85.urdf",
#     "franka": URDF_DIR / "panda_arm_hand.urdf",
# }

# # Cache for loaded models (lazy loading)
# _MODEL_CACHE = {}

# TOP_DOWN_GRASP_ROT = np.array([[1.0, 0.0, 0.0], [0.0, -1, 0], [-0.0, 0, -1.0]])


# def load_robot_model(gripper_type: str = "robotiq"):
#     """Load and cache robot model for specified gripper type.

#     Args:
#         gripper_type: Gripper type - "robotiq" or "franka" (default: "robotiq")

#     Returns:
#         Robot model from roboticstoolbox

#     Raises:
#         ValueError: If gripper_type is not recognized
#         FileNotFoundError: If URDF file doesn't exist
#     """
#     if gripper_type not in URDF_MAP:
#         raise ValueError(f"Unknown gripper_type '{gripper_type}'. Options: {list(URDF_MAP.keys())}")

#     urdf_path = URDF_MAP[gripper_type]

#     if not urdf_path.exists():
#         raise FileNotFoundError(f"URDF file not found: {urdf_path}")

#     # Return cached model if already loaded
#     if urdf_path not in _MODEL_CACHE:
#         print(f"Loading robot model from: {urdf_path}")
#         _MODEL_CACHE[urdf_path] = rtb.Robot.URDF(str(urdf_path))

#     return _MODEL_CACHE[urdf_path]


# def list_available_grippers():
#     """List all available gripper types."""
#     return list(URDF_MAP.keys())


# def contactgraspnet_to_panda(cg_grasp: np.ndarray) -> np.ndarray:
#     """Convert ContactGraspNet grasp convention to Panda convention.

#     Applies -90° rotation around Z axis to align frame conventions,
#     then +45° rotation to compensate for panda_link8 to panda_hand offset.

#     Args:
#         cg_grasp: 4x4 grasp pose matrix in ContactGraspNet convention

#     Returns:
#         4x4 grasp pose matrix in Panda convention
#     """
#     # First: -90° rotation (ContactGraspNet to standard Panda convention)
#     transform_90 = np.array([
#         [ 0, -1,  0,  0],
#         [ 1,  0,  0,  0],
#         [ 0,  0,  1,  0],
#         [ 0,  0,  0,  1]
#     ])

#     # Second: +45° rotation to compensate for URDF panda_link8->panda_hand offset
#     angle = np.pi / 4  # +45 degrees
#     cos_a = np.cos(angle)
#     sin_a = np.sin(angle)
#     transform_45 = np.array([
#         [cos_a, -sin_a,  0,  0],
#         [sin_a,  cos_a,  0,  0],
#         [    0,      0,  1,  0],
#         [    0,      0,  0,  1]
#     ])

#     # Apply both transforms
#     return cg_grasp @ transform_90 @ transform_45


# def goto_joint_angles(robot: BambooFrankaClient, q: np.ndarray, time: float) -> int:
#     # Set up logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )

#     try:
#         # Get current joint angles
#         current_joints = robot.get_joint_positions()

#         waypoints = [current_joints]
#         dt = 0.02
#         durations = [dt]
#         num_steps = int(time / dt)
#         for i in range(num_steps):
#             waypoint = current_joints + (i + 1) / num_steps * (q - current_joints)
#             waypoints.append(waypoint)
#             durations.append(dt)

#         print("\nSending trajectory to robot...")
#         result = robot.execute_joint_impedance_path(np.array(waypoints), durations=durations)

#         # Get final joint positions to calculate error
#         final_joints = robot.get_joint_positions()
#         print(f"Final joint angles: {[f'{q:.4f}' for q in final_joints]}")

#         # Calculate final position error (compare to last waypoint)
#         position_error = np.linalg.norm(np.array(final_joints) - np.array(waypoints[-1]))
#         print(f"Final position error: {position_error:.6f}")

#         if result['success']:
#             print("✓ Trajectory executed successfully!")
#         else:
#             print(f"✗ Trajectory failed: {result.get('error', 'Unknown error')}")

#     except Exception as e:
#         print(f"Error: {e}")
#         return 1

#     return 0


# def compute_twist_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
#     """Compute 6D twist error between current and target poses.

#     Args:
#         T_current: 4x4 current pose matrix
#         T_target: 4x4 target pose matrix

#     Returns:
#         6D twist error [e_pos (3), e_rot (3)] where e_rot is axis-angle
#     """
#     # Position error
#     e_pos = T_target[:3, 3] - T_current[:3, 3]

#     # Rotation error as axis-angle
#     R_current = T_current[:3, :3]
#     R_target = T_target[:3, :3]
#     R_error = R_target @ R_current.T

#     # Convert rotation matrix to axis-angle using scipy
#     rot = Rotation.from_matrix(R_error)
#     e_rot = rot.as_rotvec()  # axis-angle representation

#     return np.concatenate([e_pos, e_rot])


# def goto_hand_position_jacobian(rob: BambooFrankaClient, X_WG: np.ndarray, time: float,
#                                 gripper_type: str = "robotiq",
#                                 damping: float = 0.05,
#                                 max_joint_delta: float = 0.5) -> int:
#     """Move hand to specified pose using Jacobian pseudoinverse.

#     This method computes joint deltas directly from Cartesian error using the
#     Jacobian, which naturally stays near the current configuration. Unlike IK
#     solvers that can jump to different arm configurations, this method is
#     incremental and predictable.

#     Args:
#         rob: Robot client
#         X_WG: 4x4 target pose matrix
#         time: Movement duration in seconds
#         gripper_type: Gripper type - "robotiq" or "franka" (default: "robotiq")
#         damping: Damping factor for pseudoinverse (default: 0.05). Higher values
#             make the solution more stable near singularities but less accurate.
#         max_joint_delta: Maximum allowed joint delta per joint in radians (default: 0.5).
#             If any joint delta exceeds this, the motion is scaled down.

#     Returns:
#         0 on success, 1 on failure
#     """
#     robot_model = load_robot_model(gripper_type)

#     s_current = rob.get_joint_states()
#     q_current = np.array(s_current["qpos"])[:7]

#     # Get current pose via forward kinematics
#     T_current_se3 = robot_model.fkine(q_current, end="panda_link8")
#     T_current = np.array(T_current_se3.A)

#     # Clean up target pose
#     X_WG_clean = np.asarray(X_WG, dtype=np.float64, order='C')
#     if X_WG_clean.shape != (4, 4):
#         raise ValueError(f"Expected 4x4 matrix, got shape {X_WG_clean.shape}")

#     # Orthonormalize rotation matrix
#     R = X_WG_clean[:3, :3].copy()
#     U, _, Vt = np.linalg.svd(R)
#     R_clean = U @ Vt
#     if np.linalg.det(R_clean) < 0:
#         Vt[-1, :] *= -1
#         R_clean = U @ Vt

#     T_target = np.eye(4)
#     T_target[:3, :3] = R_clean
#     T_target[:3, 3] = X_WG_clean[:3, 3]

#     # Compute 6D twist error
#     twist_error = compute_twist_error(T_current, T_target)

#     pos_error = np.linalg.norm(twist_error[:3])
#     rot_error = np.linalg.norm(twist_error[3:])
#     print(f"Jacobian solver: pos_error={pos_error:.4f}m, rot_error={np.degrees(rot_error):.2f}°")

#     # Get Jacobian at current configuration
#     J = robot_model.jacob0(q_current, end="panda_link8")  # 6x7 Jacobian

#     # Compute damped pseudoinverse: J^T @ (J @ J^T + λ²I)^{-1}
#     JJT = J @ J.T
#     damping_matrix = (damping ** 2) * np.eye(6)
#     J_pinv = J.T @ np.linalg.inv(JJT + damping_matrix)

#     # Compute joint delta
#     dq = J_pinv @ twist_error

#     # Check and scale if any joint delta is too large
#     max_dq = np.max(np.abs(dq))
#     if max_dq > max_joint_delta:
#         scale = max_joint_delta / max_dq
#         dq = dq * scale
#         print(f"  Scaled joint delta by {scale:.3f} (max was {max_dq:.3f} rad)")

#     # Compute goal joints
#     q_goal = q_current + dq

#     # Check joint limits (approximate Franka limits)
#     joint_limits = np.array([
#         [-2.90, 2.90],   # q0
#         [-1.76, 1.76],   # q1
#         [-2.90, 2.90],   # q2
#         [-3.07, -0.07],  # q3
#         [-2.90, 2.90],   # q4
#         [-0.02, 3.75],   # q5
#         [-2.90, 2.90],   # q6
#     ])

#     for i in range(7):
#         if q_goal[i] < joint_limits[i, 0]:
#             print(f"  Warning: q{i} would exceed lower limit, clamping")
#             q_goal[i] = joint_limits[i, 0] + 0.01
#         if q_goal[i] > joint_limits[i, 1]:
#             print(f"  Warning: q{i} would exceed upper limit, clamping")
#             q_goal[i] = joint_limits[i, 1] - 0.01

#     joint_dist = np.linalg.norm(dq)
#     print(f"  Joint delta: {np.round(dq, 4)}")
#     print(f"  Joint delta norm: {joint_dist:.4f} rad ({np.degrees(joint_dist):.1f}°)")

#     return goto_joint_angles(rob, q_goal, time)


# def goto_hand_position(rob: BambooFrankaClient, X_WG: np.ndarray, time: float,
#                       gripper_type: str = "robotiq",
#                       q_reference: np.ndarray = None,
#                       ik_solver: Literal["jacobian", "jacobian_pinv", "ik_lm"] = "ik_lm") -> int:
#     """Move hand to specified pose using inverse kinematics.

#     Args:
#         rob: Robot client
#         X_WG: 4x4 target pose matrix
#         time: Movement duration in seconds
#         gripper_type: Gripper type - "robotiq" or "franka" (default: "robotiq")
#         q_reference: Optional 7D reference joint config for posture optimization.
#             IK solutions are scored by a blend of distance to q_current (reachability)
#             and distance to q_reference (posture). If None, only q_current is used.
#         ik_solver: IK solver to use:
#             - "jacobian_pinv": Direct Jacobian pseudoinverse. Computes joint delta from
#               Cartesian error. Best for small movements, always stays near current config.
#             - "jacobian": Newton-Raphson with damped pseudoinverse (rtb.ik_NR).
#             - "ik_lm": Levenberg-Marquardt with random restarts (default).

#     Returns:
#         0 on success, 1 on failure
#     """
#     # Use direct Jacobian pseudoinverse method
#     if ik_solver == "jacobian_pinv":
#         return goto_hand_position_jacobian(rob, X_WG, time, gripper_type)
#     # Load appropriate robot model
#     robot_model = load_robot_model(gripper_type)

#     s_current = rob.get_joint_states()
#     q_current = np.array(s_current["qpos"])
#     X_current = s_current["ee_pose"]

#     # Ensure X_WG is a proper 4x4 float64 contiguous array for SE3
#     X_WG_clean = np.asarray(X_WG, dtype=np.float64, order='C')
#     if X_WG_clean.shape != (4, 4):
#         raise ValueError(f"Expected 4x4 matrix, got shape {X_WG_clean.shape}")

#     # Extract rotation and translation for SE3
#     R = X_WG_clean[:3, :3].copy()
#     t = X_WG_clean[:3, 3].copy()

#     # Ensure R is a proper rotation matrix (orthonormalize using SVD)
#     U, _, Vt = np.linalg.svd(R)
#     R_clean = U @ Vt

#     # Ensure det(R) = +1 (proper rotation, not reflection)
#     if np.linalg.det(R_clean) < 0:
#         Vt[-1, :] *= -1
#         R_clean = U @ Vt

#     # Create SE3 object from rotation matrix and translation vector
#     T_target = SE3.Rt(R_clean, t)

#     MAX_JOINT_DIST = 1.0  # radians — reject IK solutions further than this (from q_current)

#     # --- Jacobian (damped pseudoinverse Newton-Raphson) solver ---
#     if ik_solver == "jacobian":
#         solution = robot_model.ik_NR(
#             T_target,
#             q0=q_current[:7],
#             end="panda_link8",
#             mask=[1, 1, 1, 1, 1, 1],
#             pinv=True,
#             pinv_damping=0.05,
#             slimit=1,       # single search from q0, no random restarts
#             ilimit=100,     # enough iterations for convergence
#         )

#         if solution[1]:
#             q_sol = solution[0]
#             dist = np.linalg.norm(q_sol - q_current[:7])
#             if dist <= MAX_JOINT_DIST:
#                 print(f"Jacobian IK solved (joint_dist={dist:.4f} rad)")
#                 return goto_joint_angles(rob, q_sol, time)
#             else:
#                 print(f"Jacobian IK solution too far: joint_dist={dist:.4f} rad (max={MAX_JOINT_DIST}), falling back to ik_lm")
#         else:
#             print("Jacobian IK did not converge, falling back to ik_lm")

#     # --- Levenberg-Marquardt solver with random restarts ---
#     NUM_IK_RETRIES = 10   # number of random restarts to try
#     POSTURE_ALPHA = 0.35  # blend weight: 0=only posture, 1=only reachability

#     best_q = None
#     best_score = float("inf")
#     best_dist_current = float("inf")

#     for attempt in range(NUM_IK_RETRIES):
#         if attempt == 0:
#             q0 = q_current[:7]
#         else:
#             # Small random perturbation around current joints
#             q0 = q_current[:7] + np.random.randn(7) * 0.1

#         solution = robot_model.ik_LM(
#             T_target,
#             q0=q0,
#             end="panda_link8",
#             mask=[1, 1, 1, 1, 1, 1],
#         )

#         if solution[1]:
#             q_candidate = solution[0]
#             dist_current = np.linalg.norm(q_candidate - q_current[:7])

#             # Score: blend distance to current config (reachability) and
#             # distance to reference config (posture), if reference is provided.
#             if q_reference is not None:
#                 dist_reference = np.linalg.norm(q_candidate - q_reference)
#                 score = POSTURE_ALPHA * dist_current + (1 - POSTURE_ALPHA) * dist_reference
#             else:
#                 score = dist_current

#             if score < best_score:
#                 best_score = score
#                 best_q = q_candidate
#                 best_dist_current = dist_current
#                 # Good enough — very close to current config
#                 if dist_current < 0.3:
#                     break

#     if best_q is not None and best_dist_current <= MAX_JOINT_DIST:
#         if q_reference is not None:
#             dist_ref = np.linalg.norm(best_q - q_reference)
#             print(f"IK solved (attempt {attempt+1}, dist_current={best_dist_current:.4f}, dist_reference={dist_ref:.4f} rad)")
#         else:
#             print(f"IK solved (attempt {attempt+1}, joint_dist={best_dist_current:.4f} rad)")
#         return goto_joint_angles(rob, best_q, time)
#     elif best_q is not None:
#         print(f"IK best solution too far: joint_dist={best_dist_current:.4f} rad (max={MAX_JOINT_DIST})")
#         print(f"  q_current: {np.round(q_current[:7], 4)}")
#         print(f"  q_ik:      {np.round(best_q, 4)}")
#         raise RuntimeError(f"IK solution jumped to different configuration (joint_dist={best_dist_current:.2f} > {MAX_JOINT_DIST})")
#     else:
#         print(f"IK failed after {NUM_IK_RETRIES} attempts")
#         raise RuntimeError("Failed to find IK solution")


# if __name__ == "__main__":
#     q_neutral = np.array([-0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, -0.14])
#     # with BambooFrankaClient(server_ip="128.30.224.88") as rob:
#     with BambooFrankaClient(server_ip="192.168.1.3") as rob:
#         goto_joint_angles(rob, q_neutral, 5)

import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Literal
import roboticstoolbox as rtb
from spatialmath import SE3
from scipy.spatial.transform import Rotation

from bamboo.client import BambooFrankaClient

# URDF directory and mapping for different gripper types
URDF_DIR = Path(__file__).parent / "urdfs"
URDF_MAP = {
    "robotiq": URDF_DIR / "fr3_robotiq_2f_85.urdf",
    "franka": URDF_DIR / "panda_arm_hand.urdf",
}

# Cache for loaded models (lazy loading)
_MODEL_CACHE = {}

TOP_DOWN_GRASP_ROT = np.array([[1.0, 0.0, 0.0], [0.0, -1, 0], [-0.0, 0, -1.0]])

Q_NEUTRAL = np.array([-0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, -0.14])


def load_robot_model(gripper_type: str = "robotiq"):
    """Load and cache robot model for specified gripper type.

    Args:
        gripper_type: Gripper type - "robotiq" or "franka" (default: "robotiq")

    Returns:
        Robot model from roboticstoolbox

    Raises:
        ValueError: If gripper_type is not recognized
        FileNotFoundError: If URDF file doesn't exist
    """
    if gripper_type not in URDF_MAP:
        raise ValueError(f"Unknown gripper_type '{gripper_type}'. Options: {list(URDF_MAP.keys())}")

    urdf_path = URDF_MAP[gripper_type]

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    # Return cached model if already loaded
    if urdf_path not in _MODEL_CACHE:
        print(f"Loading robot model from: {urdf_path}")
        _MODEL_CACHE[urdf_path] = rtb.Robot.URDF(str(urdf_path))

    return _MODEL_CACHE[urdf_path]


def list_available_grippers():
    """List all available gripper types."""
    return list(URDF_MAP.keys())


def contactgraspnet_to_panda(cg_grasp: np.ndarray) -> np.ndarray:
    """Convert ContactGraspNet grasp convention to Panda convention.

    Applies -90° rotation around Z axis to align frame conventions,
    then +45° rotation to compensate for panda_link8 to panda_hand offset.

    Args:
        cg_grasp: 4x4 grasp pose matrix in ContactGraspNet convention

    Returns:
        4x4 grasp pose matrix in Panda convention
    """
    # First: -90° rotation (ContactGraspNet to standard Panda convention)
    transform_90 = np.array([
        [ 0, -1,  0,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]
    ])

    # Second: +45° rotation to compensate for URDF panda_link8->panda_hand offset
    angle = np.pi / 4  # +45 degrees
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    transform_45 = np.array([
        [cos_a, -sin_a,  0,  0],
        [sin_a,  cos_a,  0,  0],
        [    0,      0,  1,  0],
        [    0,      0,  0,  1]
    ])

    # Apply both transforms
    return cg_grasp @ transform_90 @ transform_45


def goto_joint_angles(robot: BambooFrankaClient, q: np.ndarray, time: float) -> int:
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Get current joint angles
        current_joints = robot.get_joint_positions()

        waypoints = [current_joints]
        dt = 0.02
        durations = [dt]
        num_steps = int(time / dt)
        for i in range(num_steps):
            waypoint = current_joints + (i + 1) / num_steps * (q - current_joints)
            waypoints.append(waypoint)
            durations.append(dt)

        print("\nSending trajectory to robot...")
        result = robot.execute_joint_impedance_path(np.array(waypoints), durations=durations)

        # Get final joint positions to calculate error
        final_joints = robot.get_joint_positions()
        print(f"Final joint angles: {[f'{q:.4f}' for q in final_joints]}")

        # Calculate final position error (compare to last waypoint)
        position_error = np.linalg.norm(np.array(final_joints) - np.array(waypoints[-1]))
        print(f"Final position error: {position_error:.6f}")

        if result['success']:
            print("✓ Trajectory executed successfully!")
        else:
            print(f"✗ Trajectory failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def rtb_IK(X_WG: np.ndarray, q0: np.ndarray, gripper_type: str = "robotiq", clean: bool = True):
    robot_model = load_robot_model(gripper_type)
    if clean:
        # Ensure X_WG is a proper 4x4 float64 contiguous array for SE3
        X_WG_clean = np.asarray(X_WG, dtype=np.float64, order='C')
        if X_WG_clean.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got shape {X_WG_clean.shape}")

        # Extract rotation and translation for SE3
        R = X_WG_clean[:3, :3].copy()
        t = X_WG_clean[:3, 3].copy()

        # Ensure R is a proper rotation matrix (orthonormalize using SVD)
        U, _, Vt = np.linalg.svd(R)
        R_clean = U @ Vt

        # Ensure det(R) = +1 (proper rotation, not reflection)
        if np.linalg.det(R_clean) < 0:
            Vt[-1, :] *= -1
            R_clean = U @ Vt

        # Create SE3 object from rotation matrix and translation vector
        T_target = SE3.Rt(R_clean, t)
    else:
        T_target = SE3.Rt(X_WG[:3, :3].copy(), X_WG[:3, 3].copy())

    solution = robot_model.ik_LM(
        T_target,
        q0=q0[:7],  # Use provided joint positions as initial guess
        end="panda_link8",  # Target the end-effector frame (same as kEndEffector in libfranka)
        mask=[1, 1, 1, 1, 1, 1]  # Full 6-DOF constraint (x, y, z, roll, pitch, yaw)
    )
    if solution[1]:
        return solution[0]
    else:
        return None


def compute_twist_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """Compute 6D twist error between current and target poses.

    Args:
        T_current: 4x4 current pose matrix
        T_target: 4x4 target pose matrix

    Returns:
        6D twist error [e_pos (3), e_rot (3)] where e_rot is axis-angle
    """
    # Position error
    e_pos = T_target[:3, 3] - T_current[:3, 3]

    # Rotation error as axis-angle
    R_current = T_current[:3, :3]
    R_target = T_target[:3, :3]
    R_error = R_target @ R_current.T

    # Convert rotation matrix to axis-angle using scipy
    rot = Rotation.from_matrix(R_error)
    e_rot = rot.as_rotvec()  # axis-angle representation

    return np.concatenate([e_pos, e_rot])


def goto_hand_position_jacobian(rob: BambooFrankaClient, X_WG: np.ndarray, time: float,
                                gripper_type: str = "robotiq",
                                damping: float = 0.01,
                                max_joint_delta: float = 0.5,
                                max_iters: int = 100,
                                pos_tol: float = 1e-4,
                                rot_tol: float = 1e-3,
                                nullspace_gain: float = 1.0,
                                q_ref: np.ndarray = None,
                                max_joint_dist: float = 1.5) -> int:
    """Move hand to specified pose using hybrid Jacobian + LM IK.

    Phase 1 (Jacobian pseudoinverse): iteratively linearises the FK map with
    null-space posture control to find a joint configuration near q_current.
    Phase 2 (LM polish): uses the Jacobian result as the seed for
    Levenberg-Marquardt IK to converge to the exact Cartesian target.

    Args:
        rob: Robot client
        X_WG: 4x4 target pose matrix
        time: Movement duration in seconds
        gripper_type: Gripper type - "robotiq" or "franka" (default: "robotiq")
        damping: Damping factor for Jacobian pseudoinverse (default: 0.01).
        max_joint_delta: Maximum per-joint delta per Jacobian iteration (default: 0.5).
        max_iters: Maximum Jacobian iterations (default: 100).
        pos_tol: Convergence tolerance for position error in metres (default: 1e-4).
        rot_tol: Convergence tolerance for rotation error in radians (default: 1e-3).
        nullspace_gain: Gain for null-space posture control (default: 1.0).
        q_ref: 7D reference joint config for null-space. Defaults to Q_NEUTRAL.
        max_joint_dist: Maximum allowed total joint distance from q_current (default: 1.5).

    Returns:
        0 on success, 1 on failure
    """
    robot_model = load_robot_model(gripper_type)

    s_current = rob.get_joint_states()
    q_current = np.array(s_current["qpos"])[:7]

    if q_ref is None:
        q_ref = Q_NEUTRAL

    # Clean up target pose
    X_WG_clean = np.asarray(X_WG, dtype=np.float64, order='C')
    if X_WG_clean.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {X_WG_clean.shape}")

    # Orthonormalize rotation matrix
    R = X_WG_clean[:3, :3].copy()
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        Vt[-1, :] *= -1
        R_clean = U @ Vt

    T_target = np.eye(4)
    T_target[:3, :3] = R_clean
    T_target[:3, 3] = X_WG_clean[:3, 3]

    # Joint limits (approximate Franka limits)
    joint_limits_lo = np.array([-2.90, -1.76, -2.90, -3.07, -2.90, -0.02, -2.90])
    joint_limits_hi = np.array([ 2.90,  1.76,  2.90, -0.07,  2.90,  3.75,  2.90])

    damping_matrix = (damping ** 2) * np.eye(6)
    I7 = np.eye(7)

    # ── Phase 1: Jacobian pseudoinverse (get near the solution) ──────
    q = q_current.copy()
    jacobian_converged = False
    for it in range(max_iters):
        T_current = np.array(robot_model.fkine(q, end="panda_link8").A)
        twist_error = compute_twist_error(T_current, T_target)

        pos_error = np.linalg.norm(twist_error[:3])
        rot_error = np.linalg.norm(twist_error[3:])

        if pos_error < pos_tol and rot_error < rot_tol:
            jacobian_converged = True
            print(f"Jacobian IK converged in {it} iters: "
                  f"pos_err={pos_error:.6f}m, rot_err={np.degrees(rot_error):.4f}°")
            break

        J = robot_model.jacob0(q, end="panda_link8")
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + damping_matrix)

        # Primary task: reach Cartesian target
        dq_task = J_pinv @ twist_error

        # Secondary task: pull toward q_ref in the null space
        J_pinv_true = np.linalg.pinv(J)
        N = I7 - J_pinv_true @ J
        dq_null = N @ (nullspace_gain * (q_ref - q))

        dq = dq_task + dq_null

        # Clamp step size
        max_dq = np.max(np.abs(dq))
        if max_dq > max_joint_delta:
            dq = dq * (max_joint_delta / max_dq)

        q = q + dq

        # Clamp to joint limits
        q = np.clip(q, joint_limits_lo + 0.01, joint_limits_hi - 0.01)

    if not jacobian_converged:
        print(f"Jacobian IK stopped after {max_iters} iters: "
              f"pos_err={pos_error:.4f}m, rot_err={np.degrees(rot_error):.2f}° — handing off to LM")

    # ── Phase 2: LM polish (converge to exact target) ────────────────
    if not jacobian_converged:
        q_jacobian = q.copy()
        q_lm = rtb_IK(X_WG_clean, q_jacobian, gripper_type=gripper_type)

        if q_lm is not None:
            dist_from_current = np.linalg.norm(q_lm - q_current)
            dist_from_jacobian = np.linalg.norm(q_lm - q_jacobian)
            if dist_from_current <= max_joint_dist:
                print(f"LM polish converged: "
                      f"dist_from_current={dist_from_current:.4f} rad, "
                      f"dist_from_jacobian={dist_from_jacobian:.4f} rad")
                q = q_lm
            else:
                print(f"LM solution too far from current config: "
                      f"{dist_from_current:.4f} rad (max={max_joint_dist}). "
                      f"Using Jacobian result.")
        else:
            print("LM polish failed to converge. Using Jacobian result.")

    total_dq = q - q_current
    joint_dist = np.linalg.norm(total_dq)
    print(f"  Joint delta: {np.round(total_dq, 4)}")
    print(f"  Joint delta norm: {joint_dist:.4f} rad ({np.degrees(joint_dist):.1f}°)")

    # Scale trajectory duration so max joint velocity stays within safe limits.
    # Use a conservative limit to avoid trajectory rejection by the
    # impedance controller (which may have tighter limits than the robot's
    # hardware velocity limits of ~2.1 rad/s).
    MAX_JOINT_VEL = 0.5  # rad/s — conservative for impedance control
    max_joint_delta = np.max(np.abs(total_dq))
    min_time = max_joint_delta / MAX_JOINT_VEL
    actual_time = max(time, min_time)
    if actual_time > time:
        print(f"  Scaled move time: {time:.2f}s → {actual_time:.2f}s "
              f"(max joint delta {max_joint_delta:.3f} rad)")

    return goto_joint_angles(rob, q, actual_time)


def goto_hand_position(rob: BambooFrankaClient, X_WG: np.ndarray, time: float,
                      gripper_type: str = "robotiq", n_ik_attempts: int = 5,
                      max_joint_dist: float = 1.0,
                      ik_solver: Literal["jacobian_pinv", "ik_lm"] = "ik_lm") -> int:
    if ik_solver == "jacobian_pinv":
        return goto_hand_position_jacobian(rob, X_WG, time, gripper_type)

    s_current = rob.get_joint_states()
    q_current = np.array(s_current["qpos"])
    best_soln = None
    best_dist = float("inf")
    for ik_attempt in range(n_ik_attempts):
        if ik_attempt == 0:
            q0 = q_current
        else:
            q0 = q_current.copy()
            q0[:7] += np.random.randn(7) * 0.1
        ik_soln = rtb_IK(X_WG, q0, gripper_type=gripper_type)
        if ik_soln is not None:
            dist = np.linalg.norm(ik_soln - q_current[:7])
            if dist < best_dist:
                best_dist = dist
                best_soln = ik_soln
            if dist < 0.3:
                break

    if best_soln is not None and best_dist <= max_joint_dist:
        print(f"IK solved (joint_dist={best_dist:.4f} rad)")
        return goto_joint_angles(rob, best_soln, time)
    elif best_soln is not None:
        raise RuntimeError(
            f"IK solution too far from current config: joint_dist={best_dist:.2f} rad "
            f"(max={max_joint_dist}). Skipping to avoid drastic motion."
        )
    else:
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='move robot to home configuration')
    parser.add_argument('--server-ip', type=str, default='128.30.224.88',
                        help='Robot IP address')
    args = parser.parse_args()
    q_neutral = np.array([-0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, -0.14])
    with BambooFrankaClient(server_ip=args.server_ip) as rob:
        goto_joint_angles(rob, q_neutral, 5)