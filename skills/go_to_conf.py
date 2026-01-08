import logging
import numpy as np
from pathlib import Path
import roboticstoolbox as rtb
from spatialmath import SE3

from bamboo.client import BambooFrankaClient

# Load the robot model from URDF
URDF_PATH = Path(__file__).parent / "fr3_robotiq_2f_85.urdf"
robot_model = rtb.Robot.URDF(str(URDF_PATH))


def goto_joint_angles(robot: BambooFrankaClient, q: np.ndarray, time: float) -> int:
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("Creating BambooFrankaClient...")

    try:
        # Get current joint angles
        print("\nGetting current joint angles...")
        current_joints = robot.get_joint_positions()
        print(f"Current joint angles: {[f'{q:.4f}' for q in current_joints]}")

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
        print("\nGetting final joint angles...")
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


def goto_hand_position(rob: BambooFrankaClient, X_WG: np.ndarray, time: float) -> int:

    s_current = rob.get_joint_states()
    q_current = np.array(s_current["qpos"])
    X_current = s_current["ee_pose"]
    print(f"{X_current=}")

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
    solution = robot_model.ik_LM(
        T_target,
        q0=q_current[:7],  # Use current arm joint positions as initial guess
        end="panda_link7",  # Target the flange frame (robot mounting flange)
        mask=[1, 1, 1, 1, 1, 1]  # Full 6-DOF constraint (x, y, z, roll, pitch, yaw)
    )

    if solution[1]:
        q_next = solution[0]
        print(f"{q_next=}")
        return goto_joint_angles(rob, q_next, time)
    else:
        print(f"IK solution failed: {solution}")
        raise RuntimeError("Failed to find IK solution")


if __name__ == "__main__":
    q_neutral = np.array([-0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, -0.14])
    with BambooFrankaClient(server_ip="128.30.224.88") as rob:
        goto_joint_angles(rob, q_neutral, 5)
