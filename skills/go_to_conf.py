import logging
import numpy as np

from bamboo_client import BambooFrankaClient
from frantik import cc_ik


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


def goto_hand_position(rob: BambooFrankaClient, X_WG: np.ndarray, time: float) -> None:
    q_current = rob.get_joint_positions()
    q_next = cc_ik(
        X_WG,     # 4x4 numpy array, base -> flange
        q_current[6],   # redundancy resolution parameter
        q_current     # current joint configuration (7,)
    )
    goto_joint_angles(rob, q_next, time)


if __name__ == "__main__":
    q_neutral = np.array([-0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])
    goto_joint_angles(q_neutral, 5)
