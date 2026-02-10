#!/usr/bin/env python3
"""
Diagnose why a grasp pose fails IK.
"""

import numpy as np
import argparse
from pathlib import Path
from bamboo.client import BambooFrankaClient
import roboticstoolbox as rtb
from spatialmath import SE3

# Load robot model
URDF_PATH = Path(__file__).parent / "fr3_robotiq_2f_85.urdf"
robot_model = rtb.Robot.URDF(str(URDF_PATH))


def test_ik_at_position(position, q_current):
    """Test if position is reachable with a simple top-down orientation."""
    # Top-down grasp orientation (common for grasping)
    R_top_down = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])

    T_test = SE3.Rt(R_top_down, position)
    solution = robot_model.ik_LM(
        T_test,
        q0=q_current[:7],
        end="panda_link8",
        mask=[1, 1, 1, 1, 1, 1]
    )
    return solution[1], solution[0]  # success, joints


def main():
    parser = argparse.ArgumentParser(description="Diagnose grasp IK failure")
    parser.add_argument('--grasp-file', type=str, required=True)
    parser.add_argument('--server-ip', type=str, required=True)
    parser.add_argument('--extrinsics', type=str, required=True)
    args = parser.parse_args()

    # Load grasp and transform to world frame
    data = np.load(args.grasp_file)
    T_camera_grasp = data['pose']
    grasp_score = data['score']

    X_WE = np.load(args.extrinsics)
    T_world_grasp = X_WE @ T_camera_grasp

    grasp_position = T_world_grasp[:3, 3]
    grasp_rotation = T_world_grasp[:3, :3]

    print("="*70)
    print("GRASP POSE DIAGNOSIS")
    print("="*70)
    print(f"\nGrasp score: {grasp_score:.4f}")
    print(f"\nGrasp position (world frame): [{grasp_position[0]:.4f}, {grasp_position[1]:.4f}, {grasp_position[2]:.4f}]")
    print(f"  x: {grasp_position[0]*100:.1f} cm")
    print(f"  y: {grasp_position[1]*100:.1f} cm")
    print(f"  z: {grasp_position[2]*100:.1f} cm ← HEIGHT ABOVE ROBOT BASE")

    print(f"\nGrasp orientation (rotation matrix):")
    print(grasp_rotation)

    # Extract approach direction (typically -z axis of grasp)
    approach_dir = -grasp_rotation[:, 2]
    print(f"\nGrasp approach direction: [{approach_dir[0]:.3f}, {approach_dir[1]:.3f}, {approach_dir[2]:.3f}]")

    # Connect to robot
    print(f"\n{'='*70}")
    print("ROBOT STATE")
    print("="*70)
    with BambooFrankaClient(server_ip=args.server_ip) as robot:
        state = robot.get_joint_states()
        q_current = np.array(state['qpos'])
        ee_pose_current = np.array(state['ee_pose'])
        current_position = ee_pose_current[:3, 3]

        print(f"\nCurrent robot end-effector position:")
        print(f"  x: {current_position[0]*100:.1f} cm")
        print(f"  y: {current_position[1]*100:.1f} cm")
        print(f"  z: {current_position[2]*100:.1f} cm")

        print(f"\nCurrent joint angles:")
        print(f"  {[f'{q:.3f}' for q in q_current]}")

        # Test IK feasibility
        print(f"\n{'='*70}")
        print("IK FEASIBILITY TESTS")
        print("="*70)

        # Test 1: Can we reach the position with top-down orientation?
        print(f"\n1. Testing position reachability (top-down orientation)...")
        success, joints = test_ik_at_position(grasp_position, q_current)
        if success:
            print(f"   ✓ Position IS reachable with simple top-down grasp")
            print(f"   Joint solution: {[f'{q:.3f}' for q in joints]}")
        else:
            print(f"   ✗ Position NOT reachable even with simple orientation")
            print(f"   → Grasp position is outside robot workspace!")

        # Test 2: Try with offsets
        print(f"\n2. Testing with vertical offsets...")
        for offset in [0.05, 0.10, 0.20, 0.30, 0.50]:
            test_pos = grasp_position.copy()
            test_pos[2] += offset
            success, _ = test_ik_at_position(test_pos, q_current)
            status = "✓" if success else "✗"
            print(f"   {status} +{offset*100:.0f}cm offset (z={test_pos[2]*100:.1f}cm): {'REACHABLE' if success else 'unreachable'}")

        # Test 3: Try actual grasp orientation at different heights
        print(f"\n3. Testing actual grasp orientation at different heights...")
        for offset in [0.05, 0.10, 0.20, 0.30, 0.50]:
            T_test = T_world_grasp.copy()
            T_test[2, 3] += offset

            # Clean rotation matrix
            R = T_test[:3, :3].copy()
            U, _, Vt = np.linalg.svd(R)
            R_clean = U @ Vt
            if np.linalg.det(R_clean) < 0:
                Vt[-1, :] *= -1
                R_clean = U @ Vt

            T_target = SE3.Rt(R_clean, T_test[:3, 3])
            solution = robot_model.ik_LM(T_target, q0=q_current[:7], end="panda_link8", mask=[1, 1, 1, 1, 1, 1])
            success = solution[1]
            status = "✓" if success else "✗"
            print(f"   {status} +{offset*100:.0f}cm offset (z={T_test[2,3]*100:.1f}cm): {'REACHABLE' if success else 'unreachable'}")

        print(f"\n{'='*70}")
        print("RECOMMENDATIONS")
        print("="*70)

        if grasp_position[2] < 0.15:
            print("\n⚠ Grasp height is VERY LOW (<15cm above robot base)")
            print("  → This is likely below the robot's reachable workspace")
            print("  → Check camera calibration (X_WE)")
            print("  → Verify object is on a table/surface, not on ground")

        if grasp_position[2] < current_position[2] - 0.20:
            print("\n⚠ Grasp is significantly below current robot height")
            print("  → Robot may not be able to reach down that far")

        # Check if position is too far
        distance = np.linalg.norm(grasp_position[:2])  # xy distance from base
        if distance > 0.8:
            print(f"\n⚠ Grasp is {distance*100:.1f}cm away from robot base (xy distance)")
            print("  → This is near/beyond maximum reach")

        if distance < 0.1:
            print(f"\n⚠ Grasp is only {distance*100:.1f}cm from robot base")
            print("  → This is too close, robot cannot reach")


if __name__ == "__main__":
    main()
