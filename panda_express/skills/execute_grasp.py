#!/usr/bin/env python3
"""
Execute grasp from ContactGraspNet prediction.

Usage:
    python execute_grasp.py --grasp-file <path> --server-ip <ip> [--extrinsics <path>]
"""

import numpy as np
import argparse
import time
from pathlib import Path
from bamboo.client import BambooFrankaClient
from go_to_conf import goto_hand_position, list_available_grippers, contactgraspnet_to_panda


def main():
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="Execute grasp from file")
    parser.add_argument('--grasp-file', type=str, required=True,
                        help='Path to .npz grasp file')
    parser.add_argument('--server-ip', type=str, required=True,
                        help='Robot server IP address')
    parser.add_argument('--extrinsics', type=str,
                        default='perception/calibrate/X_WE.npy',
                        help='Path to X_WE.npy extrinsics file')
    parser.add_argument('--offset', type=float, default=0.05,
                        help='Vertical offset above grasp in meters (default: 0.05)')
    parser.add_argument('--move-time', type=float, default=5.0,
                        help='Movement duration in seconds (default: 5.0)')
    parser.add_argument('--gripper-type', type=str, default='franka',
                        choices=['robotiq', 'franka'],
                        help='Gripper type (default: franka)')
    parser.add_argument('--list-grippers', action='store_true',
                        help='List available gripper types and exit')
    args = parser.parse_args()

    # Handle --list-grippers
    if args.list_grippers:
        print("Available gripper types:")
        for gripper in list_available_grippers():
            print(f"  - {gripper}")
        return 0

    # 2. Load grasp pose from .npz file (4x4 matrix in camera frame)
    print(f"Loading grasp from {args.grasp_file}...")
    data = np.load(args.grasp_file)
    T_camera_grasp = data['pose']  # 4x4 matrix
    grasp_score = data['score']
    print(f"Loaded grasp with score: {grasp_score:.4f}")
    print(f"Grasp pose (camera frame):\n{T_camera_grasp}")

    # 3. Load extrinsics (X_WE: external camera to world transform)
    extrinsics_path = Path(__file__).parent.parent / args.extrinsics
    print(f"\nLoading extrinsics from {extrinsics_path}...")
    X_WE = np.load(extrinsics_path)
    print(f"Camera pose (world frame):\n{X_WE}")

    # 4. Transform grasp from camera frame to world frame
    T_world_grasp = X_WE @ T_camera_grasp

    # 4.5. Apply ContactGraspNet to Panda frame convention conversion
    T_world_grasp = contactgraspnet_to_panda(T_world_grasp)
    print(f"\nGrasp pose (world frame, Panda convention):\n{T_world_grasp}")

    # 5. Apply offset along world vertical axis (z-up)
    T_world_pregrasp = T_world_grasp.copy()
    T_world_pregrasp[2, 3] += args.offset  # Add offset to z-coordinate
    print(f"\nPre-grasp pose ({args.offset*100:.1f}cm above, world frame):")
    print(T_world_pregrasp)

    # 6. Connect to robot
    print(f"\nConnecting to robot at {args.server_ip}...")
    print(f"Using gripper type: {args.gripper_type}")
    with BambooFrankaClient(server_ip=args.server_ip, gripper_type="franka") as robot:
        # 7. Execute motion to pre-grasp pose
        print(f"Moving to pre-grasp pose over {args.move_time} seconds...")
        result = goto_hand_position(robot, T_world_pregrasp, args.move_time,
                                   gripper_type=args.gripper_type)

        if result == 0:
            print("✓ Successfully reached pre-grasp pose!")

            # Get and print final end-effector position
            final_state = robot.get_joint_states()
            final_ee_pose = np.array(final_state['ee_pose'])
            final_position = final_ee_pose[:3, 3]
            print(f"\nFinal end-effector position:")
            print(f"  x: {final_position[0]*100:.2f} cm")
            print(f"  y: {final_position[1]*100:.2f} cm")
            print(f"  z: {final_position[2]*100:.2f} cm")
            print(f"\nFinal end-effector orientation (rotation matrix):")
            print(final_ee_pose[:3, :3])

            # 8. Calculate grasp pose (8cm below pre-grasp)
            T_world_grasp_final = T_world_pregrasp.copy()
            T_world_grasp_final[2, 3] -= 0.08  # Move down 8cm
            print(f"\nTarget grasp pose (8.0cm below pre-grasp):")
            print(T_world_grasp_final)

            # 9. Move to grasp pose
            print(f"\nMoving to grasp pose over 5.0 seconds...")
            result = goto_hand_position(robot, T_world_grasp_final, 5.0,
                                       gripper_type=args.gripper_type)

            if result != 0:
                print("✗ Failed to reach grasp pose")
                return 1

            print("✓ Successfully reached grasp pose!")

            # Get and print final grasp position
            final_grasp_state = robot.get_joint_states()
            final_grasp_ee_pose = np.array(final_grasp_state['ee_pose'])
            final_grasp_position = final_grasp_ee_pose[:3, 3]
            print(f"\nFinal grasp position:")
            print(f"  x: {final_grasp_position[0]*100:.2f} cm")
            print(f"  y: {final_grasp_position[1]*100:.2f} cm")
            print(f"  z: {final_grasp_position[2]*100:.2f} cm")
            print(f"\nFinal grasp orientation (rotation matrix):")
            print(final_grasp_ee_pose[:3, :3])

            # 10. Stabilization delay
            print("\nStabilizing robot position...")
            time.sleep(0.5)

            # 11. Close gripper
            print("Closing gripper...")
            gripper_result = robot.close_gripper()
            print(f"Gripper close result: {gripper_result}")

            # Get final gripper state
            final_gripper_state = robot.get_gripper_state()
            print(f"Final gripper state: {final_gripper_state}")

            print("\n✓ Grasp execution complete!")

            # 12. Move 30cm upward (lift)
            T_world_lift = T_world_grasp_final.copy()
            T_world_lift[2, 3] += 0.30  # Move up 30cm
            print(f"\nTarget lift pose (30.0cm above grasp):")
            print(T_world_lift)

            print(f"\nMoving upward 30cm over 5.0 seconds...")
            result = goto_hand_position(robot, T_world_lift, 5.0,
                                       gripper_type=args.gripper_type)

            if result != 0:
                print("✗ Failed to reach lift pose")
                return 1

            print("✓ Successfully reached lift pose!")

            # Get and print final lift position
            final_lift_state = robot.get_joint_states()
            final_lift_ee_pose = np.array(final_lift_state['ee_pose'])
            final_lift_position = final_lift_ee_pose[:3, 3]
            print(f"\nFinal lift position:")
            print(f"  x: {final_lift_position[0]*100:.2f} cm")
            print(f"  y: {final_lift_position[1]*100:.2f} cm")
            print(f"  z: {final_lift_position[2]*100:.2f} cm")

            print("\n✓ Full grasp and lift sequence complete!")
        else:
            print("✗ Failed to reach pre-grasp pose")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
