#!/usr/bin/env python3
"""
Dual-arm Joy-Con teleoperation for two SO100/SO101 follower arms.

This is the current single-arm Joy-Con file duplicated for left and right arms:
- left Joy-Con controls the left follower arm
- right Joy-Con controls the right follower arm
"""

import argparse
import logging
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

try:
    from joycon_hid_compat import ensure_hid_backend, patch_joycon_open
except ModuleNotFoundError:
    from examples.joycon_hid_compat import ensure_hid_backend, patch_joycon_open

ensure_hid_backend()
patch_joycon_open()
from joyconrobotics import JoyconRobotics


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CALIBRATION_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots"
ARM_TO_DEFAULT_ID = {
    "left": "left_follower_arm",
    "right": "right_follower_arm",
}

JOINT_CALIBRATION = [
    ["shoulder_pan", 6.0, 1.0],
    ["shoulder_lift", 2.0, 0.97],
    ["elbow_flex", 0.0, 1.05],
    ["wrist_flex", 0.0, 0.94],
    ["wrist_roll", 0.0, 0.5],
    ["gripper", 0.0, 1.0],
]


@dataclass
class ArmRuntimeConfig:
    label: str
    arm_side: str
    port: str
    robot_id: str
    robot_type: str
    calibration_dir: Path
    calibration_file: Path


@dataclass
class ArmControlContext:
    runtime: ArmRuntimeConfig
    robot: object
    joycon_controller: JoyconRobotics
    start_positions: dict[str, float]
    target_positions: dict[str, float]
    last_control_button: int = 0


class FixedAxesJoyconRobotics(JoyconRobotics):
    def common_update(self):
        speed_scale = 0.0008

        joycon_stick_v = (
            self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        )
        joycon_stick_v_0 = 1800
        joycon_stick_v_threshold = 300
        joycon_stick_v_range = 1000
        if joycon_stick_v > joycon_stick_v_threshold + joycon_stick_v_0:
            delta = speed_scale * (joycon_stick_v - joycon_stick_v_0) / joycon_stick_v_range
            self.position[0] += delta * self.dof_speed[0] * self.direction_reverse[0] * self.direction_vector[0]
            self.position[2] += delta * self.dof_speed[1] * self.direction_reverse[1] * self.direction_vector[2]
        elif joycon_stick_v < joycon_stick_v_0 - joycon_stick_v_threshold:
            delta = speed_scale * (joycon_stick_v - joycon_stick_v_0) / joycon_stick_v_range
            self.position[0] += delta * self.dof_speed[0] * self.direction_reverse[0] * self.direction_vector[0]
            self.position[2] += delta * self.dof_speed[1] * self.direction_reverse[1] * self.direction_vector[2]

        joycon_stick_h = (
            self.joycon.get_stick_right_horizontal()
            if self.joycon.is_right()
            else self.joycon.get_stick_left_horizontal()
        )
        joycon_stick_h_0 = 2000
        joycon_stick_h_threshold = 300
        joycon_stick_h_range = 1000
        if joycon_stick_h > joycon_stick_h_threshold + joycon_stick_h_0:
            delta = speed_scale * (joycon_stick_h - joycon_stick_h_0) / joycon_stick_h_range
            self.position[1] += delta * self.dof_speed[1] * self.direction_reverse[1]
        elif joycon_stick_h < joycon_stick_h_0 - joycon_stick_h_threshold:
            delta = speed_scale * (joycon_stick_h - joycon_stick_h_0) / joycon_stick_h_range
            self.position[1] += delta * self.dof_speed[1] * self.direction_reverse[1]

        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        if joycon_button_up == 1:
            self.position[2] += speed_scale * self.dof_speed[2] * self.direction_reverse[2]

        joycon_button_down = (
            self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        )
        if joycon_button_down == 1:
            self.position[2] -= speed_scale * self.dof_speed[2] * self.direction_reverse[2]

        joycon_button_xup = self.joycon.get_button_x() if self.joycon.is_right() else self.joycon.get_button_up()
        joycon_button_xback = self.joycon.get_button_b() if self.joycon.is_right() else self.joycon.get_button_down()
        if joycon_button_xup == 1:
            self.position[0] += 0.001 * self.dof_speed[0]
        elif joycon_button_xback == 1:
            self.position[0] -= 0.001 * self.dof_speed[0]

        joycon_button_home = (
            self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        )
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()
            self.return_home_button = 1
        else:
            self.return_home_button = 0

        for event_type, status in self.button.events():
            if (self.joycon.is_right() and event_type == "plus" and status == 1) or (
                self.joycon.is_left() and event_type == "minus" and status == 1
            ):
                self.reset_button = 1
                self.reset_joycon()
            elif self.joycon.is_right() and event_type == "a":
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == "y":
                self.restart_episode_button = status
            elif (
                (self.joycon.is_right() and event_type == "zr")
                or (self.joycon.is_left() and event_type == "zl")
            ) and not self.change_down_to_gripper:
                self.gripper_toggle_button = status
            elif (
                (self.joycon.is_right() and event_type == "stick_r_btn")
                or (self.joycon.is_left() and event_type == "stick_l_btn")
            ) and self.change_down_to_gripper:
                self.gripper_toggle_button = status
            else:
                self.reset_button = 0

        if self.gripper_toggle_button == 1:
            if self.gripper_state == self.gripper_open:
                self.gripper_state = self.gripper_close
            else:
                self.gripper_state = self.gripper_open
            self.gripper_toggle_button = 0

        if self.return_home_button == 1:
            self.button_control = 9
        elif self.joycon.is_right():
            if self.next_episode_button == 1:
                self.button_control = 1
            elif self.restart_episode_button == 1:
                self.button_control = -1
            elif self.reset_button == 1:
                self.button_control = 8
            else:
                self.button_control = 0
        else:
            self.button_control = 0

        return self.position, self.gripper_state, self.button_control


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-arm SO100/SO101 Joy-Con teleop")
    parser.add_argument("--left-port", type=str, default=None, help="USB serial port for the left arm")
    parser.add_argument("--right-port", type=str, default=None, help="USB serial port for the right arm")
    parser.add_argument(
        "--left-robot-type",
        choices=["auto", "so100", "so101"],
        default="auto",
        help="Left follower robot type. Default auto-selects from the calibration files.",
    )
    parser.add_argument(
        "--right-robot-type",
        choices=["auto", "so100", "so101"],
        default="auto",
        help="Right follower robot type. Default auto-selects from the calibration files.",
    )
    parser.add_argument(
        "--debug-controls",
        action="store_true",
        help="Print raw Joy-Con sticks/buttons periodically for mapping verification.",
    )
    return parser.parse_args()


def apply_joint_calibration(joint_name, raw_position):
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]
            scale = joint_cal[2]
            return (raw_position - offset) * scale
    return raw_position


def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max

    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min

    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta2 = math.pi - math.acos(cos_theta2)

    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    joint2 = max(-0.1, min(3.45, theta1 + theta1_offset))
    joint3 = max(-0.2, min(math.pi, theta2 + theta2_offset))

    joint2_deg = 90 - math.degrees(joint2)
    joint3_deg = math.degrees(joint3) - 90
    return joint2_deg, joint3_deg


def move_to_zero_position(label, robot, duration=3.0, kp=0.5):
    print(f"[{label}] Using P control to slowly move robot to zero position...")
    zero_positions = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    }

    control_freq = 50
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq
    print(
        f"[{label}] Will use P control to move to zero position in {duration} seconds, "
        f"control frequency: {control_freq}Hz, proportional gain: {kp}"
    )

    for step in range(total_steps):
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                current_positions[motor_name] = apply_joint_calibration(motor_name, value)

        robot_action = {}
        for joint_name, target_pos in zero_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                robot_action[f"{joint_name}.pos"] = current_pos + kp * (target_pos - current_pos)

        if robot_action:
            robot.send_action(robot_action)

        if step % (control_freq // 2) == 0:
            progress = (step / total_steps) * 100
            print(f"[{label}] Moving to zero position progress: {progress:.1f}%")

        time.sleep(step_time)

    print(f"[{label}] Robot has moved to zero position")


def return_to_start_position(label, robot, start_positions, kp=0.5, control_freq=50):
    print(f"[{label}] Returning to start position...")
    control_period = 1.0 / control_freq

    for _ in range(int(5.0 * control_freq)):
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith(".pos"):
                current_positions[key.removesuffix(".pos")] = value

        robot_action = {}
        total_error = 0.0
        for joint_name, target_pos in start_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                total_error += abs(error)
                robot_action[f"{joint_name}.pos"] = current_pos + kp * error

        if robot_action:
            robot.send_action(robot_action)

        if total_error < 2.0:
            print(f"[{label}] Returned to start position")
            break

        time.sleep(control_period)

    print(f"[{label}] Return to start position completed")


def resolve_robot_type(robot_id, requested_type):
    so101_calib = CALIBRATION_ROOT / "so101_follower" / f"{robot_id}.json"
    so100_calib = CALIBRATION_ROOT / "so100_follower" / f"{robot_id}.json"

    if requested_type == "auto":
        if so101_calib.exists():
            robot_type = "so101"
        elif so100_calib.exists():
            robot_type = "so100"
        else:
            robot_type = "so101"
    else:
        robot_type = requested_type

    calibration_dir = CALIBRATION_ROOT / f"{robot_type}_follower"
    calibration_file = calibration_dir / f"{robot_id}.json"
    return robot_type, calibration_dir, calibration_file


def resolve_arm_runtime_config(label, arm_side, port, requested_type):
    robot_id = ARM_TO_DEFAULT_ID[arm_side]
    robot_type, calibration_dir, calibration_file = resolve_robot_type(robot_id, requested_type)
    return ArmRuntimeConfig(
        label=label,
        arm_side=arm_side,
        port=port,
        robot_id=robot_id,
        robot_type=robot_type,
        calibration_dir=calibration_dir,
        calibration_file=calibration_file,
    )


def make_robot(runtime):
    if runtime.robot_type == "so101":
        from lerobot.robots.so101_follower import SO101Follower as FollowerRobot
        from lerobot.robots.so101_follower import SO101FollowerConfig as FollowerRobotConfig
    else:
        from lerobot.robots.so100_follower import SO100Follower as FollowerRobot
        from lerobot.robots.so100_follower import SO100FollowerConfig as FollowerRobotConfig

    robot_config = FollowerRobotConfig(
        port=runtime.port,
        id=runtime.robot_id,
        calibration_dir=runtime.calibration_dir,
    )
    return FollowerRobot(robot_config)


def maybe_calibrate_robot(label, robot):
    if not robot.is_calibrated:
        print(f"[{label}] No valid calibration found for this robot. Calibration is required.")
        robot.calibrate()
        print(f"[{label}] Calibration completed!")
        return

    while True:
        calibrate_choice = input(f"[{label}] Do you want to recalibrate the robot? (y/n): ").strip().lower()
        if calibrate_choice in ["y", "yes"]:
            print(f"[{label}] Starting recalibration...")
            robot.calibrate()
            print(f"[{label}] Calibration completed!")
            return
        if calibrate_choice in ["n", "no"]:
            print(f"[{label}] Using existing calibration")
            return
        print(f"[{label}] Please enter y or n")


def build_target_positions(pose, gripper):
    x, y, z, roll_, pitch_, _ = pose
    pitch = -pitch_ * 60 + 20
    current_x = 0.1629 + x
    current_y = 0.1131 + z
    roll = roll_ * 50

    joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
    return {
        "shoulder_pan": y * 300.0,
        "shoulder_lift": joint2_target,
        "elbow_flex": joint3_target,
        "wrist_flex": -joint2_target - joint3_target + pitch,
        "wrist_roll": roll,
        "gripper": 60 if gripper == 1 else 0,
    }


def build_robot_action(robot, target_positions, kp=0.5):
    current_obs = robot.get_observation()
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith(".pos"):
            motor_name = key.removesuffix(".pos")
            current_positions[motor_name] = apply_joint_calibration(motor_name, value)

    robot_action = {}
    for joint_name, target_pos in target_positions.items():
        if joint_name in current_positions:
            current_pos = current_positions[joint_name]
            robot_action[f"{joint_name}.pos"] = current_pos + kp * (target_pos - current_pos)
    return robot_action


def maybe_print_debug(joycon_controller, label, last_debug_dump_time, debug_controls):
    if not debug_controls or (time.time() - last_debug_dump_time) <= 0.4:
        return last_debug_dump_time

    joycon = joycon_controller.joycon
    if joycon.is_right():
        stick_v = joycon.get_stick_right_vertical()
        stick_h = joycon.get_stick_right_horizontal()
        up_btn = joycon.get_button_r()
        down_btn = joycon.get_button_r_stick()
        grip_btn = joycon.get_button_zr()
        home_btn = joycon.get_button_home()
        device = "right"
    else:
        stick_v = joycon.get_stick_left_vertical()
        stick_h = joycon.get_stick_left_horizontal()
        up_btn = joycon.get_button_l()
        down_btn = joycon.get_button_l_stick()
        grip_btn = joycon.get_button_zl()
        home_btn = joycon.get_button_capture()
        device = "left"
    print(
        f"[joycon-debug:{label}] dev={device} stick_v={stick_v} stick_h={stick_h} "
        f"up={up_btn} stick_press={down_btn} grip={grip_btn} home/capture={home_btn}"
    )
    return time.time()


def teleop_loop(left_context, right_context, debug_controls=False, kp=0.5, control_freq=50):
    control_period = 1.0 / control_freq
    last_debug_dump_time = 0.0
    contexts = (left_context, right_context)

    print("Dual-arm Joy-Con control mapping:")
    print("- Purple left Joy-Con controls the LEFT arm")
    print("- Green right Joy-Con controls the RIGHT arm")
    print("- Stick vertical: coupled X/Z motion")
    print("- Stick horizontal: shoulder_pan")
    print("- L/R: move up")
    print("- Stick press: move down")
    print("- Left D-pad Up/Down or Right X/B: fine X nudges")
    print("- ZL/ZR: toggle gripper open/close")
    print("- Capture/Home: reset Joy-Con pose and return that arm to startup pose")
    print("- Ctrl+C: stop")
    print()

    while True:
        try:
            for context in contexts:
                pose, gripper, control_button = context.joycon_controller.get_control()
                if control_button == 9 and context.last_control_button != 9:
                    return_to_start_position(
                        context.runtime.label,
                        context.robot,
                        context.start_positions,
                        kp=kp,
                        control_freq=control_freq,
                    )
                context.last_control_button = control_button

                context.target_positions.update(build_target_positions(pose, gripper))
                robot_action = build_robot_action(context.robot, context.target_positions, kp=kp)
                if robot_action:
                    context.robot.send_action(robot_action)

                last_debug_dump_time = maybe_print_debug(
                    context.joycon_controller,
                    context.runtime.label,
                    last_debug_dump_time,
                    debug_controls,
                )

            time.sleep(control_period)
        except KeyboardInterrupt:
            print("User interrupted program")
            for context in contexts:
                return_to_start_position(
                    context.runtime.label,
                    context.robot,
                    context.start_positions,
                    kp=kp,
                    control_freq=control_freq,
                )
            break
        except Exception as e:
            print(f"P control loop error: {e}")
            traceback.print_exc()
            break


def main():
    print("LeRobot Dual-Arm Joy-Con Control Example (P Control)")
    print("=" * 50)

    left_robot = None
    right_robot = None
    left_joycon = None
    right_joycon = None

    try:
        args = parse_args()

        left_port = args.left_port.strip() if args.left_port else ""
        if not left_port:
            left_port = input("Please enter the USB port for LEFT follower robot (e.g., /dev/ttyACM0): ").strip()
        if not left_port:
            left_port = "/dev/ttyACM0"
            print(f"Using default LEFT port: {left_port}")

        right_port = args.right_port.strip() if args.right_port else ""
        if not right_port:
            right_port = input("Please enter the USB port for RIGHT follower robot (e.g., /dev/ttyACM1): ").strip()
        if not right_port:
            right_port = "/dev/ttyACM1"
            print(f"Using default RIGHT port: {right_port}")

        left_runtime = resolve_arm_runtime_config("LEFT", "left", left_port, args.left_robot_type)
        right_runtime = resolve_arm_runtime_config("RIGHT", "right", right_port, args.right_robot_type)

        for runtime in (left_runtime, right_runtime):
            print(f"{runtime.label} arm side: {runtime.arm_side}")
            print(f"{runtime.label} robot type: {runtime.robot_type}")
            print(f"{runtime.label} calibration id: {runtime.robot_id}")
            print(f"{runtime.label} calibration directory: {runtime.calibration_dir}")
            print(
                f"{runtime.label} calibration file: {runtime.calibration_file} "
                f"({'found' if runtime.calibration_file.exists() else 'missing'})"
            )

        left_robot = make_robot(left_runtime)
        right_robot = make_robot(right_runtime)

        print(f"[{left_runtime.label}] Connecting to port: {left_runtime.port}")
        left_robot.connect(calibrate=False)
        print(f"[{right_runtime.label}] Connecting to port: {right_runtime.port}")
        right_robot.connect(calibrate=False)

        print("Dual-arm device connection successful!")

        maybe_calibrate_robot(left_runtime.label, left_robot)
        maybe_calibrate_robot(right_runtime.label, right_robot)

        print(f"[{left_runtime.label}] Reading initial joint angles...")
        left_start_positions = {
            key.removesuffix(".pos"): float(value)
            for key, value in left_robot.get_observation().items()
            if key.endswith(".pos")
        }
        print(f"[{right_runtime.label}] Reading initial joint angles...")
        right_start_positions = {
            key.removesuffix(".pos"): float(value)
            for key, value in right_robot.get_observation().items()
            if key.endswith(".pos")
        }

        move_to_zero_position(left_runtime.label, left_robot, duration=3.0)
        move_to_zero_position(right_runtime.label, right_robot, duration=3.0)

        left_joycon = FixedAxesJoyconRobotics("left", dof_speed=[2, 2, 2, 1, 1, 1])
        right_joycon = FixedAxesJoyconRobotics("right", dof_speed=[2, 2, 2, 1, 1, 1])

        left_context = ArmControlContext(
            runtime=left_runtime,
            robot=left_robot,
            joycon_controller=left_joycon,
            start_positions=left_start_positions,
            target_positions={
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            },
        )
        right_context = ArmControlContext(
            runtime=right_runtime,
            robot=right_robot,
            joycon_controller=right_joycon,
            start_positions=right_start_positions,
            target_positions={
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            },
        )

        teleop_loop(left_context, right_context, debug_controls=args.debug_controls, kp=0.5, control_freq=50)

    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Whether both robots are properly connected")
        print("2. Whether both USB ports are correct")
        print("3. Whether you have sufficient permissions to access USB devices")
        print("4. Whether both robots are properly configured")
    finally:
        if left_joycon is not None:
            left_joycon.disconnect()
        if right_joycon is not None:
            right_joycon.disconnect()
        if left_robot is not None and left_robot.is_connected:
            left_robot.disconnect()
        if right_robot is not None and right_robot.is_connected:
            right_robot.disconnect()
        print("Program ended")


if __name__ == "__main__":
    main()
