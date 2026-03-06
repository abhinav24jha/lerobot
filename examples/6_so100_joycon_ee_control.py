#!/usr/bin/env python3
"""
Single-arm Joy-Con teleoperation for SO100/SO101 follower arms.

This keeps the control behavior from the working local script while adding:
- `--left` / `--right` arm selection
- `--port` CLI override
- automatic calibration-file wiring to left/right follower IDs
"""

import argparse
import logging
import math
import time
import traceback
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
    parser = argparse.ArgumentParser(description="Single-arm SO100/SO101 Joy-Con teleop")
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument("--left", action="store_true", help="Use left arm calibration and left Joy-Con")
    arm_group.add_argument("--right", action="store_true", help="Use right arm calibration and right Joy-Con")
    parser.add_argument("--port", type=str, default=None, help="USB serial port for the selected arm")
    parser.add_argument(
        "--robot-type",
        choices=["auto", "so100", "so101"],
        default="auto",
        help="Follower robot type. Default auto-selects from the calibration files.",
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


def move_to_zero_position(robot, duration=3.0, kp=0.5):
    print("Using P control to slowly move robot to zero position...")
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
        f"Will use P control to move to zero position in {duration} seconds, "
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
            print(f"Moving to zero position progress: {progress:.1f}%")

        time.sleep(step_time)

    print("Robot has moved to zero position")


def return_to_start_position(robot, start_positions, kp=0.5, control_freq=50):
    print("Returning to start position...")
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
            print("Returned to start position")
            break

        time.sleep(control_period)

    print("Return to start position completed")


def resolve_runtime_config(args):
    arm_side = "left" if args.left else "right"
    robot_id = ARM_TO_DEFAULT_ID[arm_side]

    so101_calib = CALIBRATION_ROOT / "so101_follower" / f"{robot_id}.json"
    so100_calib = CALIBRATION_ROOT / "so100_follower" / f"{robot_id}.json"

    if args.robot_type == "auto":
        if so101_calib.exists():
            robot_type = "so101"
        elif so100_calib.exists():
            robot_type = "so100"
        else:
            robot_type = "so101"
    else:
        robot_type = args.robot_type

    calibration_dir = CALIBRATION_ROOT / f"{robot_type}_follower"
    calibration_file = calibration_dir / f"{robot_id}.json"
    return arm_side, robot_id, robot_type, calibration_dir, calibration_file


def make_robot(robot_type, port, robot_id, calibration_dir):
    if robot_type == "so101":
        from lerobot.robots.so101_follower import SO101Follower as FollowerRobot
        from lerobot.robots.so101_follower import SO101FollowerConfig as FollowerRobotConfig
    else:
        from lerobot.robots.so100_follower import SO100Follower as FollowerRobot
        from lerobot.robots.so100_follower import SO100FollowerConfig as FollowerRobotConfig

    robot_config = FollowerRobotConfig(port=port, id=robot_id, calibration_dir=calibration_dir)
    return FollowerRobot(robot_config)


def maybe_calibrate_robot(robot):
    if not robot.is_calibrated:
        print("No valid calibration found for this robot. Calibration is required.")
        robot.calibrate()
        print("Calibration completed!")
        return

    while True:
        calibrate_choice = input("Do you want to recalibrate the robot? (y/n): ").strip().lower()
        if calibrate_choice in ["y", "yes"]:
            print("Starting recalibration...")
            robot.calibrate()
            print("Calibration completed!")
            return
        if calibrate_choice in ["n", "no"]:
            print("Using existing calibration")
            return
        print("Please enter y or n")


def p_control_loop(robot, target_positions, start_positions, joycon_controller, debug_controls=False, kp=0.5, control_freq=50):
    control_period = 1.0 / control_freq
    last_control_button = 0
    last_debug_dump_time = 0.0

    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")

    while True:
        try:
            pose, gripper, control_button = joycon_controller.get_control()
            if control_button == 9 and last_control_button != 9:
                print("Joy-Con Home/Capture pressed: returning robot to startup pose...")
                return_to_start_position(robot, start_positions, kp=kp, control_freq=control_freq)
            last_control_button = control_button

            x, y, z, roll_, pitch_, _ = pose
            pitch = -pitch_ * 60 + 20
            current_x = 0.1629 + x
            current_y = 0.1131 + z
            roll = roll_ * 50

            target_positions["shoulder_pan"] = y * 300.0
            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
            target_positions["shoulder_lift"] = joint2_target
            target_positions["elbow_flex"] = joint3_target
            target_positions["wrist_flex"] = -joint2_target - joint3_target + pitch
            target_positions["wrist_roll"] = roll
            target_positions["gripper"] = 60 if gripper == 1 else 0

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

            if robot_action:
                robot.send_action(robot_action)

            if debug_controls and (time.time() - last_debug_dump_time) > 0.4:
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
                    f"[joycon-debug] dev={device} stick_v={stick_v} stick_h={stick_h} "
                    f"up={up_btn} stick_press={down_btn} grip={grip_btn} home/capture={home_btn}"
                )
                last_debug_dump_time = time.time()

            time.sleep(control_period)

        except KeyboardInterrupt:
            print("User interrupted program")
            return_to_start_position(robot, start_positions, kp=kp, control_freq=control_freq)
            break
        except Exception as e:
            print(f"P control loop error: {e}")
            traceback.print_exc()
            break


def main():
    print("LeRobot Joy-Con Control Example (P Control)")
    print("=" * 50)

    robot = None
    joycon_controller = None
    try:
        args = parse_args()
        arm_side, robot_id, robot_type, calibration_dir, calibration_file = resolve_runtime_config(args)

        if args.port:
            port = args.port.strip()
            print(f"Using CLI port: {port}")
        else:
            port = input("Please enter the USB port for follower robot (e.g., /dev/ttyACM0): ").strip()

        if not port:
            port = "/dev/ttyACM0"
            print(f"Using default port: {port}")
        else:
            print(f"Connecting to port: {port}")

        print(f"Arm side: {arm_side}")
        print(f"Robot type: {robot_type}")
        print(f"Calibration id: {robot_id}")
        print(f"Calibration directory: {calibration_dir}")
        print(f"Calibration file: {calibration_file} ({'found' if calibration_file.exists() else 'missing'})")

        robot = make_robot(robot_type, port, robot_id, calibration_dir)
        robot.connect(calibrate=False)

        joycon_controller = FixedAxesJoyconRobotics(arm_side, dof_speed=[2, 2, 2, 1, 1, 1])

        print("Joy-Con control mapping:")
        print(f"- Active Joy-Con side: {arm_side}")
        print("- Stick vertical: coupled X/Z motion")
        print("- Stick horizontal: shoulder_pan")
        print("- L/R: move up")
        print("- Stick press: move down")
        print("- D-pad Up/Down on left or X/B on right: fine X nudges")
        print("- ZL/ZR: toggle gripper open/close")
        print("- Capture/Home: reset Joy-Con pose and return robot to startup pose")
        print("- Ctrl+C: stop")
        print()

        print("Device connection successful!")
        maybe_calibrate_robot(robot)

        print("Reading initial joint angles...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith(".pos"):
                start_positions[key.removesuffix(".pos")] = float(value)

        print("Initial joint angles:")
        for joint_name, position in start_positions.items():
            print(f"  {joint_name}: {position:.1f}°")

        move_to_zero_position(robot, duration=3.0)

        target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }

        p_control_loop(
            robot,
            target_positions,
            start_positions,
            joycon_controller,
            debug_controls=args.debug_controls,
            kp=0.5,
            control_freq=50,
        )

    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Whether the robot is properly connected")
        print("2. Whether the USB port is correct")
        print("3. Whether you have sufficient permissions to access USB devices")
        print("4. Whether the robot is properly configured")
    finally:
        if joycon_controller is not None:
            joycon_controller.disconnect()
        if robot is not None and robot.is_connected:
            robot.disconnect()
        print("Program ended")


if __name__ == "__main__":
    main()
