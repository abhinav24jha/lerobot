#!/usr/bin/env python

"""Detect and test a single motor connected to a servo board.

This script is intended for a quick hardware sanity check:
1) Scan a serial port to find connected motor IDs.
2) Pick one motor (auto if only one is found).
3) Move it back-and-forth and verify that position feedback changes.

Examples:
    python examples/test_single_motor_servo.py \
        --port /dev/tty.usbmodem575E0031751 \
        --bus feetech

    python examples/test_single_motor_servo.py \
        --port /dev/ttyUSB0 \
        --bus dynamixel \
        --motor-id 1 \
        --motor-model xl330-m077
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from lerobot.motors.dynamixel.dynamixel import DynamixelMotorsBus
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode


@dataclass
class DetectedMotor:
    baudrate: int
    motor_id: int
    model_number: int


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and test one motor on a servo board.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0 or /dev/tty.usbmodem...")
    parser.add_argument(
        "--bus",
        choices=["feetech", "dynamixel"],
        default="feetech",
        help="Motor protocol family used by your board.",
    )
    parser.add_argument(
        "--motor-id",
        type=int,
        default=None,
        help="Motor ID to test. If omitted, auto-select when exactly one motor is detected.",
    )
    parser.add_argument(
        "--motor-model",
        type=str,
        default=None,
        help="Motor model string (e.g. sts3215, xl330-m077). Optional if the model number can be inferred.",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=None,
        help="Known baudrate. If omitted, the script scans all supported baudrates.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=40,
        help="Motion step in encoder ticks for each direction (default is intentionally conservative).",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.8,
        help="Delay after each move command.",
    )
    parser.add_argument(
        "--read-threshold",
        type=int,
        default=10,
        help="Minimum position change (ticks) to consider a move successful.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass safety checks (operating mode and step-size cap). Use only if you know your setup.",
    )
    return parser


def _pick_bus_class(bus_name: str):
    if bus_name == "feetech":
        return FeetechMotorsBus
    if bus_name == "dynamixel":
        return DynamixelMotorsBus
    raise ValueError(f"Unknown bus type: {bus_name}")


def _reverse_model_number_table(model_number_table: dict[str, int]) -> dict[int, str]:
    reverse = {}
    for model_name, model_number in model_number_table.items():
        reverse.setdefault(model_number, model_name)
    return reverse


def _scan_motors(bus_class, port: str, baudrate: int | None) -> list[DetectedMotor]:
    scanner = bus_class(port=port, motors={})
    scanner.connect(handshake=False)
    detections: list[DetectedMotor] = []

    try:
        baudrates = [baudrate] if baudrate is not None else list(scanner.available_baudrates)
        for rate in baudrates:
            scanner.set_baudrate(rate)
            ids_to_models = scanner.broadcast_ping() or {}
            for motor_id, model_number in ids_to_models.items():
                detections.append(DetectedMotor(baudrate=rate, motor_id=motor_id, model_number=model_number))
    finally:
        scanner.disconnect(disable_torque=False)

    return detections


def _select_detection(detections: list[DetectedMotor], requested_id: int | None) -> DetectedMotor:
    if not detections:
        raise RuntimeError("No motors detected on this port. Check cable/power and try a specific --baudrate.")

    if requested_id is not None:
        for detection in detections:
            if detection.motor_id == requested_id:
                return detection
        raise RuntimeError(f"Motor ID {requested_id} not found on this port.")

    unique_ids = sorted({d.motor_id for d in detections})
    if len(unique_ids) != 1:
        raise RuntimeError(
            "Multiple motor IDs detected; specify which one to test with --motor-id. "
            f"Detected IDs: {unique_ids}"
        )

    selected_id = unique_ids[0]
    for detection in detections:
        if detection.motor_id == selected_id:
            return detection

    raise RuntimeError("Unexpected selection failure while choosing detected motor.")


def _infer_motor_model(bus_class, selected: DetectedMotor, model_arg: str | None) -> str:
    if model_arg is not None:
        return model_arg

    reverse_table = _reverse_model_number_table(bus_class.model_number_table)
    inferred = reverse_table.get(selected.model_number)
    if inferred is None:
        raise RuntimeError(
            "Could not infer motor model from model number. "
            f"Pass it explicitly with --motor-model (model number={selected.model_number})."
        )

    return inferred


def _assert_safe_step(step: int, force: bool) -> None:
    max_safe_step = 120
    if not force and step > max_safe_step:
        raise RuntimeError(
            f"--step={step} is too large for safe default testing. Use --step <= {max_safe_step} "
            "or pass --force to override."
        )


def _ensure_position_mode(bus, motor_name: str, bus_name: str, force: bool) -> None:
    try:
        operating_mode = int(bus.read("Operating_Mode", motor_name, normalize=False))
    except Exception:
        # Some models may not expose a readable operating mode; keep going unless strict checking is needed.
        return

    if bus_name == "feetech":
        # Feetech position mode enum value.
        safe_modes = {0}
    else:
        # Dynamixel position(3) and extended-position(4).
        safe_modes = {3, 4}

    if operating_mode not in safe_modes:
        msg = (
            f"Motor is in Operating_Mode={operating_mode}, expected one of {sorted(safe_modes)} "
            "for safe position testing."
        )
        if force:
            print(f"WARNING: {msg} Continuing because --force is set.")
        else:
            raise RuntimeError(msg + " Reconfigure the motor mode first, or re-run with --force.")


def _apply_safe_speed_profile(bus, motor_name: str, bus_name: str) -> None:
    if bus_name == "feetech":
        # Use the profile/time fields available on many Feetech models to avoid abrupt motion.
        for reg_name, value in (("Acceleration", 15), ("Goal_Time", 600), ("Running_Time", 600)):
            try:
                bus.write(reg_name, motor_name, value, normalize=False)
            except Exception:
                continue
    else:
        # Dynamixel X-series style profile limits (RAM area) for gentle motion.
        for reg_name, value in (("Profile_Acceleration", 8), ("Profile_Velocity", 20)):
            try:
                bus.write(reg_name, motor_name, value, normalize=False)
            except Exception:
                continue


def _run_motion_test(args: argparse.Namespace, selected: DetectedMotor, motor_model: str) -> None:
    bus_class = _pick_bus_class(args.bus)
    motor_name = "test_motor"
    bus = bus_class(
        port=args.port,
        motors={motor_name: Motor(id=selected.motor_id, model=motor_model, norm_mode=MotorNormMode.DEGREES)},
    )

    print(
        "\nTesting motor "
        f"id={selected.motor_id}, model={motor_model}, model_number={selected.model_number}, baudrate={selected.baudrate}"
    )

    bus.connect(handshake=False)
    moved_ok = False

    try:
        _assert_safe_step(args.step, args.force)
        bus.set_baudrate(selected.baudrate)
        _ensure_position_mode(bus, motor_name, args.bus, args.force)
        _apply_safe_speed_profile(bus, motor_name, args.bus)
        present = int(bus.read("Present_Position", motor_name, normalize=False))
        print(f"Start position: {present}")

        max_pos = bus.model_resolution_table[motor_model] - 1
        min_pos = 0

        # Avoid false failures when starting near hard limits.
        if present >= max_pos - args.step:
            first_goal = max(min_pos, present - args.step)
            second_goal = min(max_pos, first_goal + args.step)
        elif present <= min_pos + args.step:
            first_goal = min(max_pos, present + args.step)
            second_goal = max(min_pos, first_goal - args.step)
        else:
            first_goal = min(max_pos, present + args.step)
            second_goal = max(min_pos, present - args.step)

        bus.enable_torque(motor_name)

        bus.write("Goal_Position", motor_name, first_goal, normalize=False)
        time.sleep(args.hold_seconds)
        pos_after_first = int(bus.read("Present_Position", motor_name, normalize=False))
        print(f"After move 1: goal={first_goal}, read={pos_after_first}")

        bus.write("Goal_Position", motor_name, second_goal, normalize=False)
        time.sleep(args.hold_seconds)
        pos_after_second = int(bus.read("Present_Position", motor_name, normalize=False))
        print(f"After move 2: goal={second_goal}, read={pos_after_second}")

        bus.write("Goal_Position", motor_name, present, normalize=False)
        time.sleep(args.hold_seconds)

        delta_1 = abs(pos_after_first - present)
        delta_2 = abs(pos_after_second - pos_after_first)
        moved_ok = delta_1 >= args.read_threshold or delta_2 >= args.read_threshold

        print(
            "\nResult: "
            + (
                "PASS - motor responded to move command(s)."
                if moved_ok
                else "FAIL - position feedback barely changed. Motor/gear/power may be faulty."
            )
        )
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=True)

    if not moved_ok:
        raise RuntimeError(
            "Motor response check failed. Verify power supply, wiring, horn obstruction, and correct model/baudrate."
        )


def main() -> None:
    args = _build_arg_parser().parse_args()
    bus_class = _pick_bus_class(args.bus)

    print(f"Scanning port={args.port} on bus={args.bus}...")
    detections = _scan_motors(bus_class, args.port, args.baudrate)

    if detections:
        print("Detected motors (possibly repeated across baudrates):")
        for detection in detections:
            print(
                f"  - baudrate={detection.baudrate}, id={detection.motor_id}, model_number={detection.model_number}"
            )

    selected = _select_detection(detections, args.motor_id)
    motor_model = _infer_motor_model(bus_class, selected, args.motor_model)

    _run_motion_test(args, selected, motor_model)


if __name__ == "__main__":
    main()
