"""Compatibility shim to force the working hid backend on macOS.

`joyconrobotics` imports `hid`, but some environments install two different
providers under the same module name. If the first one fails to load native
libraries, we explicitly load the compiled `hid` extension from `hidapi`.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import time
from pathlib import Path


def _hid_extension_candidates() -> list[Path]:
    candidates: list[Path] = []
    for path_entry in sys.path:
        if not path_entry:
            continue
        path = Path(path_entry)
        if not path.is_dir():
            continue
        candidates.extend(sorted(path.glob("hid.cpython-*.so")))
    return candidates


def ensure_hid_backend() -> None:
    """Ensure `import hid` resolves to a loadable backend."""
    if "hid" in sys.modules:
        return

    try:
        import hid  # noqa: F401

        return
    except Exception as import_error:
        last_error = import_error

    for candidate in _hid_extension_candidates():
        try:
            loader = importlib.machinery.ExtensionFileLoader("hid", str(candidate))
            spec = importlib.util.spec_from_file_location("hid", str(candidate), loader=loader)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules["hid"] = module
            return
        except Exception as load_error:
            last_error = load_error

    raise ImportError(
        "Failed to load a working 'hid' backend. "
        "Ensure `hidapi` is installed in the active Python environment."
    ) from last_error


def patch_joycon_open(max_attempts: int = 40, sleep_s: float = 0.2) -> None:
    """Patch joyconrobotics for flaky/limited HID open behavior on macOS."""
    import hid
    from joyconrobotics import joycon as joycon_mod

    if getattr(joycon_mod.JoyCon, "_mac_open_patch_applied", False):
        return

    def _open_with_retry(self, vendor_id, product_id, serial):
        last_error = None
        for _ in range(max_attempts):
            try:
                if hasattr(hid, "device"):
                    dev = hid.device()
                    try:
                        dev.open(vendor_id, product_id, serial)
                        return dev
                    except Exception as open_exc:  # noqa: BLE001
                        last_error = open_exc
                        # Fallback to open_path using enumerate in case direct open is flaky.
                        for info in hid.enumerate(vendor_id, product_id):
                            candidate_serial = info.get("serial_number")
                            if serial and candidate_serial != serial:
                                continue
                            path = info.get("path")
                            if not path:
                                continue
                            try:
                                dev2 = hid.device()
                                dev2.open_path(path)
                                return dev2
                            except Exception as path_exc:  # noqa: BLE001
                                last_error = path_exc
                elif hasattr(hid, "Device"):
                    return hid.Device(vendor_id, product_id, serial)
                else:
                    raise RuntimeError("Unsupported hid backend")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(sleep_s)

        raise IOError("joycon connect failed") from last_error

    joycon_mod.JoyCon._open = _open_with_retry
    joycon_mod.JoyCon._mac_open_patch_applied = True

    # On macOS, opening the same Joy-Con multiple times often fails.
    # joyconrobotics normally opens three handles per controller
    # (JoyCon + GyroTrackingJoyCon + ButtonEventJoyCon). Patch it to a
    # single-handle init with lightweight local button-event tracking.
    from joyconrobotics import joyconrobotics as jr_mod

    if getattr(jr_mod.JoyconRobotics, "_mac_single_handle_patch_applied", False):
        return

    class _SingleHandleButtonEvents:
        def __init__(self, joycon, track_sticks: bool = True):
            self.joycon = joycon
            self.track_sticks = track_sticks
            self._prev: dict[str, int] = {}

        def _emit_if_changed(self, out: list[tuple[str, int]], name: str, value: int) -> None:
            previous = self._prev.get(name, 0)
            if previous != value:
                self._prev[name] = value
                out.append((name, value))

        def events(self):
            out: list[tuple[str, int]] = []
            j = self.joycon
            if j.is_right():
                if self.track_sticks:
                    self._emit_if_changed(out, "stick_r_btn", j.stick_r_btn)
                self._emit_if_changed(out, "r", j.r)
                self._emit_if_changed(out, "zr", j.zr)
                self._emit_if_changed(out, "plus", j.plus)
                self._emit_if_changed(out, "a", j.a)
                self._emit_if_changed(out, "b", j.b)
                self._emit_if_changed(out, "x", j.x)
                self._emit_if_changed(out, "y", j.y)
                self._emit_if_changed(out, "home", j.home)
                self._emit_if_changed(out, "right_sr", j.right_sr)
                self._emit_if_changed(out, "right_sl", j.right_sl)
            else:
                if self.track_sticks:
                    self._emit_if_changed(out, "stick_l_btn", j.stick_l_btn)
                self._emit_if_changed(out, "l", j.l)
                self._emit_if_changed(out, "zl", j.zl)
                self._emit_if_changed(out, "minus", j.minus)
                self._emit_if_changed(out, "up", j.up)
                self._emit_if_changed(out, "down", j.down)
                self._emit_if_changed(out, "left", j.left)
                self._emit_if_changed(out, "right", j.right)
                self._emit_if_changed(out, "capture", j.capture)
                self._emit_if_changed(out, "left_sr", j.left_sr)
                self._emit_if_changed(out, "left_sl", j.left_sl)
            return iter(out)

    def _single_handle_init(
        self,
        device: str = "right",
        gripper_open: float = 1.0,
        gripper_close: float = 0.0,
        gripper_state: float = 1.0,
        horizontal_stick_mode: str = "y",
        close_y: bool = False,
        limit_dof: bool = False,
        glimit: list = [[0.125, -0.4, 0.046, -3.1, -1.5, -1.57], [0.380, 0.4, 0.23, 3.1, 1.5, 1.57]],
        offset_position_m: list = [0.0, 0.0, 0.0],
        offset_euler_rad: list = [0.0, 0.0, 0.0],
        euler_reverse: list = [1, 1, 1],
        direction_reverse: list = [1, 1, 1],
        dof_speed: list = [1, 1, 1, 1, 1, 1],
        rotation_filter_alpha_rate=1,
        common_rad: bool = True,
        lerobot: bool = False,
        pitch_down_double: bool = False,
        without_rest_init: bool = False,
        pure_xz: bool = True,
        change_down_to_gripper: bool = False,
        lowpassfilter_alpha_rate=0.05,
    ):
        if device == "right":
            self.joycon_id = jr_mod.get_R_id()
        elif device == "left":
            self.joycon_id = jr_mod.get_L_id()
        else:
            raise ValueError("wrong device name of joycon")

        if not self.joycon_id or self.joycon_id[0] is None:
            raise IOError("There is no joycon for robotics")

        serial = self.joycon_id[2] or ""
        device_serial = serial[:6]

        # Single HID handle for this Joy-Con
        self.joycon = jr_mod.GyroTrackingJoyCon(*self.joycon_id)
        self.gyro = self.joycon
        self.button = _SingleHandleButtonEvents(self.joycon, track_sticks=True)

        self.lerobot = lerobot
        self.pitch_down_double = pitch_down_double
        self.rotation_filter_alpha_rate = rotation_filter_alpha_rate
        self.orientation_sensor = jr_mod.AttitudeEstimator(
            common_rad=common_rad,
            lerobot=self.lerobot,
            pitch_down_double=self.pitch_down_double,
            lowpassfilter_alpha_rate=self.rotation_filter_alpha_rate,
        )
        self.without_rest_init = without_rest_init

        print(f"\033[32mconnect to {device} joycon successful.\033[0m")
        if not self.without_rest_init:
            self.reset_joycon()
        print()

        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.gripper_state = gripper_state

        self.position = offset_position_m.copy()
        self.orientation_rad = offset_euler_rad.copy()
        self.direction_vector = []
        self.direction_vector_right = []
        self.yaw_diff = 0.0

        self.offset_position_m = offset_position_m.copy()
        self.posture = offset_position_m.copy()

        self.horizontal_stick_mode = horizontal_stick_mode
        self.if_close_y = close_y
        self.if_limit_dof = limit_dof
        self.dof_speed = dof_speed.copy()
        self.glimit = glimit
        self.offset_euler_rad = offset_euler_rad
        self.euler_reverse = euler_reverse
        self.pure_xz = pure_xz
        self.direction_reverse = direction_reverse
        self.change_down_to_gripper = change_down_to_gripper
        self.gripper_toggle_button = 0

        self.reset_button = 0
        self.next_episode_button = 0
        self.restart_episode_button = 0
        self.button_control = 0

        if device_serial != jr_mod.JOYCON_SERIAL_SUPPORT:
            if len(serial) != 17 or serial.count(":") != 5:
                raise IOError("There is no joycon for robotics")

        self.running = True
        self.lock = jr_mod.threading.Lock()
        self.thread = jr_mod.threading.Thread(target=self.solve_loop, daemon=True)
        self.thread.start()

    jr_mod.JoyconRobotics.__init__ = _single_handle_init
    jr_mod.JoyconRobotics._mac_single_handle_patch_applied = True
