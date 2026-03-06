#!/usr/bin/env python3
"""Diagnose Joy-Con HID accessibility on macOS/Linux."""

from __future__ import annotations

from joycon_hid_compat import ensure_hid_backend

ensure_hid_backend()

import hid

JOYCON_VENDOR_ID = 0x057E


def _try_open(device_info: dict) -> dict:
    vid = device_info["vendor_id"]
    pid = device_info["product_id"]
    serial = device_info.get("serial_number")
    path = device_info.get("path")

    results = {}

    dev = hid.device()
    try:
        dev.open(vid, pid)
        results["open(vid,pid)"] = "OK"
        dev.close()
    except Exception as exc:  # noqa: BLE001
        results["open(vid,pid)"] = f"FAIL: {type(exc).__name__}: {exc}"

    if serial:
        dev = hid.device()
        try:
            dev.open(vid, pid, serial)
            results["open(vid,pid,serial)"] = "OK"
            dev.close()
        except Exception as exc:  # noqa: BLE001
            results["open(vid,pid,serial)"] = f"FAIL: {type(exc).__name__}: {exc}"

    if path:
        dev = hid.device()
        try:
            dev.open_path(path)
            results["open_path(path)"] = "OK"
            dev.close()
        except Exception as exc:  # noqa: BLE001
            results["open_path(path)"] = f"FAIL: {type(exc).__name__}: {exc}"

    return results


def main() -> None:
    all_devices = hid.enumerate()
    joycons = [d for d in all_devices if d.get("vendor_id") == JOYCON_VENDOR_ID]

    print(f"Total HID devices visible: {len(all_devices)}")
    print(f"Joy-Con-like devices visible: {len(joycons)}")
    print()

    if not joycons:
        print("No Joy-Cons enumerated. Pair/connect Joy-Con(s) first.")
        return

    any_open_ok = False
    for idx, devinfo in enumerate(joycons, start=1):
        print(f"[JoyCon #{idx}]")
        print(f"  product: {devinfo.get('product_string')}")
        print(f"  vid:pid: {hex(devinfo.get('vendor_id', 0))}:{hex(devinfo.get('product_id', 0))}")
        print(f"  serial: {devinfo.get('serial_number')}")
        print(f"  path: {devinfo.get('path')}")

        open_results = _try_open(devinfo)
        for method, result in open_results.items():
            print(f"  {method}: {result}")
            if result == "OK":
                any_open_ok = True
        print()

    if any_open_ok:
        print("Result: At least one Joy-Con open method works. HID access is functional.")
    else:
        print("Result: Joy-Cons enumerate but cannot be opened.")
        print("Most likely causes on macOS:")
        print("1. Terminal app lacks Input Monitoring permission.")
        print("2. Another app has grabbed the controllers (Steam/SDL/browser/game tools).")
        print("3. Controllers are paired but not in a usable state; disconnect/re-pair.")


if __name__ == "__main__":
    main()
