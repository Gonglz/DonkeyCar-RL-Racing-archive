#!/usr/bin/env python3
"""Mock active-mode checks for DeploymentSafetyGate."""

import argparse
import importlib.util
import json
import os
import sys
import time


def load_runtime_monitor(path: str):
    path = os.path.abspath(os.path.expanduser(path))
    spec = importlib.util.spec_from_file_location("runtime_monitor_gate_mock", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class MockVehicle:
    def __init__(self):
        self.on = True


class MockLidarReader:
    def __init__(self, connected=True, frame_count=10, age_ms=100.0):
        self.is_connected = connected
        self.frame_count = frame_count
        self.age_ms = age_ms

    def get_data(self):
        return {
            "frame_count": self.frame_count,
            "scan_age_ms": self.age_ms,
        }


class MockSerialReader:
    def __init__(self, connected=True, frame_count=10, age_ms=100.0):
        self.is_connected = connected
        self.frame_count = frame_count
        self.last_update = time.time() - (age_ms / 1000.0)

    def get_data(self):
        return {
            "frame_count": self.frame_count,
            "last_update": self.last_update,
        }


def run_case(runtime_monitor, case):
    vehicle = MockVehicle()
    gate = runtime_monitor.DeploymentSafetyGate(
        control_mode="active",
        serial_reader=case["serial_reader"],
        lidar_reader=case["lidar_reader"],
        require_lidar=case.get("require_lidar", True),
        require_rp2040=case.get("require_rp2040", True),
        max_lidar_age_ms=case.get("max_lidar_age_ms", 350.0),
        max_inference_ms=case.get("max_inference_ms", 350.0),
        max_rp2040_age_ms=case.get("max_rp2040_age_ms", 1000.0),
    )
    gate.vehicle = vehicle
    result = gate.run(0.31, 0.42, case.get("pilot_latency_ms", 120.0))
    return {
        "name": case["name"],
        "safe_angle": result[0],
        "safe_throttle": result[1],
        "blocked": result[2],
        "block_reason": result[3],
        "inference_timeout_count": result[4],
        "lidar_missing_count": result[5],
        "lidar_stale_count": result[6],
        "rp2040_missing_count": result[7],
        "vehicle_on": bool(vehicle.on),
        "pass": (
            result[0] == 0.0
            and result[1] == 0.0
            and result[2] is True
            and vehicle.on is False
            and bool(result[3])
        ),
    }


def write_markdown(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# DeploymentSafetyGate Active Mock Result\n\n")
        f.write("| case | blocked | vehicle_on | reason | safe_angle | safe_throttle | pass |\n")
        f.write("|---|---:|---:|---|---:|---:|---:|\n")
        for row in payload["cases"]:
            f.write(
                "| {name} | {blocked} | {vehicle_on} | `{reason}` | {angle:.3f} | {throttle:.3f} | {passed} |\n".format(
                    name=row["name"],
                    blocked=str(row["blocked"]).lower(),
                    vehicle_on=str(row["vehicle_on"]).lower(),
                    reason=row["block_reason"],
                    angle=float(row["safe_angle"]),
                    throttle=float(row["safe_throttle"]),
                    passed=str(row["pass"]).lower(),
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-monitor", default="/home/jetson/mycar/runtime_monitor.py")
    parser.add_argument("--out-json", default="active_safety_gate_mock.json")
    parser.add_argument("--out-md", default="active_safety_gate_mock.md")
    args = parser.parse_args()

    runtime_monitor = load_runtime_monitor(args.runtime_monitor)
    cases = [
        {
            "name": "lidar_stale",
            "lidar_reader": MockLidarReader(age_ms=1200.0),
            "serial_reader": MockSerialReader(age_ms=100.0),
            "pilot_latency_ms": 120.0,
        },
        {
            "name": "inference_timeout",
            "lidar_reader": MockLidarReader(age_ms=100.0),
            "serial_reader": MockSerialReader(age_ms=100.0),
            "pilot_latency_ms": 999.0,
            "max_inference_ms": 350.0,
        },
        {
            "name": "rp2040_stale",
            "lidar_reader": MockLidarReader(age_ms=100.0),
            "serial_reader": MockSerialReader(age_ms=2500.0),
            "pilot_latency_ms": 120.0,
        },
    ]
    results = [run_case(runtime_monitor, case) for case in cases]
    payload = {
        "runtime_monitor": os.path.abspath(os.path.expanduser(args.runtime_monitor)),
        "cases": results,
        "pass": all(row["pass"] for row in results),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    write_markdown(args.out_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
