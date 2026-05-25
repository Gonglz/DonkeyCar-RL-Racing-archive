#!/usr/bin/env python3
"""Run V17 TensorRT preflight mismatch negative cases.

Each case writes a mutated metadata copy into its own evidence directory and
expects runtime_monitor.py to fail before Vehicle.start.
"""

import argparse
import copy
import glob
import json
import os
import subprocess
import sys
from datetime import datetime


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def mutate_lidar_dim_72(metadata):
    metadata.setdefault("shape", {})["lidar_dim"] = 72


def mutate_lstm_hidden_128(metadata):
    metadata.setdefault("shape", {})["lstm_hidden_size"] = 128


def mutate_remove_lidar_meta_input(metadata):
    inputs = list(metadata.get("inputs") or [])
    metadata["inputs"] = [name for name in inputs if name != "lidar_meta"]


def mutate_rename_next_h_output(metadata):
    outputs = list(metadata.get("outputs") or [])
    metadata["outputs"] = ["bad_next_h" if name == "next_h" else name for name in outputs]


CASES = [
    ("metadata_lidar_dim_72", mutate_lidar_dim_72, "shape mismatch for lidar_dim"),
    ("metadata_lstm_hidden_128", mutate_lstm_hidden_128, "binding shape mismatch"),
    ("metadata_missing_lidar_meta", mutate_remove_lidar_meta_input, "missing inputs"),
    ("metadata_bad_next_h_output", mutate_rename_next_h_output, "missing outputs"),
]


def command_for_case(args, case_dir, bad_metadata_path):
    return [
        args.python,
        args.runtime_monitor,
        "drive",
        "--model",
        args.model,
        "--type",
        "v17",
        "--control-mode",
        "shadow",
        "--shadow-duration",
        "1",
        "--log-dir",
        case_dir,
        "--run-label",
        os.path.basename(case_dir),
        "--track-condition",
        "preflight_negative",
        "--shadow-engine",
        args.engine,
        "--shadow-engine-metadata",
        bad_metadata_path,
        "--no-start-lidar-driver",
        "--no-require-lidar",
        "--no-require-rp2040",
        "--shadow-cpu",
    ]


def run_case(args, base_metadata, name, mutator, expected_text):
    case_dir = os.path.join(args.out_dir, name)
    os.makedirs(case_dir, exist_ok=True)
    bad_metadata = copy.deepcopy(base_metadata)
    mutator(bad_metadata)
    bad_metadata_path = os.path.join(case_dir, "metadata_bad.json")
    write_json(bad_metadata_path, bad_metadata)

    cmd = command_for_case(args, case_dir, bad_metadata_path)
    command_path = os.path.join(case_dir, "command.txt")
    with open(command_path, "w", encoding="utf-8") as f:
        f.write(" ".join(subprocess.list2cmdline([part]) for part in cmd))
        f.write("\n")

    proc = subprocess.run(
        cmd,
        cwd=args.cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=args.timeout_sec,
    )
    log_path = os.path.join(case_dir, "runtime.log")
    with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(proc.stdout)

    preflight_path = os.path.join(case_dir, "preflight_report.json")
    preflight = read_json(preflight_path) if os.path.exists(preflight_path) else {}
    csv_paths = sorted(glob.glob(os.path.join(case_dir, "run_*.csv")))
    entered_vehicle_loop = (
        "DataCollector 已注入 Vehicle 循环" in proc.stdout
        or "Vehicle.start" in proc.stdout
        or bool(csv_paths)
    )
    error_text = str(preflight.get("error") or proc.stdout)
    passed = (
        proc.returncode == args.expected_exit
        and not entered_vehicle_loop
        and expected_text.lower() in error_text.lower()
    )
    return {
        "name": name,
        "case_dir": case_dir,
        "bad_metadata": bad_metadata_path,
        "command_file": command_path,
        "runtime_log": log_path,
        "preflight_report": preflight_path if os.path.exists(preflight_path) else None,
        "exit_code": proc.returncode,
        "expected_exit": args.expected_exit,
        "entered_vehicle_loop": entered_vehicle_loop,
        "csv_files": csv_paths,
        "expected_error_text": expected_text,
        "observed_error": preflight.get("error") or "",
        "pass": passed,
    }


def write_markdown(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# V17 Preflight Negative Cases\n\n")
        f.write("| case | exit | entered vehicle loop | expected text | pass |\n")
        f.write("|---|---:|---:|---|---:|\n")
        for row in payload["cases"]:
            f.write(
                "| {name} | {exit_code} | {entered} | `{expected}` | {passed} |\n".format(
                    name=row["name"],
                    exit_code=row["exit_code"],
                    entered=str(row["entered_vehicle_loop"]).lower(),
                    expected=row["expected_error_text"],
                    passed=str(row["pass"]).lower(),
                )
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-monitor", default="/home/jetson/mycar/runtime_monitor.py")
    parser.add_argument("--cwd", default="/home/jetson/mycar")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", default="/home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip")
    parser.add_argument("--engine", default="/home/jetson/mycar/models/v17_actor_fp16.engine")
    parser.add_argument("--metadata", default="/home/jetson/mycar/models/v17_actor_export.json")
    parser.add_argument("--out-dir", default="preflight_negative_cases")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--expected-exit", type=int, default=2)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    args = parser.parse_args()

    args.runtime_monitor = os.path.abspath(os.path.expanduser(args.runtime_monitor))
    args.cwd = os.path.abspath(os.path.expanduser(args.cwd))
    args.model = os.path.abspath(os.path.expanduser(args.model))
    args.engine = os.path.abspath(os.path.expanduser(args.engine))
    args.metadata = os.path.abspath(os.path.expanduser(args.metadata))
    args.out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(args.out_dir, exist_ok=True)

    base_metadata = read_json(args.metadata)
    results = [run_case(args, base_metadata, name, mutator, expected) for name, mutator, expected in CASES]
    payload = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "runtime_monitor": args.runtime_monitor,
        "model": args.model,
        "engine": args.engine,
        "metadata": args.metadata,
        "out_dir": args.out_dir,
        "cases": results,
        "pass": all(row["pass"] for row in results),
    }

    out_json = args.out_json or os.path.join(args.out_dir, "preflight_negative_cases.json")
    out_md = args.out_md or os.path.join(args.out_dir, "preflight_negative_cases.md")
    write_json(out_json, payload)
    write_markdown(out_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
