#!/usr/bin/env python3
"""Write a reproducibility manifest for V17 endpoint deployment runs."""

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Dict, Optional


def run_cmd(cmd, cwd=None) -> Dict[str, object]:
    try:
        proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=10,
    )
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "output": proc.stdout.strip()}
    except Exception as exc:
        return {"ok": False, "returncode": None, "output": str(exc)}


def sha256_file(path: Optional[str]) -> Optional[Dict[str, object]]:
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        return {"path": path, "exists": False, "sha256": None, "size_bytes": None}
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "path": path,
        "exists": True,
        "sha256": h.hexdigest(),
        "size_bytes": os.path.getsize(path),
    }


def git_info(cwd: str, source_branch: Optional[str], source_commit: Optional[str]) -> Dict[str, object]:
    branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    commit = run_cmd(["git", "rev-parse", "HEAD"], cwd=cwd)
    status = run_cmd(["git", "status", "--short"], cwd=cwd)
    if not branch["ok"] and source_branch:
        branch = {"ok": True, "returncode": 0, "output": source_branch}
    if not commit["ok"] and source_commit:
        commit = {"ok": True, "returncode": 0, "output": source_commit}
    return {
        "branch": branch["output"] if branch["ok"] else "unknown",
        "commit": commit["output"] if commit["ok"] else "unknown",
        "dirty": bool(status["output"]) if status["ok"] else None,
        "status_short": status["output"] if status["ok"] else "unknown",
        "source": "git" if branch["ok"] and commit["ok"] else "env_or_unknown",
    }


def module_versions() -> Dict[str, str]:
    versions = {}
    for name in ("numpy", "torch", "cv2", "tensorrt", "stable_baselines3", "donkeycar"):
        try:
            mod = importlib.import_module(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            versions[name] = "unavailable: %s" % exc.__class__.__name__
    return versions


def read_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="/home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip")
    parser.add_argument("--onnx", default="/home/jetson/mycar/models/v17_actor.onnx")
    parser.add_argument("--engine", default="/home/jetson/mycar/models/v17_actor_fp16.engine")
    parser.add_argument("--metadata", default="/home/jetson/mycar/models/v17_actor_export.json")
    parser.add_argument("--command-file", default=None)
    parser.add_argument("--runtime-command", default=None)
    parser.add_argument("--engine-build-command", default=None)
    parser.add_argument("--source-branch", default=os.environ.get("V17_SOURCE_BRANCH"))
    parser.add_argument("--source-commit", default=os.environ.get("V17_SOURCE_COMMIT"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    os.makedirs(run_dir, exist_ok=True)
    command_text = args.runtime_command
    if not command_text and args.command_file:
        command_text = read_file(args.command_file)

    payload = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_dir": run_dir,
        "git": git_info(os.getcwd(), args.source_branch, args.source_commit),
        "runtime_command": command_text or "unknown",
        "command_file": os.path.abspath(os.path.expanduser(args.command_file)) if args.command_file else None,
        "files": {
            "model": sha256_file(args.model),
            "onnx": sha256_file(args.onnx),
            "engine": sha256_file(args.engine),
            "metadata": sha256_file(args.metadata),
        },
        "engine_build_command": args.engine_build_command or "unknown",
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "ld_preload": os.environ.get("LD_PRELOAD", ""),
            "nv_tegra_release": read_file("/etc/nv_tegra_release"),
            "cuda_version_txt": read_file("/usr/local/cuda/version.txt"),
            "nvcc_version": run_cmd(["bash", "-lc", "nvcc --version 2>/dev/null || true"])["output"],
            "tegrastats_sample": run_cmd(["bash", "-lc", "timeout 2 tegrastats 2>/dev/null | head -n 1 || true"])["output"],
            "python_modules": module_versions(),
        },
    }
    out_path = args.out or os.path.join(run_dir, "repro_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
