#!/usr/bin/env python3
"""
Run a small online gating sweep for learned obstacle-context integration.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import eval_obstacle_context_compare as occ  # type: ignore


def _parse_list(spec: str, cast):
    vals = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(cast(tok))
    if not vals:
        raise ValueError("empty sweep list")
    return vals


def _score_result(result: Dict[str, Any]) -> float:
    score = 0.0
    for scene_key, scene_res in result["results"].items():
        runtime = scene_res["runtime"]
        learned = scene_res["learned_v1"]
        learned_err = learned["obstacle_error_summary"]
        runtime_reward = float(runtime["reward_mean"])
        learned_reward = float(learned["reward_mean"])
        reward_ratio = learned_reward / max(1.0, abs(runtime_reward))
        offtrack_pen = float(learned["offtrack_rate"]) * 2.0
        collision_pen = float(learned["collision_rate"]) * 1.5
        fp_pen = float(learned_err["false_positive_visible_rate"]) * 5.0
        fn_pen = float(learned_err["false_negative_visible_rate"]) * 2.0
        score += reward_ratio - offtrack_pen - collision_pen - fp_pen - fn_pen
    return float(score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--phase", default="lane_pid_intro")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thresholds", default="0.55,0.60,0.65")
    ap.add_argument("--off-thresholds", default="0.40,0.45,0.50")
    ap.add_argument("--activations", default="2,3")
    ap.add_argument("--deactivations", default="2,3")
    ap.add_argument("--obstacle-context-checkpoint", required=True)
    ap.add_argument("--obstacle-context-device", default="cpu")
    ap.add_argument("--obstacle-context-seq-len", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    thresholds = _parse_list(args.thresholds, float)
    off_thresholds = _parse_list(args.off_thresholds, float)
    activations = _parse_list(args.activations, int)
    deactivations = _parse_list(args.deactivations, int)

    results: List[Dict[str, Any]] = []
    for thr, off_thr, act, deact in itertools.product(thresholds, off_thresholds, activations, deactivations):
        if off_thr >= thr:
            continue
        run = {"threshold": float(thr), "off_threshold": float(off_thr), "activation": int(act), "deactivation": int(deact)}
        print(f"[sweep] thr={thr:.2f} off={off_thr:.2f} act={act} deact={deact}")
        aggregate: Dict[str, Any] = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "phase": str(args.phase),
            "episodes": int(args.episodes),
            "results": {},
        }
        for scene_key, env_id in {"ws": "donkey-waveshare-v0", "gt": "donkey-generated-track-v0"}.items():
            aggregate["results"][scene_key] = {}
            for source in ("runtime", "learned_v1"):
                res = occ._evaluate(
                    model_path=args.checkpoint,
                    scene_env_id=env_id,
                    phase=args.phase,
                    obstacle_context_source=source,
                    obstacle_context_checkpoint=str(args.obstacle_context_checkpoint),
                    obstacle_context_device=str(args.obstacle_context_device),
                    obstacle_context_seq_len=int(args.obstacle_context_seq_len),
                    obstacle_context_present_threshold=float(thr),
                    obstacle_context_present_off_threshold=float(off_thr),
                    obstacle_context_activation_consecutive=int(act),
                    obstacle_context_deactivation_consecutive=int(deact),
                    port=int(args.port),
                    episodes=int(args.episodes),
                    seed=int(args.seed),
                    dump_error_dir="",
                    dump_top_k=0,
                )
                aggregate["results"][scene_key][source] = res
        run["score"] = _score_result(aggregate)
        run["summary"] = {
            scene: {
                "runtime_reward": float(aggregate["results"][scene]["runtime"]["reward_mean"]),
                "learned_reward": float(aggregate["results"][scene]["learned_v1"]["reward_mean"]),
                "learned_offtrack": float(aggregate["results"][scene]["learned_v1"]["offtrack_rate"]),
                "learned_collision": float(aggregate["results"][scene]["learned_v1"]["collision_rate"]),
                "learned_fp_visible": float(aggregate["results"][scene]["learned_v1"]["obstacle_error_summary"]["false_positive_visible_rate"]),
                "learned_fn_visible": float(aggregate["results"][scene]["learned_v1"]["obstacle_error_summary"]["false_negative_visible_rate"]),
                "learned_cv_present_rate": float(aggregate["results"][scene]["learned_v1"]["obstacle_error_summary"]["cv_present_rate"]),
                "runtime_visible_present_rate": float(aggregate["results"][scene]["runtime"]["obstacle_error_summary"]["runtime_visible_present_rate"]),
            }
            for scene in ("ws", "gt")
        }
        results.append(run)

    results.sort(key=lambda x: float(x["score"]), reverse=True)
    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "phase": str(args.phase),
        "episodes": int(args.episodes),
        "results": results,
        "best": results[0] if results else None,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[saved] {output_path}")
    if results:
        print(json.dumps(results[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
