#!/usr/bin/env python3
"""
Evaluate V17 world-model readiness artifacts.

Capabilities:
- merge exported world-model datasets into one formal-training dataset
- validate merged dataset structure and label distribution
- validate train_summary.json against readiness thresholds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_DATASET_KEYS: Tuple[str, ...] = (
    "ego8",
    "lidar",
    "async_meta",
    "target_rel",
    "target_rel_mask",
    "target_gap",
    "target_collision",
    "target_ttc",
    "target_safety_valid",
    "target_passable",
    "target_closing_rate",
    "target_overtake_progress",
    "target_opportunity_valid",
    "episode_id",
    "step_in_episode",
    "scene_id",
    "done",
)

OPTIONAL_DATASET_KEYS: Tuple[str, ...] = (
    "camera",
)

DATASET_THRESHOLD_DEFAULTS: Dict[str, Any] = {
    "expected_samples": 5120,
    "expected_scenes": ["generated_track", "waveshare"],
    "collision_pos_rate_min": 0.05,
    "collision_pos_rate_max": 0.30,
    "opportunity_valid_rate_min": 0.70,
    "passable_rate_min": 0.10,
    "passable_rate_max": 0.90,
}

TRAINING_THRESHOLD_DEFAULTS: Dict[str, float] = {
    "stage_a_mae_target_rel_max": 2.50,
    "stage_a_mae_gap_max": 2.00,
    "stage_b_geom_regression_max_ratio": 0.05,
    "stage_c_guard_geom_regression_max_ratio": 0.03,
}


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _manifest_dataset_thresholds(manifest: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    thresholds = dict(DATASET_THRESHOLD_DEFAULTS)
    if manifest:
        thresholds.update(dict(manifest.get("phase_c", {}).get("thresholds", {})))
    return thresholds


def _manifest_training_thresholds(manifest: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    thresholds = dict(TRAINING_THRESHOLD_DEFAULTS)
    if manifest:
        thresholds.update(dict(manifest.get("phase_d", {}).get("thresholds", {})))
    return thresholds


def _infer_meta_path(npz_path: str | Path) -> Optional[Path]:
    candidate = Path(npz_path).expanduser().with_suffix(".json")
    return candidate if candidate.exists() else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if np.isfinite(out) else float(default)


def _scene_name_maps(meta: Optional[Mapping[str, Any]], scene_ids: np.ndarray) -> Tuple[Dict[int, str], Dict[str, int]]:
    id_to_name: Dict[int, str] = {}
    if meta:
        raw_scene_to_id = dict(meta.get("scene_to_id", {}) or {})
        for name, scene_id in raw_scene_to_id.items():
            try:
                id_to_name[int(scene_id)] = str(name)
            except Exception:
                continue
    if not id_to_name:
        for scene_id in np.unique(scene_ids).tolist():
            id_to_name[int(scene_id)] = f"scene_{int(scene_id)}"
    scene_to_id = {name: int(scene_id) for scene_id, name in sorted(id_to_name.items())}
    return id_to_name, scene_to_id


def _dataset_arrays(npz_path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(Path(npz_path).expanduser(), allow_pickle=False)
    return {key: np.asarray(data[key]) for key in data.files}


def summarize_world_model_dataset(
    npz_path: str | Path,
    meta_json: Optional[str | Path] = None,
) -> Dict[str, Any]:
    arrays = _dataset_arrays(npz_path)
    meta = _load_json(meta_json) if meta_json else (_load_json(_infer_meta_path(npz_path)) if _infer_meta_path(npz_path) else {})
    missing_keys = [key for key in REQUIRED_DATASET_KEYS if key not in arrays]
    samples = int(arrays.get("ego8", np.asarray([], dtype=np.float32)).shape[0]) if "ego8" in arrays else 0

    key_shapes: Dict[str, List[int]] = {}
    shape_mismatches: Dict[str, int] = {}
    nonfinite_keys: List[str] = []
    for key, arr in arrays.items():
        key_shapes[key] = list(arr.shape)
        if arr.ndim == 0:
            shape_mismatches[key] = -1
        elif samples and int(arr.shape[0]) != samples:
            shape_mismatches[key] = int(arr.shape[0])
        if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
            nonfinite_keys.append(str(key))

    scene_id = np.asarray(arrays.get("scene_id", np.asarray([], dtype=np.int64))).reshape(-1)
    id_to_name, _scene_to_id = _scene_name_maps(meta, scene_id)
    scene_counts: Dict[str, int] = {}
    for raw_scene_id, count in zip(*np.unique(scene_id, return_counts=True)):
        scene_counts[id_to_name.get(int(raw_scene_id), f"scene_{int(raw_scene_id)}")] = int(count)

    target_collision = np.asarray(arrays.get("target_collision", np.asarray([], dtype=np.float32))).reshape(-1)
    target_opportunity_valid = np.asarray(
        arrays.get("target_opportunity_valid", np.asarray([], dtype=np.float32))
    ).reshape(-1)
    target_passable = np.asarray(arrays.get("target_passable", np.asarray([], dtype=np.float32)))
    if target_passable.ndim == 1:
        target_passable = target_passable.reshape(-1, 2) if target_passable.size else target_passable.reshape(0, 2)

    return {
        "npz_path": str(_as_path(npz_path)),
        "meta_json": str(_as_path(meta_json)) if meta_json else None,
        "samples": int(samples),
        "episodes": int(np.unique(np.asarray(arrays.get("episode_id", np.asarray([], dtype=np.int64))).reshape(-1)).size),
        "keys_present": sorted(arrays.keys()),
        "missing_keys": missing_keys,
        "key_shapes": key_shapes,
        "shape_mismatches": shape_mismatches,
        "nonfinite_keys": nonfinite_keys,
        "scene_counts": scene_counts,
        "collision_pos_rate": float(np.mean(target_collision > 0.5)) if target_collision.size else 0.0,
        "opportunity_valid_rate": float(np.mean(target_opportunity_valid > 0.5)) if target_opportunity_valid.size else 0.0,
        "passable_left_rate": (
            float(np.mean(target_passable[:, 0] > 0.5))
            if target_passable.ndim == 2 and target_passable.shape[1] >= 2 and target_passable.shape[0] > 0
            else 0.0
        ),
        "passable_right_rate": (
            float(np.mean(target_passable[:, 1] > 0.5))
            if target_passable.ndim == 2 and target_passable.shape[1] >= 2 and target_passable.shape[0] > 0
            else 0.0
        ),
    }


def merge_world_model_datasets(
    sources: Sequence[Mapping[str, Any]],
    output_npz: str | Path,
    output_json: str | Path,
) -> Dict[str, Any]:
    if not sources:
        raise ValueError("at least one source dataset is required")

    common_optional_keys = set(OPTIONAL_DATASET_KEYS)
    for source in sources:
        with np.load(_as_path(str(source["path"])), allow_pickle=False) as data:
            common_optional_keys &= set(data.files)

    merge_keys: Tuple[str, ...] = tuple(REQUIRED_DATASET_KEYS) + tuple(sorted(common_optional_keys))
    merged_lists: Dict[str, List[np.ndarray]] = {key: [] for key in merge_keys}
    merged_scene_to_id: Dict[str, int] = {}
    merged_sources: List[Dict[str, Any]] = []
    next_episode_id = 0

    for source in sources:
        name = str(source.get("name") or Path(str(source["path"])).stem)
        npz_path = _as_path(str(source["path"]))
        meta_path = source.get("meta_json")
        meta_path_resolved = _as_path(str(meta_path)) if meta_path else _infer_meta_path(npz_path)
        arrays = _dataset_arrays(npz_path)
        missing_keys = [key for key in REQUIRED_DATASET_KEYS if key not in arrays]
        if missing_keys:
            raise ValueError(f"source {name!r} missing keys: {missing_keys}")
        meta = _load_json(meta_path_resolved) if meta_path_resolved and meta_path_resolved.exists() else {}

        local_scene_id = np.asarray(arrays["scene_id"], dtype=np.int64).reshape(-1)
        local_episode_id = np.asarray(arrays["episode_id"], dtype=np.int64).reshape(-1)
        local_scene_to_name, _ = _scene_name_maps(meta, local_scene_id)

        remapped_scene_id = np.zeros_like(local_scene_id)
        for raw_scene_id in np.unique(local_scene_id).tolist():
            scene_name = local_scene_to_name.get(int(raw_scene_id), f"scene_{int(raw_scene_id)}")
            if scene_name not in merged_scene_to_id:
                merged_scene_to_id[scene_name] = len(merged_scene_to_id)
            remapped_scene_id[local_scene_id == int(raw_scene_id)] = int(merged_scene_to_id[scene_name])

        remapped_episode_id = np.zeros_like(local_episode_id)
        local_unique_eps = sorted(int(x) for x in np.unique(local_episode_id).tolist())
        for offset, raw_episode_id in enumerate(local_unique_eps):
            remapped_episode_id[local_episode_id == raw_episode_id] = int(next_episode_id + offset)
        next_episode_id += len(local_unique_eps)

        for key in merge_keys:
            arr = np.asarray(arrays[key])
            if key == "scene_id":
                arr = remapped_scene_id.astype(np.int64)
            elif key == "episode_id":
                arr = remapped_episode_id.astype(np.int64)
            merged_lists[key].append(arr)

        merged_sources.append(
            {
                "name": name,
                "path": str(npz_path),
                "meta_json": str(meta_path_resolved) if meta_path_resolved else None,
                "samples": int(local_scene_id.shape[0]),
                "episodes": int(len(local_unique_eps)),
                "scene_names": sorted(set(local_scene_to_name.values())),
            }
        )

    merged_arrays = {
        key: np.concatenate(chunks, axis=0) if chunks else np.asarray([], dtype=np.float32)
        for key, chunks in merged_lists.items()
    }

    output_npz_path = Path(output_npz).expanduser()
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz_path, **merged_arrays)

    merged_scene_counts: Dict[str, int] = {}
    merged_scene_id = np.asarray(merged_arrays["scene_id"], dtype=np.int64).reshape(-1)
    merged_id_to_scene = {scene_id: name for name, scene_id in merged_scene_to_id.items()}
    for raw_scene_id, count in zip(*np.unique(merged_scene_id, return_counts=True)):
        merged_scene_counts[merged_id_to_scene.get(int(raw_scene_id), f"scene_{int(raw_scene_id)}")] = int(count)
    target_collision = np.asarray(merged_arrays["target_collision"], dtype=np.float32).reshape(-1)
    target_opportunity_valid = np.asarray(merged_arrays["target_opportunity_valid"], dtype=np.float32).reshape(-1)
    target_passable = np.asarray(merged_arrays["target_passable"], dtype=np.float32).reshape(-1, 2)
    payload = {
        "output_npz": str(output_npz_path.resolve()),
        "sources": merged_sources,
        "samples": int(merged_scene_id.shape[0]),
        "episodes": int(np.unique(np.asarray(merged_arrays["episode_id"], dtype=np.int64).reshape(-1)).size),
        "scene_to_id": dict(sorted(merged_scene_to_id.items(), key=lambda kv: kv[1])),
        "scene_counts": merged_scene_counts,
        "optional_keys": sorted(common_optional_keys),
        "collision_pos_rate": float(np.mean(target_collision > 0.5)) if target_collision.size else 0.0,
        "opportunity_valid_rate": (
            float(np.mean(target_opportunity_valid > 0.5)) if target_opportunity_valid.size else 0.0
        ),
        "passable_left_rate": float(np.mean(target_passable[:, 0] > 0.5)) if target_passable.size else 0.0,
        "passable_right_rate": float(np.mean(target_passable[:, 1] > 0.5)) if target_passable.size else 0.0,
    }
    _write_json(output_json, payload)
    return payload


def evaluate_dataset_readiness(
    npz_path: str | Path,
    meta_json: Optional[str | Path] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    thresholds = _manifest_dataset_thresholds(manifest)
    summary = summarize_world_model_dataset(npz_path=npz_path, meta_json=meta_json)
    fail_reasons: List[str] = []

    if summary["missing_keys"]:
        fail_reasons.append(f"missing required keys: {summary['missing_keys']}")
    if summary["shape_mismatches"]:
        fail_reasons.append(f"inconsistent first dimension: {summary['shape_mismatches']}")
    if summary["nonfinite_keys"]:
        fail_reasons.append(f"non-finite values detected: {summary['nonfinite_keys']}")
    if summary["episodes"] < 2:
        fail_reasons.append("dataset must contain at least 2 episodes for train/val split")

    expected_samples = int(thresholds.get("expected_samples", 0) or 0)
    if expected_samples > 0 and int(summary["samples"]) != expected_samples:
        fail_reasons.append(
            f"samples={summary['samples']} does not match expected_samples={expected_samples}"
        )

    expected_scenes = [str(x) for x in thresholds.get("expected_scenes", [])]
    for scene_name in expected_scenes:
        if int(summary["scene_counts"].get(scene_name, 0)) <= 0:
            fail_reasons.append(f"scene {scene_name!r} has no samples")

    collision_pos_rate = float(summary["collision_pos_rate"])
    if collision_pos_rate < float(thresholds["collision_pos_rate_min"]) or collision_pos_rate > float(
        thresholds["collision_pos_rate_max"]
    ):
        fail_reasons.append(
            f"collision_pos_rate={collision_pos_rate:.6f} outside "
            f"[{float(thresholds['collision_pos_rate_min']):.3f}, {float(thresholds['collision_pos_rate_max']):.3f}]"
        )

    opportunity_valid_rate = float(summary["opportunity_valid_rate"])
    if opportunity_valid_rate < float(thresholds["opportunity_valid_rate_min"]):
        fail_reasons.append(
            f"opportunity_valid_rate={opportunity_valid_rate:.6f} below "
            f"{float(thresholds['opportunity_valid_rate_min']):.3f}"
        )

    for side_key in ("passable_left_rate", "passable_right_rate"):
        side_value = float(summary[side_key])
        if side_value < float(thresholds["passable_rate_min"]) or side_value > float(thresholds["passable_rate_max"]):
            fail_reasons.append(
                f"{side_key}={side_value:.6f} outside "
                f"[{float(thresholds['passable_rate_min']):.3f}, {float(thresholds['passable_rate_max']):.3f}]"
            )

    return {
        "status": "passed" if not fail_reasons else "failed",
        "artifacts": {
            "npz_path": str(_as_path(npz_path)),
            "meta_json": str(_as_path(meta_json)) if meta_json else None,
        },
        "metrics": {
            "samples": int(summary["samples"]),
            "episodes": int(summary["episodes"]),
            "scene_counts": summary["scene_counts"],
            "collision_pos_rate": collision_pos_rate,
            "opportunity_valid_rate": opportunity_valid_rate,
            "passable_left_rate": float(summary["passable_left_rate"]),
            "passable_right_rate": float(summary["passable_right_rate"]),
        },
        "summary": summary,
        "fail_reasons": fail_reasons,
    }


def _is_finite_metric_dict(metrics: Mapping[str, Any], keys: Iterable[str]) -> Tuple[bool, List[str]]:
    bad_keys: List[str] = []
    for key in keys:
        value = metrics.get(key, None)
        if not np.isfinite(_safe_float(value, float("nan"))):
            bad_keys.append(str(key))
    return (len(bad_keys) == 0), bad_keys


def evaluate_training_readiness(
    summary_json: str | Path,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    thresholds = _manifest_training_thresholds(manifest)
    summary = _load_json(summary_json)
    fail_reasons: List[str] = []

    stage_a = dict(summary.get("stage_a", {}) or {})
    stage_b = dict(summary.get("stage_b", {}) or {})
    stage_c = dict(summary.get("stage_c", {}) or {})
    stage_c_pre_metrics = dict(summary.get("stage_c_pre_metrics", {}) or {})
    stage_c_best_metrics = dict(stage_c.get("best_metrics", {}) or {})
    final_path = summary.get("final_path")

    required_paths = {
        "stage_a_best": stage_a.get("best_path"),
        "stage_b_best": stage_b.get("best_path"),
        "final_checkpoint": final_path,
        "train_summary": str(_as_path(summary_json)),
    }
    if stage_c.get("best_path"):
        required_paths["stage_c_best"] = stage_c.get("best_path")

    missing_artifacts = {
        name: str(path)
        for name, path in required_paths.items()
        if (not path) or (not Path(str(path)).expanduser().exists())
    }
    if missing_artifacts:
        fail_reasons.append(f"missing training artifacts: {missing_artifacts}")

    stage_a_best_metrics = dict(stage_a.get("best_metrics", {}) or {})
    stage_b_best_metrics = dict(stage_b.get("best_metrics", {}) or {})
    finite_ok_a, bad_metric_keys_a = _is_finite_metric_dict(
        stage_a_best_metrics,
        ("mae_target_rel", "mae_gap"),
    )
    finite_ok_b, bad_metric_keys_b = _is_finite_metric_dict(
        stage_b_best_metrics,
        (
            "mae_target_rel",
            "mae_gap",
            "loss_collision",
            "loss_ttc",
            "loss_passable",
            "loss_closing",
            "loss_overtake_gain",
        ),
    )
    if not finite_ok_a:
        fail_reasons.append(f"stage_a best metrics contain non-finite values: {bad_metric_keys_a}")
    if not finite_ok_b:
        fail_reasons.append(f"stage_b best metrics contain non-finite values: {bad_metric_keys_b}")

    stage_a_mae_target_rel = _safe_float(stage_a_best_metrics.get("mae_target_rel"), float("nan"))
    stage_a_mae_gap = _safe_float(stage_a_best_metrics.get("mae_gap"), float("nan"))
    if stage_a_mae_target_rel > float(thresholds["stage_a_mae_target_rel_max"]):
        fail_reasons.append(
            f"stage_a mae_target_rel={stage_a_mae_target_rel:.6f} exceeds "
            f"{float(thresholds['stage_a_mae_target_rel_max']):.3f}"
        )
    if stage_a_mae_gap > float(thresholds["stage_a_mae_gap_max"]):
        fail_reasons.append(
            f"stage_a mae_gap={stage_a_mae_gap:.6f} exceeds {float(thresholds['stage_a_mae_gap_max']):.3f}"
        )

    geom_regression_ratio = float(thresholds["stage_b_geom_regression_max_ratio"])
    geom_regressions: Dict[str, Dict[str, float]] = {}
    stage_b_geom_failures: List[str] = []
    for key in ("mae_target_rel", "mae_gap"):
        stage_a_value = _safe_float(stage_a_best_metrics.get(key), float("nan"))
        stage_b_value = _safe_float(stage_b_best_metrics.get(key), float("nan"))
        allow = stage_a_value * (1.0 + geom_regression_ratio)
        geom_regressions[key] = {
            "stage_a": stage_a_value,
            "stage_b": stage_b_value,
            "allowed_max": allow,
        }
        if np.isfinite(stage_a_value) and np.isfinite(stage_b_value) and stage_b_value > allow:
            stage_b_geom_failures.append(
                f"stage_b {key}={stage_b_value:.6f} exceeds allowed regression max {allow:.6f}"
            )

    stage_c_guard_triggered = bool(stage_c.get("guard_triggered", False))
    stage_c_guard_failures: List[str] = []
    stage_c_final_recovery_ok = False
    if stage_c_guard_triggered:
        guard_geom_ratio = float(thresholds["stage_c_guard_geom_regression_max_ratio"])
        if not stage_c_pre_metrics:
            stage_c_guard_failures.append("stage_c guard triggered but stage_c_pre_metrics missing")
        for key in ("mae_target_rel", "mae_gap"):
            stage_b_value = _safe_float(stage_b_best_metrics.get(key), float("nan"))
            final_value = _safe_float(stage_c_pre_metrics.get(key), float("nan"))
            allowed = stage_b_value * (1.0 + guard_geom_ratio)
            if np.isfinite(stage_b_value) and np.isfinite(final_value) and final_value > allowed:
                stage_c_guard_failures.append(
                    f"guard fallback final {key}={final_value:.6f} exceeds {allowed:.6f}"
                )
    else:
        guard_geom_ratio = float(thresholds["stage_c_guard_geom_regression_max_ratio"])
        recovered = True
        for key in ("mae_target_rel", "mae_gap"):
            stage_a_value = _safe_float(stage_a_best_metrics.get(key), float("nan"))
            stage_c_value = _safe_float(stage_c_best_metrics.get(key), float("nan"))
            allowed = stage_a_value * (1.0 + guard_geom_ratio)
            if not (np.isfinite(stage_a_value) and np.isfinite(stage_c_value) and stage_c_value <= allowed):
                recovered = False
                break
        stage_c_final_recovery_ok = recovered
    if stage_c_guard_failures:
        fail_reasons.extend(stage_c_guard_failures)
    elif stage_b_geom_failures and not stage_c_final_recovery_ok:
        fail_reasons.extend(stage_b_geom_failures)

    return {
        "status": "passed" if not fail_reasons else "failed",
        "artifacts": {
            "train_summary": str(_as_path(summary_json)),
            "stage_a_best": stage_a.get("best_path"),
            "stage_b_best": stage_b.get("best_path"),
            "stage_c_best": stage_c.get("best_path"),
            "final_checkpoint": final_path,
        },
        "metrics": {
            "stage_a": {
                "best_val_loss": stage_a.get("best_val_loss"),
                "mae_target_rel": stage_a_best_metrics.get("mae_target_rel"),
                "mae_gap": stage_a_best_metrics.get("mae_gap"),
            },
            "stage_b": {
                "best_val_loss": stage_b.get("best_val_loss"),
                "mae_target_rel": stage_b_best_metrics.get("mae_target_rel"),
                "mae_gap": stage_b_best_metrics.get("mae_gap"),
                "loss_collision": stage_b_best_metrics.get("loss_collision"),
                "loss_ttc": stage_b_best_metrics.get("loss_ttc"),
                "loss_passable": stage_b_best_metrics.get("loss_passable"),
                "loss_closing": stage_b_best_metrics.get("loss_closing"),
                "loss_overtake_gain": stage_b_best_metrics.get("loss_overtake_gain"),
            },
            "stage_c": {
                "guard_triggered": stage_c_guard_triggered,
                "best_path": stage_c.get("best_path"),
                "pre_metrics": stage_c_pre_metrics,
                "best_metrics": stage_c_best_metrics,
                "final_recovery_ok": stage_c_final_recovery_ok,
            },
            "geom_regressions": geom_regressions,
        },
        "summary": summary,
        "fail_reasons": fail_reasons,
    }


def _parse_source_specs(specs: Sequence[str]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for raw in specs:
        if "=" not in raw:
            raise ValueError(f"source spec must look like name=path, got {raw!r}")
        name, path = raw.split("=", 1)
        sources.append({"name": str(name).strip(), "path": str(path).strip()})
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V17 world-model readiness artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_merge = subparsers.add_parser("merge", help="merge world-model datasets")
    p_merge.add_argument("--source", action="append", required=True, help="dataset source spec: name=/path/to/file.npz")
    p_merge.add_argument("--output-npz", type=str, required=True)
    p_merge.add_argument("--output-json", type=str, required=True)

    p_dataset = subparsers.add_parser("check-dataset", help="validate merged dataset readiness")
    p_dataset.add_argument("--npz", type=str, required=True)
    p_dataset.add_argument("--meta-json", type=str, default=None)
    p_dataset.add_argument("--manifest", type=str, default=None)
    p_dataset.add_argument("--output-json", type=str, default=None)

    p_train = subparsers.add_parser("check-training", help="validate train_summary readiness")
    p_train.add_argument("--summary-json", type=str, required=True)
    p_train.add_argument("--manifest", type=str, default=None)
    p_train.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args()
    manifest = _load_json(args.manifest) if getattr(args, "manifest", None) else None

    if args.command == "merge":
        payload = merge_world_model_datasets(
            sources=_parse_source_specs(args.source),
            output_npz=args.output_npz,
            output_json=args.output_json,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.command == "check-dataset":
        payload = evaluate_dataset_readiness(
            npz_path=args.npz,
            meta_json=args.meta_json,
            manifest=manifest,
        )
    else:
        payload = evaluate_training_readiness(
            summary_json=args.summary_json,
            manifest=manifest,
        )

    if args.output_json:
        _write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
