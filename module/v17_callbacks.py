"""
module/v17_callbacks.py

V17-specific callbacks.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    var_y = float(np.var(y_true))
    if var_y < 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


class CriticCalibrationCallback(BaseCallback):
    """
    Online critic calibration using the current rollout buffer.

    This avoids opening a second simulator connection while still providing the
    mechanical trigger required by V17 dual value heads.
    """

    def __init__(
        self,
        eval_freq_timesteps: int = 50_000,
        mae_ratio_threshold: float = 2.0,
        adv_std_ratio_threshold: float = 1.5,
        ev_gap_threshold: float = 0.20,
        ev_worse_threshold: float = 0.45,
        consecutive_required: int = 2,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq_timesteps = int(max(1, eval_freq_timesteps))
        self.mae_ratio_threshold = float(max(1.0, mae_ratio_threshold))
        self.adv_std_ratio_threshold = float(max(1.0, adv_std_ratio_threshold))
        self.ev_gap_threshold = float(max(0.0, ev_gap_threshold))
        self.ev_worse_threshold = float(ev_worse_threshold)
        self.consecutive_required = int(max(1, consecutive_required))
        self._last_eval_t = 0
        self._consecutive_hits = 0
        self._last_metrics: Dict[str, float] = {}

    @staticmethod
    def _domain_mask(domain_id: np.ndarray, is_gt: bool) -> np.ndarray:
        domain = np.asarray(domain_id, dtype=np.float32).reshape(-1)
        return (domain > 0.5) if is_gt else (domain <= 0.5)

    def _collect_metrics(self) -> Dict[str, float]:
        rb = getattr(self.model, "rollout_buffer", None)
        if rb is None:
            return {}
        observations = getattr(rb, "observations", None)
        if not isinstance(observations, dict) or "domain_id" not in observations:
            return {}

        domain_id = np.asarray(observations["domain_id"], dtype=np.float32).reshape(-1)
        values = np.asarray(getattr(rb, "values", None), dtype=np.float32).reshape(-1)
        returns = np.asarray(getattr(rb, "returns", None), dtype=np.float32).reshape(-1)
        advantages = np.asarray(getattr(rb, "advantages", None), dtype=np.float32).reshape(-1)
        if values.size == 0 or returns.size == 0 or advantages.size == 0:
            return {}

        metrics: Dict[str, float] = {}
        per_domain: Dict[str, Dict[str, float]] = {}
        for name, is_gt in (("ws", False), ("gt", True)):
            mask = self._domain_mask(domain_id, is_gt=is_gt)
            if not np.any(mask):
                continue
            val = values[mask]
            ret = returns[mask]
            adv = advantages[mask]
            per_domain[name] = {
                "mae": float(np.mean(np.abs(val - ret))),
                "bias": float(abs(np.mean(val - ret))),
                "adv_std": float(np.std(adv)),
                "ev": _explained_variance(val, ret),
                "count": float(mask.sum()),
            }
        if "ws" not in per_domain or "gt" not in per_domain:
            return {}

        mae_ws = per_domain["ws"]["mae"]
        mae_gt = per_domain["gt"]["mae"]
        adv_std_ws = per_domain["ws"]["adv_std"]
        adv_std_gt = per_domain["gt"]["adv_std"]
        ev_ws = per_domain["ws"]["ev"]
        ev_gt = per_domain["gt"]["ev"]

        metrics.update(
            {
                "ws_mae": mae_ws,
                "gt_mae": mae_gt,
                "ws_bias": per_domain["ws"]["bias"],
                "gt_bias": per_domain["gt"]["bias"],
                "ws_adv_std": adv_std_ws,
                "gt_adv_std": adv_std_gt,
                "ws_ev": ev_ws,
                "gt_ev": ev_gt,
                "mae_ratio": float(max(mae_ws, mae_gt) / max(min(mae_ws, mae_gt), 1e-6)),
                "adv_std_ratio": float(max(adv_std_ws, adv_std_gt) / max(min(adv_std_ws, adv_std_gt), 1e-6)),
                "ev_gap": float(abs(ev_ws - ev_gt)),
                "ev_worse": float(min(ev_ws, ev_gt)),
                "ws_count": per_domain["ws"]["count"],
                "gt_count": per_domain["gt"]["count"],
            }
        )
        return metrics

    def _maybe_activate_dual_heads(self, metrics: Dict[str, float]) -> bool:
        trigger = bool(
            metrics.get("mae_ratio", 0.0) >= self.mae_ratio_threshold
            and metrics.get("adv_std_ratio", 0.0) >= self.adv_std_ratio_threshold
            and metrics.get("ev_gap", 0.0) >= self.ev_gap_threshold
            and metrics.get("ev_worse", 1.0) <= self.ev_worse_threshold
        )
        if trigger:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        policy = getattr(self.model, "policy", None)
        if (
            trigger
            and self._consecutive_hits >= self.consecutive_required
            and hasattr(policy, "activate_dual_value_heads")
            and (not bool(getattr(policy, "use_dual_value_heads", False)))
        ):
            policy.activate_dual_value_heads()
            if self.verbose > 0:
                print(
                    "🧭 critic dual heads activated: "
                    f"mae_ratio={metrics.get('mae_ratio', 0.0):.3f}, "
                    f"adv_std_ratio={metrics.get('adv_std_ratio', 0.0):.3f}, "
                    f"ev_gap={metrics.get('ev_gap', 0.0):.3f}, "
                    f"ev_worse={metrics.get('ev_worse', 0.0):.3f}"
                )
            return True
        return False

    def _log_metrics(self, metrics: Dict[str, float], activated: bool) -> None:
        for key, value in metrics.items():
            self.logger.record(f"critic_calib/{key}", float(value))
        self.logger.record("critic_calib/consecutive_hits", float(self._consecutive_hits))
        self.logger.record("critic_calib/dual_heads_active", float(bool(getattr(self.model.policy, "use_dual_value_heads", False))))
        self.logger.record("critic_calib/dual_heads_activated_now", float(bool(activated)))

    def _on_rollout_end(self) -> None:
        if (self.num_timesteps - self._last_eval_t) < self.eval_freq_timesteps:
            return
        metrics = self._collect_metrics()
        if not metrics:
            return
        activated = self._maybe_activate_dual_heads(metrics)
        self._log_metrics(metrics, activated=activated)
        self._last_metrics = dict(metrics)
        self._last_eval_t = int(self.num_timesteps)

    def _on_step(self) -> bool:
        return True

    def summary(self) -> Dict[str, Any]:
        return {
            "last_metrics": dict(self._last_metrics),
            "consecutive_hits": int(self._consecutive_hits),
            "dual_heads_active": bool(getattr(getattr(self, "model", None), "policy", None) and getattr(self.model.policy, "use_dual_value_heads", False)),
        }
