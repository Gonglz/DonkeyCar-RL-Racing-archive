#!/usr/bin/env python3
"""
Obstacle context estimation from ego-available perception.

V1 goal:
  - Keep the current V16 policy/state interface unchanged (12D state)
  - Replace runtime truth obstacle context with a perception-side estimate
  - Use only semantic image channels already available to the policy

This first version is intentionally lightweight and heuristic:
  - ch4: vehicle_prob
  - ch5: motion_residual

Later, a learned estimator can plug into the same interface.
"""

from __future__ import annotations

import math
import os
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None


@dataclass
class ObstacleContextEstimate:
    present: float = 0.0
    longitudinal: float = 0.0
    lateral: float = 0.0
    dist: float = 0.0
    risk: float = 0.0
    confidence: float = 0.0
    age: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None

    def to_info_dict(self) -> Dict[str, Any]:
        return {
            "obstacle_present": float(self.present),
            "obstacle_longitudinal": float(self.longitudinal),
            "obstacle_lateral": float(self.lateral),
            "obstacle_dist": float(self.dist),
            "obstacle_risk": float(self.risk),
        }

    def to_debug_info_dict(self) -> Dict[str, Any]:
        return self.to_prefixed_debug_info("obstacle_cv")

    def to_prefixed_debug_info(self, prefix: str) -> Dict[str, Any]:
        prefix = str(prefix or "obstacle_ctx").strip()
        return {
            f"{prefix}_present": float(self.present),
            f"{prefix}_longitudinal": float(self.longitudinal),
            f"{prefix}_lateral": float(self.lateral),
            f"{prefix}_dist": float(self.dist),
            f"{prefix}_risk": float(self.risk),
            f"{prefix}_confidence": float(self.confidence),
            f"{prefix}_age": int(self.age),
            f"{prefix}_bbox": (
                None if self.bbox is None else [int(x) for x in self.bbox]
            ),
        }


class CVObstacleContextEstimatorV1:
    """
    First-pass self-perception obstacle context estimator.

    Inputs:
      - 6-channel semantic image already available to the policy
      - optional ego info (currently used only for future extensibility)

    Outputs:
      - obstacle_present / longitudinal / lateral / dist / risk

    Notes:
      - This is a heuristic bootstrap version, not a learned model yet.
      - It intentionally keeps the interface stable so a learned estimator can
        replace it later without touching PPO or observation wiring.
    """

    def __init__(
        self,
        vehicle_channel_idx: int = 4,
        motion_channel_idx: int = 5,
        white_channel_idx: int = 1,
        yellow_channel_idx: int = 2,
        edge_channel_idx: int = 3,
        vehicle_thresh: float = 0.16,
        vehicle_thresh_lo: float = 0.08,
        vehicle_thresh_strict: float = 0.28,
        motion_thresh: float = 0.05,
        min_area_px: int = 14,
        hold_steps: int = 4,
        camera_half_fov_deg: float = 32.0,
        ema_alpha: float = 0.35,
        fuse_motion_weight: float = 0.28,
        center_prior_sigma: float = 1.10,
    ):
        self.vehicle_channel_idx = int(max(0, vehicle_channel_idx))
        self.motion_channel_idx = int(max(0, motion_channel_idx))
        self.white_channel_idx = int(max(0, white_channel_idx))
        self.yellow_channel_idx = int(max(0, yellow_channel_idx))
        self.edge_channel_idx = int(max(0, edge_channel_idx))
        self.vehicle_thresh = float(np.clip(vehicle_thresh, 0.05, 0.95))
        self.vehicle_thresh_lo = float(np.clip(vehicle_thresh_lo, 0.02, self.vehicle_thresh))
        self.vehicle_thresh_strict = float(
            np.clip(vehicle_thresh_strict, self.vehicle_thresh + 1e-3, 0.98)
        )
        self.motion_thresh = float(np.clip(motion_thresh, 0.0, 0.95))
        self.min_area_px = int(max(1, min_area_px))
        self.hold_steps = int(max(0, hold_steps))
        self.camera_half_fov_rad = math.radians(float(np.clip(camera_half_fov_deg, 5.0, 80.0)))
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.fuse_motion_weight = float(np.clip(fuse_motion_weight, 0.0, 0.9))
        self.center_prior_sigma = float(np.clip(center_prior_sigma, 0.15, 1.5))

        self._last_estimate = ObstacleContextEstimate()
        self._missed_steps = 0

    def reset(self) -> None:
        self._last_estimate = ObstacleContextEstimate()
        self._missed_steps = 0

    @staticmethod
    def _compute_obstacle_risk(longitudinal: float, lateral: float, planar_distance: float) -> float:
        if not np.isfinite(planar_distance) or planar_distance <= 0.0:
            return 0.0
        if longitudinal < -0.4:
            return 0.0

        front_gate = float(np.clip((4.0 - max(longitudinal, 0.0)) / 3.5, 0.0, 1.0))
        planar_gate = float(np.clip((5.0 - planar_distance) / 4.5, 0.0, 1.0))
        lateral_gate = float(np.clip(1.0 - abs(lateral) / 1.25, 0.0, 1.0))
        risk = (front_gate ** 2.0) * max(0.20, lateral_gate) * planar_gate
        return float(np.clip(risk, 0.0, 1.0))

    @staticmethod
    def _bbox_center(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[float, float]]:
        if bbox is None:
            return None
        x, y, w, h = bbox
        return float(x + 0.5 * w), float(y + 0.5 * h)

    def _extract_candidate(
        self,
        image_chw: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[ObstacleContextEstimate]:
        if image_chw.ndim != 3 or image_chw.shape[0] <= self.vehicle_channel_idx:
            return None

        veh_prob = np.clip(
            np.asarray(image_chw[self.vehicle_channel_idx], dtype=np.float32), 0.0, 1.0
        )
        motion = np.zeros_like(veh_prob, dtype=np.float32)
        if image_chw.shape[0] > self.motion_channel_idx:
            motion = np.clip(
                np.asarray(image_chw[self.motion_channel_idx], dtype=np.float32), 0.0, 1.0
            )
        white_prob = np.zeros_like(veh_prob, dtype=np.float32)
        if image_chw.shape[0] > self.white_channel_idx:
            white_prob = np.clip(
                np.asarray(image_chw[self.white_channel_idx], dtype=np.float32), 0.0, 1.0
            )
        yellow_prob = np.zeros_like(veh_prob, dtype=np.float32)
        if image_chw.shape[0] > self.yellow_channel_idx:
            yellow_prob = np.clip(
                np.asarray(image_chw[self.yellow_channel_idx], dtype=np.float32), 0.0, 1.0
            )
        edge_prob = np.zeros_like(veh_prob, dtype=np.float32)
        if image_chw.shape[0] > self.edge_channel_idx:
            edge_prob = np.clip(
                np.asarray(image_chw[self.edge_channel_idx], dtype=np.float32), 0.0, 1.0
            )

        fused = np.clip(
            (1.0 - self.fuse_motion_weight) * veh_prob + self.fuse_motion_weight * motion,
            0.0,
            1.0,
        )
        line_strength = np.maximum(np.maximum(white_prob, yellow_prob), 0.85 * edge_prob)
        saliency = np.maximum(veh_prob, fused)
        adaptive_thresh = self.vehicle_thresh
        strong_values = saliency[saliency >= self.vehicle_thresh_lo]
        if strong_values.size >= 16:
            adaptive_thresh = float(
                np.clip(
                    np.percentile(strong_values, 78) * 0.64,
                    self.vehicle_thresh_lo,
                    0.42,
                )
            )

        base_mask = (
            (veh_prob >= max(self.vehicle_thresh_lo, adaptive_thresh))
            | (fused >= adaptive_thresh)
        ).astype(np.uint8) * 255
        assist_mask = (
            (veh_prob >= max(self.vehicle_thresh_lo * 0.9, 0.04))
            & (motion >= self.motion_thresh)
        ).astype(np.uint8) * 255
        strict_mask = (
            (veh_prob >= max(self.vehicle_thresh, adaptive_thresh + 0.05))
            | (fused >= min(0.95, adaptive_thresh + 0.08))
        ).astype(np.uint8) * 255
        top_cut = int(0.18 * veh_prob.shape[0])
        if top_cut > 0:
            base_mask[:top_cut, :] = 0
            assist_mask[:top_cut, :] = 0
            strict_mask[:top_cut, :] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        base_mask = cv2.bitwise_or(base_mask, assist_mask)
        base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        strict_mask = cv2.dilate(strict_mask, kernel, iterations=1)
        mask = cv2.bitwise_and(base_mask, cv2.max(base_mask, strict_mask))

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return None

        h_img, w_img = veh_prob.shape[:2]
        best_score = -1.0
        best_est: Optional[ObstacleContextEstimate] = None
        prev_center = self._bbox_center(self._last_estimate.bbox)

        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < self.min_area_px:
                continue

            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[idx]

            comp_mask = labels == idx
            mean_prob = float(np.mean(veh_prob[comp_mask])) if np.any(comp_mask) else 0.0
            max_prob = float(np.max(veh_prob[comp_mask])) if np.any(comp_mask) else 0.0
            mean_motion = float(np.mean(motion[comp_mask])) if np.any(comp_mask) else 0.0
            mean_fused = float(np.mean(fused[comp_mask])) if np.any(comp_mask) else 0.0
            mean_white = float(np.mean(white_prob[comp_mask])) if np.any(comp_mask) else 0.0
            mean_yellow = float(np.mean(yellow_prob[comp_mask])) if np.any(comp_mask) else 0.0
            mean_edge = float(np.mean(edge_prob[comp_mask])) if np.any(comp_mask) else 0.0
            veh_support_ratio = float(np.mean(veh_prob[comp_mask] >= self.vehicle_thresh_lo)) if np.any(comp_mask) else 0.0

            x_norm = ((float(cx) / max(float(w_img - 1), 1.0)) - 0.5) * 2.0
            bottom_norm = float((y + h) / max(h_img, 1))
            height_norm = float(h / max(h_img, 1))
            area_norm = float(area / max(h_img * w_img, 1))
            width_norm = float(w / max(w_img, 1))
            aspect_ratio = float(h / max(float(w), 1.0))
            center_prior = math.exp(
                -0.5 * (x_norm / max(self.center_prior_sigma, 1e-3)) ** 2
            )
            bottom_prior = float(np.clip((bottom_norm - 0.10) / 0.90, 0.0, 1.0))
            temporal_prior = 0.0
            if prev_center is not None:
                prev_cx, prev_cy = prev_center
                dx = (float(cx) - prev_cx) / max(float(w_img), 1.0)
                dy = (float(cy) - prev_cy) / max(float(h_img), 1.0)
                temporal_prior = math.exp(-8.0 * (dx * dx + dy * dy))

            line_like = max(mean_white, mean_yellow)
            weak_vehicle = (
                max_prob < max(self.vehicle_thresh_lo * 1.15, 0.10)
                and mean_prob < max(self.vehicle_thresh_lo * 0.95, 0.07)
                and veh_support_ratio < 0.16
            )
            edge_line_artifact = (
                line_like > 0.10
                and mean_edge > 0.08
                and aspect_ratio > 1.7
                and width_norm < 0.14
            )
            if weak_vehicle and temporal_prior < 0.58:
                continue
            if edge_line_artifact and mean_prob < 0.11 and temporal_prior < 0.68:
                continue

            confidence = float(
                np.clip(
                    0.40 * mean_prob
                    + 0.24 * max_prob
                    + 0.14 * mean_motion
                    + 0.14 * mean_fused
                    + 0.08 * center_prior
                    + 0.12 * bottom_prior
                    - 0.08 * line_like
                    - 0.06 * mean_edge,
                    0.0,
                    1.0,
                )
            )
            score = float(
                confidence
                + 0.22 * bottom_prior
                + 0.14 * temporal_prior
                + 0.05 * np.clip(width_norm * 10.0, 0.0, 1.0)
                + 0.04 * center_prior
                - 0.06 * line_like
                - 0.05 * mean_edge
            )

            # Monocular pseudo-distance from image geometry.
            bottom_term = 0.70 + 6.50 * (1.0 - np.clip(bottom_norm, 0.0, 1.0))
            height_term = 0.55 / max(height_norm, 0.025)
            area_term = 0.20 / max(math.sqrt(max(area_norm, 1e-4)), 1e-3)
            dist = float(np.clip(0.55 * bottom_term + 0.30 * height_term + 0.15 * area_term, 0.6, 8.0))

            angle = float(x_norm) * self.camera_half_fov_rad
            lateral = float(np.clip(dist * math.tan(angle), -2.5, 2.5))
            longitudinal = float(max(0.0, math.sqrt(max(dist * dist - lateral * lateral, 0.0))))
            risk = self._compute_obstacle_risk(longitudinal, lateral, dist)
            present = float(
                np.clip(
                    0.94 * confidence
                    + 0.12 * temporal_prior
                    + 0.12 * veh_support_ratio
                    - 0.10 * line_like
                    - 0.06 * mean_edge,
                    0.0,
                    1.0,
                )
            )

            est = ObstacleContextEstimate(
                present=present,
                longitudinal=longitudinal,
                lateral=lateral,
                dist=dist,
                risk=risk,
                confidence=confidence,
                age=0,
                bbox=(x, y, w, h),
            )
            if score > best_score:
                best_score = score
                best_est = est

        return best_est

    def _smooth(self, est: ObstacleContextEstimate) -> ObstacleContextEstimate:
        prev = self._last_estimate
        if prev.present <= 0.2:
            return est

        a = float(np.clip(self.ema_alpha, 0.0, 1.0))
        est.longitudinal = (1.0 - a) * prev.longitudinal + a * est.longitudinal
        est.lateral = (1.0 - a) * prev.lateral + a * est.lateral
        est.dist = (1.0 - a) * prev.dist + a * est.dist
        est.risk = (1.0 - a) * prev.risk + a * est.risk
        est.present = (1.0 - a) * prev.present + a * est.present
        est.confidence = max(est.confidence, 0.5 * prev.confidence)
        return est

    def estimate(
        self,
        image_chw: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        state7: Optional[np.ndarray] = None,
    ) -> ObstacleContextEstimate:
        _ = info  # reserved for future learned estimator / ego-state-aware variants
        _ = state7
        candidate = self._extract_candidate(image_chw, info=info)

        if candidate is not None:
            candidate = self._smooth(candidate)
            self._last_estimate = candidate
            self._missed_steps = 0
            return candidate

        self._missed_steps += 1
        if self._last_estimate.present > 0.2 and self._missed_steps <= self.hold_steps:
            decay = float(max(0.0, 1.0 - 0.18 * self._missed_steps))
            held = ObstacleContextEstimate(
                present=float(np.clip(self._last_estimate.present * decay, 0.0, 1.0)),
                longitudinal=float(self._last_estimate.longitudinal),
                lateral=float(self._last_estimate.lateral),
                dist=float(self._last_estimate.dist),
                risk=float(self._last_estimate.risk * decay),
                confidence=float(self._last_estimate.confidence * decay),
                age=int(self._missed_steps),
                bbox=self._last_estimate.bbox,
            )
            self._last_estimate = held
            return held

        self._last_estimate = ObstacleContextEstimate()
        return self._last_estimate


class LearnedObstacleContextEstimatorV1:
    """
    Online temporal learned obstacle-context estimator.

    Uses the same obstacle-5 interface as runtime/CV:
      - obstacle_present
      - obstacle_longitudinal
      - obstacle_lateral
      - obstacle_dist
      - obstacle_risk
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        seq_len: int = 16,
        present_threshold: float = 0.5,
        present_off_threshold: Optional[float] = None,
        activation_consecutive: int = 3,
        deactivation_consecutive: int = 2,
    ):
        if torch is None:
            raise ImportError("torch is required for learned_v1 obstacle context")
        ckpt = os.path.abspath(os.path.expanduser(str(checkpoint_path or "")))
        if not ckpt or not os.path.isfile(ckpt):
            raise FileNotFoundError(f"learned obstacle checkpoint not found: {checkpoint_path}")

        from .obstacle_context_learned import (
            ObstacleContextTemporalNet,
            compute_obstacle_risk,
        )

        self.compute_obstacle_risk = compute_obstacle_risk
        self.device = torch.device(str(device or "cpu"))
        self.present_threshold = float(np.clip(present_threshold, 0.01, 0.99))
        if present_off_threshold is None:
            present_off_threshold = max(0.01, self.present_threshold - 0.10)
        self.present_off_threshold = float(
            np.clip(present_off_threshold, 0.01, self.present_threshold)
        )
        self.seq_len = int(max(1, seq_len))
        self.activation_consecutive = int(max(1, activation_consecutive))
        self.deactivation_consecutive = int(max(1, deactivation_consecutive))

        summary_path = os.path.join(os.path.dirname(ckpt), "train_summary.json")
        if os.path.isfile(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                self.seq_len = int(max(1, summary.get("seq_len", self.seq_len)))
            except Exception:
                pass

        self.model = ObstacleContextTemporalNet()
        raw_ckpt = torch.load(ckpt, map_location=self.device)
        state_dict = raw_ckpt
        if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
            state_dict = raw_ckpt["model_state_dict"]
            try:
                self.seq_len = int(max(1, raw_ckpt.get("seq_len", self.seq_len)))
            except Exception:
                pass
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._image_seq: deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._state_seq: deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._active = False
        self._pos_streak = 0
        self._neg_streak = 0

    def reset(self) -> None:
        self._image_seq.clear()
        self._state_seq.clear()
        self._active = False
        self._pos_streak = 0
        self._neg_streak = 0

    def estimate(
        self,
        image_chw: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        state7: Optional[np.ndarray] = None,
    ) -> ObstacleContextEstimate:
        _ = info
        img = np.asarray(image_chw, dtype=np.float32)
        if img.ndim != 3:
            return ObstacleContextEstimate()
        if state7 is None:
            state7_arr = np.zeros((7,), dtype=np.float32)
        else:
            state7_arr = np.asarray(state7, dtype=np.float32).reshape(-1)
            if state7_arr.shape[0] != 7:
                state7_arr = np.zeros((7,), dtype=np.float32)

        self._image_seq.append(img)
        self._state_seq.append(state7_arr)

        img_seq = np.stack(list(self._image_seq), axis=0)
        state_seq = np.stack(list(self._state_seq), axis=0)
        length = int(img_seq.shape[0])

        with torch.no_grad():
            image_t = torch.from_numpy(img_seq[None, ...]).float().to(self.device)
            state_t = torch.from_numpy(state_seq[None, ...]).float().to(self.device)
            length_t = torch.tensor([length], dtype=torch.long, device=self.device)
            out = self.model.predict(image_seq=image_t, state7_seq=state_t, lengths=length_t)

        present_prob = float(out["present"].detach().cpu().numpy().reshape(-1)[0])
        longitudinal = float(out["longitudinal"].detach().cpu().numpy().reshape(-1)[0])
        lateral = float(out["lateral"].detach().cpu().numpy().reshape(-1)[0])
        dist = float(out["dist"].detach().cpu().numpy().reshape(-1)[0])
        if present_prob >= self.present_threshold:
            self._pos_streak += 1
        else:
            self._pos_streak = 0

        if present_prob <= self.present_off_threshold:
            self._neg_streak += 1
        else:
            self._neg_streak = 0

        if not self._active and self._pos_streak >= self.activation_consecutive:
            self._active = True
        if self._active and self._neg_streak >= self.deactivation_consecutive:
            self._active = False

        if not self._active:
            return ObstacleContextEstimate(
                present=0.0,
                longitudinal=0.0,
                lateral=0.0,
                dist=0.0,
                risk=0.0,
                confidence=present_prob,
                age=max(length, self._pos_streak),
                bbox=None,
            )
        risk = float(self.compute_obstacle_risk(longitudinal, lateral, dist))
        return ObstacleContextEstimate(
            present=1.0,
            longitudinal=longitudinal,
            lateral=lateral,
            dist=dist,
            risk=risk,
            confidence=present_prob,
            age=length,
            bbox=None,
        )
