"""
module/obv.py
V13/V17 观测空间：6 通道语义图像 + 5/7 维纯传感器状态。

观测结构
--------
image: (6, obs_size, obs_size) float32 [0, 1]
  ch0  raw_Y               原图 Y 亮度通道
  ch1  edge_line_prob      车边线软概率（跨域 canonical 语义）
  ch2  guide_line_prob     中线/引导线软概率（跨域 canonical 语义，渐进 dropout）
  ch3  sobel_edge          Sobel 边缘图（raw_Y）
  ch4  vehicle_prob        rawpink 障碍车颜色软概率（宽松固定粉色）
  ch5  motion_residual     |Y_t - Y_{t-1}|，reset 后置零

state: (5,) float32
  v_long_norm      clip(speed / v_max, 0, 2)
  yaw_rate_norm    clip(gyro_y / 8.0, -2, 2)   ← Unity Y-up: yaw=gyro[1], /8 dampened
  accel_x_norm     clip(accel_x / 9.8, -2, 2)   ← 纵向加速度（上线前需 sanity check）
  prev_steer       ∈ [-1, 1]
  prev_throttle    ∈ [-1, 1]

设计原则
--------
- ch1/ch2 直接由 original 提取线概率；对 WS 会做一次通道对齐，
  使 ch1 始终表示车边线、ch2 始终表示中线/引导线
- 软概率（Gaussian blur）而非硬二值，通道 dropout 提升鲁棒性
- motion_residual 替代帧堆叠；配合 RecurrentPPO LSTM 处理长时依赖
- 状态向量完全来自传感器 info，无 TrackGeometryManager 依赖

RGB/BGR 约定
------------
- DonkeyEnv 输出 RGB
- rawpink/obstacle-color detector 与线提取器期望 BGR
- cv2.resize 参数顺序：(width, height)，即 (W, H)
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import gym
import numpy as np


# 兼容保留：历史代码会从 obv import _CANONICAL_TGT_STATS。
# 当前 V13 观测已不依赖 canonical 路径。
_CANONICAL_TGT_STATS: Dict[str, Any] = {
    "prototypes_hsv": {
        "blue": [112.0, 125.0, 94.0],
        "yellow": [13.0, 152.0, 146.0],
        "white": [0.0, 0.0, 200.0],
    },
    "centerline_hsv": [13.0, 152.0, 146.0],
}


# ============================================================
# CanonicalSemanticWrapper
# ============================================================
class CanonicalSemanticWrapper(gym.ObservationWrapper):
    """
    V13 专用：6 通道观测 = original 辅助流 + original 语义流。

    通道布局 (6, obs_size, obs_size) float32 [0,1]:
      ch0  raw_Y               原图 Y 亮度通道
      ch1  edge_line_prob      车边线软概率（跨域 canonical 语义）
      ch2  guide_line_prob     中线/引导线软概率（跨域 canonical 语义，渐进 dropout）
      ch3  sobel_edge          Sobel 边缘图（raw_Y）
      ch4  vehicle_prob        rawpink 障碍车颜色软概率（宽松固定粉色）
      ch5  motion_residual     |Y_t - Y_{t-1}|，reset 后清零
    """

    def __init__(
        self,
        env,
        domain: str = "ws",
        obs_size: int = 128,
        augment: bool = False,
        dropout_start_step: int = 0,
        dropout_ramp_steps: int = 200_000,
        dropout_max_prob: float = 0.20,
        edge_preblur_sigma: float = 1.0,
        edge_support_thresh: float = 0.035,
        edge_support_dilate: int = 1,
        edge_comp_min_area: int = 10,
        edge_comp_min_height: int = 3,
        edge_comp_max_compactness: float = 0.80,
        edge_comp_aspect_ratio: float = 1.3,
        line_comp_min_area: int = 8,
        line_comp_min_height: int = 2,
        line_comp_max_compactness: float = 0.72,
        line_comp_tall_ratio: float = 1.8,
        line_edge_filter_floor: float = 0.0,
        line_edge_boost_gain: float = 0.8,
        line_edge_min_strength: float = 0.06,
        line_edge_strength_power: float = 1.4,
        line_edge_hard_floor: float = 0.25,
        line_close_kernel: int = 1,
        line_close_iter: int = 0,
        ws_white_simple: Optional[bool] = None,
        ws_white_edge_soft_floor: float = 0.30,
        ws_white_edge_soft_far_boost: float = 0.22,
        ws_white_edge_support_thresh: float = 0.020,
        ws_white_edge_min_strength: float = 0.030,
        ws_white_edge_strength_power: float = 1.00,
        ws_white_close_kernel: int = 3,
        ws_white_close_iter: int = 1,
        ws_white_skip_prob_min: bool = True,
        ws_white_dash_close_h: int = 9,
        ws_white_dash_close_w: int = 3,
        ws_white_dash_min_height: int = 6,
        ws_white_dash_top_min_height: int = 4,
        ws_white_dash_max_width: int = 20,
        ws_white_dash_fill_weight: float = 0.45,
        ws_white_loose_s_max: int = 68,
        ws_white_loose_v_min: int = 158,
        ws_white_loose_weight: float = 0.55,
        ws_white_loose_edge_min: float = 0.06,
        ws_white_corridor_near: int = 9,
        ws_white_corridor_far: int = 18,
        ws_white_edge_grow_weight: float = 0.45,
        ws_white_temporal_blend: float = 0.18,
        ws_white_temporal_dilate: int = 1,
        ws_white_temporal_motion_scale: float = 0.35,
        line_prob_min: float = 0.001,
    ):
        super().__init__(env)
        self._domain = str(domain).lower()
        self.obs_size = int(obs_size)
        self.H = self.obs_size
        self.W = self.obs_size
        self.augment = bool(augment)
        self.dropout_start_step = int(max(0, dropout_start_step))
        self.dropout_ramp_steps = int(max(1, dropout_ramp_steps))
        self.dropout_max_prob = float(np.clip(dropout_max_prob, 0.0, 1.0))
        self.edge_preblur_sigma = float(np.clip(edge_preblur_sigma, 0.0, 5.0))
        self.edge_support_thresh = float(np.clip(edge_support_thresh, 0.0, 1.0))
        self.edge_support_dilate = int(max(0, edge_support_dilate))
        self.edge_comp_min_area = int(max(1, edge_comp_min_area))
        self.edge_comp_min_height = int(max(1, edge_comp_min_height))
        self.edge_comp_max_compactness = float(np.clip(edge_comp_max_compactness, 0.05, 1.0))
        self.edge_comp_aspect_ratio = float(max(1.0, edge_comp_aspect_ratio))
        self.line_comp_min_area = int(max(1, line_comp_min_area))
        self.line_comp_min_height = int(max(1, line_comp_min_height))
        self.line_comp_max_compactness = float(np.clip(line_comp_max_compactness, 0.05, 1.0))
        self.line_comp_tall_ratio = float(max(1.0, line_comp_tall_ratio))
        self.line_edge_filter_floor = float(np.clip(line_edge_filter_floor, 0.0, 1.0))
        self.line_edge_boost_gain = float(np.clip(line_edge_boost_gain, 0.0, 3.0))
        self.line_edge_min_strength = float(np.clip(line_edge_min_strength, 0.0, 0.8))
        self.line_edge_strength_power = float(np.clip(line_edge_strength_power, 0.5, 4.0))
        self.line_edge_hard_floor = float(np.clip(line_edge_hard_floor, 0.0, 1.0))
        self.line_close_kernel = int(max(1, line_close_kernel))
        self.line_close_iter = int(max(0, line_close_iter))
        # WS white/yellow extraction is fully delegated to WS2NewTrack.
        # Keep ws_white_simple argument only for backward compatibility.
        self.ws_white_simple = (self._domain == "ws")
        self.ws_white_edge_soft_floor = float(np.clip(ws_white_edge_soft_floor, 0.0, 1.0))
        self.ws_white_edge_soft_far_boost = float(np.clip(ws_white_edge_soft_far_boost, 0.0, 0.9))
        self.ws_white_edge_support_thresh = float(np.clip(ws_white_edge_support_thresh, 0.0, 1.0))
        self.ws_white_edge_min_strength = float(np.clip(ws_white_edge_min_strength, 0.0, 0.8))
        self.ws_white_edge_strength_power = float(np.clip(ws_white_edge_strength_power, 0.5, 4.0))
        self.ws_white_close_kernel = int(max(1, ws_white_close_kernel))
        self.ws_white_close_iter = int(max(0, ws_white_close_iter))
        self.ws_white_skip_prob_min = bool(ws_white_skip_prob_min)
        self.ws_white_dash_close_h = int(max(1, ws_white_dash_close_h))
        self.ws_white_dash_close_w = int(max(1, ws_white_dash_close_w))
        self.ws_white_dash_min_height = int(max(1, ws_white_dash_min_height))
        self.ws_white_dash_top_min_height = int(max(1, ws_white_dash_top_min_height))
        self.ws_white_dash_max_width = int(max(2, ws_white_dash_max_width))
        self.ws_white_dash_fill_weight = float(np.clip(ws_white_dash_fill_weight, 0.0, 1.0))
        self.ws_white_loose_s_max = int(np.clip(ws_white_loose_s_max, 10, 160))
        self.ws_white_loose_v_min = int(np.clip(ws_white_loose_v_min, 80, 255))
        self.ws_white_loose_weight = float(np.clip(ws_white_loose_weight, 0.0, 1.0))
        self.ws_white_loose_edge_min = float(np.clip(ws_white_loose_edge_min, 0.0, 0.6))
        self.ws_white_corridor_near = int(max(2, ws_white_corridor_near))
        self.ws_white_corridor_far = int(max(self.ws_white_corridor_near, ws_white_corridor_far))
        self.ws_white_edge_grow_weight = float(np.clip(ws_white_edge_grow_weight, 0.0, 1.0))
        self.ws_white_temporal_blend = float(np.clip(ws_white_temporal_blend, 0.0, 1.0))
        self.ws_white_temporal_dilate = int(max(0, ws_white_temporal_dilate))
        self.ws_white_temporal_motion_scale = float(np.clip(ws_white_temporal_motion_scale, 0.05, 2.0))
        self.line_prob_min = float(np.clip(line_prob_min, 0.0, 0.2))

        self._step = 0
        self._prev_y: Optional[np.ndarray] = None
        self._prev_ws_white_prob: Optional[np.ndarray] = None

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(6, self.obs_size, self.obs_size),
            dtype=np.float32,
        )
        print(
            f"✅ CanonicalSemanticWrapper [{domain}]: "
            f"(6,{obs_size},{obs_size}) float32, "
            f"dropout_max={dropout_max_prob:.2f} ramp={dropout_ramp_steps}, "
            f"edge_blur={self.edge_preblur_sigma:.2f}, "
            f"edge_support>{self.edge_support_thresh:.3f}, "
            f"line_edge=floor{self.line_edge_filter_floor:.2f}+"
            f"boost{self.line_edge_boost_gain:.2f}, "
            f"edge_min={self.line_edge_min_strength:.3f}, "
            f"edge_pow={self.line_edge_strength_power:.2f}, "
            f"hard_floor={self.line_edge_hard_floor:.2f}, "
            f"close=k{self.line_close_kernel}/i{self.line_close_iter}, "
            f"ws_white_simple={int(self.ws_white_simple)}, "
            f"ws_w_floor={self.ws_white_edge_soft_floor:.2f}+{self.ws_white_edge_soft_far_boost:.2f}, "
            f"ws_e_thr={self.ws_white_edge_support_thresh:.3f}, "
            f"line_prob_min={self.line_prob_min:.4f}"
        )

    # ------------------------------------------------------------------
    # Dropout 渐进 schedule
    # ------------------------------------------------------------------
    def _get_yellow_dropout(self) -> float:
        """渐进式 dropout 概率：step < start → 0；线性增至 dropout_max_prob。"""
        if self._step < self.dropout_start_step:
            return 0.0
        ramp = float(self._step - self.dropout_start_step) / self.dropout_ramp_steps
        return float(np.clip(ramp * self.dropout_max_prob, 0.0, self.dropout_max_prob))

    @staticmethod
    def _extract_generic_lane_masks(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Domain fallback: direct HSV threshold for white/yellow."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 180], np.uint8), np.array([179, 70, 255], np.uint8)) > 0
        yellow = cv2.inRange(hsv, np.array([10, 55, 70], np.uint8), np.array([45, 255, 255], np.uint8)) > 0
        yellow &= ~white
        return white.astype(np.float32), yellow.astype(np.float32)

    def _extract_raw_lane_masks(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract white/yellow line masks directly from original frame."""
        if self._domain == "gt":
            from .GT2NewTrack import detect_gt_road_support, detect_white_line, detect_yellow_line

            road = detect_gt_road_support(img_bgr)
            white = (detect_white_line(img_bgr, road_support=road, edge_enhance=True) > 0).astype(np.float32)
            yellow = (detect_yellow_line(img_bgr, road_support=road, edge_enhance=True) > 0).astype(np.float32)
            return white, yellow

        if self._domain == "rrl":
            from .RRL2NewTrack import detect_rrl_white_lines

            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            white = (detect_rrl_white_lines(img_bgr) > 0).astype(np.float32)
            yellow = (
                cv2.inRange(hsv, np.array([10, 55, 70], np.uint8), np.array([45, 255, 255], np.uint8)) > 0
            ).astype(np.float32)
            yellow *= (white < 0.5).astype(np.float32)
            return white, yellow

        return self._extract_generic_lane_masks(img_bgr)

    def _refine_vehicle_prob(
        self,
        veh_prob: np.ndarray,
        white_prob: np.ndarray,
        yellow_prob: np.ndarray,
        edge: np.ndarray,
        motion: np.ndarray,
    ) -> np.ndarray:
        """
        Shared line-structure suppression for vehicle_prob.

        Goal:
          - keep one unified logic for WS/GT
          - suppress false vehicle responses that are really lane/edge structures
          - preserve true compact moving blobs when vehicle evidence is strong
        """
        veh_prob = np.clip(np.asarray(veh_prob, dtype=np.float32), 0.0, 1.0)
        if float(np.max(veh_prob)) < 1e-4:
            return veh_prob

        white_prob = np.clip(np.asarray(white_prob, dtype=np.float32), 0.0, 1.0)
        yellow_prob = np.clip(np.asarray(yellow_prob, dtype=np.float32), 0.0, 1.0)
        edge = np.clip(np.asarray(edge, dtype=np.float32), 0.0, 1.0)
        motion = np.clip(np.asarray(motion, dtype=np.float32), 0.0, 1.0)

        h_img, w_img = veh_prob.shape[:2]
        line_map = np.maximum(white_prob, yellow_prob)
        line_support = cv2.GaussianBlur(line_map, (5, 5), sigmaX=1.0)
        edge_support = cv2.GaussianBlur(edge, (5, 5), sigmaX=1.0)
        motion_support = cv2.GaussianBlur(motion, (5, 5), sigmaX=1.0)

        refined = veh_prob.copy()
        seed_mask = (veh_prob >= 0.06).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(seed_mask, connectivity=8)
        if n <= 1:
            penalty = np.clip(0.80 * line_support + 0.35 * edge_support, 0.0, 1.0)
            keep = np.clip(0.30 + 0.70 * motion_support, 0.0, 1.0)
            refined = np.clip(refined * (1.0 - penalty) + refined * keep * 0.25, 0.0, 1.0)
            return refined.astype(np.float32)

        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 6:
                refined[labels == idx] *= 0.2
                continue

            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            comp = labels == idx

            mean_veh = float(np.mean(veh_prob[comp]))
            max_veh = float(np.max(veh_prob[comp]))
            mean_line = float(np.mean(line_support[comp]))
            mean_edge = float(np.mean(edge_support[comp]))
            mean_motion = float(np.mean(motion_support[comp]))
            support_ratio = float(np.mean(veh_prob[comp] >= 0.10))
            aspect = float(max(h, w) / max(min(h, w), 1.0))
            width_norm = float(w / max(w_img, 1))
            bottom_norm = float((y + h) / max(h_img, 1))

            line_like = (
                mean_line > 0.10
                and mean_edge > 0.08
                and aspect > 1.6
                and width_norm < 0.16
            )
            weak_vehicle = (
                max_veh < 0.22
                and mean_veh < 0.12
                and support_ratio < 0.22
            )
            likely_artifact = line_like and weak_vehicle and mean_motion < 0.10

            if likely_artifact:
                refined[comp] *= 0.08
                continue

            penalty = np.clip(0.75 * mean_line + 0.28 * mean_edge, 0.0, 0.92)
            keep_gain = np.clip(
                0.25
                + 0.55 * max_veh
                + 0.20 * support_ratio
                + 0.12 * mean_motion
                + 0.08 * np.clip(bottom_norm, 0.0, 1.0),
                0.15,
                1.0,
            )
            refined[comp] *= max(0.06, 1.0 - penalty) * keep_gain

        refined = cv2.GaussianBlur(refined, (5, 5), sigmaX=1.2)
        return np.clip(refined, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 主观测构建
    # ------------------------------------------------------------------
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        输入: DonkeyEnv 原始 RGB 帧 (H, W, C) uint8
        输出: (6, obs_size, obs_size) float32 [0, 1]
        """
        self._step += 1
        img = np.asarray(obs, dtype=np.uint8)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]

        # DonkeyEnv 输出 RGB；cv2.resize 参数为 (width, height)
        raw_rgb = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

        # ch0: raw Y  —— RGB→YCrCb，取第 0 分量（亮度）
        raw_y = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32) / 255.0

        # 线提取器和 GreenDetector 均期望 BGR
        raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        motion_hint = (
            np.clip(np.abs(raw_y - self._prev_y) * 4.0, 0.0, 1.0)
            if self._prev_y is not None
            else np.zeros_like(raw_y, dtype=np.float32)
        ).astype(np.float32)

        if self._domain == "ws":
            from .WS2NewTrack import build_ws_observation_line_probs

            ws_out = build_ws_observation_line_probs(
                raw_bgr,
                raw_y,
                prev_y=self._prev_y,
                prev_white_prob=self._prev_ws_white_prob,
                edge_preblur_sigma=self.edge_preblur_sigma,
                edge_support_thresh=self.edge_support_thresh,
                edge_support_dilate=self.edge_support_dilate,
                edge_comp_min_area=self.edge_comp_min_area,
                edge_comp_min_height=self.edge_comp_min_height,
                edge_comp_max_compactness=self.edge_comp_max_compactness,
                edge_comp_aspect_ratio=self.edge_comp_aspect_ratio,
                line_comp_min_area=self.line_comp_min_area,
                line_comp_min_height=self.line_comp_min_height,
                line_comp_max_compactness=self.line_comp_max_compactness,
                line_comp_tall_ratio=self.line_comp_tall_ratio,
                line_edge_filter_floor=self.line_edge_filter_floor,
                line_edge_boost_gain=self.line_edge_boost_gain,
                line_edge_min_strength=self.line_edge_min_strength,
                line_edge_strength_power=self.line_edge_strength_power,
                line_edge_hard_floor=self.line_edge_hard_floor,
                line_close_kernel=self.line_close_kernel,
                line_close_iter=self.line_close_iter,
                ws_white_edge_soft_floor=self.ws_white_edge_soft_floor,
                ws_white_edge_soft_far_boost=self.ws_white_edge_soft_far_boost,
                ws_white_edge_support_thresh=self.ws_white_edge_support_thresh,
                ws_white_edge_min_strength=self.ws_white_edge_min_strength,
                ws_white_edge_strength_power=self.ws_white_edge_strength_power,
                ws_white_close_kernel=self.ws_white_close_kernel,
                ws_white_close_iter=self.ws_white_close_iter,
                ws_white_skip_prob_min=self.ws_white_skip_prob_min,
                ws_white_dash_close_h=self.ws_white_dash_close_h,
                ws_white_dash_close_w=self.ws_white_dash_close_w,
                ws_white_dash_min_height=self.ws_white_dash_min_height,
                ws_white_dash_top_min_height=self.ws_white_dash_top_min_height,
                ws_white_dash_max_width=self.ws_white_dash_max_width,
                ws_white_dash_fill_weight=self.ws_white_dash_fill_weight,
                ws_white_loose_s_max=self.ws_white_loose_s_max,
                ws_white_loose_v_min=self.ws_white_loose_v_min,
                ws_white_loose_weight=self.ws_white_loose_weight,
                ws_white_loose_edge_min=self.ws_white_loose_edge_min,
                ws_white_corridor_near=self.ws_white_corridor_near,
                ws_white_corridor_far=self.ws_white_corridor_far,
                ws_white_edge_grow_weight=self.ws_white_edge_grow_weight,
                ws_white_temporal_blend=self.ws_white_temporal_blend,
                ws_white_temporal_dilate=self.ws_white_temporal_dilate,
                ws_white_temporal_motion_scale=self.ws_white_temporal_motion_scale,
                line_prob_min=self.line_prob_min,
            )
            edge = ws_out["edge"].astype(np.float32)
            # WS 原始提取里：
            #   white_prob  = 白色中心虚线（引导线）
            #   yellow_prob = 黄色边线
            # 这里对调到与 GT 一致的 canonical 语义：
            #   ch1 = 车边线, ch2 = 中线/引导线
            white_prob = ws_out["yellow_prob"].astype(np.float32)
            yellow_prob = ws_out["white_prob"].astype(np.float32)
        else:
            if self._domain == "gt":
                from .GT2NewTrack import build_gt_observation_line_probs

                gt_out = build_gt_observation_line_probs(
                    raw_bgr=raw_bgr,
                    raw_y=raw_y,
                    preset_name="relax1",
                    line_conf_mode="none",
                )
                edge = gt_out["edge"].astype(np.float32)
                white_prob = gt_out["white_prob"].astype(np.float32)
                yellow_prob = gt_out["yellow_prob"].astype(np.float32)
                yellow_raw = gt_out.get("yellow_raw", np.zeros_like(yellow_prob, dtype=np.float32))
                if np.max(yellow_raw) < 0.01:
                    yellow_prob = np.zeros_like(yellow_prob, dtype=np.float32)
            else:
                from .GT2NewTrack import build_observation_line_probs

                white_raw, yellow_raw = self._extract_raw_lane_masks(raw_bgr)
                generic_out = build_observation_line_probs(
                    raw_y,
                    white_raw,
                    yellow_raw,
                    raw_bgr=raw_bgr,
                    line_conf_mode="none",
                    edge_preblur_sigma=self.edge_preblur_sigma,
                    edge_support_thresh=self.edge_support_thresh,
                    edge_support_dilate=self.edge_support_dilate,
                    edge_comp_min_area=self.edge_comp_min_area,
                    edge_comp_min_height=self.edge_comp_min_height,
                    edge_comp_max_compactness=self.edge_comp_max_compactness,
                    edge_comp_aspect_ratio=self.edge_comp_aspect_ratio,
                    line_comp_min_area=self.line_comp_min_area,
                    line_comp_min_height=self.line_comp_min_height,
                    line_comp_max_compactness=self.line_comp_max_compactness,
                    line_comp_tall_ratio=self.line_comp_tall_ratio,
                    line_edge_filter_floor=self.line_edge_filter_floor,
                    line_edge_boost_gain=self.line_edge_boost_gain,
                    line_edge_min_strength=self.line_edge_min_strength,
                    line_edge_strength_power=self.line_edge_strength_power,
                    line_edge_hard_floor=self.line_edge_hard_floor,
                    line_close_kernel=self.line_close_kernel,
                    line_close_iter=self.line_close_iter,
                    line_prob_min=self.line_prob_min,
                )
                edge = generic_out["edge"].astype(np.float32)
                white_prob = generic_out["white_prob"].astype(np.float32)
                yellow_prob = generic_out["yellow_prob"].astype(np.float32)
                # 地图先天无中线 → 全零，与 augment 无关
                if np.max(yellow_raw) < 0.01:
                    yellow_prob = np.zeros_like(yellow_prob, dtype=np.float32)

        if self.augment and np.random.random() < self._get_yellow_dropout():
            yellow_prob = np.zeros_like(yellow_prob, dtype=np.float32)   # 渐进 dropout

        # ch4: rawpink vehicle prob. Keep this as a loose color cue rather
        # than a hard detection so lighting/color shifts degrade gracefully.
        from .green_vehicle_detect import raw_pink_prob

        veh_prob = raw_pink_prob(raw_bgr, roi_top_frac=0.10)

        # ch5: motion residual |Y_t - Y_{t-1}|，放大小运动
        if self._prev_y is not None:
            motion = motion_hint
        else:
            motion = np.zeros_like(raw_y)
        self._prev_y = raw_y.copy()
        if self._domain == "ws":
            self._prev_ws_white_prob = white_prob.copy()
        else:
            self._prev_ws_white_prob = None

        return np.stack(
            [raw_y, white_prob, yellow_prob, edge, veh_prob, motion], axis=0
        ).astype(np.float32)

    def reset(self, **kwargs):
        self._prev_y = None   # reset 时清除运动历史
        self._prev_ws_white_prob = None
        return super().reset(**kwargs)


# ============================================================
# _build_state_v13 / _build_state_v17 / _build_state_v16
# ============================================================
def _build_state_v13(
    info: Dict[str, Any],
    action_safety_wrapper,
    control_wrapper,
    v_max: float = 2.2,
) -> np.ndarray:
    """
    返回 7 维状态向量，完全来自传感器 info + adapter 内部状态，无 TrackGeometryManager 依赖：
      [v_long_norm, yaw_rate_norm, accel_x_norm, prev_steer, prev_throttle,
       steer_core, bias_smooth]

    归一化范围：
      v_long_norm      = clip(speed / v_max,      0,   2)
      yaw_rate_norm    = clip(gyro_y / 8.0,      -2,   2)   ← Unity Y-up: yaw=gyro[1], /8 dampened
      accel_x_norm     = clip(accel_x / 9.8,     -2,   2)   ← 纵向加减速
      prev_steer       ∈ [-1, 1]   ← 上一已执行低层 steer（safety 约束后）
      prev_throttle    ∈ [-1, 1]   ← 上一已执行低层 throttle（adapter 输出）
      steer_core       ∈ [-1, 1]   ← ActionAdapterWrapper 积分器内部状态
      bias_smooth      ∈ [-1, 1]   ← ActionAdapterWrapper 占位意图（低通滤波后原始值）

    ⚠️  accel_x 上线前需 sanity check（直线大油门起步应显著为正）。
        若不可信，可替换为 delta_speed 或退化为 6 维状态。

    兼容性：若 control_wrapper 无 steer_core/bias_smooth 属性（V12 path），
    对应维度退化为 0.0。
    """
    v = float(info.get("speed", 0.0) or 0.0)
    gyro  = info.get("gyro",  (0.0, 0.0, 0.0))
    accel = info.get("accel", (0.0, 0.0, 0.0))
    # DonkeySim uses Unity coords (Y-up): yaw rotation = gyro[1] (gyro_y).
    # Previously used gyro[2] (gyro_z) which is near-zero in sim.
    # On physical car, the sensor adapter should map -rp2040/gyro_z → gyro[1].
    try:
        gy = float(gyro[1])
    except Exception:
        gy = 0.0
    try:
        ax = float(accel[0])
    except Exception:
        ax = 0.0

    prev_steer = 0.0
    prev_throttle = 0.0
    if action_safety_wrapper is not None:
        try:
            prev_steer = float(action_safety_wrapper.diag.get("steer_exec", 0.0))
        except Exception:
            pass
    if control_wrapper is not None:
        try:
            prev_throttle = float(control_wrapper.last_low_level_action[1])
            if action_safety_wrapper is None:
                prev_steer = float(control_wrapper.last_low_level_action[0])
        except Exception:
            pass

    # adapter internal state (V13 ActionAdapterWrapper exposes these)
    steer_core = 0.0
    bias_smooth = 0.0
    if control_wrapper is not None:
        try:
            steer_core = float(getattr(control_wrapper, "steer_core", 0.0))
        except Exception:
            steer_core = 0.0
        try:
            bias_smooth = float(getattr(control_wrapper, "bias_smooth", 0.0))
        except Exception:
            bias_smooth = 0.0

    return np.array([
        float(np.clip(v   / v_max, 0.0,  2.0)),       # v_long_norm
        float(np.clip(gy  / 8.0,  -2.0,  2.0)),       # yaw_rate_norm (dampened: /8)
        float(np.clip(ax  / 9.8,  -2.0,  2.0)),       # accel_x_norm
        float(np.clip(prev_steer,    -1.0,  1.0)),     # prev_steer_exec
        float(np.clip(prev_throttle, -1.0,  1.0)),     # prev_throttle_exec
        float(np.clip(steer_core,    -1.0,  1.0)),     # steer_core
        float(np.clip(bias_smooth,   -1.0,  1.0)),     # bias_smooth
    ], dtype=np.float32)


def _build_state_v17(
    info: Dict[str, Any],
    action_safety_wrapper,
    control_wrapper,
    v_max: float = 2.2,
) -> np.ndarray:
    """
    V17 policy core state.

    保持 V13 的 7D 控制/动力学核心，不再把 obstacle-5 压入 policy state。
    LiDAR 及其异步 meta 在独立观测键中提供，由特征提取器内部再与 state 融合。
    """
    return _build_state_v13(
        info=info,
        action_safety_wrapper=action_safety_wrapper,
        control_wrapper=control_wrapper,
        v_max=v_max,
    )


def _build_state_v16(
    info: Dict[str, Any],
    action_safety_wrapper,
    control_wrapper,
    v_max: float = 2.2,
) -> np.ndarray:
    """
    V16 state = V13 7D core + 5D obstacle context:
      [base_7,
       obstacle_present,
       obstacle_longitudinal_norm,
       obstacle_lateral_norm,
       obstacle_dist_norm,
       obstacle_risk]

    设计原则：
    - 保留 V13 的控制内态，避免 action adapter 学习重置
    - 障碍信息来自 runtime wrapper 注入的 info，不依赖赛道几何
    - 保持与历史 V16 checkpoint 兼容，state 固定为 12 维
    - 当本步无障碍信号时，新增 5 维退化为 0
    """
    base = _build_state_v13(
        info=info,
        action_safety_wrapper=action_safety_wrapper,
        control_wrapper=control_wrapper,
        v_max=v_max,
    )

    def _get_float(name: str) -> float:
        try:
            return float(info.get(name, 0.0) or 0.0)
        except Exception:
            return 0.0

    obstacle_present = _get_float("obstacle_present")
    obstacle_longitudinal = _get_float("obstacle_longitudinal")
    obstacle_lateral = _get_float("obstacle_lateral")
    obstacle_dist = _get_float("obstacle_dist")
    obstacle_risk = _get_float("obstacle_risk")

    obstacle_state = np.array([
        float(np.clip(obstacle_present, 0.0, 1.0)),
        float(np.clip(obstacle_longitudinal / 6.0, -2.0, 2.0)),
        float(np.clip(obstacle_lateral / 2.5, -2.0, 2.0)),
        float(np.clip(obstacle_dist / 6.0, 0.0, 2.0)),
        float(np.clip(obstacle_risk, 0.0, 1.0)),
    ], dtype=np.float32)

    return np.concatenate([base, obstacle_state], axis=0).astype(np.float32)
