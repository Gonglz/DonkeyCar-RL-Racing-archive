#!/usr/bin/env python3
"""
Learned obstacle-context estimator for V16.

V1 goal:
  - input: 6ch semantic image + 7D ego/base state
  - output: present, longitudinal, lateral, dist
  - keep downstream obstacle-5 interface stable

V2 extension target:
  - add short-horizon prediction heads on the same shared backbone
  - predict future relative pose / risk / TTC style quantities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObstacleContextPrediction:
    present: float
    longitudinal: float
    lateral: float
    dist: float
    risk: float

    def to_info_dict(self) -> Dict[str, float]:
        return {
            "obstacle_present": float(self.present),
            "obstacle_longitudinal": float(self.longitudinal),
            "obstacle_lateral": float(self.lateral),
            "obstacle_dist": float(self.dist),
            "obstacle_risk": float(self.risk),
        }


def compute_obstacle_risk(longitudinal: float, lateral: float, planar_distance: float) -> float:
    if not np.isfinite(planar_distance) or planar_distance <= 0.0:
        return 0.0
    if longitudinal < -0.4:
        return 0.0
    front_gate = float(np.clip((4.0 - max(longitudinal, 0.0)) / 3.5, 0.0, 1.0))
    planar_gate = float(np.clip((5.0 - planar_distance) / 4.5, 0.0, 1.0))
    lateral_gate = float(np.clip(1.0 - abs(lateral) / 1.25, 0.0, 1.0))
    return float(np.clip((front_gate ** 2.0) * max(0.20, lateral_gate) * planar_gate, 0.0, 1.0))


def _make_spatial_image_encoder(
    image_channels: int,
    image_feat_dim: int,
    pooled_hw: int = 4,
) -> nn.Sequential:
    spatial_channels = 64
    flat_dim = spatial_channels * pooled_hw * pooled_hw
    return nn.Sequential(
        nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, spatial_channels, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((pooled_hw, pooled_hw)),
        nn.Flatten(),
        nn.Linear(flat_dim, image_feat_dim),
        nn.ReLU(inplace=True),
    )


class ObstacleContextFrameNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 6,
        state_dim: int = 7,
        image_feat_dim: int = 96,
        state_feat_dim: int = 32,
        hidden_dim: int = 128,
        pooled_hw: int = 4,
    ):
        super().__init__()
        self.image_encoder = _make_spatial_image_encoder(
            image_channels=image_channels,
            image_feat_dim=image_feat_dim,
            pooled_hw=pooled_hw,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(state_feat_dim, state_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(image_feat_dim + state_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.present_head = nn.Linear(hidden_dim, 1)
        self.geom_head = nn.Linear(hidden_dim, 3)

    def forward(self, image: torch.Tensor, state7: torch.Tensor) -> Dict[str, torch.Tensor]:
        img_feat = self.image_encoder(image)
        state_feat = self.state_encoder(state7)
        fused = torch.cat([img_feat, state_feat], dim=-1)
        trunk = self.trunk(fused)
        present_logit = self.present_head(trunk).squeeze(-1)
        geom_raw = self.geom_head(trunk)
        longitudinal = F.softplus(geom_raw[:, 0])  # >= 0
        lateral = 2.5 * torch.tanh(geom_raw[:, 1])
        dist = F.softplus(geom_raw[:, 2])          # >= 0
        return {
            "present_logit": present_logit,
            "longitudinal": longitudinal,
            "lateral": lateral,
            "dist": dist,
        }

    @torch.no_grad()
    def predict(self, image: torch.Tensor, state7: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        out = self.forward(image, state7)
        out["present"] = torch.sigmoid(out["present_logit"])
        return out


class ObstacleContextTemporalNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 6,
        state_dim: int = 7,
        image_feat_dim: int = 96,
        state_feat_dim: int = 32,
        fused_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        pooled_hw: int = 4,
    ):
        super().__init__()
        self.image_encoder = _make_spatial_image_encoder(
            image_channels=image_channels,
            image_feat_dim=image_feat_dim,
            pooled_hw=pooled_hw,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(state_feat_dim, state_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.step_fuser = nn.Sequential(
            nn.Linear(image_feat_dim + state_feat_dim, fused_dim),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRU(
            input_size=fused_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head_trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.present_head = nn.Linear(hidden_dim, 1)
        self.geom_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        image_seq: torch.Tensor,
        state7_seq: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = image_seq.shape[:2]
        flat_img = image_seq.reshape(batch_size * seq_len, *image_seq.shape[2:])
        flat_state = state7_seq.reshape(batch_size * seq_len, state7_seq.shape[-1])
        img_feat = self.image_encoder(flat_img)
        state_feat = self.state_encoder(flat_state)
        step_feat = self.step_fuser(torch.cat([img_feat, state_feat], dim=-1))
        step_feat = step_feat.view(batch_size, seq_len, -1)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                step_feat,
                lengths=lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h_n = self.rnn(packed)
            trunk = self.head_trunk(h_n[-1])
        else:
            _, h_n = self.rnn(step_feat)
            trunk = self.head_trunk(h_n[-1])

        present_logit = self.present_head(trunk).squeeze(-1)
        geom_raw = self.geom_head(trunk)
        longitudinal = F.softplus(geom_raw[:, 0])
        lateral = 2.5 * torch.tanh(geom_raw[:, 1])
        dist = F.softplus(geom_raw[:, 2])
        return {
            "present_logit": present_logit,
            "longitudinal": longitudinal,
            "lateral": lateral,
            "dist": dist,
        }

    @torch.no_grad()
    def predict(
        self,
        image_seq: torch.Tensor,
        state7_seq: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        out = self.forward(image_seq=image_seq, state7_seq=state7_seq, lengths=lengths)
        out["present"] = torch.sigmoid(out["present_logit"])
        return out


def obstacle_context_loss(
    pred: Dict[str, torch.Tensor],
    target_present: torch.Tensor,
    target_longitudinal: torch.Tensor,
    target_lateral: torch.Tensor,
    target_dist: torch.Tensor,
    present_pos_weight: Optional[torch.Tensor] = None,
    present_focal_gamma: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_present = target_present.float().view(-1)
    bce_raw = F.binary_cross_entropy_with_logits(
        pred["present_logit"],
        target_present,
        pos_weight=present_pos_weight,
        reduction="none",
    )
    if float(present_focal_gamma) > 0.0:
        probs = torch.sigmoid(pred["present_logit"])
        p_t = torch.where(target_present > 0.5, probs, 1.0 - probs)
        focal = torch.pow(torch.clamp(1.0 - p_t, min=1e-6), float(present_focal_gamma))
        bce = torch.mean(focal * bce_raw)
    else:
        bce = torch.mean(bce_raw)
    positive = target_present > 0.5
    if torch.any(positive):
        long_loss = F.smooth_l1_loss(pred["longitudinal"][positive], target_longitudinal[positive])
        lat_loss = F.smooth_l1_loss(pred["lateral"][positive], target_lateral[positive])
        dist_loss = F.smooth_l1_loss(pred["dist"][positive], target_dist[positive])
    else:
        long_loss = pred["longitudinal"].sum() * 0.0
        lat_loss = pred["lateral"].sum() * 0.0
        dist_loss = pred["dist"].sum() * 0.0

    total = bce + long_loss + 0.8 * lat_loss + 0.7 * dist_loss
    stats = {
        "loss_total": float(total.detach().cpu().item()),
        "loss_present": float(bce.detach().cpu().item()),
        "loss_longitudinal": float(long_loss.detach().cpu().item()),
        "loss_lateral": float(lat_loss.detach().cpu().item()),
        "loss_dist": float(dist_loss.detach().cpu().item()),
        "positive_rate": float(target_present.mean().detach().cpu().item()),
    }
    return total, stats


@torch.no_grad()
def decode_prediction_dict(pred: Dict[str, torch.Tensor], index: int = 0) -> ObstacleContextPrediction:
    present = float(torch.sigmoid(pred["present_logit"][index]).cpu().item())
    longitudinal = float(pred["longitudinal"][index].cpu().item())
    lateral = float(pred["lateral"][index].cpu().item())
    dist = float(pred["dist"][index].cpu().item())
    risk = compute_obstacle_risk(longitudinal, lateral, dist)
    return ObstacleContextPrediction(
        present=present,
        longitudinal=longitudinal,
        lateral=lateral,
        dist=dist,
        risk=risk,
    )
