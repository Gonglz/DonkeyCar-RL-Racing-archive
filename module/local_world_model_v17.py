"""
module/local_world_model_v17.py

V17 local interaction world model for offline training.

This model is separate from the bootstrap ego-only `world_model.py` used by the
current predictive safety filter. Its purpose is to support the staged V17
training plan with LiDAR sequence inputs and multiple task heads.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class SharedSideLidarEncoder(nn.Module):
    """
    Per-side LiDAR encoder used by the local world model.

    Input per side: (B, 2, 18)
    Output per side: (B, 32)
    """

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.res1 = ResidualBlock1D(32, kernel_size=3, padding=1)
        self.res2 = ResidualBlock1D(32, kernel_size=3, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(32 * 18, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv_in(x))
        h = self.res1(h)
        h = self.res2(h)
        h = torch.flatten(h, start_dim=1)
        return self.mlp(h)


class CameraFrameEncoder(nn.Module):
    """
    Lightweight semantic camera encoder for reduced CHW frames.

    Input:
      (B, C, H, W)
    Output:
      (B, camera_feat_dim)
    """

    def __init__(self, in_channels: int, camera_feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, camera_feat_dim),
            nn.LayerNorm(camera_feat_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class InteractionHead(nn.Module):
    """
    Foreground interaction head for the key obstacle / opponent target.

    Outputs:
      - target_rel         (4,)
      - closing_rate       (1,)
      - overtake_progress  (1,)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.target_rel_head = nn.Linear(hidden_dim, 4)
        self.closing_rate_head = nn.Linear(hidden_dim, 1)
        self.overtake_progress_head = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "target_rel": self.target_rel_head(latent),
            "closing_rate": self.closing_rate_head(latent).squeeze(-1),
            "overtake_progress": self.overtake_progress_head(latent).squeeze(-1),
        }


class SafetyPassabilityHead(nn.Module):
    """
    Local safety / passability head.

    Outputs:
      - gap               (2,)
      - collision_logit   (1,)
      - ttc_proxy         (1,)
      - passable_logits   (2,)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.pre = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )
        self.gap_head = nn.Linear(self.hidden_dim, 2)
        self.risk_head = nn.Linear(self.hidden_dim, 2)
        self.passable_logits_head = nn.Linear(self.hidden_dim, 2)

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.pre(latent)
        risk = self.risk_head(latent)
        return {
            "gap": self.gap_head(latent),
            "collision_logit": risk[:, 0],
            "ttc_proxy": risk[:, 1],
            "passable_logits": self.passable_logits_head(latent),
        }


class LocalWorldModelV17(nn.Module):
    """
    Input:
      ego_seq        (B, T, 8)
      camera_seq     (B, T, C, H, W) optional reduced semantic camera tensor
      lidar_seq      (B, T, 72)
      async_meta_seq (B, T, 4)
      lengths        (B,)

    Output heads:
      target_rel           (B, 4)
      gap                  (B, 2)
      collision_logit      (B,)
      ttc_proxy            (B,)
      passable_logits      (B, 2)
      closing_rate         (B,)
      overtake_progress    (B,)
    """

    def __init__(
        self,
        ego_dim: int = 8,
        camera_channels: int = 0,
        camera_feat_dim: int = 64,
        lidar_dim: int = 72,
        async_meta_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        if lidar_dim != 72:
            raise ValueError(f"LocalWorldModelV17 expects lidar_dim=72, got {lidar_dim}")
        self.ego_dim = int(ego_dim)
        self.camera_channels = int(max(0, camera_channels))
        self.camera_feat_dim = int(camera_feat_dim if self.camera_channels > 0 else 0)
        self.lidar_dim = int(lidar_dim)
        self.async_meta_dim = int(async_meta_dim)
        self.hidden_dim = int(hidden_dim)

        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_dim + self.async_meta_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        self.lidar_side_encoder = SharedSideLidarEncoder()
        self.lidar_post = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        self.camera_encoder: Optional[CameraFrameEncoder]
        if self.camera_channels > 0:
            self.camera_encoder = CameraFrameEncoder(
                in_channels=self.camera_channels,
                camera_feat_dim=self.camera_feat_dim,
            )
        else:
            self.camera_encoder = None

        self.step_fusion = nn.Sequential(
            nn.Linear(64 + 64, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=float(max(0.0, dropout))),
        )
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.trunk = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )

        self.interaction_head = InteractionHead(self.hidden_dim)
        self.safety_head = SafetyPassabilityHead(
            input_dim=self.hidden_dim + self.camera_feat_dim,
            hidden_dim=self.hidden_dim,
        )

    @property
    def target_head(self) -> nn.Linear:
        return self.interaction_head.target_rel_head

    @property
    def gap_head(self) -> nn.Linear:
        return self.safety_head.gap_head

    @property
    def risk_head(self) -> nn.Linear:
        return self.safety_head.risk_head

    @property
    def passable_head(self) -> nn.Linear:
        return self.safety_head.passable_logits_head

    def _resize_linear_weight(
        self,
        old_weight: torch.Tensor,
        new_weight: torch.Tensor,
    ) -> torch.Tensor:
        resized = new_weight.detach().clone()
        rows = min(int(old_weight.shape[0]), int(new_weight.shape[0]))
        cols = min(int(old_weight.shape[1]), int(new_weight.shape[1]))
        resized[:rows, :cols] = old_weight[:rows, :cols]
        return resized

    def _inject_current_defaults(
        self,
        remapped: Dict[str, torch.Tensor],
        prefixes: Tuple[str, ...],
    ) -> Dict[str, torch.Tensor]:
        current = self.state_dict()
        for key, value in current.items():
            if key in remapped:
                continue
            if any(key.startswith(prefix) for prefix in prefixes):
                remapped[key] = value.detach().clone()
        return remapped

    def _remap_legacy_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        remapped = dict(state_dict)
        legacy_keys = {
            "target_head.weight",
            "target_head.bias",
            "gap_head.weight",
            "gap_head.bias",
            "safety_head.weight",
            "safety_head.bias",
            "opportunity_head.weight",
            "opportunity_head.bias",
        }
        if any(key in remapped for key in legacy_keys):
            if "target_head.weight" in remapped:
                remapped["interaction_head.target_rel_head.weight"] = remapped.pop("target_head.weight")
                remapped["interaction_head.target_rel_head.bias"] = remapped.pop("target_head.bias")
            if "gap_head.weight" in remapped:
                remapped["safety_head.gap_head.weight"] = remapped.pop("gap_head.weight")
                remapped["safety_head.gap_head.bias"] = remapped.pop("gap_head.bias")
            if "safety_head.weight" in remapped:
                remapped["safety_head.risk_head.weight"] = remapped.pop("safety_head.weight")
                remapped["safety_head.risk_head.bias"] = remapped.pop("safety_head.bias")
            if "opportunity_head.weight" in remapped:
                weight = remapped.pop("opportunity_head.weight")
                bias = remapped.pop("opportunity_head.bias")
                remapped["safety_head.passable_logits_head.weight"] = weight[:2].clone()
                remapped["safety_head.passable_logits_head.bias"] = bias[:2].clone()
                remapped["interaction_head.closing_rate_head.weight"] = weight[2:3].clone()
                remapped["interaction_head.closing_rate_head.bias"] = bias[2:3].clone()
                remapped["interaction_head.overtake_progress_head.weight"] = weight[3:4].clone()
                remapped["interaction_head.overtake_progress_head.bias"] = bias[3:4].clone()

        remapped.pop("ego_head.weight", None)
        remapped.pop("ego_head.bias", None)
        for key in [k for k in remapped.keys() if k.startswith("ego_head.")]:
            remapped.pop(key, None)

        if "step_fusion.0.weight" in remapped:
            old_weight = remapped["step_fusion.0.weight"]
            new_weight = self.step_fusion[0].weight.detach().clone()
            if tuple(old_weight.shape) != tuple(new_weight.shape):
                remapped["step_fusion.0.weight"] = self._resize_linear_weight(old_weight, new_weight)
        if "step_fusion.0.bias" not in remapped:
            remapped["step_fusion.0.bias"] = self.step_fusion[0].bias.detach().clone()

        if self.camera_encoder is not None:
            remapped = self._inject_current_defaults(
                remapped,
                prefixes=("camera_encoder.",),
            )
        remapped = self._inject_current_defaults(
            remapped,
            prefixes=("safety_head.pre.",),
        )
        if "safety_head.pre.0.weight" in remapped:
            old_weight = remapped["safety_head.pre.0.weight"]
            new_weight = self.safety_head.pre[0].weight.detach().clone()
            if tuple(old_weight.shape) != tuple(new_weight.shape):
                remapped["safety_head.pre.0.weight"] = self._resize_linear_weight(old_weight, new_weight)
        if "safety_head.gap_head.weight" in remapped:
            old_weight = remapped["safety_head.gap_head.weight"]
            new_weight = self.safety_head.gap_head.weight.detach().clone()
            if tuple(old_weight.shape) != tuple(new_weight.shape):
                remapped["safety_head.gap_head.weight"] = self._resize_linear_weight(old_weight, new_weight)
        if "safety_head.risk_head.weight" in remapped:
            old_weight = remapped["safety_head.risk_head.weight"]
            new_weight = self.safety_head.risk_head.weight.detach().clone()
            if tuple(old_weight.shape) != tuple(new_weight.shape):
                remapped["safety_head.risk_head.weight"] = self._resize_linear_weight(old_weight, new_weight)
        if "safety_head.passable_logits_head.weight" in remapped:
            old_weight = remapped["safety_head.passable_logits_head.weight"]
            new_weight = self.safety_head.passable_logits_head.weight.detach().clone()
            if tuple(old_weight.shape) != tuple(new_weight.shape):
                remapped["safety_head.passable_logits_head.weight"] = self._resize_linear_weight(old_weight, new_weight)
        return remapped

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        return super().load_state_dict(self._remap_legacy_state_dict(state_dict), strict=strict)

    def _encode_lidar(self, lidar_flat: torch.Tensor) -> torch.Tensor:
        lidar_range, lidar_valid = torch.chunk(lidar_flat.float(), 2, dim=-1)
        lidar_2c = torch.stack([lidar_range, lidar_valid], dim=1)
        left = lidar_2c[:, :, :18]
        right = lidar_2c[:, :, 18:]
        left_feat = self.lidar_side_encoder(left)
        right_feat = self.lidar_side_encoder(right)
        return self.lidar_post(torch.cat([left_feat, right_feat], dim=-1))

    def _encode_camera_last(
        self,
        camera_seq: Optional[torch.Tensor],
        lengths: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self.camera_feat_dim <= 0:
            return torch.zeros(batch_size, 0, dtype=dtype, device=device)
        if camera_seq is None:
            return torch.zeros(batch_size, self.camera_feat_dim, dtype=dtype, device=device)
        if self.camera_encoder is None:
            raise RuntimeError("camera_seq was provided but camera encoder is disabled")
        if camera_seq.ndim != 5:
            raise ValueError(f"expected camera_seq=(B,T,C,H,W), got shape={tuple(camera_seq.shape)}")
        if int(camera_seq.shape[2]) != self.camera_channels:
            raise ValueError(
                f"expected camera_seq channels={self.camera_channels}, got {int(camera_seq.shape[2])}"
            )
        camera_feat_seq = self.camera_encoder(
            camera_seq.reshape(batch_size * seq_len, *camera_seq.shape[2:])
        ).reshape(batch_size, seq_len, -1)
        if lengths is None:
            return camera_feat_seq[:, -1, :]
        last_idx = torch.clamp(lengths.to(device=device, dtype=torch.long) - 1, min=0)
        gather_idx = last_idx.view(batch_size, 1, 1).expand(batch_size, 1, camera_feat_seq.shape[-1])
        return camera_feat_seq.gather(dim=1, index=gather_idx).squeeze(1)

    def forward_grouped(
        self,
        ego_seq: torch.Tensor,
        lidar_seq: torch.Tensor,
        async_meta_seq: torch.Tensor,
        camera_seq: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = ego_seq.shape

        ego_in = torch.cat([ego_seq.float(), async_meta_seq.float()], dim=-1)
        ego_feat = self.ego_encoder(ego_in.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, -1)
        lidar_feat = self._encode_lidar(lidar_seq.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, -1)
        step_feat = self.step_fusion(torch.cat([ego_feat, lidar_feat], dim=-1))

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                step_feat,
                lengths=lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
            latent = hidden[-1]
        else:
            _, hidden = self.gru(step_feat)
            latent = hidden[-1]

        latent = self.trunk(latent)
        camera_feat = self._encode_camera_last(
            camera_seq=camera_seq,
            lengths=lengths,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=latent.dtype,
            device=latent.device,
        )
        safety_input = torch.cat([latent, camera_feat], dim=-1)
        return {
            "interaction": self.interaction_head(latent),
            "safety": self.safety_head(safety_input),
        }

    def forward(
        self,
        ego_seq: torch.Tensor,
        lidar_seq: torch.Tensor,
        async_meta_seq: torch.Tensor,
        camera_seq: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        grouped = self.forward_grouped(
            ego_seq=ego_seq,
            lidar_seq=lidar_seq,
            async_meta_seq=async_meta_seq,
            camera_seq=camera_seq,
            lengths=lengths,
        )
        flat: Dict[str, torch.Tensor] = {}
        for outputs in grouped.values():
            flat.update(outputs)
        return flat


def local_world_model_loss(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    stage: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    stage = str(stage).strip().lower()
    if stage not in ("a", "b", "c"):
        raise ValueError(f"unsupported stage={stage!r}")

    zero = pred["target_rel"].sum() * 0.0
    gap_loss = F.smooth_l1_loss(pred["gap"], batch["target_gap"])

    target_mask = batch["target_rel_mask"].float()
    target_denom = torch.clamp(target_mask.sum(), min=1.0)
    target_loss = (
        F.smooth_l1_loss(pred["target_rel"], batch["target_rel"], reduction="none") * target_mask
    ).sum() / target_denom

    collision_loss = F.binary_cross_entropy_with_logits(
        pred["collision_logit"], batch["target_collision"].float()
    )
    safety_mask = batch["target_safety_valid"].float()
    safety_denom = torch.clamp(safety_mask.sum(), min=1.0)
    ttc_loss = (
        F.smooth_l1_loss(pred["ttc_proxy"], batch["target_ttc"], reduction="none") * safety_mask
    ).sum() / safety_denom

    passable_loss = F.binary_cross_entropy_with_logits(
        pred["passable_logits"], batch["target_passable"].float()
    )
    opp_mask = batch["target_opportunity_valid"].float()
    opp_denom = torch.clamp(opp_mask.sum(), min=1.0)
    closing_loss = (
        F.smooth_l1_loss(pred["closing_rate"], batch["target_closing_rate"], reduction="none") * opp_mask
    ).sum() / opp_denom
    overtake_gain_loss = (
        F.smooth_l1_loss(pred["overtake_progress"], batch["target_overtake_progress"], reduction="none") * opp_mask
    ).sum() / opp_denom

    if stage == "a":
        total = target_loss + gap_loss
    elif stage == "b":
        # Keep Stage B focused on risk/interaction auxiliaries while retaining
        # a meaningful geometry anchor so gap/target predictions do not drift.
        total = (
            collision_loss
            + ttc_loss
            + passable_loss
            + closing_loss
            + overtake_gain_loss
            + 0.5 * target_loss
            + 0.5 * gap_loss
        )
    else:
        total = (
            target_loss
            + gap_loss
            + collision_loss
            + ttc_loss
            + passable_loss
            + closing_loss
            + overtake_gain_loss
        )

    stats = {
        "loss_total": float(total.detach().cpu().item()),
        "loss_target": float(target_loss.detach().cpu().item()),
        "loss_gap": float(gap_loss.detach().cpu().item()),
        "loss_collision": float(collision_loss.detach().cpu().item()),
        "loss_ttc": float(ttc_loss.detach().cpu().item()),
        "loss_passable": float(passable_loss.detach().cpu().item()),
        "loss_closing": float(closing_loss.detach().cpu().item()),
        "loss_overtake_gain": float(overtake_gain_loss.detach().cpu().item()),
        "stage": float({"a": 1, "b": 2, "c": 3}[stage]),
    }
    return total + zero, stats


__all__ = ["LocalWorldModelV17", "local_world_model_loss"]
