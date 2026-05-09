"""
module/v17_policy.py

V17 policy components:
- LiDAR-first FiLM feature extractor
- Optional critic-only dual value heads with domain routing
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
import torch as th
import torch.nn as nn
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )
        self.act = nn.ReLU()

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.act(x + self.block(x))


class SharedSideLidarEncoder(nn.Module):
    """
    Shared-weight side-separated encoder.

    Input per side: (B, 2, sectors_per_side)
    Output per side: (B, 48)
    """

    def __init__(self, sectors_per_side: int = 18):
        super().__init__()
        self.sectors_per_side = int(sectors_per_side)
        if self.sectors_per_side <= 0:
            raise ValueError(f"sectors_per_side must be > 0, got {sectors_per_side}")
        self.conv_in = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.res1 = ResidualBlock1D(32, kernel_size=3, padding=1)
        self.res2 = ResidualBlock1D(32, kernel_size=3, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(32 * self.sectors_per_side, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
        )

    def forward(self, side_obs: th.Tensor) -> th.Tensor:
        if side_obs.shape[-1] != self.sectors_per_side:
            raise ValueError(
                f"side_obs sectors mismatch: expected {self.sectors_per_side}, "
                f"got {side_obs.shape[-1]}"
            )
        h = th.relu(self.conv_in(side_obs))
        h = self.res1(h)
        h = self.res2(h)
        h = th.flatten(h, start_dim=1)
        return self.mlp(h)


class PooledLidarEncoder(nn.Module):
    """
    Simpler pooled LiDAR baseline for ablation.

    Input:  (B, 2, sectors)
    Output: (B, 96)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 96),
            nn.ReLU(),
        )

    def forward(self, lidar_obs: th.Tensor) -> th.Tensor:
        return self.net(lidar_obs)


class LiDARFiLMFeatureExtractor(BaseFeaturesExtractor):
    """
    V17 feature extractor for Dict obs with keys:
      - image:     (C, H, W), default C=6 including rawpink vehicle_prob
      - state:     (7,)
      - lidar:     (2 * sectors,) canonical full LiDAR, or (12,) target token mode
      - lidar_meta:(2,)
      - domain_id: (1,)   optional, ignored by actor features
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        image_feat_dim: int = 128,
        state_feat_dim: int = 32,
        lidar_feat_dim: int = 96,
        lidar_encoder_mode: str = "side_separated",
        disable_lidar_meta: bool = False,
    ):
        features_dim = int(image_feat_dim + state_feat_dim + lidar_feat_dim)
        super().__init__(observation_space, features_dim=features_dim)

        image_shape = observation_space["image"].shape
        state_dim = int(observation_space["state"].shape[0])
        lidar_dim = int(observation_space["lidar"].shape[0])
        lidar_meta_dim = int(observation_space["lidar_meta"].shape[0])
        if lidar_dim != 12 and (lidar_dim < 4 or lidar_dim % 4 != 0):
            raise ValueError(
                "V17 full lidar dim must be 2 * an even sector count "
                f"(or 12 for target_token), got {lidar_dim}"
            )

        n_ch = int(image_shape[0])
        self.image_feat_dim = int(image_feat_dim)
        self.state_feat_dim = int(state_feat_dim)
        self.lidar_feat_dim = int(lidar_feat_dim)
        self.disable_lidar_meta = bool(disable_lidar_meta)
        self.lidar_dim = int(lidar_dim)
        self.lidar_num_sectors = 0 if self.lidar_dim == 12 else self.lidar_dim // 2
        self.lidar_side_sectors = 0 if self.lidar_dim == 12 else self.lidar_num_sectors // 2
        self.lidar_encoder_mode = str(lidar_encoder_mode).strip().lower()
        if self.lidar_encoder_mode not in ("side_separated", "pooled"):
            raise ValueError(
                f"unsupported lidar_encoder_mode={lidar_encoder_mode!r}; "
                "expected 'side_separated' or 'pooled'"
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, image_feat_dim),
            nn.ReLU(),
        )
        self.image_norm = nn.LayerNorm(image_feat_dim)

        state_input_dim = int(state_dim + (0 if self.disable_lidar_meta else lidar_meta_dim))
        self.state_enc = nn.Sequential(
            nn.Linear(state_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_feat_dim),
            nn.ReLU(),
        )

        self.lidar_side_encoder = SharedSideLidarEncoder(
            sectors_per_side=max(1, self.lidar_side_sectors)
        )
        self.lidar_pooled_encoder = PooledLidarEncoder()
        self.lidar_target_encoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, lidar_feat_dim),
            nn.ReLU(),
        )

        self.film_hidden = nn.Linear(state_feat_dim + lidar_feat_dim, 192)
        self.film_act = nn.ReLU()
        self.film_out = nn.Linear(192, image_feat_dim * 2)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)
        self.fused_norm = nn.LayerNorm(image_feat_dim)

    def _encode_lidar(self, lidar_flat: th.Tensor) -> th.Tensor:
        lidar_flat = lidar_flat.float()
        if self.lidar_dim == 12:
            return self.lidar_target_encoder(lidar_flat)
        lidar_range, lidar_valid = th.chunk(lidar_flat, 2, dim=-1)
        lidar_2c = th.stack([lidar_range, lidar_valid], dim=1)  # (B, 2, sectors)
        if self.lidar_encoder_mode == "pooled":
            return self.lidar_pooled_encoder(lidar_2c)
        mid = self.lidar_side_sectors
        left = lidar_2c[:, :, :mid]
        right = lidar_2c[:, :, mid:]
        left_feat = self.lidar_side_encoder(left)
        right_feat = self.lidar_side_encoder(right)
        return th.cat([left_feat, right_feat], dim=-1)

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        image_feat = self.image_norm(self.cnn(obs["image"].float()))

        if self.disable_lidar_meta:
            state_aug = obs["state"].float()
        else:
            state_aug = th.cat([obs["state"].float(), obs["lidar_meta"].float()], dim=-1)
        state_feat = self.state_enc(state_aug)

        lidar_feat = self._encode_lidar(obs["lidar"])
        film_in = th.cat([state_feat, lidar_feat], dim=-1)
        film_hidden = self.film_act(self.film_hidden(film_in))
        gamma_raw, beta_raw = self.film_out(film_hidden).chunk(2, dim=-1)

        gamma = 1.0 + 0.1 * th.tanh(gamma_raw)
        beta = 0.1 * th.tanh(beta_raw)
        fused = self.fused_norm(gamma * image_feat + beta)

        return th.cat([fused, state_feat, lidar_feat], dim=-1)


class V17RecurrentMultiInputPolicy(RecurrentMultiInputActorCriticPolicy):
    """
    Recurrent MultiInput policy with optional critic-only dual value heads.

    The actor path never consumes domain_id. If dual value heads are activated,
    the critic routes WS/GT samples to separate scalar heads.
    """

    def __init__(self, *args, domain_obs_key: str = "domain_id", **kwargs):
        self.domain_obs_key = str(domain_obs_key)
        self.use_dual_value_heads = False
        super().__init__(*args, **kwargs)

        self.value_head_ws = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.value_head_gt = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.value_head_ws.to(self.device)
        self.value_head_gt.to(self.device)
        self._copy_shared_value_to_dual()
        self.optimizer.add_param_group({"params": self.value_head_ws.parameters()})
        self.optimizer.add_param_group({"params": self.value_head_gt.parameters()})

    def _copy_shared_value_to_dual(self) -> None:
        with th.no_grad():
            self.value_head_ws.weight.copy_(self.value_net.weight)
            self.value_head_ws.bias.copy_(self.value_net.bias)
            self.value_head_gt.weight.copy_(self.value_net.weight)
            self.value_head_gt.bias.copy_(self.value_net.bias)

    def activate_dual_value_heads(self) -> None:
        self._copy_shared_value_to_dual()
        self.use_dual_value_heads = True

    def deactivate_dual_value_heads(self) -> None:
        self.use_dual_value_heads = False

    def _domain_is_gt(self, obs: Any, batch_size: int, device: th.device) -> Optional[th.Tensor]:
        if not isinstance(obs, dict):
            return None
        domain = obs.get(self.domain_obs_key)
        if domain is None:
            return None
        if not isinstance(domain, th.Tensor):
            domain = th.as_tensor(domain, device=device)
        domain = domain.float().reshape(batch_size, -1)
        return (domain[:, 0] > 0.5).float().unsqueeze(-1)

    def _value_from_latent(self, latent_vf: th.Tensor, obs: Any) -> th.Tensor:
        if not self.use_dual_value_heads:
            return self.value_net(latent_vf)

        mask_gt = self._domain_is_gt(obs, batch_size=latent_vf.shape[0], device=latent_vf.device)
        if mask_gt is None:
            return self.value_net(latent_vf)

        value_ws = self.value_head_ws(latent_vf)
        value_gt = self.value_head_gt(latent_vf)
        return (1.0 - mask_gt) * value_ws + mask_gt * value_gt

    def forward(
        self,
        obs: th.Tensor,
        lstm_states,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self._value_from_latent(latent_vf, obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, type(lstm_states)(lstm_states_pi, lstm_states_vf)

    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self._value_from_latent(latent_vf, obs)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self._value_from_latent(latent_vf, obs)
        return values, log_prob, distribution.entropy()


__all__ = [
    "LiDARFiLMFeatureExtractor",
    "V17RecurrentMultiInputPolicy",
]
