#!/usr/bin/env python3
"""V17 shadow pilot for Jetson DonkeyCar deployment.

This part is designed for measurement runs. It reads the camera frame and
optional RP2040/LiDAR state, runs the latest V17 RecurrentPPO policy, and
publishes pilot outputs plus per-frame latency. Runtime monitor can mount it in
shadow mode while the vehicle actuator path stays in user/manual mode.
"""

import argparse
import io
import importlib.util
import math
import os
import sys
import time
import types
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - deployment dependency
    torch = None

try:
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - deployment dependency
    gym = None
    class _FallbackBox:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

    class _FallbackDict(dict):
        def __init__(self, spaces_dict):
            super().__init__(spaces_dict)
            self.spaces = dict(spaces_dict)

    class _FallbackSpaces:
        Box = _FallbackBox
        Dict = _FallbackDict

    spaces = _FallbackSpaces()

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception:  # pragma: no cover - deployment dependency
    if torch is not None:
        class BaseFeaturesExtractor(torch.nn.Module):
            def __init__(self, observation_space, features_dim: int = 0):
                super().__init__()
                self.observation_space = observation_space
                self._features_dim = int(features_dim)

            @property
            def features_dim(self) -> int:
                return self._features_dim
    else:
        BaseFeaturesExtractor = object

try:
    from stable_baselines3.common.policies import ActorCriticPolicy
    from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
except Exception:  # pragma: no cover - deployment dependency
    ActorCriticPolicy = object
    RecurrentMultiInputActorCriticPolicy = object


class FiLMFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor used by the V13/V14/V17 multi-input policies."""

    def __init__(self, observation_space, image_feat_dim: int = 128,
                 state_feat_dim: int = 32):
        if torch is None:
            raise ImportError("PyTorch is required for V17Pilot")
        features_dim = int(image_feat_dim) + int(state_feat_dim)
        super().__init__(observation_space, features_dim=features_dim)

        n_ch = int(observation_space["image"].shape[0])
        state_dim = int(observation_space["state"].shape[0])

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_ch, 32, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 4 * 4, image_feat_dim), torch.nn.ReLU(),
        )
        self.image_norm = torch.nn.LayerNorm(image_feat_dim)
        self.state_enc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64), torch.nn.ReLU()
        )
        self.film_head = torch.nn.Linear(64, image_feat_dim * 2)
        torch.nn.init.zeros_(self.film_head.weight)
        torch.nn.init.zeros_(self.film_head.bias)
        self.state_feat_head = torch.nn.Sequential(
            torch.nn.Linear(64, state_feat_dim), torch.nn.ReLU()
        )
        self.fused_norm = torch.nn.LayerNorm(image_feat_dim)

    def forward(self, obs: Dict[str, Any]):
        image_feat = self.image_norm(self.cnn(obs["image"]))
        state_h = self.state_enc(obs["state"])
        gamma_raw, beta_raw = self.film_head(state_h).chunk(2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)
        beta = 0.1 * torch.tanh(beta_raw)
        fused = self.fused_norm(gamma * image_feat + beta)
        state_feat = self.state_feat_head(state_h)
        return torch.cat([fused, state_feat], dim=-1)


class ResidualBlock1D(torch.nn.Module if torch is not None else object):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )
        self.act = torch.nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class SharedSideLidarEncoder(torch.nn.Module if torch is not None else object):
    def __init__(self, sectors_per_side: int = 18):
        super().__init__()
        self.sectors_per_side = int(sectors_per_side)
        self.conv_in = torch.nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.res1 = ResidualBlock1D(32, kernel_size=3, padding=1)
        self.res2 = ResidualBlock1D(32, kernel_size=3, padding=1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32 * self.sectors_per_side, 96),
            torch.nn.ReLU(),
            torch.nn.Linear(96, 48),
            torch.nn.ReLU(),
        )

    def forward(self, side_obs):
        h = torch.relu(self.conv_in(side_obs))
        h = self.res1(h)
        h = self.res2(h)
        h = torch.flatten(h, start_dim=1)
        return self.mlp(h)


class PooledLidarEncoder(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(2, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 96),
            torch.nn.ReLU(),
        )

    def forward(self, lidar_obs):
        return self.net(lidar_obs)


class LiDARFiLMFeatureExtractor(BaseFeaturesExtractor):
    """V17 feature extractor: image + state + canonical LiDAR + metadata."""

    def __init__(self, observation_space, image_feat_dim: int = 128,
                 state_feat_dim: int = 32, lidar_feat_dim: int = 96,
                 lidar_encoder_mode: str = "side_separated",
                 disable_lidar_meta: bool = False):
        if torch is None:
            raise ImportError("PyTorch is required for V17Pilot")
        features_dim = int(image_feat_dim + state_feat_dim + lidar_feat_dim)
        super().__init__(observation_space, features_dim=features_dim)

        image_shape = observation_space["image"].shape
        state_dim = int(observation_space["state"].shape[0])
        lidar_dim = int(observation_space["lidar"].shape[0])
        lidar_meta_dim = int(observation_space["lidar_meta"].shape[0])
        n_ch = int(image_shape[0])

        self.disable_lidar_meta = bool(disable_lidar_meta)
        self.lidar_dim = int(lidar_dim)
        self.lidar_num_sectors = 0 if self.lidar_dim == 12 else self.lidar_dim // 2
        self.lidar_side_sectors = 0 if self.lidar_dim == 12 else self.lidar_num_sectors // 2
        self.lidar_encoder_mode = str(lidar_encoder_mode).strip().lower()

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_ch, 32, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 4 * 4, image_feat_dim),
            torch.nn.ReLU(),
        )
        self.image_norm = torch.nn.LayerNorm(image_feat_dim)

        state_input_dim = int(state_dim + (0 if self.disable_lidar_meta else lidar_meta_dim))
        self.state_enc = torch.nn.Sequential(
            torch.nn.Linear(state_input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, state_feat_dim),
            torch.nn.ReLU(),
        )
        self.lidar_side_encoder = SharedSideLidarEncoder(
            sectors_per_side=max(1, self.lidar_side_sectors)
        )
        self.lidar_pooled_encoder = PooledLidarEncoder()
        self.lidar_target_encoder = torch.nn.Sequential(
            torch.nn.Linear(12, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, lidar_feat_dim),
            torch.nn.ReLU(),
        )
        self.film_hidden = torch.nn.Linear(state_feat_dim + lidar_feat_dim, 192)
        self.film_act = torch.nn.ReLU()
        self.film_out = torch.nn.Linear(192, image_feat_dim * 2)
        torch.nn.init.zeros_(self.film_out.weight)
        torch.nn.init.zeros_(self.film_out.bias)
        self.fused_norm = torch.nn.LayerNorm(image_feat_dim)

    def _encode_lidar(self, lidar_flat):
        lidar_flat = lidar_flat.float()
        if self.lidar_dim == 12:
            return self.lidar_target_encoder(lidar_flat)
        lidar_range, lidar_valid = torch.chunk(lidar_flat, 2, dim=-1)
        lidar_2c = torch.stack([lidar_range, lidar_valid], dim=1)
        if self.lidar_encoder_mode == "pooled":
            return self.lidar_pooled_encoder(lidar_2c)
        mid = self.lidar_side_sectors
        left = lidar_2c[:, :, :mid]
        right = lidar_2c[:, :, mid:]
        left_feat = self.lidar_side_encoder(left)
        right_feat = self.lidar_side_encoder(right)
        return torch.cat([left_feat, right_feat], dim=-1)

    def forward(self, obs: Dict[str, Any]):
        image_feat = self.image_norm(self.cnn(obs["image"].float()))
        if self.disable_lidar_meta:
            state_aug = obs["state"].float()
        else:
            state_aug = torch.cat([obs["state"].float(), obs["lidar_meta"].float()], dim=-1)
        state_feat = self.state_enc(state_aug)
        lidar_feat = self._encode_lidar(obs["lidar"])
        film_in = torch.cat([state_feat, lidar_feat], dim=-1)
        film_hidden = self.film_act(self.film_hidden(film_in))
        gamma_raw, beta_raw = self.film_out(film_hidden).chunk(2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)
        beta = 0.1 * torch.tanh(beta_raw)
        fused = self.fused_norm(gamma * image_feat + beta)
        return torch.cat([fused, state_feat, lidar_feat], dim=-1)


class V17RecurrentMultiInputPolicy(RecurrentMultiInputActorCriticPolicy):
    """V17 recurrent policy class with optional critic-only domain routing."""

    def __init__(self, *args, domain_obs_key: str = "domain_id", **kwargs):
        self.domain_obs_key = str(domain_obs_key)
        self.use_dual_value_heads = False
        super().__init__(*args, **kwargs)
        self.value_head_ws = torch.nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.value_head_gt = torch.nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.value_head_ws.to(self.device)
        self.value_head_gt.to(self.device)
        self._copy_shared_value_to_dual()
        self.optimizer.add_param_group({"params": self.value_head_ws.parameters()})
        self.optimizer.add_param_group({"params": self.value_head_gt.parameters()})

    def _copy_shared_value_to_dual(self) -> None:
        with torch.no_grad():
            self.value_head_ws.weight.copy_(self.value_net.weight)
            self.value_head_ws.bias.copy_(self.value_net.bias)
            self.value_head_gt.weight.copy_(self.value_net.weight)
            self.value_head_gt.bias.copy_(self.value_net.bias)

    def activate_dual_value_heads(self) -> None:
        self._copy_shared_value_to_dual()
        self.use_dual_value_heads = True

    def deactivate_dual_value_heads(self) -> None:
        self.use_dual_value_heads = False

    def _domain_is_gt(self, obs, batch_size: int, device):
        if not isinstance(obs, dict):
            return None
        domain = obs.get(self.domain_obs_key)
        if domain is None:
            return None
        if not isinstance(domain, torch.Tensor):
            domain = torch.as_tensor(domain, device=device)
        domain = domain.float().reshape(batch_size, -1)
        return (domain[:, 0] > 0.5).float().unsqueeze(-1)

    def _value_from_latent(self, latent_vf, obs):
        if not self.use_dual_value_heads:
            return self.value_net(latent_vf)
        mask_gt = self._domain_is_gt(obs, batch_size=latent_vf.shape[0], device=latent_vf.device)
        if mask_gt is None:
            return self.value_net(latent_vf)
        value_ws = self.value_head_ws(latent_vf)
        value_gt = self.value_head_gt(latent_vf)
        return (1.0 - mask_gt) * value_ws + mask_gt * value_gt

    def forward(self, obs, lstm_states, episode_starts, deterministic: bool = False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, lstm_states_pi = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
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

    def predict_values(self, obs, lstm_states, episode_starts):
        features = super(ActorCriticPolicy, self).extract_features(
            obs, self.vf_features_extractor
        )
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                features, lstm_states, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_pi, _ = self._process_sequence(
                features, lstm_states, episode_starts, self.lstm_actor
            )
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self._value_from_latent(latent_vf, obs)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
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


class V17ManualPolicy(torch.nn.Module if torch is not None else object):
    """Pure PyTorch deterministic actor for Jetson envs without RecurrentPPO."""

    def __init__(self, observation_space, device: str = "cpu"):
        super().__init__()
        self.device_name = str(device)
        self.observation_space = observation_space
        self.features_extractor = LiDARFiLMFeatureExtractor(
            observation_space,
            image_feat_dim=128,
            state_feat_dim=32,
            lidar_feat_dim=96,
            lidar_encoder_mode="side_separated",
            disable_lidar_meta=False,
        )
        self.lstm_actor = torch.nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
        )
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
        )
        self.action_net = torch.nn.Linear(64, 3)
        self.to(self.device_name)
        self.reset()

    def reset(self) -> None:
        dev = torch.device(self.device_name)
        self.h = torch.zeros((2, 1, 256), dtype=torch.float32, device=dev)
        self.c = torch.zeros((2, 1, 256), dtype=torch.float32, device=dev)

    def load_sb3_policy_state_dict(self, state_dict: Dict[str, Any]) -> None:
        fx = {}
        for key, value in state_dict.items():
            if key.startswith("pi_features_extractor."):
                fx[key.replace("pi_features_extractor.", "", 1)] = value
            elif key.startswith("features_extractor."):
                fx[key.replace("features_extractor.", "", 1)] = value
        self.features_extractor.load_state_dict(fx, strict=False)

        own = self.state_dict()
        copy_keys = [
            key for key in own.keys()
            if key.startswith("lstm_actor.")
            or key.startswith("policy_net.")
            or key.startswith("action_net.")
        ]
        sub = {}
        for key in copy_keys:
            src_key = key
            if key.startswith("policy_net."):
                src_key = "mlp_extractor." + key
            if src_key in state_dict:
                sub[key] = state_dict[src_key]
        missing = sorted(set(copy_keys) - set(sub.keys()))
        if missing:
            raise ValueError(f"manual V17 policy missing keys: {missing[:5]}")
        self.load_state_dict(sub, strict=False)
        self.eval()

    def _tensor_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        dev = torch.device(self.device_name)
        out = {}
        for key, value in obs.items():
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == len(self.observation_space[key].shape):
                arr = arr[None]
            out[key] = torch.as_tensor(arr, dtype=torch.float32, device=dev)
        return out

    def predict_np(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            obs_t = self._tensor_obs(obs)
            features = self.features_extractor(obs_t)
            seq = features.unsqueeze(0)
            out, (self.h, self.c) = self.lstm_actor(seq, (self.h, self.c))
            latent = out.squeeze(0)
            latent = self.policy_net(latent)
            action = self.action_net(latent)
            action = torch.clamp(action, -1.0, 1.0)
            return action.detach().cpu().numpy()[0]


def _install_actor_module_alias() -> None:
    """Allow SB3 archives saved with module.actor.FiLMFeatureExtractor to load."""
    if "module.actor" in sys.modules:
        return
    pkg = sys.modules.get("module")
    if pkg is None:
        pkg = types.ModuleType("module")
        sys.modules["module"] = pkg
    actor_mod = types.ModuleType("module.actor")
    actor_mod.FiLMFeatureExtractor = FiLMFeatureExtractor
    sys.modules["module.actor"] = actor_mod
    setattr(pkg, "actor", actor_mod)


def _install_v17_policy_module_alias() -> None:
    pkg = sys.modules.get("module")
    if pkg is None:
        pkg = types.ModuleType("module")
        pkg.__path__ = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "module")]
        sys.modules["module"] = pkg
    v17_mod = types.ModuleType("module.v17_policy")
    v17_mod.ResidualBlock1D = ResidualBlock1D
    v17_mod.SharedSideLidarEncoder = SharedSideLidarEncoder
    v17_mod.PooledLidarEncoder = PooledLidarEncoder
    v17_mod.LiDARFiLMFeatureExtractor = LiDARFiLMFeatureExtractor
    v17_mod.V17RecurrentMultiInputPolicy = V17RecurrentMultiInputPolicy
    sys.modules["module.v17_policy"] = v17_mod
    setattr(pkg, "v17_policy", v17_mod)


def _load_canonical_semantic_wrapper():
    """Load module.obv without executing module/__init__.py on Python 3.6 Jetson."""
    candidate_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "module"),
        os.path.join(os.getcwd(), "module"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "module"),
    ]
    for module_dir in candidate_dirs:
        obv_path = os.path.join(module_dir, "obv.py")
        if not os.path.exists(obv_path):
            continue
        pkg = sys.modules.get("module")
        if pkg is None or not hasattr(pkg, "__path__"):
            pkg = types.ModuleType("module")
            pkg.__path__ = [module_dir]
            sys.modules["module"] = pkg
        elif module_dir not in pkg.__path__:
            pkg.__path__.append(module_dir)

        existing = sys.modules.get("module.obv")
        if existing is not None and hasattr(existing, "CanonicalSemanticWrapper"):
            return existing.CanonicalSemanticWrapper

        spec = importlib.util.spec_from_file_location("module.obv", obv_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["module.obv"] = mod
        spec.loader.exec_module(mod)
        return mod.CanonicalSemanticWrapper

    from module.obv import CanonicalSemanticWrapper
    return CanonicalSemanticWrapper


def _import_recurrent_ppo():
    try:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    except Exception:
        from sb3_contrib.ppo_recurrent import RecurrentPPO
        return RecurrentPPO


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def _wrap_angle_deg_pm180(angle_deg: float) -> float:
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


@dataclass
class ShadowActionAdapter:
    """Standalone copy of the V13 3D high-level action adapter."""

    k_delta: float = 0.15
    lambda_bias: float = 0.20
    k_bias: float = 0.15
    steer_core_decay: float = 0.0
    v_nominal: float = 1.15
    k_turn: float = 0.55
    k_bias_speed: float = 0.0
    alpha_speed: float = 0.55
    v_min: float = 0.35
    v_max: float = 2.0
    speed_kp: float = 0.35
    speed_ki: float = 0.08
    speed_kff: float = 0.10
    control_dt: float = 0.05
    integral_limit: float = 3.0
    max_throttle: float = 0.8
    allow_reverse: bool = False

    def __post_init__(self) -> None:
        self.steer_core = 0.0
        self.bias_smooth = 0.0
        self.i_term = 0.0
        self.last_speed_mps = 0.0
        self.last_low_level_action = np.array([0.0, 0.0], dtype=np.float32)
        self.diag: Dict[str, float] = {}

    def reset(self) -> None:
        self.__post_init__()

    def action(self, action: Sequence[float], speed_mps: float) -> np.ndarray:
        delta_steer = _clip(float(action[0]), -1.0, 1.0)
        speed_scale = _clip(float(action[1]), -1.0, 1.0)
        line_bias = _clip(float(action[2]), -1.0, 1.0)

        if self.steer_core_decay > 0.0:
            self.steer_core *= 1.0 - self.steer_core_decay
        self.steer_core = _clip(
            self.steer_core + self.k_delta * delta_steer, -1.0, 1.0
        )
        self.bias_smooth = (
            (1.0 - self.lambda_bias) * self.bias_smooth
            + self.lambda_bias * line_bias
        )
        bias_offset = self.k_bias * self.bias_smooth
        steer_target = _clip(self.steer_core + bias_offset, -1.0, 1.0)

        v_base = _clip(
            self.v_nominal
            - self.k_turn * abs(steer_target)
            - self.k_bias_speed * abs(self.bias_smooth),
            self.v_min,
            self.v_max,
        )
        v_ref = _clip(
            v_base * (1.0 + self.alpha_speed * speed_scale),
            self.v_min,
            self.v_max,
        )
        self.last_speed_mps = max(0.0, float(speed_mps))
        v_err = v_ref - self.last_speed_mps
        self.i_term = _clip(
            self.i_term + v_err * self.control_dt,
            -self.integral_limit,
            self.integral_limit,
        )
        throttle = (
            self.speed_kff * v_ref
            + self.speed_kp * v_err
            + self.speed_ki * self.i_term
        )
        if self.allow_reverse:
            throttle = _clip(throttle, -self.max_throttle, self.max_throttle)
        else:
            throttle = _clip(throttle, 0.0, self.max_throttle)

        low_level = np.array([steer_target, throttle], dtype=np.float32)
        self.last_low_level_action = low_level
        self.diag = {
            "v_target": float(v_ref),
            "v_meas": float(self.last_speed_mps),
            "target_steer": float(steer_target),
            "steer_core": float(self.steer_core),
            "bias_smooth": float(self.bias_smooth),
        }
        return low_level


@dataclass
class ShadowActionSafety:
    """Standalone ActionSafetyWrapper for the policy -> adapter -> safety chain."""

    delta_max: float = 0.35
    enable_lpf: bool = True
    beta: float = 0.6

    def __post_init__(self) -> None:
        self.delta_max = _clip(float(self.delta_max), 0.0, 1.0)
        self.beta = _clip(float(self.beta), 0.0, 1.0)
        self.steer_prev_limited = 0.0
        self.steer_prev_exec = 0.0
        self.delta_steer_prev = 0.0
        self.diag: Dict[str, Any] = self._zero_diag()

    def _zero_diag(self) -> Dict[str, Any]:
        return {
            "steer_raw": 0.0,
            "steer_exec": 0.0,
            "delta_steer": 0.0,
            "delta_steer_prev": 0.0,
            "rate_limit_hit": False,
            "rate_excess_raw": 0.0,
            "rate_excess_bounded": 0.0,
            "steer_clip_hit": False,
            "mismatch": 0.0,
            "effective_delta_max": float(self.delta_max),
        }

    def reset(self) -> None:
        self.__post_init__()

    def action(self, action: Sequence[float]) -> np.ndarray:
        steer_raw = float(action[0])
        throttle = float(action[1])
        effective_delta_max = float(self.delta_max)
        delta = steer_raw - self.steer_prev_limited
        rate_limit_hit = abs(delta) > effective_delta_max
        rate_excess_raw = (
            max(0.0, abs(delta) - effective_delta_max)
            / max(effective_delta_max, 1e-6)
            if effective_delta_max > 0.0
            else 0.0
        )
        if effective_delta_max > 0.0 and abs(delta) > effective_delta_max:
            delta = _clip(delta, -effective_delta_max, effective_delta_max)
        steer_limited = self.steer_prev_limited + delta

        if self.enable_lpf:
            steer_exec = (1.0 - self.beta) * self.steer_prev_exec + self.beta * steer_limited
        else:
            steer_exec = steer_limited
        steer_exec = _clip(steer_exec, -1.0, 1.0)

        actual_delta = steer_exec - self.steer_prev_exec
        self.diag = {
            "steer_raw": steer_raw,
            "steer_exec": steer_exec,
            "delta_steer": actual_delta,
            "delta_steer_prev": self.delta_steer_prev,
            "rate_limit_hit": rate_limit_hit,
            "rate_excess_raw": rate_excess_raw,
            "rate_excess_bounded": float(math.tanh(rate_excess_raw)),
            "steer_clip_hit": abs(steer_exec) > 0.95,
            "mismatch": steer_raw - steer_exec,
            "effective_delta_max": effective_delta_max,
        }
        self.delta_steer_prev = actual_delta
        self.steer_prev_limited = steer_limited
        self.steer_prev_exec = steer_exec
        return np.array([steer_exec, throttle], dtype=np.float32)


class V17SemanticPreprocessor:
    """Build the 6-channel semantic image expected by the V17 policy."""

    channels = (
        "raw_Y",
        "edge_line_prob",
        "guide_line_prob",
        "sobel_edge",
        "vehicle_prob",
        "motion_residual",
    )

    def __init__(self, obs_size: int = 128, domain: str = "ws",
                 prefer_official: bool = True):
        self.obs_size = int(obs_size)
        self.domain = str(domain).lower()
        self.prev_gray: Optional[np.ndarray] = None
        self.kernel = np.ones((3, 3), np.uint8)
        self.official = None
        if prefer_official:
            self.official = self._try_make_official_wrapper()

    def _try_make_official_wrapper(self):
        if gym is None or spaces is None:
            return None
        try:
            CanonicalSemanticWrapper = _load_canonical_semantic_wrapper()
        except Exception as exc:
            print(f"V17 semantic fallback: module.obv unavailable ({exc})")
            return None

        class DummyEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(
                    low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
                )

            def reset(self):
                return np.zeros((224, 224, 3), dtype=np.uint8)

            def step(self, action):
                return self.reset(), 0.0, False, {}

        try:
            wrapper = CanonicalSemanticWrapper(
                DummyEnv(), domain=self.domain, obs_size=self.obs_size,
                augment=False
            )
            print(
                f"V17 semantic preprocessor: using CanonicalSemanticWrapper "
                f"domain={self.domain}"
            )
            return wrapper
        except Exception as exc:
            print(f"V17 semantic fallback: CanonicalSemanticWrapper init failed ({exc})")
            return None

    def reset(self) -> None:
        self.prev_gray = None
        if self.official is not None:
            try:
                self.official.reset()
            except Exception:
                pass

    def __call__(self, img_arr: np.ndarray) -> np.ndarray:
        img = np.asarray(img_arr)
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError("Expected RGB image with shape (H, W, 3)")
        if self.official is not None:
            return np.asarray(self.official.observation(img), dtype=np.float32)
        rgb = img[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        resized = cv2.resize(
            rgb, (self.obs_size, self.obs_size), interpolation=cv2.INTER_LINEAR
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        gray_f = gray.astype(np.float32) / 255.0

        hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        yellow = cv2.inRange(
            hsv, np.array([15, 60, 60], dtype=np.uint8),
            np.array([40, 255, 255], dtype=np.uint8)
        )
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, self.kernel)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, self.kernel)
        guide_line = yellow.astype(np.float32) / 255.0

        edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
        sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
        sobel = sobel / (float(np.max(sobel)) + 1e-6)

        green = cv2.inRange(
            hsv, np.array([35, 45, 35], dtype=np.uint8),
            np.array([90, 255, 255], dtype=np.uint8)
        )
        vehicle_prob = green.astype(np.float32) / 255.0

        if self.prev_gray is None:
            motion = np.zeros_like(gray_f, dtype=np.float32)
        else:
            motion = np.abs(gray_f - self.prev_gray)
        self.prev_gray = gray_f.copy()

        stacked = np.stack(
            [gray_f, edges, guide_line, sobel, vehicle_prob, motion],
            axis=0,
        ).astype(np.float32)
        return np.clip(stacked, 0.0, 1.0)


class V17Pilot:
    """DonkeyCar part that publishes V17 policy outputs and latency."""

    def __init__(
        self,
        model_path: str,
        obs_size: int = 128,
        state_dim: Optional[int] = None,
        domain: str = "ws",
        max_throttle: float = 0.8,
        delta_max: float = 0.35,
        enable_lpf: bool = True,
        beta: float = 0.6,
        use_cuda: bool = True,
        serial_reader: Any = None,
        lidar_reader: Any = None,
        warmup_frames: int = 5,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for V17Pilot")
        self.model_path = os.path.abspath(os.path.expanduser(model_path))
        self.model_name = os.path.basename(self.model_path)
        self.obs_size = int(obs_size)
        self.image_channels = 6
        self.state_dim = int(state_dim) if state_dim else 7
        self.lidar_dim = 144
        self.lidar_meta_dim = 2
        self.lidar_fov_deg = 360.0
        self.observation_keys = ["image", "state", "lidar", "lidar_meta", "domain_id"]
        self.max_throttle = float(max_throttle)
        self.domain = str(domain).lower()
        self.serial_reader = serial_reader
        self.lidar_reader = lidar_reader
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.backend = "PyTorch/SB3 RecurrentPPO"

        self.preprocessor = V17SemanticPreprocessor(
            obs_size=self.obs_size, domain=self.domain, prefer_official=True
        )
        self.adapter = ShadowActionAdapter(max_throttle=self.max_throttle)
        self.safety = ShadowActionSafety(
            delta_max=delta_max, enable_lpf=enable_lpf, beta=beta
        )
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.inference_count = 0
        self.total_latency_ms = 0.0
        self.total_preprocess_ms = 0.0
        self._last_error_print = 0.0
        self._last_lidar_frame = -1
        self._lidar_stale_steps = 999

        self.model = self._load_model()
        self._warmup(max(0, int(warmup_frames)))

        print("V17Pilot initialized")
        print(f"   model: {self.model_path}")
        print(f"   device: {self.device}")
        print(f"   obs: image=6x{self.obs_size}x{self.obs_size}, state={self.state_dim}")
        print(f"   output: pilot/angle, pilot/throttle, latency")

    @property
    def metadata(self) -> Dict[str, Any]:
        size_mb = -1.0
        try:
            size_mb = os.path.getsize(self.model_path) / (1024.0 * 1024.0)
        except Exception:
            pass
        modalities = "image+state"
        if "lidar" in self.observation_keys:
            modalities = "image+state+LiDAR"
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_size_MB": round(size_mb, 3) if size_mb >= 0 else -1.0,
            "backend": self.backend,
            "input_resolution": f"{self.obs_size}x{self.obs_size}",
            "input_modalities": modalities,
            "control_mode": "shadow",
            "policy_chain": "CanonicalSemanticWrapper -> RecurrentPPO -> ActionAdapterWrapper -> ActionSafetyWrapper",
        }

    def _make_dummy_vec_env(self, obs_size: int, state_dim: int):
        if gym is None or spaces is None:
            raise ImportError("gym is required to load V17 RecurrentPPO models")
        from stable_baselines3.common.vec_env import DummyVecEnv

        class DummyEnv(gym.Env):
            def __init__(self):
                self.observation_space = self_make_obs_space(obs_size, state_dim)
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(3,), dtype=np.float32
                )

            def reset(self):
                return {
                    "image": np.zeros((6, int(obs_size), int(obs_size)), dtype=np.float32),
                    "state": np.zeros((int(state_dim),), dtype=np.float32),
                    "lidar": np.concatenate([
                        np.full((72,), 20.0, dtype=np.float32),
                        np.zeros((72,), dtype=np.float32),
                    ]),
                    "lidar_meta": np.array([0.0, 1.0], dtype=np.float32),
                    "domain_id": np.array([0.0], dtype=np.float32),
                }

            def step(self, action):
                return self.reset(), 0.0, False, {}

        def self_make_obs_space(_obs_size, _state_dim):
            return self._make_v17_observation_space(_obs_size, _state_dim)

        return DummyVecEnv([lambda: DummyEnv()])

    def _make_v17_observation_space(self, obs_size: int, state_dim: int):
        if spaces is None:
            raise ImportError("gym.spaces is required for V17 observation space")
        return spaces.Dict({
            "image": spaces.Box(
                low=0.0, high=1.0,
                shape=(6, int(obs_size), int(obs_size)),
                dtype=np.float32,
            ),
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(int(state_dim),),
                dtype=np.float32,
            ),
            "lidar": spaces.Box(
                low=0.0, high=20.0, shape=(144,), dtype=np.float32
            ),
            "lidar_meta": spaces.Box(
                low=0.0, high=4.0, shape=(2,), dtype=np.float32
            ),
            "domain_id": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
        })

    def _infer_state_dim(self, state_dict: Dict[str, Any]) -> int:
        for key, value in state_dict.items():
            if key.endswith("features_extractor.state_enc.0.weight"):
                try:
                    dim = int(value.shape[1])
                    return dim - 2 if dim > 7 else dim
                except Exception:
                    pass
            if key.endswith("pi_features_extractor.state_enc.0.weight"):
                try:
                    dim = int(value.shape[1])
                    return dim - 2 if dim > 7 else dim
                except Exception:
                    pass
        return self.state_dim

    def _load_policy_state_from_zip(self, model_path: str):
        with zipfile.ZipFile(model_path, "r") as archive:
            payload = archive.read("policy.pth")
        return torch.load(io.BytesIO(payload), map_location=self.device)

    def _load_manual_policy(self):
        if os.path.splitext(self.model_path)[1].lower() == ".zip":
            state_dict = self._load_policy_state_from_zip(self.model_path)
        else:
            state_dict = torch.load(self.model_path, map_location=self.device)
        self.state_dim = self._infer_state_dim(state_dict)
        obs_space = self._make_v17_observation_space(self.obs_size, self.state_dim)
        manual = V17ManualPolicy(obs_space, device=self.device)
        manual.load_sb3_policy_state_dict(state_dict)
        self.backend = "PyTorch/manual SB3 RecurrentPPO actor"
        self.observation_keys = list(obs_space.spaces.keys())
        self.lidar_dim = int(obs_space["lidar"].shape[0])
        self.lidar_meta_dim = int(obs_space["lidar_meta"].shape[0])
        print("V17Pilot: using pure PyTorch manual RecurrentPPO actor")
        return manual

    def _load_model(self):
        _install_actor_module_alias()
        _install_v17_policy_module_alias()
        try:
            RecurrentPPO = _import_recurrent_ppo()
        except Exception as exc:
            print(f"V17Pilot: RecurrentPPO unavailable ({exc}); using manual actor")
            return self._load_manual_policy()

        suffix = os.path.splitext(self.model_path)[1].lower()
        if suffix == ".pth":
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.state_dim = self._infer_state_dim(state_dict)
            env = self._make_dummy_vec_env(self.obs_size, self.state_dim)
            policy_kwargs = dict(
                features_extractor_class=LiDARFiLMFeatureExtractor,
                features_extractor_kwargs=dict(
                    image_feat_dim=128,
                    state_feat_dim=32,
                    lidar_feat_dim=96,
                    lidar_encoder_mode="side_separated",
                    disable_lidar_meta=False,
                ),
                lstm_hidden_size=256,
                n_lstm_layers=2,
                shared_lstm=False,
                enable_critic_lstm=True,
            )
            model = RecurrentPPO(
                V17RecurrentMultiInputPolicy,
                env,
                policy_kwargs=policy_kwargs,
                device=self.device,
                verbose=0,
            )
            model.policy.load_state_dict(state_dict, strict=False)
            model.policy.to(self.device)
            model.policy.eval()
            return model

        try:
            model = RecurrentPPO.load(self.model_path, device=self.device)
        except Exception as exc:
            print(f"V17Pilot: SB3 load failed ({exc}); using manual actor")
            return self._load_manual_policy()
        obs_space = getattr(model, "observation_space", None)
        try:
            self.observation_keys = list(obs_space.spaces.keys())
            self.obs_size = int(obs_space["image"].shape[-1])
            self.image_channels = int(obs_space["image"].shape[0])
            self.state_dim = int(obs_space["state"].shape[0])
            if "lidar" in obs_space.spaces:
                self.lidar_dim = int(obs_space["lidar"].shape[0])
            if "lidar_meta" in obs_space.spaces:
                self.lidar_meta_dim = int(obs_space["lidar_meta"].shape[0])
            if self.obs_size != self.preprocessor.obs_size:
                self.preprocessor = V17SemanticPreprocessor(
                    obs_size=self.obs_size, domain=self.domain,
                    prefer_official=True
                )
        except Exception:
            pass
        return model

    def _warmup(self, n_frames: int) -> None:
        if n_frames <= 0:
            return
        fake = np.zeros((224, 224, 3), dtype=np.uint8)
        for _ in range(n_frames):
            self.run(fake, 0.0, 0.0, 0.0, 0.0)
        self.reset()

    def reset(self) -> None:
        self.preprocessor.reset()
        self.adapter.reset()
        self.safety.reset()
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        if hasattr(self.model, "reset"):
            self.model.reset()

    def _sensor_snapshot(self) -> Dict[str, float]:
        if self.serial_reader is not None and getattr(self.serial_reader, "is_connected", False):
            try:
                return dict(self.serial_reader.get_data())
            except Exception:
                pass
        return {}

    def _lidar_snapshot(self) -> Dict[str, float]:
        if self.lidar_reader is not None and getattr(self.lidar_reader, "is_connected", False):
            try:
                return dict(self.lidar_reader.get_data())
            except Exception:
                pass
        return {}

    def _estimate_speed_mps(self, sensors: Dict[str, Any],
                            user_throttle: Optional[float],
                            final_throttle: Optional[float]) -> float:
        left = float(sensors.get("motor_lvel", 0.0) or 0.0)
        right = float(sensors.get("motor_rvel", 0.0) or 0.0)
        enc = abs(0.5 * (left + right))
        if enc > 1e-6:
            return _clip(enc / 1000.0, 0.0, 3.0)
        throttle = final_throttle if final_throttle is not None else user_throttle
        return _clip(abs(float(throttle or 0.0)) * 2.2, 0.0, 3.0)

    def _build_lidar_obs(self, lidar: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        if self.lidar_dim == 12:
            nearest = float(lidar.get("nearest_min", -1.0) or -1.0)
            present = 1.0 if 0.0 < nearest < 20.0 else 0.0
            token = np.zeros((12,), dtype=np.float32)
            token[0] = present
            if present:
                token[1] = nearest
                token[5] = 1.0
            return token, self._build_lidar_meta(lidar)

        sectors = max(1, self.lidar_dim // 2)
        max_range = 20.0
        min_range = 0.18
        ranges_out = np.full((sectors,), max_range, dtype=np.float32)
        valid_out = np.zeros((sectors,), dtype=np.float32)

        raw_ranges = lidar.get("ranges") or []
        if raw_ranges:
            try:
                arr = np.asarray(raw_ranges, dtype=np.float32).reshape(-1)
            except Exception:
                values = []
                for value in raw_ranges:
                    try:
                        v = float(value)
                    except Exception:
                        v = float("nan")
                    values.append(v)
                arr = np.asarray(values, dtype=np.float32)
            finite = np.isfinite(arr)
            good = np.zeros_like(finite, dtype=bool)
            good[finite] = arr[finite] >= min_range
            n = int(arr.shape[0])
            angle_inc = float(lidar.get("angle_increment", 0.0) or 0.0)
            if abs(angle_inc) > 1e-9:
                angle_min = float(lidar.get("angle_min", 0.0) or 0.0)
                beam_idx = np.arange(n, dtype=np.float32)
                angles_deg = np.degrees(angle_min + beam_idx * angle_inc)
                angles_deg = ((angles_deg + 180.0) % 360.0 - 180.0).astype(np.float32)
                half_fov = 0.5 * float(self.lidar_fov_deg)
                edges = np.linspace(half_fov, -half_fov, sectors + 1, dtype=np.float32)
                clipped = np.clip(arr, min_range, max_range)
                for idx in range(sectors):
                    hi = edges[idx]
                    lo = edges[idx + 1]
                    if idx == 0:
                        mask = good & (angles_deg <= hi) & (angles_deg >= lo)
                    else:
                        mask = good & (angles_deg < hi) & (angles_deg >= lo)
                    if np.any(mask):
                        ranges_out[idx] = float(np.quantile(clipped[mask], 0.20))
                        valid_out[idx] = 1.0
            else:
                for idx in range(sectors):
                    start = int(round(idx * n / float(sectors)))
                    end = int(round((idx + 1) * n / float(sectors)))
                    chunk = arr[start:max(start + 1, end)]
                    mask = good[start:max(start + 1, end)]
                    if chunk.size and np.any(mask):
                        vals = np.clip(chunk[mask], min_range, max_range)
                        ranges_out[idx] = float(np.quantile(vals, 0.20))
                        valid_out[idx] = 1.0

        return np.concatenate([ranges_out, valid_out]).astype(np.float32), self._build_lidar_meta(lidar)

    def _build_lidar_meta(self, lidar: Dict[str, Any]) -> np.ndarray:
        frame = int(lidar.get("frame_count", 0) or 0)
        if frame > 0 and frame != self._last_lidar_frame:
            is_new = 1.0
            self._lidar_stale_steps = 0
            self._last_lidar_frame = frame
        else:
            is_new = 0.0
            self._lidar_stale_steps += 1
        stale_norm = _clip(self._lidar_stale_steps / 4.0, 0.0, 1.0)
        return np.array([is_new, stale_norm], dtype=np.float32)

    def _build_state(self, sensors: Dict[str, Any], lidar: Dict[str, Any],
                     user_throttle: Optional[float],
                     final_throttle: Optional[float]) -> Tuple[np.ndarray, float]:
        speed_mps = self._estimate_speed_mps(sensors, user_throttle, final_throttle)
        gyro_z = float(sensors.get("gyro_z", 0.0) or 0.0)
        accel_x = float(sensors.get("accel_x", 0.0) or 0.0)
        prev_throttle = float(self.adapter.last_low_level_action[1])
        prev_steer_exec = float(self.safety.steer_prev_exec)

        state = [
            _clip(speed_mps / 2.2, 0.0, 2.0),
            _clip((-gyro_z) / 8.0, -2.0, 2.0),
            _clip(accel_x / 9.8, -2.0, 2.0),
            _clip(prev_steer_exec, -1.0, 1.0),
            _clip(prev_throttle, -1.0, 1.0),
            _clip(float(self.adapter.steer_core), -1.0, 1.0),
            _clip(float(self.adapter.bias_smooth), -1.0, 1.0),
        ]

        if self.state_dim > 7:
            nearest = float(lidar.get("nearest_min", -1.0) or -1.0)
            present = 1.0 if 0.0 < nearest < 1.8 else 0.0
            obstacle_extra = [
                present,
                _clip(nearest / 6.0 if nearest > 0 else 0.0, -2.0, 2.0),
                0.0,
                _clip(nearest / 6.0 if nearest > 0 else 0.0, 0.0, 2.0),
                _clip(1.0 - nearest / 1.8 if present else 0.0, 0.0, 1.0),
            ]
            state.extend(obstacle_extra)

        if len(state) < self.state_dim:
            state.extend([0.0] * (self.state_dim - len(state)))
        return np.asarray(state[:self.state_dim], dtype=np.float32), speed_mps

    def _predict_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        if hasattr(self.model, "predict_np"):
            return np.asarray(self.model.predict_np(obs), dtype=np.float32).reshape(-1)
        action, self.lstm_states = self.model.predict(
            obs,
            state=self.lstm_states,
            episode_start=self.episode_starts,
            deterministic=True,
        )
        self.episode_starts = np.zeros((1,), dtype=bool)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def _build_model_obs(self, image: np.ndarray, state: np.ndarray,
                         lidar_obs: np.ndarray, lidar_meta: np.ndarray) -> Dict[str, np.ndarray]:
        obs = {}
        if "image" in self.observation_keys:
            obs["image"] = image
        if "state" in self.observation_keys:
            obs["state"] = state
        if "lidar" in self.observation_keys:
            obs["lidar"] = lidar_obs
        if "lidar_meta" in self.observation_keys:
            obs["lidar_meta"] = lidar_meta
        if "domain_id" in self.observation_keys:
            domain_id = 0.0 if self.domain == "ws" else 1.0
            obs["domain_id"] = np.array([domain_id], dtype=np.float32)
        return obs

    def run(self, img_arr, user_angle=None, user_throttle=None,
            final_angle=None, final_throttle=None):
        if img_arr is None:
            return 0.0, 0.0, -1.0, 0.0, 0.0, -1.0

        t0 = time.time()
        try:
            image = self.preprocessor(img_arr)
            sensors = self._sensor_snapshot()
            lidar = self._lidar_snapshot()
            state, speed_mps = self._build_state(
                sensors=sensors,
                lidar=lidar,
                user_throttle=user_throttle,
                final_throttle=final_throttle,
            )
            lidar_obs, lidar_meta = self._build_lidar_obs(lidar)
            preprocess_ms = (time.time() - t0) * 1000.0

            obs = self._build_model_obs(image, state, lidar_obs, lidar_meta)
            raw_action = self._predict_action(obs)
            if raw_action.size >= 3:
                target_low_level = self.adapter.action(raw_action[:3], speed_mps=speed_mps)
                low_level = self.safety.action(target_low_level)
                raw_angle = float(raw_action[0])
                raw_throttle = float(raw_action[1])
                pilot_angle = float(low_level[0])
                pilot_throttle = float(low_level[1])
            elif raw_action.size >= 2:
                raw_angle = _clip(float(raw_action[0]), -1.0, 1.0)
                raw_throttle = _clip(float(raw_action[1]), 0.0, 1.0)
                pilot_angle = raw_angle
                pilot_throttle = min(raw_throttle, self.max_throttle)
            else:
                raw_angle = _clip(float(raw_action[0]), -1.0, 1.0)
                raw_throttle = 0.0
                pilot_angle = raw_angle
                pilot_throttle = 0.0

            latency_ms = (time.time() - t0) * 1000.0
            self.inference_count += 1
            self.total_latency_ms += latency_ms
            self.total_preprocess_ms += preprocess_ms
            if self.inference_count % 100 == 0:
                avg = self.total_latency_ms / max(1, self.inference_count)
                print(
                    "V17Pilot: frames=%d avg_latency=%.1fms last=%.1fms "
                    "pilot=(%.3f, %.3f)"
                    % (self.inference_count, avg, latency_ms,
                       pilot_angle, pilot_throttle)
                )
            return (
                pilot_angle,
                pilot_throttle,
                latency_ms,
                raw_angle,
                raw_throttle,
                preprocess_ms,
            )
        except Exception as exc:
            now = time.time()
            if now - self._last_error_print > 5.0:
                print(f"V17Pilot inference error: {exc}")
                self._last_error_print = now
            latency_ms = (time.time() - t0) * 1000.0
            return 0.0, 0.0, latency_ms, 0.0, 0.0, -1.0

    def shutdown(self) -> None:
        if self.inference_count:
            avg = self.total_latency_ms / self.inference_count
            pre = self.total_preprocess_ms / self.inference_count
            print(
                "V17Pilot shutdown: frames=%d avg_latency=%.1fms "
                "avg_preprocess=%.1fms"
                % (self.inference_count, avg, pre)
            )


def _run_smoke(args: argparse.Namespace) -> None:
    pilot = V17Pilot(
        model_path=args.model,
        obs_size=args.obs_size,
        state_dim=args.state_dim,
        domain=args.domain,
        max_throttle=args.max_throttle,
        delta_max=args.delta_max,
        enable_lpf=not args.disable_lpf,
        beta=args.beta,
        use_cuda=not args.cpu,
        warmup_frames=0,
    )
    fake = np.zeros((224, 224, 3), dtype=np.uint8)
    latencies = []
    for _ in range(int(args.frames)):
        out = pilot.run(fake, 0.0, 0.0, 0.0, 0.0)
        latencies.append(float(out[2]))
    pilot.shutdown()
    latencies = np.asarray(latencies, dtype=np.float32)
    print("smoke_frames:", len(latencies))
    print("latency_ms_p50: %.3f" % float(np.percentile(latencies, 50)))
    print("latency_ms_p95: %.3f" % float(np.percentile(latencies, 95)))
    print("metadata:", pilot.metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="V17 shadow pilot smoke test")
    parser.add_argument("--model", required=True, help="V17 .zip or *_policy.pth")
    parser.add_argument("--obs-size", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--domain", default="ws", choices=["ws", "gt", "rrl", "generic"])
    parser.add_argument("--max-throttle", type=float, default=0.8)
    parser.add_argument("--delta-max", type=float, default=0.35)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--disable-lpf", action="store_true")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    _run_smoke(args)


if __name__ == "__main__":
    main()
