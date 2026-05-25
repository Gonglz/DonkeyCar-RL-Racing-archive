#!/usr/bin/env python3
"""Export V17 RecurrentPPO actor to ONNX for TensorRT.

This script intentionally exports only the deterministic actor path used during
real-time driving:

  image/state/lidar/lidar_meta + LSTM h/c -> action + next h/c

It does not export SB3, the critic, value heads, optimizer, or training-only
distribution logic.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import zipfile
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class V17ActorShape:
    def __init__(
        self,
        image_channels: int,
        obs_size: int,
        state_dim: int,
        lidar_dim: int,
        lidar_meta_dim: int,
        lstm_layers: int,
        lstm_hidden_size: int,
    ):
        self.image_channels = int(image_channels)
        self.obs_size = int(obs_size)
        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)
        self.lidar_meta_dim = int(lidar_meta_dim)
        self.lstm_layers = int(lstm_layers)
        self.lstm_hidden_size = int(lstm_hidden_size)

    def as_dict(self) -> Dict[str, int]:
        return {
            "image_channels": self.image_channels,
            "obs_size": self.obs_size,
            "state_dim": self.state_dim,
            "lidar_dim": self.lidar_dim,
            "lidar_meta_dim": self.lidar_meta_dim,
            "lstm_layers": self.lstm_layers,
            "lstm_hidden_size": self.lstm_hidden_size,
        }

    @property
    def lidar_num_sectors(self) -> int:
        return self.lidar_dim // 2 if self.lidar_dim != 12 else 0

    @property
    def lidar_side_sectors(self) -> int:
        return self.lidar_num_sectors // 2 if self.lidar_dim != 12 else 0


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
    def __init__(self, sectors_per_side: int):
        super().__init__()
        self.sectors_per_side = int(sectors_per_side)
        self.conv_in = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.res1 = ResidualBlock1D(32, kernel_size=3, padding=1)
        self.res2 = ResidualBlock1D(32, kernel_size=3, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(32 * self.sectors_per_side, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
        )

    def forward(self, side_obs: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv_in(side_obs))
        h = self.res1(h)
        h = self.res2(h)
        h = torch.flatten(h, start_dim=1)
        return self.mlp(h)


class PooledLidarEncoder(nn.Module):
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

    def forward(self, lidar_obs: torch.Tensor) -> torch.Tensor:
        return self.net(lidar_obs)


class FixedLayerNorm1D(nn.Module):
    """LayerNorm over feature dim using static axis 1 for TensorRT 7 ONNX import."""

    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(features)))
        self.bias = nn.Parameter(torch.zeros(int(features)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=1, keepdim=True)
        normed = centered * torch.rsqrt(var + self.eps)
        return normed * self.weight + self.bias


class LiDARFiLMFeatureExtractor(nn.Module):
    def __init__(
        self,
        shape: V17ActorShape,
        image_feat_dim: int = 128,
        state_feat_dim: int = 32,
        lidar_feat_dim: int = 96,
        lidar_encoder_mode: str = "side_separated",
        disable_lidar_meta: bool = False,
    ):
        super().__init__()
        self.image_feat_dim = int(image_feat_dim)
        self.disable_lidar_meta = bool(disable_lidar_meta)
        self.lidar_dim = int(shape.lidar_dim)
        self.lidar_num_sectors = int(shape.lidar_num_sectors)
        self.lidar_side_sectors = int(shape.lidar_side_sectors)
        self.lidar_encoder_mode = str(lidar_encoder_mode).strip().lower()

        self.cnn = nn.Sequential(
            nn.Conv2d(shape.image_channels, 32, 3, stride=2, padding=1),
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
        self.image_norm = FixedLayerNorm1D(image_feat_dim)

        state_input_dim = int(shape.state_dim + (0 if self.disable_lidar_meta else shape.lidar_meta_dim))
        self.state_enc = nn.Sequential(
            nn.Linear(state_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_feat_dim),
            nn.ReLU(),
        )

        self.lidar_side_encoder = SharedSideLidarEncoder(max(1, self.lidar_side_sectors))
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
        self.fused_norm = FixedLayerNorm1D(image_feat_dim)

    def _encode_lidar(self, lidar_flat: torch.Tensor) -> torch.Tensor:
        lidar_flat = lidar_flat.float()
        if self.lidar_dim == 12:
            return self.lidar_target_encoder(lidar_flat)
        lidar_range = lidar_flat[:, :self.lidar_num_sectors]
        lidar_valid = lidar_flat[:, self.lidar_num_sectors:self.lidar_dim]
        lidar_2c = torch.stack([lidar_range, lidar_valid], dim=1)
        if self.lidar_encoder_mode == "pooled":
            return self.lidar_pooled_encoder(lidar_2c)
        mid = self.lidar_side_sectors
        left = lidar_2c[:, :, :mid]
        right = lidar_2c[:, :, mid:]
        left_feat = self.lidar_side_encoder(left)
        right_feat = self.lidar_side_encoder(right)
        return torch.cat([left_feat, right_feat], dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        lidar: torch.Tensor,
        lidar_meta: torch.Tensor,
    ) -> torch.Tensor:
        image_feat = self.image_norm(self.cnn(image.float()))
        if self.disable_lidar_meta:
            state_aug = state.float()
        else:
            state_aug = torch.cat([state.float(), lidar_meta.float()], dim=-1)
        state_feat = self.state_enc(state_aug)
        lidar_feat = self._encode_lidar(lidar)
        film_in = torch.cat([state_feat, lidar_feat], dim=-1)
        film_hidden = self.film_act(self.film_hidden(film_in))
        film_params = self.film_out(film_hidden)
        gamma_raw = film_params[:, :self.image_feat_dim]
        beta_raw = film_params[:, self.image_feat_dim:self.image_feat_dim * 2]
        gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)
        beta = 0.1 * torch.tanh(beta_raw)
        fused = self.fused_norm(gamma * image_feat + beta)
        return torch.cat([fused, state_feat, lidar_feat], dim=-1)


class V17ActorONNX(nn.Module):
    def __init__(self, shape: V17ActorShape):
        super().__init__()
        self.shape = shape
        self.features_extractor = LiDARFiLMFeatureExtractor(shape)
        self.lstm_actor = nn.LSTM(
            input_size=256,
            hidden_size=shape.lstm_hidden_size,
            num_layers=shape.lstm_layers,
        )
        self.policy_net = nn.Sequential(
            nn.Linear(shape.lstm_hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.action_net = nn.Linear(64, 3)

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        lidar: torch.Tensor,
        lidar_meta: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features_extractor(image, state, lidar, lidar_meta)
        seq = features.unsqueeze(0)
        out, (next_h, next_c) = self.lstm_actor(seq, (h, c))
        latent = out.squeeze(0)
        latent = self.policy_net(latent)
        action = torch.clamp(self.action_net(latent), -1.0, 1.0)
        return action, next_h, next_c


def load_policy_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as archive:
            payload = archive.read("policy.pth")
        return torch.load(io.BytesIO(payload), map_location="cpu")
    return torch.load(path, map_location="cpu")


def infer_shape(state_dict: Dict[str, torch.Tensor], obs_size: int) -> V17ActorShape:
    image_channels = int(state_dict["pi_features_extractor.cnn.0.weight"].shape[1])
    state_input_dim = int(state_dict["pi_features_extractor.state_enc.0.weight"].shape[1])
    lidar_side_flat = int(state_dict["pi_features_extractor.lidar_side_encoder.mlp.0.weight"].shape[1])
    lidar_side_sectors = lidar_side_flat // 32
    lidar_dim = int(lidar_side_sectors * 4)
    lstm_layers = len([k for k in state_dict if k.startswith("lstm_actor.weight_ih_l")])
    lstm_hidden_size = int(state_dict["lstm_actor.weight_hh_l0"].shape[1])
    return V17ActorShape(
        image_channels=image_channels,
        obs_size=int(obs_size),
        state_dim=state_input_dim - 2,
        lidar_dim=lidar_dim,
        lidar_meta_dim=2,
        lstm_layers=lstm_layers,
        lstm_hidden_size=lstm_hidden_size,
    )


def load_actor_weights(actor: V17ActorONNX, state_dict: Dict[str, torch.Tensor]) -> None:
    feature_state = {}
    for key, value in state_dict.items():
        if key.startswith("pi_features_extractor."):
            feature_state[key.replace("pi_features_extractor.", "", 1)] = value
        elif key.startswith("features_extractor."):
            feature_state[key.replace("features_extractor.", "", 1)] = value
    actor.features_extractor.load_state_dict(feature_state, strict=False)

    actor_state = actor.state_dict()
    mapped = {}
    for key in actor_state:
        if key.startswith("features_extractor."):
            continue
        source_key = key
        if key.startswith("policy_net."):
            source_key = "mlp_extractor." + key
        if source_key in state_dict:
            mapped[key] = state_dict[source_key]

    missing = sorted(
        key for key in actor_state
        if not key.startswith("features_extractor.") and key not in mapped
    )
    if missing:
        raise RuntimeError("missing actor weights: " + ", ".join(missing[:8]))

    actor.load_state_dict(mapped, strict=False)
    actor.eval()


def make_dummy_inputs(shape: V17ActorShape, device: torch.device):
    torch.manual_seed(17)
    image = torch.rand(1, shape.image_channels, shape.obs_size, shape.obs_size, device=device)
    state = torch.zeros(1, shape.state_dim, device=device)
    lidar = torch.zeros(1, shape.lidar_dim, device=device)
    if shape.lidar_dim != 12:
        half = shape.lidar_dim // 2
        lidar[:, :half] = 20.0
    lidar_meta = torch.zeros(1, shape.lidar_meta_dim, device=device)
    h = torch.zeros(shape.lstm_layers, 1, shape.lstm_hidden_size, device=device)
    c = torch.zeros(shape.lstm_layers, 1, shape.lstm_hidden_size, device=device)
    return image, state, lidar, lidar_meta, h, c


def export_onnx(actor: V17ActorONNX, shape: V17ActorShape, output_path: str, opset: int) -> None:
    device = next(actor.parameters()).device
    dummy_inputs = make_dummy_inputs(shape, device)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        actor,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["image", "state", "lidar", "lidar_meta", "h", "c"],
        output_names=["action", "next_h", "next_c"],
        dynamic_axes=None,
    )


def run_trtexec(trtexec: str, onnx_path: str, engine_path: str, fp16: bool, workspace: int) -> int:
    cmd = [
        trtexec,
        "--onnx=" + onnx_path,
        "--explicitBatch",
        "--workspace=" + str(int(workspace)),
        "--saveEngine=" + engine_path,
        "--verbose",
    ]
    if fp16:
        cmd.insert(-2, "--fp16")
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export V17 actor to ONNX/TensorRT")
    parser.add_argument("--model", default="/home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip")
    parser.add_argument("--onnx", default="/home/jetson/mycar/models/v17_actor.onnx")
    parser.add_argument("--engine", default="/home/jetson/mycar/models/v17_actor_fp16.engine")
    parser.add_argument("--obs-size", type=int, default=128)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--workspace", type=int, default=512)
    parser.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
    parser.add_argument("--metadata", default="/home/jetson/mycar/models/v17_actor_export.json")
    args = parser.parse_args()

    state_dict = load_policy_state_dict(args.model)
    shape = infer_shape(state_dict, obs_size=args.obs_size)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    actor = V17ActorONNX(shape).to(device)
    load_actor_weights(actor, state_dict)

    with torch.no_grad():
        outputs = actor(*make_dummy_inputs(shape, device))
    action = outputs[0].detach().cpu().numpy()
    print("shape:", json.dumps(shape.as_dict(), sort_keys=True))
    print("dummy_action:", np.array2string(action, precision=6, suppress_small=False))

    if args.check and not args.export and not args.build_engine:
        return 0

    if args.export:
        export_onnx(actor, shape, args.onnx, args.opset)
        metadata = {
            "model": os.path.abspath(args.model),
            "onnx": os.path.abspath(args.onnx),
            "shape": shape.as_dict(),
            "opset": int(args.opset),
            "outputs": ["action", "next_h", "next_c"],
            "inputs": ["image", "state", "lidar", "lidar_meta", "h", "c"],
        }
        with open(args.metadata, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        print("exported_onnx:", args.onnx)
        print("metadata:", args.metadata)

    if args.build_engine:
        if not os.path.exists(args.onnx):
            raise FileNotFoundError(args.onnx)
        rc = run_trtexec(
            trtexec=args.trtexec,
            onnx_path=args.onnx,
            engine_path=args.engine,
            fp16=not args.no_fp16,
            workspace=args.workspace,
        )
        if rc != 0:
            return rc
        print("exported_engine:", args.engine)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
