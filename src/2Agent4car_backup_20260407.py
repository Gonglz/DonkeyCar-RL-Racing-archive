#!/usr/bin/env python3
"""
Team-vs-team generated_track training with synchronized multi-car race sessions.
"""

import argparse
import math
import os
import random
import sys
import time
import uuid
from datetime import datetime
from multiprocessing import Manager, Process
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import gym
import gym_donkeycar  # noqa: F401
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _candidate_torch_lib_dirs() -> Sequence[Path]:
    pyver = "python%d.%d" % (sys.version_info.major, sys.version_info.minor)
    exe = Path(sys.executable).resolve()
    prefixes = []
    for prefix in (Path(sys.prefix), exe.parents[1]):
        if prefix not in prefixes:
            prefixes.append(prefix)
    candidates = []
    for prefix in prefixes:
        candidates.append(prefix / "lib" / pyver / "site-packages" / "torch" / "lib")
    return candidates


def _ensure_torch_cuda_lib_priority() -> None:
    if os.environ.get("DONKEY_TORCH_LIB_PRIORITY_DONE") == "1":
        return

    torch_lib = None
    for candidate in _candidate_torch_lib_dirs():
        if (candidate / "libcudnn.so.8").is_file() and any(candidate.glob("libcudart*.so*")):
            torch_lib = candidate
            break
    if torch_lib is None:
        return

    env_root = Path(sys.executable).resolve().parents[1]
    env_lib = env_root / "lib"
    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    desired = []
    for path in (str(torch_lib), str(env_lib)):
        if path and path not in desired:
            desired.append(path)
    for path in existing:
        if path not in desired:
            desired.append(path)

    new_ld = ":".join(desired)
    if new_ld == os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["DONKEY_TORCH_LIB_PRIORITY_DONE"] = "1"
        return

    os.environ["LD_LIBRARY_PATH"] = new_ld
    os.environ["DONKEY_TORCH_LIB_PRIORITY_DONE"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


_ensure_torch_cuda_lib_priority()

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
try:
    import torch
except Exception:
    torch = None

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from module.action_adapter import ActionAdapterWrapper
from module.actor import FiLMFeatureExtractor
from module.control import ActionSafetyWrapper
from module.multi_scene_env import MultiInputObsWrapper
from module.obv import CanonicalSemanticWrapper, _build_state_v13
from module.reward import DonkeyRewardWrapper
from module.team_race_wrapper import CAR_KEYS, RaceGridResetWrapper, TeamRaceRewardWrapper, team_car_key
from module.track import MODULE_TRACK_DATA_DIR, TrackGeometryManager
from module.utils import MONITOR_INFO_KEYS, _seed_everything


ENV_ID = "donkey-generated-track-v0"
SCENE_KEY = "generated_track"
TRACK_FILE = "manual_width_generated_track.json"
CONTROL_DT = 0.05
DEFAULT_SAVE_ROOT = str(REPO_ROOT / "models" / "team_race_gt")
TEAM_BODY_RGBS = {
    "blue": (0, 96, 255),
    "red": (255, 48, 48),
}
RACE_MONITOR_INFO_KEYS = MONITOR_INFO_KEYS + (
    "race/epoch",
    "race/session_end",
    "race/final_rank",
    "race/team_blue_total",
    "race/team_red_total",
    "race/summary_reason",
    "race/fastest_lap_car",
)
SCENE_SPECS = {
    ENV_ID: {
        "scene_key": SCENE_KEY,
        "logging_key": "gt",
        "level_name": SCENE_KEY,
        "track_file": TRACK_FILE,
    }
}
RACE_TB_MEAN_KEYS = (
    "race/rank",
    "race/rank_reward",
    "race/team_progress_reward",
    "race/cover_bonus",
    "race/pack_pressure_bonus",
    "race/overtake_reward",
    "race/collision_penalty",
    "race/progress_laps",
    "race/session_progress_laps",
    "race/track_limit_offenses",
)
RACE_TB_EVENT_KEYS = (
    "race/overtake_success",
    "race/finished",
    "race/retired",
    "race/session_end",
    "race/contact_penalty",
    "race/rear_end_penalty",
    "race/danger_squeeze_penalty",
)
REWARD_DEBUG_TB_MEAN_KEYS = (
    "reward_debug/progress_reward",
    "reward_debug/progress_reward_raw",
    "reward_debug/progress_center_gate",
    "reward_debug/progress_forward_gain",
    "reward_debug/cte_abs",
    "reward_debug/cte_over_in",
    "reward_debug/cte_over_out",
    "reward_debug/r_near_offtrack",
    "reward_debug/r_near_collision",
    "reward_debug/near_collision_risk",
)
GT_BASE_REWARD_KWARGS = dict(
    # Keep multi-car training anchored to the proven V13 GT-style dense reward.
    w_time=0.01,
    w_center=0.03,
    w_heading=0.015,
    w_speed_ref=0.0,
    lap_reward_scale=1.0,
    progress_reward_scale=80.0,
    survival_reward_scale=0.30,
    collision_penalty_base=8.0,
    offtrack_penalty_base=5.0,
    w_near_offtrack=0.55,
    near_offtrack_start_ratio=0.50,
    w_near_collision=0.20,
    near_collision_start_ratio=0.65,
)



def _apply_training_preset(args) -> None:
    preset = str(getattr(args, "training_preset", "standard") or "standard").strip().lower()
    args.training_preset = preset

    if preset == "coop-race":
        args.use_default_spawn = True
        args.grid_randomize = False
        args.default_spawn_order_mode = "mixed"
        args.team_priority_mode = "strong"
        args.allow_light_collision = True
        args.race_laps_schedule = (1, 2, 3)
        args.race_step_budget_scale = int(max(args.race_step_budget_scale, 900))
        args.spawn_stabilize_steps = int(max(args.spawn_stabilize_steps, 3))
        args.max_cte = float(max(args.max_cte, 9.0))
        args.save_root = str(Path(args.save_root).expanduser())


def _default_spawn_order(mode: str, rng: random.Random) -> Sequence[str]:
    mode = str(mode or "mixed").strip().lower()

    reds = ["red_1", "red_2"]
    blues = ["blue_1", "blue_2"]

    if mode == "shuffle":
        out = list(CAR_KEYS)
        rng.shuffle(out)
        return out

    if mode == "red-front":
        rng.shuffle(reds)
        rng.shuffle(blues)
        return reds + blues

    if mode == "blue-front":
        rng.shuffle(reds)
        rng.shuffle(blues)
        return blues + reds

    if mode == "split-front":
        rng.shuffle(reds)
        rng.shuffle(blues)
        front = [reds.pop(), blues.pop()]
        rear = [reds.pop(), blues.pop()]
        rng.shuffle(front)
        rng.shuffle(rear)
        return front + rear

    # mixed: randomize between team-dominant front row and one-red-one-blue front row.
    if rng.random() < 0.5:
        return _default_spawn_order("red-front", rng)
    return _default_spawn_order("split-front", rng)


def _resolve_track_dir(track_dir: str) -> str:
    candidates = []
    if track_dir:
        candidates.append(Path(track_dir).expanduser())

    env_override = os.environ.get("MYSIM_TRACK_DIR", "").strip()
    if env_override:
        candidates.append(Path(env_override).expanduser())

    candidates.extend(
        [
            REPO_ROOT / "track_profiles",
            REPO_ROOT / "track",
            Path(MODULE_TRACK_DATA_DIR),
        ]
    )

    seen = set()
    tried = []
    for candidate in candidates:
        abs_candidate = os.path.abspath(os.path.expanduser(str(candidate)))
        if abs_candidate in seen:
            continue
        seen.add(abs_candidate)
        track_path = Path(abs_candidate) / TRACK_FILE
        if track_path.is_file():
            return abs_candidate
        tried.append(abs_candidate)

    raise FileNotFoundError(
        "Could not find %s in track_dir candidates: %s"
        % (TRACK_FILE, ", ".join(tried))
    )


def _make_episode_over_fn(allow_light_collision: bool):
    def _episode_over(handler) -> None:
        if math.fabs(handler.cte) > 2.0 * handler.max_cte:
            return
        if math.fabs(handler.cte) > handler.max_cte:
            handler.over = True
        elif (not allow_light_collision) and handler.hit != "none":
            handler.over = True

    return _episode_over


def _donkey_episode_over_no_checkpoint(handler) -> None:
    if math.fabs(handler.cte) > 2.0 * handler.max_cte:
        return
    if math.fabs(handler.cte) > handler.max_cte:
        handler.over = True
    elif handler.hit != "none":
        handler.over = True


def _install_custom_episode_over(base_env, allow_light_collision: bool) -> None:
    if hasattr(base_env, "set_episode_over_fn"):
        base_env.set_episode_over_fn(_make_episode_over_fn(bool(allow_light_collision)))


def _set_handler_max_cte(base_env, max_cte: float) -> None:
    try:
        base_env.viewer.handler.max_cte = float(max_cte)
    except Exception:
        pass


def _track_geometry(track_dir: str) -> TrackGeometryManager:
    return TrackGeometryManager(
        track_dir=track_dir,
        env_ids=[ENV_ID],
        scene_specs=SCENE_SPECS,
    )


def _make_car_conf(team_name: str, car_id: int, port: int, max_cte: float) -> Dict[str, Any]:
    body_rgb = TEAM_BODY_RGBS[team_name]
    car_name = "%s_%d" % (team_name.upper(), int(car_id))
    return {
        "host": "127.0.0.1",
        "port": int(port),
        "body_style": "donkey",
        "body_rgb": body_rgb,
        "car_name": car_name,
        "racer_name": car_name,
        "bio": "team-race-%s-%d" % (team_name, int(car_id)),
        "country": "CN",
        "guid": "%s-%d-%s" % (team_name, int(car_id), uuid.uuid4().hex[:8]),
        "max_cte": float(max_cte),
        "level": SCENE_KEY,
        "font_size": 50,
    }


def _policy_kwargs() -> Dict[str, Any]:
    return dict(
        features_extractor_class=FiLMFeatureExtractor,
        features_extractor_kwargs=dict(
            image_feat_dim=128,
            state_feat_dim=32,
        ),
        log_std_init=-1.0,
        lstm_hidden_size=128,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
    )


def _build_model(
    team_name: str,
    vec_env,
    lr: float,
    args,
    team_seed: int,
    save_dir: str,
):
    requested_device = str(args.device).strip().lower()
    explicit_device = requested_device not in ("", "auto")
    if requested_device in ("", "auto"):
        if torch is not None and torch.cuda.is_available():
            requested_device = "cuda"
        else:
            requested_device = "cpu"

    model_kwargs = dict(
        policy="MultiInputLstmPolicy",
        env=vec_env,
        learning_rate=float(lr),
        n_steps=int(args.ppo_n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.ppo_n_epochs),
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=float(args.ent_coef),
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=(None if float(args.target_kl) <= 0.0 else float(args.target_kl)),
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tensorboard"),
        policy_kwargs=_policy_kwargs(),
        seed=team_seed,
        device=requested_device,
    )

    try:
        print("[%s] requested device: %s" % (team_name.upper(), requested_device))
        return RecurrentPPO(**model_kwargs)
    except RuntimeError as exc:
        msg = str(exc).lower()
        can_fallback = (not explicit_device) and requested_device != "cpu" and (
            "cudnn" in msg or "cuda" in msg or "cublas" in msg
        )
        if not can_fallback:
            raise
        print("[%s] device init failed on %s: %s" % (team_name.upper(), requested_device, exc))
        print("[%s] falling back to CPU" % team_name.upper())
        model_kwargs["device"] = "cpu"
        return RecurrentPPO(**model_kwargs)


class SharedProgressCallback(BaseCallback):
    def __init__(self, shared_state, team_name: str, sync_freq: int = 32, verbose: int = 0):
        super().__init__(verbose)
        self.shared_state = shared_state
        self.team_name = str(team_name).strip().lower()
        self.sync_freq = int(max(1, sync_freq))
        self._last_synced = -1

    def _sync(self) -> None:
        steps = int(self.num_timesteps)
        if steps != self._last_synced:
            self.shared_state["train:%s_num_timesteps" % self.team_name] = steps
            self._last_synced = steps

    def _on_step(self) -> bool:
        if self.num_timesteps == 0:
            return True
        if self._last_synced < 0 or (self.num_timesteps - self._last_synced) >= self.sync_freq:
            self._sync()
        return True

    def _on_training_end(self) -> None:
        self._sync()


class RaceInfoStatsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._reset_stats()

    def _reset_stats(self) -> None:
        all_keys = RACE_TB_MEAN_KEYS + RACE_TB_EVENT_KEYS + REWARD_DEBUG_TB_MEAN_KEYS
        self._sum = {key: 0.0 for key in all_keys}
        self._count = {key: 0 for key in all_keys}

    @staticmethod
    def _tb_name(key: str) -> str:
        return str(key).replace("/", "_")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            for key in RACE_TB_MEAN_KEYS + RACE_TB_EVENT_KEYS + REWARD_DEBUG_TB_MEAN_KEYS:
                if key not in info:
                    continue
                self._sum[key] += float(info.get(key, 0.0))
                self._count[key] += 1
        return True

    def _on_rollout_end(self) -> None:
        for key in RACE_TB_MEAN_KEYS:
            count = int(self._count.get(key, 0))
            if count <= 0:
                continue
            self.logger.record("race_stats/%s" % self._tb_name(key), self._sum[key] / float(count))
        for key in RACE_TB_EVENT_KEYS:
            count = int(self._count.get(key, 0))
            if count <= 0:
                continue
            self.logger.record("race_events/%s" % self._tb_name(key), self._sum[key] / float(count))
        for key in REWARD_DEBUG_TB_MEAN_KEYS:
            count = int(self._count.get(key, 0))
            if count <= 0:
                continue
            self.logger.record("reward_debug/%s" % self._tb_name(key.split("/", 1)[-1]), self._sum[key] / float(count))
        self._reset_stats()


class TeamCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_dir: str, team_name: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = int(max(1, save_freq))
        self.save_dir = save_dir
        self.team_name = str(team_name)
        self.ckpt_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._last_save = 0

    def _on_step(self) -> bool:
        if int(self.num_timesteps) - int(self._last_save) < self.save_freq:
            return True
        self._last_save = int(self.num_timesteps)
        path = os.path.join(self.ckpt_dir, "step_%d.zip" % int(self.num_timesteps))
        self.model.save(path)
        if self.verbose:
            print("[%s] checkpoint -> %s" % (self.team_name, path))
        return True


def build_team_env(
    team_name: str,
    car_id: int,
    args,
    shared_state,
    shared_lock,
    track_geometry: TrackGeometryManager,
    connect_not_before: Optional[float] = None,
):
    if connect_not_before is not None:
        wait_s = float(connect_not_before) - time.time()
        if wait_s > 1e-3:
            print(
                "[%s_%d] default-spawn connect wait: %.2fs"
                % (team_name.upper(), int(car_id), wait_s)
            )
            time.sleep(wait_s)

    conf = _make_car_conf(team_name, car_id, args.port, args.max_cte)
    opponent_team = "red" if team_name == "blue" else "blue"
    logging_key = "gt_%s_%d" % (team_name, int(car_id))

    base_env = gym.make(ENV_ID, conf=conf)
    _install_custom_episode_over(base_env, allow_light_collision=bool(getattr(args, "allow_light_collision", False)))
    _set_handler_max_cte(base_env, args.max_cte)

    env = CanonicalSemanticWrapper(
        base_env,
        domain="gt",
        obs_size=args.obs_size,
        augment=False,
        ally_rgb=TEAM_BODY_RGBS[team_name],
        opponent_rgb=TEAM_BODY_RGBS[opponent_team],
    )

    geo = track_geometry.scenes[SCENE_KEY]
    reward_wrapper = DonkeyRewardWrapper(
        env,
        total_timesteps=args.steps,
        action_safety_wrapper=None,
        **GT_BASE_REWARD_KWARGS,
        track_geometry=track_geometry,
        scene_key=SCENE_KEY,
        logging_key=logging_key,
        cte_left=float(geo.cte_left),
        cte_right=float(geo.cte_right),
        cte_left_out=float(geo.cte_left_out),
        cte_right_out=float(geo.cte_right_out),
        coord_scale=float(geo.coord_scale),
        cte_half_width=float(geo.cte_half_width),
    )
    env = reward_wrapper

    grid_wrapper = RaceGridResetWrapper(
        env,
        shared_state=shared_state,
        shared_lock=shared_lock,
        track_geometry=track_geometry,
        scene_key=SCENE_KEY,
        team_name=team_name,
        car_id=car_id,
        total_timesteps=args.steps,
        race_lap_schedule=args.race_laps_schedule,
        race_step_budget_scale=args.race_step_budget_scale,
        grid_randomize=args.grid_randomize,
        use_default_spawn=args.use_default_spawn,
        spawn_stabilize_steps=args.spawn_stabilize_steps,
        low_level_action_dim=2,
    )
    env = grid_wrapper

    action_safety = ActionSafetyWrapper(
        env,
        delta_max=args.delta_max,
        enable_lpf=True,
        beta=args.safety_beta,
    )
    env = action_safety
    reward_wrapper.action_safety_wrapper = action_safety

    adapter = ActionAdapterWrapper(
        env,
        k_delta=0.10,
        lambda_bias=0.20,
        k_bias=0.10,
        v_nominal=1.4,
        v_min=0.6,
        v_max=1.8,
        speed_kp=0.35,
        speed_ki=0.08,
        speed_kff=0.10,
        control_dt=CONTROL_DT,
        max_throttle=args.max_throttle,
        allow_reverse=False,
    )
    env = adapter

    env = MultiInputObsWrapper(
        env,
        track_geometry=None,
        scene_key=SCENE_KEY,
        logging_key=logging_key,
        domain="gt",
        obs_size=args.obs_size,
        image_channels=7,
        include_cte_in_obs=False,
        speed_vmax=2.2,
        control_wrapper=adapter,
        action_safety_wrapper=action_safety,
        state_builder=_build_state_v13,
        state_dim=7,
    )

    env = TeamRaceRewardWrapper(
        env,
        grid_wrapper=grid_wrapper,
        shared_state=shared_state,
        shared_lock=shared_lock,
        track_geometry=track_geometry,
        scene_key=SCENE_KEY,
        team_name=team_name,
        car_id=car_id,
        domain="gt",
        logging_key=logging_key,
        step_dt=CONTROL_DT,
        team_priority_mode=args.team_priority_mode,
    )

    env = Monitor(
        env,
        filename=None,
        allow_early_resets=True,
        info_keywords=RACE_MONITOR_INFO_KEYS,
    )
    return env


def train_team(team_name: str, save_dir: str, args, shared_state, shared_lock) -> None:
    if RecurrentPPO is None:
        raise ImportError("sb3_contrib is required: pip install sb3_contrib==1.8.0")

    team_seed = int(args.seed) + (0 if team_name == "blue" else 10_000)
    _seed_everything(team_seed)
    track_dir = _resolve_track_dir(args.track_dir)
    track_geometry = _track_geometry(track_dir)
    os.makedirs(save_dir, exist_ok=True)

    connect_schedule = dict(getattr(args, "default_spawn_connect_schedule", {}))
    local_car_ids = [1, 2]
    if args.use_default_spawn and connect_schedule:
        local_car_ids.sort(
            key=lambda cid: float(connect_schedule.get(team_car_key(team_name, cid), 0.0))
        )

    def make_env(car_id: int):
        def _factory():
            connect_not_before = None
            if args.use_default_spawn and connect_schedule:
                connect_not_before = float(
                    connect_schedule.get(team_car_key(team_name, car_id), 0.0)
                )
            return build_team_env(
                team_name=team_name,
                car_id=car_id,
                args=args,
                shared_state=shared_state,
                shared_lock=shared_lock,
                track_geometry=track_geometry,
                connect_not_before=connect_not_before,
            )

        return _factory

    vec_env = DummyVecEnv([make_env(car_id) for car_id in local_car_ids])

    lr = args.lr_blue if team_name == "blue" else args.lr_red
    model = _build_model(
        team_name=team_name,
        vec_env=vec_env,
        lr=lr,
        args=args,
        team_seed=team_seed,
        save_dir=save_dir,
    )

    callbacks = CallbackList(
        [
            SharedProgressCallback(shared_state, team_name=team_name, sync_freq=32, verbose=0),
            RaceInfoStatsCallback(verbose=0),
            TeamCheckpointCallback(
                save_freq=args.save_freq,
                save_dir=save_dir,
                team_name=team_name.upper(),
                verbose=1,
            ),
        ]
    )

    print("[%s] training start: steps=%d save_dir=%s" % (team_name.upper(), int(args.steps), save_dir))
    try:
        model.learn(
            total_timesteps=int(args.steps),
            callback=callbacks,
            progress_bar=False,
        )
        final_path = os.path.join(save_dir, "final_model.zip")
        model.save(final_path)
        print("[%s] training done -> %s" % (team_name.upper(), final_path))
    finally:
        try:
            vec_env.close()
        except Exception:
            pass


def launch_team_race_training(args) -> int:
    _apply_training_preset(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = "%s_%s" % (args.save_root, timestamp)
    blue_dir = os.path.join(root_dir, "blue_team")
    red_dir = os.path.join(root_dir, "red_team")
    os.makedirs(blue_dir, exist_ok=True)
    os.makedirs(red_dir, exist_ok=True)

    manager = Manager()
    shared_state = manager.dict()
    shared_lock = manager.RLock()
    shared_state["race_epoch"] = 0
    shared_state["race:registered_car_keys"] = list(CAR_KEYS)
    shared_state["train:blue_num_timesteps"] = 0
    shared_state["train:red_num_timesteps"] = 0

    default_spawn_order = []
    if args.use_default_spawn:
        rng = random.SystemRandom()
        default_spawn_order = list(_default_spawn_order(args.default_spawn_order_mode, rng))
        connect_base_time = time.time() + max(3.0, float(args.process_stagger_sec) + 1.5)
        connect_gap_s = 2.0
        connect_schedule = {
            car_key: float(connect_base_time + idx * connect_gap_s)
            for idx, car_key in enumerate(default_spawn_order)
        }
        args.default_spawn_connect_schedule = connect_schedule
        shared_state["spawn:default_connect_order"] = list(default_spawn_order)
        shared_state["spawn:default_connect_schedule"] = dict(connect_schedule)
    else:
        args.default_spawn_connect_schedule = {}

    print("=" * 72)
    print("4-car team race training")
    print("=" * 72)
    print("scene: %s" % SCENE_KEY)
    print("steps per team: %d" % int(args.steps))
    print("race_laps_schedule: %s" % ",".join(str(v) for v in args.race_laps_schedule))
    print("race_step_budget_scale: %d" % int(args.race_step_budget_scale))
    print("obs: image=7x%dx%d state=7" % (int(args.obs_size), int(args.obs_size)))
    print("device: %s" % args.device)
    print("training_preset: %s" % args.training_preset)
    print("ppo: n_steps=%d batch=%d epochs=%d ent=%.4f target_kl=%s" % (
        int(args.ppo_n_steps),
        int(args.batch_size),
        int(args.ppo_n_epochs),
        float(args.ent_coef),
        ("off" if float(args.target_kl) <= 0.0 else "%.4f" % float(args.target_kl)),
    ))
    print("use_default_spawn: %s" % ("on" if args.use_default_spawn else "off"))
    print("grid_randomize: %s" % ("on" if args.grid_randomize else "off"))
    if default_spawn_order:
        print("default_spawn_order_mode: %s" % str(args.default_spawn_order_mode))
        print("default_spawn_connect_order: %s" % ", ".join(default_spawn_order))
    print("team_priority_mode: %s" % args.team_priority_mode)
    print("output: %s" % root_dir)
    print("=" * 72)

    blue_proc = Process(
        target=train_team,
        args=("blue", blue_dir, args, shared_state, shared_lock),
        daemon=False,
    )
    red_proc = Process(
        target=train_team,
        args=("red", red_dir, args, shared_state, shared_lock),
        daemon=False,
    )

    blue_proc.start()
    time.sleep(float(args.process_stagger_sec))
    red_proc.start()

    exit_code = 0
    try:
        blue_proc.join()
        red_proc.join()
    except KeyboardInterrupt:
        print("interrupt received, terminating worker processes")
        for proc in (blue_proc, red_proc):
            if proc.is_alive():
                proc.terminate()
        for proc in (blue_proc, red_proc):
            proc.join(timeout=5.0)
        return 130

    for proc in (blue_proc, red_proc):
        if int(proc.exitcode or 0) != 0:
            exit_code = int(proc.exitcode or 1)

    if exit_code != 0:
        print("training failed: blue_exit=%s red_exit=%s" % (blue_proc.exitcode, red_proc.exitcode))
    else:
        print("training finished successfully")
    return exit_code


def parse_args():
    parser = argparse.ArgumentParser(description="4-car generated_track team race training")
    parser.add_argument("--steps", type=int, default=1_000_000, help="training timesteps per team")
    parser.add_argument("--port", type=int, default=9091, help="DonkeySim port")
    parser.add_argument("--save-root", type=str, default=DEFAULT_SAVE_ROOT, help="output directory prefix")
    parser.add_argument("--track-dir", type=str, default="", help="track profile directory")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    parser.add_argument("--max-cte", type=float, default=8.0, help="handler max_cte")
    parser.add_argument("--obs-size", type=int, default=128, help="semantic observation size")
    parser.add_argument("--lr-blue", type=float, default=3e-4, help="blue team learning rate")
    parser.add_argument("--lr-red", type=float, default=3e-4, help="red team learning rate")
    parser.add_argument("--save-freq", type=int, default=50_000, help="checkpoint save frequency")
    parser.add_argument("--ppo-n-steps", type=int, default=2048, help="RecurrentPPO rollout steps")
    parser.add_argument("--batch-size", type=int, default=128, help="RecurrentPPO batch size")
    parser.add_argument("--ppo-n-epochs", type=int, default=10, help="RecurrentPPO epochs per update")
    parser.add_argument("--ent-coef", type=float, default=0.003, help="entropy coefficient")
    parser.add_argument("--target-kl", type=float, default=0.02, help="target KL for PPO early stop; <=0 disables")
    parser.add_argument("--device", type=str, default="auto", help="training device: auto, cpu, cuda, cuda:0")
    parser.add_argument("--race-laps-schedule", type=str, default="2,3,5", help="lap curriculum as a,b,c")
    parser.add_argument("--race-step-budget-scale", type=int, default=750, help="step budget scale per lap")
    parser.add_argument("--use-default-spawn", dest="use_default_spawn", action="store_true", help="use simulator default spawn instead of manual set_position grid")
    parser.add_argument("--manual-grid-spawn", dest="use_default_spawn", action="store_false", help="use manual 2x2 grid teleport placement")
    parser.set_defaults(use_default_spawn=True)
    parser.add_argument(
        "--default-spawn-order-mode",
        choices=["mixed", "red-front", "blue-front", "split-front", "shuffle"],
        default="mixed",
        help="ordering strategy when using simulator default spawn",
    )
    parser.add_argument("--grid-randomize", dest="grid_randomize", action="store_true", help="randomize 2x2 slot assignment when manual grid spawn is enabled")
    parser.add_argument("--fixed-grid", dest="grid_randomize", action="store_false", help="disable slot randomization when manual grid spawn is enabled")
    parser.set_defaults(grid_randomize=True)
    parser.add_argument("--team-priority-mode", choices=["strong", "balanced"], default="strong", help="team reward emphasis")
    parser.add_argument("--spawn-stabilize-steps", type=int, default=2, help="post-teleport settle steps")
    parser.add_argument("--process-stagger-sec", type=float, default=4.0, help="delay before starting the second team")
    parser.add_argument("--delta-max", type=float, default=0.35, help="steering rate limit")
    parser.add_argument("--safety-beta", type=float, default=0.6, help="steering LPF beta")
    parser.add_argument("--max-throttle", type=float, default=0.30, help="adapter max throttle")
    parser.add_argument(
        "--allow-light-collision",
        dest="allow_light_collision",
        action="store_true",
        help="do not terminate/reset on light contact hits; keep running unless offtrack/max-cte",
    )
    parser.set_defaults(allow_light_collision=False)
    parser.add_argument(
        "--training-preset",
        choices=["standard", "coop-race"],
        default="standard",
        help="training preset: standard or coop-race",
    )

    args = parser.parse_args()
    args.race_laps_schedule = tuple(
        int(v) for v in str(args.race_laps_schedule).split(",") if str(v).strip()
    )
    if len(args.race_laps_schedule) != 3:
        raise ValueError("--race-laps-schedule must contain exactly three integers")
    return args


def main() -> int:
    args = parse_args()
    return launch_team_race_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
