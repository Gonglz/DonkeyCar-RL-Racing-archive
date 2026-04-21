# module package for ppo_waveshare_v12
from .utils import (
    load_config, ENV_DOMAIN_MAP, MONITOR_INFO_KEYS,
    _seed_everything, _safe_seed_env, _find_latest_checkpoint,
    _wrap_pi, _clip_float, _get_domain_for_env,
)
from .track import MODULE_TRACK_DATA_DIR, SceneGeometry, TrackGeometryManager
from .track_generator import TrackProfile, available_gym_envs, available_scenes, load_track
from .dynamics_wrapper import DynamicsAlignedGymEnv
from .obstacle import (
    ObstacleFleetPreset, DonkeyObstacleFleet,
    PoseState, RelativeState, TrackTarget, PositionJitterConfig, InPlaceNudgeConfig, LanePIDConfig, ObstacleSnapshot,
    DonkeyObstacleCar, infer_scene_key, pose_from_info,
    compute_relative_state, sample_track_target, sample_random_track_targets,
    resolve_obstacle_fleet_preset, build_obstacle_track_geometry, default_obstacle_layout,
    spawn_preset_obstacle_fleet, spawn_gt_obstacles, spawn_ws_obstacles,
)
from .obstacle_runtime import ObstacleRuntimeConfig, ObstacleRuntimeManager, ScenarioObstacleWrapper
from .reward import DonkeyRewardWrapper, ImprovedRewardWrapperV3, V9DomainRewardWrapper
from .control import (
    HighLevelControlWrapper,
    ActionSafetyWrapper, ThrottleControlWrapper, CurvatureAwareThrottleWrapper,
)
from .action_adapter import ActionAdapterWrapper
from .world_model import (
    NeuralPhysicsDynamics,
    build_input_5d,
    build_input_8d,
    INPUT_DIM_BASELINE,
    INPUT_DIM_FULL,
    PHYS_DIM,
    STATE_LO,
    STATE_HI,
)
from .world_model_dataset import (
    CatalogTransitionDataset,
    CatalogTransitionDatasetV2,
    SimTransitionDataset,
    CombinedTransitionDataset,
    chronological_split,
)
from .predictive_safety_filter import PredictiveSafetyFilter, PhysState
from .robust_lane_detector import RobustLaneDetector, RobustYellowLaneEnhancer
from .wrappers import (
    GeneralizationWrapper, TransposeWrapper, NormalizeWrapper,
    V9YellowLaneWrapper, GTResetPerturbWrapper,
    RGBResizeWrapper,
    CanonicalSemanticWrapper,
)
from .callbacks import (
    PTHExportCallback, CoverageLoggingCallback,
    PerSceneStatsCallback, PerDomainStatsCallback,   # PerDomainStatsCallback 为别名
    AdaptiveLearningRateCallback, TrainingMetricsFileLoggerCallback,
    BestModelCallback, DomainAwareBestModelCallback,  # DomainAwareBestModelCallback 为别名
    ShortEpisodeLoggerCallback, CrashRecoveryCallback,
)
from .multi_scene_env import (
    MultiSceneEnv, MultiSceneEnvV12, MultiSceneEnvV13, MultiSceneEnvV16,
    MultiInputObsWrapper,
    _build_v12_wrapper_chain,
    _build_state_v13, _build_state_v16,
)
# All V8/V9 components unified here so v12 has no cross-file imports.
