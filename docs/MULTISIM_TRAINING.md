# Multi-Simulator Parallel Training

`src/ppo_multitrack_v16_multisim.py` 是 `ppo_multitrack_v16.py` 的并行扩展版本，
通过同时连接多个 DonkeySim 实例（`SubprocVecEnv`）将 rollout 采集效率提升 N 倍。

**设计原则：不修改任何现有文件**，全部逻辑通过 import 复用原有代码。

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `src/ppo_multitrack_v16_multisim.py` | 训练入口，支持 `--ports` 多端口参数 |
| `~/bin/start_donkey_vnc_multi.sh` | 启动 3 个模拟器实例的 bash 脚本 |

---

## 快速开始

```bash
# 1. 启动 3 个模拟器（Xorg + VNC + 3x DonkeySim）
bash ~/bin/start_donkey_vnc_multi.sh

# 2. 等待约 15 秒，脚本会自动做端口连通性检查

# 3. 开始多路并行训练
cd /home/longzhao/mysim_public
python src/ppo_multitrack_v16_multisim.py \
    --ports 9093 9095 9097 \
    --auto-curriculum \
    --steps 2000000
```

仍然支持单端口模式（等价于原 v16 脚本）：

```bash
python src/ppo_multitrack_v16_multisim.py --port 9091 --steps 2000000
```

---

## 架构差异（与 v16 单路训练的区别）

### VecEnv 选择

| 条件 | VecEnv 类型 | 行为 |
|------|------------|------|
| `len(ports) == 1` | `DummyVecEnv` | 单进程顺序执行，与原 v16 完全一致 |
| `len(ports) > 1` | `SubprocVecEnv(start_method="fork")` | 每个 port 独立子进程，真并行采集 |

### 模型初始化

RecurrentPPO 的 `n_envs` 必须在构建时确定，因此 dummy 占位环境也需对应数量：

```python
n_envs = len(ports)
dummy_vec_env = DummyVecEnv([lambda: DummyEnv()] * n_envs)
model = RecurrentPPO(..., env=dummy_vec_env)
model.set_env(SubprocVecEnv(env_fns, start_method="fork"))
```

### 工厂函数模式

每个子进程环境持有独立的端口配置：

```python
def make_env_factory(env_port: int):
    env_conf = dict(conf)
    env_conf["port"] = env_port
    def _make():
        return MultiSceneEnvV16(env_ids=env_ids, conf=env_conf, ...)
    return _make

env_fns = [make_env_factory(p) for p in ports]
```

---

## 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_steps` | 4096 | per-env，总 rollout = n_envs × n_steps |
| `batch_size` | 256 | 3 envs × 4096 = 12288；12288/256 = 48 minibatches |
| `learning_rate` | 8e-5 | 与原 v16 相同；可按 √n_envs 缩放至 ~1.4e-4 |

其余超参数（`n_epochs`、`gamma`、`gae_lambda`、`ent_coef`、`clip_range` 等）与 `ppo_multitrack_v16.py` 完全一致。

---

## Callback 兼容性

所有 v16 原有 callbacks 均通过 `model.ep_info_buffer` 或 `model.env` 读取统计，
与 `SubprocVecEnv` 天然兼容，无需修改：

- `PerSceneStatsCallback` — 多 env 的 episode info 自动聚合到同一 buffer
- `CrashRecoveryCallback` — 检测所有子进程 env 的崩溃状态
- `BestModelCallback`、`PTHExportCallback` — 只监控 reward，与 env 数量无关
- `AdaptiveLearningRateCallback` — 读 `ep_info_buffer`，多 env 下样本更充分

---

## 模拟器启动脚本说明

`start_donkey_vnc_multi.sh` 逻辑：

1. 清理旧进程（`pkill donkey_sim`、`X1` 锁文件等）
2. 启动 Xorg `:1`（GPU 硬件加速，与单实例脚本相同）
3. 启动 VNC（仅本地，端口 5901）
4. 依次启动 3 个 DonkeySim 实例，端口 **9093 / 9095 / 9097**，间隔 3 秒
5. 等待 15 秒后检查端口连通性
6. `wait` 保持脚本运行直到 Ctrl+C

分辨率使用 **800×600**（原单实例为 1280×720），以减少 VRAM 占用。

---

## 端口约定

| 端口 | 用途 |
|------|------|
| 9091 | 原 v16 单路训练（默认） |
| 9093 | 多路训练 env-0 |
| 9095 | 多路训练 env-1 |
| 9097 | 多路训练 env-2 |

奇数端口对应 sim TCP 监听；偶数端口（+1）由 Unity 内部用于 telemetry，外部不直接访问。

---

## 资源需求

| 资源 | 单路 | 3 路并行 | 备注 |
|------|------|---------|------|
| VRAM | ~2 GB | ~5-6 GB | 每个 Unity 实例 ~1.5-2 GB |
| CPU | ~4 核 | ~10 核 | 每个子进程 env + sim |
| 训练时间 | 1x | ~0.35-0.45x | 受 sim 启动/帧率限制，非线性 |

若 GPU 显存不足，可将 `-screen-width` 降至 640，在 `start_donkey_vnc_multi.sh` 中修改 `SIM_WIDTH=640 SIM_HEIGHT=480`。

---

## 故障排查

**`AssertionError: The number of environments ... (3 != 1)`**
- 原因：`RecurrentPPO` 初始化时 `n_envs=1`，`set_env` 时传入 n_envs=3 的 VecEnv
- 修复：`dummy_vec_env = DummyVecEnv([lambda: DummyEnv()] * n_envs)`

**某个端口不可达**
```bash
nc -zv 127.0.0.1 9093
```
- 检查 `/tmp/donkey_sim_9093.log` 定位 Unity 崩溃原因
- 重新运行 `start_donkey_vnc_multi.sh`

**SubprocVecEnv 子进程挂起**
- 确认 `start_method="fork"`（Linux 默认，无需 pickle）
- 若使用 `spawn`，需确保 `MultiSceneEnvV16` 及其依赖可序列化
