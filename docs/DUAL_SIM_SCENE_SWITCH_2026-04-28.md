# 双 sim 持久 env 训练方案(方案 B)

记录日期:2026-04-28
影响版本:v17(`src/ppo_multitrack_v17.py`)
关联代码:`module/v17_env.py`、`module/multi_scene_env.py`、`src/ppo_multitrack_v17.py`

---

## 1. 起因:场景切换 100% 失败

### 1.1 现象

正式训练里日志反复出现:

```
🔄 切换场景：目标场景 generated_track（复用模拟器进程）
⏳ 等待场景加载中: generated_track (剩余超时 8s)
⏳ 等待场景加载中: generated_track (剩余超时 3s)
⚠️  场景切换失败: TimeoutError: 场景切换超时（>8s）: generated_track
🔁 尝试重启模拟器并恢复目标场景...
```

某次正式训练的 console.log 里 25+ 次切换 **每一次都走超时回退**,成功率 0%。

### 1.2 关键证据

`gym_donkeycar/envs/donkey_sim.py:645` 在 sim 回包 `scene_names` 时打印 `loading scene XXX`。
日志里这一行**只出现在 gym.make 重连之后**(108-110 行段),**从未出现在 8s 等待段**(103-106 行)。

也就是说 `exit_scene` 之后,sim 端的 `scene_names` 回包**从来没回来过**。

### 1.3 根因

donkey_sim 这个 Unity 构建在 `exit_scene` 时**会断开/挂起 per-vehicle TCP 连接**。SDClient(`gym_donkeycar/core/client.py`)的 `proc_msg` 线程对半开 socket 不报错(`select` 仍报 readable/writable,`recv` 拿 0 字节即 break),`aborted` 也不翻 True,所以 Python 端无法感知死链路 — 只能靠应用层 `wait_until_loaded` 超时兜底。

每次切换的实际成本:**8s 等待 + ~5s 重连 = ~13s 浪费**。

### 1.4 单 sim 重建路径的副作用:zombie 车

第二次 smoke 时观察到"agent 不动 + 障碍车不断 born + 跳来跳去"。诊断:

- `env.close()` → `viewer.quit()` → `client.stop()` 只是关 socket,**不发"卸载车辆"消息**给 Unity
- 这个 Unity 构建在 socket 断开时**不清理对应车辆实体**
- 每次切换 `gym.make` 重连:agent 车 +1 zombie,2 个 obstacle 车 +2 zombie
- 旧切换路径每次 8s 超时后也走 close+rebuild,**也在泄漏**,只是切换频率较低,堆积慢

---

## 2. 方案选型

按"训练总吞吐 + 长跑稳定性 + 跨域泛化"三轴算账:

| 方案 | 切换成本 | 长跑稳态 | 跨域泛化 | 实施代价 |
|---|---|---|---|---|
| A:重启 sim 进程 | ~17s/次 | ✅ 零 zombie | 中 | ~30 行 |
| **B:双 sim 双端口持久 env** | **~0s/次** | ✅ 零 zombie | ✅ 最佳 | ~50 行 |
| C:单场景分两进程 | 0(无切换) | ✅ 零 zombie | ❌ 失去 cross-domain | 0(但放弃 v17 dual-domain 设计) |
| D:单 sim + 增大切换间隔 | ~5s/次 | ❌ zombie 累积慢 | ❌ rollout 单域主导 | 1 行(改默认值) |

**方案 B 是三轴都不输的唯一选项。** 关键洞察:

- 方案 D 的隐性代价是**4096 步 rollout 内场景段数从 8 降到 1-2**,gradient 偏单域 → LSTM 漂移 → sim2real 跨域迁移退化
- 方案 B 因为切换 0 成本,可以**保持 `min_episodes_per_scene=5`**,rollout 内 8 段交替,gradient 信号稳

---

## 3. 实施

### 3.1 代码改动

**`module/v17_env.py`** — `MultiSceneEnvV17`:

新增构造参数 `port_per_scene: Optional[Sequence[int]] = None`。

新增缓存字典:
```python
self._scene_base_envs: Dict[int, Any] = {}
self._scene_obstacle_runtimes: Dict[int, Any] = {}
```

`_create_env(scene_idx)` 入口分两路:

- **`port_per_scene` 设了**:走持久缓存路径
  - 首次访问 scene_idx:用 `port_per_scene[scene_idx]` 覆盖 conf.port,`gym.make` 一次,缓存到 `_scene_base_envs[scene_idx]`
  - 之后访问:直接 `self._base_env = self._scene_base_envs[scene_idx]`(字典查表)
  - obstacle_runtime 同样按 scene_idx 缓存,且 runtime 的 `conf.port` 也使用该 scene 的专属端口

- **`port_per_scene` 未设**:走原 try/except + reload + fallback 链路(完全保留)

`close()` 覆盖:Option B 模式下遍历两个缓存字典,关掉所有持久 env / runtime。

### 3.2 obstacle_runtime 缓存的关键 bug 修复

**第一次实现的版本有 bug**:切换到一个**新**的 scene_idx 时,`self._obstacle_runtime` 仍指向旧 scene 的 runtime(因为没在分支里清空),后面的 `if self._obstacle_runtime is None:` 判断不成立,导致复用旧 runtime → `attach_scene` 检测到 scene_key 变了 → `close()` 旧 fleet → `_ensure_fleet` 重建 → obstacle 车重新 `gym.make`。

症状:每次切换都打印 `starting DonkeyGym env`,fps 跳水到 1-3 步/s。

**修复**:在持久路径里,如果当前 scene_idx 没有缓存的 runtime,**显式把 `self._obstacle_runtime` 置 None**,让下面的 lazy 创建块为这个 scene 建一个全新的:

```python
cached_runtime = self._scene_obstacle_runtimes.get(scene_idx)
if cached_runtime is not None:
    self._obstacle_runtime = cached_runtime
else:
    # 首次访问这个 scene_idx,清掉残留引用,让下面的懒创建走新路径
    self._obstacle_runtime = None
```

修复后:每个 scene 只在**首次访问**时做一次 obstacle 车 `gym.make`,之后纯查表。

### 3.3 Timeout clamp 调整

由于回退方案(超时 fallback)仍保留作为 `--ports` 未设时的路径,把超时从默认 8s 降到 2s:

- `src/ppo_multitrack_v17.py` 默认 `scene_reload_timeout_s: float = 2.0`
- `module/multi_scene_env.py` `_force_reload_scene` 的 `max(3.0, ...)` clamp 改为 `max(1.0, ...)`
- 同 file `__init__` 里 `max(3.0, scene_reload_timeout_s)` clamp 同步改为 `max(1.0, ...)`

### 3.4 CLI

新增 flag(只在方案 B 下使用):

```
--ports 9091 9093
```

`nargs="+"`,长度必须等于 `--env-ids`(默认 2 个)。

---

## 4. Sim 进程编排

### 4.1 双 sim 启动

donkey_sim Unity 构建支持 `--port N --host 0.0.0.0` 命令行参数。

**sim #1**:`9091/9092`(用户原本就跑着,`start_donkey_vnc.sh` 启的,在 `pts/3`)

**sim #2**:迁到 tmux 持久会话:

```bash
tmux new-session -d -s sim-9093 \
  "DISPLAY=:1 /home/longzhao/DonkeySim/DonkeySimLinux/donkey_sim.x86_64 \
   --port 9093 --host 0.0.0.0 \
   -screen-width 800 -screen-height 600 \
   -logFile /tmp/donkey_sim_9093.log 2>&1 | tee /tmp/donkey_sim_9093.stdout"
```

每个 sim 进程会**自动开两个相邻端口**(主控 + 副控),所以:
- sim #1 占 9091 + 9092
- sim #2 占 9093 + 9094

只用主控端口(9091, 9093)给 v17 训练。

### 4.2 端口与场景映射

`DEFAULT_ENV_IDS` 顺序:
```python
["donkey-waveshare-v0", "donkey-generated-track-v0"]
# ↑ scene_idx=0 (ws)        ↑ scene_idx=1 (gt)
```

CLI `--ports 9091 9093` 表示:
- ws 走 9091(sim #1)
- gt 走 9093(sim #2)

### 4.3 状态查看

```bash
# 查端口
ss -tlnp | grep ":909"

# 查进程
ps -ef | grep donkey_sim

# attach sim-9093 看输出
tmux attach -t sim-9093
# Ctrl-B 然后 d 脱离

# 看 sim #2 日志
tail -f /tmp/donkey_sim_9093.log
```

### 4.4 重启机器后

两个 sim 都需要手工启动(没做 systemd unit)。建议把 sim #1 也搬进 tmux 一起管理(目前还在 pts/3 的旧会话里跑了 7+ 天)。

---

## 5. 测试结果

### 5.1 Smoke #2(修复 obstacle_runtime bug 后)

命令:
```bash
python -u src/ppo_multitrack_v17.py \
  --port 9091 --ports 9091 9093 \
  --sim remote --steps 6000 \
  --ppo-n-steps 1024 --ppo-batch-size 256 \
  --disable-preflight-checks \
  --curriculum-phase warmup \
  --resume-path .../v17_1x_full_fixall_from_warmup_20260428_044059/final_model.zip
```

| 指标 | 数值 |
|---|---|
| 总耗时 | 7.2 min |
| 净训练步数 | 6143 |
| 切换次数 | 12 |
| `场景切换失败` | **0** |
| `场景切换超时` | **0** |
| Traceback / Error | **0** |
| 短命 episode(len<15) | 2 |
| 聚合 fps | **14.2 步/s** |
| tqdm 稳态 fps | 16-19 步/s |
| explained_variance | 0.785 |

### 5.2 切换路径验证(grep 日志)

正确的初始化序列:

```
starting DonkeyGym env             ← agent 主 env 连 9091
✅ [scene_idx=0 port=9091] 已加载持久场景: waveshare
   双 sim 模式: ports=[9091, 9093] (持久 env，无切换重建)
starting DonkeyGym env             ← obstacle car 在 9091

starting DonkeyGym env             ← agent 主 env 连 9093
✅ [scene_idx=1 port=9093] 已加载持久场景: generated_track
starting DonkeyGym env             ← obstacle car 在 9093

🔁 [scene_idx=0 port=9091] 切换到持久场景: waveshare
🔁 [scene_idx=1 port=9093] 切换到持久场景: generated_track
🔁 [scene_idx=0 port=9091] 切换到持久场景: waveshare
... (12 次 🔁,完全无 starting DonkeyGym env)
```

### 5.3 对比基线

| 项 | 旧链路(8s reload + close+rebuild fallback) | 方案 B |
|---|---|---|
| 聚合 fps | 11 | **14.2** |
| 切换成本 | 13s | **~0s** |
| zombie 累积 | +3 辆/次切换 | **零** |
| rollout 内场景混合 | 8 段 | 8 段(保持) |

吞吐 +29%,且长跑无衰减。

### 5.4 一个保留的小问题

切换瞬间 fps 会瞬时落到 1-3 步/s 持续 ~5-10s 才回到稳态。这是"对端 sim 长期 idle 后第一个 step + reset 的握手延迟",**不是**修复链路问题(已确认无 gym.make,无 attach_scene 重建)。

可优化方向(尚未实施):
- 切换时主动给目标 sim 发一个 noop step "唤醒"
- 或在 `_create_env` 末尾对新激活的 env 做一次预热 reset

---

## 6. 关于 domain_id 的澄清(与方案 B 无关,但易混淆)

`domain_id` 是 v17 obs Dict 的一个字段,常被误以为是"双域泛化"的关键。事实:

- **v17 原始设计的一部分**(v16 没有)
- **actor 完全不消费**:`v17_policy.py:96` "domain_id: (1,) optional, ignored by actor features"
- 只在两处使用,均与 actor 无关:
  - critic 的 dual value heads(默认关,由 `CriticCalibrationCallback` 在训练中自动检测后激活)
  - `CriticCalibrationCallback` 自身的诊断日志
- **删掉 obs 里的 domain_id,actor 输出一字节都不变**

这是非对称 actor-critic 的标准做法:训练时给 critic 用上帝视角额外信息,但策略本身被强制泛化(因为 actor 看不到 domain_id)。

**结论**:方案 B 不影响 domain_id 的语义,跨域泛化由 "actor 看不到 domain_id" 这个事实保证,而不是由场景切换机制保证。

---

## 7. 文件备份

回退原版本(v17_env.py 早期"始终重建"修复版):

- `module/v17_env.py.bak_reloadfix_20260428_053845`
- `src/ppo_multitrack_v17.py.bak_reloadfix_20260428_053845`

这两份是我在引入方案 B 之前的中间状态(close + gym.make 直接重建,跳过 8s 等待)。如果发现方案 B 有未预料问题,可以回退到这版,但仍有 zombie 累积风险,不建议长跑用。

当前版本(方案 B + 默认 timeout 2s)是推荐的正式训练版本。

---

## 8. 正式训练命令模板

```bash
python -u src/ppo_multitrack_v17.py \
  --ports 9091 9093 \
  --sim remote \
  --steps 2000000 \
  --auto-curriculum \
  --auto-curriculum-no-hard-min-gate \
  --save-dir models/v17_optionB_full_$(date +%Y%m%d_%H%M%S) \
  --exp-tag optionB_full
```

前置条件:
- sim #1 跑在 9091/9092(任何位置)
- sim #2 跑在 9093/9094(推荐 tmux session `sim-9093`)
- 两个 sim 都基于同一个 DonkeySim build,DISPLAY 共享 `:1`

回退到旧路径(如果方案 B 出问题):去掉 `--ports`,会走 close+rebuild + 2s timeout fallback。
