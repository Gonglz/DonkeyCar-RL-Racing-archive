# V17 Formal Readiness Matrix

本文档定义 `scripts/run_v17_formal_readiness.py` 的阶段门禁和顶层签字语义。

## 1. 顶层结论

- `formal_train_ready`
  - 含义：允许启动 `V17` 正式 long-train
  - 阻塞项：`Phase A` 到 `Phase D`
- `runtime_ready`
  - 含义：允许进入在线 `log-only` 接线验证
  - 当前状态：本 readiness 程序**不评估**，固定为 `false`
- `deployment_ready`
  - 含义：real-first / deployment-prep 的 LiDAR 基础层已通过 formal 验收
  - 阻塞项：`Phase F`

顶层 `launch_decision` 只由 `formal_train_ready` 决定：

- `pass`
  - `Phase A-D` 全部通过
- `fail`
  - `Phase A-D` 任一失败

因此允许出现以下组合：

- `formal_train_ready=true`
- `runtime_ready=false`
- `deployment_ready=false`

这表示：

- 可以启动正式 long-train
- 不能声称 runtime `log-only` 已验证
- 不能声称 real-first / deployment 已就绪

## 2. Phase Matrix

### Phase A: 环境与输入发现

必须通过：

- `python_bin` 存在
- manifest 中列出的关键脚本存在
- pinned policy/checkpoint 路径存在
- real LiDAR log 可解析出至少一个 canonical sample
- `V17` preflight / contract tests 通过

失败后：

- `Phase B/C/D/F` 全部跳过
- `formal_train_ready=false`

### Phase B: PPO Smoke

必须通过：

- smoke 命令返回码为 `0`
- 产出：
  - `final_model.zip`
  - `final_model_policy.pth`
  - `v17_config.json`
  - `train_metrics.jsonl`
- `train_metrics.jsonl` 中存在非零 `train/n_updates`

失败后：

- `formal_train_ready=false`

### Phase C: Dataset Export + Merge

必须通过：

- 三份导出全部成功
- merged dataset 成功生成
- 必需 key 齐全
- 所有数组第一维一致
- 无 NaN / Inf
- `samples == 5120`
- `generated_track` 与 `waveshare` 都有样本
- `collision_pos_rate` 在 `[0.05, 0.30]`
- `opportunity_valid_rate >= 0.70`
- `passable_left_rate` 与 `passable_right_rate` 都在 `[0.10, 0.90]`

失败后：

- `Phase D` 跳过
- `formal_train_ready=false`

### Phase D: World-Model Training

必须通过：

- `Stage A` / `Stage B` best checkpoint 存在
- `local_world_model_v17_final.pth` 存在
- `train_summary.json` 存在
- `Stage A`
  - `mae_ego <= 0.06`
  - `mae_target_rel <= 2.50`
  - `mae_gap <= 2.00`
- `Stage B` 相比 `Stage A` 的几何退化不超过 `5%`
- `Stage B` 关键 loss / MAE 全部有限
- `Stage C`
  - 正常 best checkpoint 通过，或
  - guard fallback 成功保存，且回退后几何不差于 `Stage B + 3%`

失败后：

- `formal_train_ready=false`

### Phase F: Deployment-Prep

这是独立门禁，不阻塞 `formal_train_ready`。

必须通过：

- sim raw-LiDAR monitor 成功采集
- `eval_lidar_domain_gap.py` 满足：
  - `valid_ratio_mae <= 0.10`
  - `wasserstein_median <= 0.08`
  - `wasserstein_p95 <= 0.20`
  - `scene_js_divergence <= 0.15`

失败后：

- `deployment_ready=false`
- readiness report 中必须明确写出：
  - `real-first blocked by LiDAR domain gap`

## 3. Runner Outputs

标准输出目录下至少应包含：

- `readiness_manifest_resolved.json`
- `readiness_report.json`
- `phase_a/provenance.json`
- `phase_c_dataset/dataset_readiness.json`
- `phase_d_world_model_train/training_readiness.json`
- `phase_f_deployment/lidar_domain_gap_eval.json`

每个执行 phase 还应写出对应命令日志。
