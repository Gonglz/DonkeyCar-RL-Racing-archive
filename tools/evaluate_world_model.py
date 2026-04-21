"""
tools/evaluate_world_model.py

世界模型评估脚本：
  1. 单步 RMSE（per-dim，按归一化空间）
  2. k 步自回归误差（k = 1, 2, 4, 8, 16），在 sim 测试集上评估
  3. Cross-domain 评估（量化 sim2real gap）：
       wm_sim 在真实数据上测试  → sim→real gap
       wm_real 在 sim 数据上测试 → real→sim gap

用法示例
--------
# 单模型评估
python3 tools/evaluate_world_model.py \\
    --model models/world_model/wm_real.pth \\
    --catalog-dirs data \\
    --mode single

# Cross-domain gap 分析
python3 tools/evaluate_world_model.py \\
    --model-real  models/world_model/wm_real.pth \\
    --model-sim   models/world_model/wm_sim.pth \\
    --catalog-dirs data \\
    --sim-dirs dynamics_data/sim_transitions \\
    --mode cross_domain
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_SCRIPT_DIR = Path(__file__).parent
_REPO_DIR   = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_DIR))

from module.world_model import NeuralPhysicsDynamics, PHYS_DIM, STATE_LO, STATE_HI
from module.world_model_dataset import (
    CatalogTransitionDatasetV2,
    SimTransitionDataset,
    chronological_split,
)

DIM_NAMES   = ["v_long", "yaw_rate", "accel_x"]
DIM_TARGETS = [0.05, 0.05, 0.10]   # RMSE 目标值


# ─── 单步评估 ────────────────────────────────────────────────────

@torch.no_grad()
def eval_single_step(
    model: NeuralPhysicsDynamics,
    loader: DataLoader,
    device: str,
) -> dict:
    model.eval()
    sq_errs = torch.zeros(PHYS_DIM)
    n = 0
    for batch in loader:
        x, delta = batch[0].to(device), batch[1].to(device)
        pred_delta, _ = model(x, x[:, :PHYS_DIM])
        sq_errs += ((pred_delta.cpu() - delta.cpu()) ** 2).sum(dim=0)
        n += x.shape[0]
    n = max(n, 1)
    rmse = {name: math.sqrt(float(sq_errs[i] / n))
            for i, name in enumerate(DIM_NAMES)}
    return rmse


# ─── k 步自回归误差 ──────────────────────────────────────────────

@torch.no_grad()
def eval_kstep_rollout(
    model: NeuralPhysicsDynamics,
    dataset,
    device: str,
    k_list: List[int] = (1, 2, 4, 8, 16),
    n_seeds: int = 200,
) -> dict:
    """
    从测试集中随机采 n_seeds 个起始点，向前自回归 max(k_list) 步。
    返回每个 k 的 per-dim RMSE 字典。

    注意：k 步展开需要连续样本，这里使用"假滚动"：
    只保持 phys_t 自回归更新，action 从测试集依次读入
    （即"open-loop"预测，action 使用 ground truth）。
    """
    model.eval()
    max_k  = max(k_list)
    n      = len(dataset)
    seeds  = np.random.choice(max(n - max_k - 1, 1), size=min(n_seeds, n - max_k - 1), replace=False)

    # 对每个 k，累积误差
    sq_errs = {k: torch.zeros(PHYS_DIM) for k in k_list}
    counts  = {k: 0 for k in k_list}

    for seed in seeds:
        # 取起始真实物理状态
        x0, _ = dataset[int(seed)]
        phys_cur = x0[:PHYS_DIM].clone().to(device)

        for step in range(1, max_k + 1):
            idx = int(seed) + step
            if idx >= n:
                break
            x_step, delta_step = dataset[idx]
            x_step = x_step.to(device)

            # 预测残差（open-loop：使用真实 action 特征）
            x_pred = x_step.clone()
            x_pred[:PHYS_DIM] = phys_cur
            pred_delta, phys_next = model(x_pred.unsqueeze(0), phys_cur.unsqueeze(0))
            phys_cur = phys_next.squeeze(0).detach()

            if step in k_list:
                # 目标：真实下一步物理状态
                true_phys = x_step[:PHYS_DIM] + delta_step.to(device)
                true_phys = torch.clamp(
                    true_phys,
                    STATE_LO.to(device),
                    STATE_HI.to(device),
                )
                sq_errs[step] += ((phys_cur.cpu() - true_phys.cpu()) ** 2)
                counts[step]  += 1

    result = {}
    for k in k_list:
        c = max(counts[k], 1)
        rmse_per_dim = {name: math.sqrt(float(sq_errs[k][i] / c))
                        for i, name in enumerate(DIM_NAMES)}
        result[k] = rmse_per_dim
    return result


# ─── 打印结果 ────────────────────────────────────────────────────

def print_rmse_table(rmse: dict, title: str, targets: Optional[List[float]] = None):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    for i, (name, val) in enumerate(rmse.items()):
        tgt = targets[i] if targets else None
        ok  = ""
        if tgt is not None:
            ok = "  ✓" if val < tgt else f"  ✗ (target < {tgt:.2f})"
        print(f"  {name:12s}: RMSE = {val:.5f}{ok}")


def print_kstep_table(kstep: dict):
    print(f"\n{'─'*60}")
    print(f"  k-step 自回归 RMSE（open-loop, ground-truth action）")
    print(f"{'─'*60}")
    print(f"  {'k':>3}  {'v_long':>9}  {'yaw_rate':>9}  {'accel_x':>9}")
    k1_rmse = {n: kstep[1][n] for n in DIM_NAMES} if 1 in kstep else None
    for k, rmse in sorted(kstep.items()):
        ratio_str = ""
        if k1_rmse is not None and k > 1:
            ratios = [f"{rmse[n] / max(k1_rmse[n], 1e-9):.1f}x" for n in DIM_NAMES]
            ratio_str = "  (" + " / ".join(ratios) + " vs k=1)"
        print(
            f"  {k:>3}  {rmse['v_long']:>9.5f}  "
            f"{rmse['yaw_rate']:>9.5f}  {rmse['accel_x']:>9.5f}{ratio_str}"
        )


# ─── 主函数 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate world model")
    parser.add_argument("--mode", choices=["single", "cross_domain"], default="single")
    parser.add_argument("--model",      default="", help="单模型路径（--mode single）")
    parser.add_argument("--model-real", default="", help="wm_real.pth（--mode cross_domain）")
    parser.add_argument("--model-sim",  default="", help="wm_sim.pth（--mode cross_domain）")
    parser.add_argument("--catalog-dirs", nargs="+", default=[],
                        help="真实车 catalog 目录列表")
    parser.add_argument("--sim-dirs",   nargs="+", default=[],
                        help="Sim CSV 目录列表")
    parser.add_argument("--input-dim",  type=int, choices=[5, 8], default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--kstep-seeds", type=int, default=200,
                        help="k 步评估的起始点数量")
    parser.add_argument("--output-json", default="",
                        help="评估结果保存路径（可选）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─ 加载数据集 ─
    real_ds = sim_ds = None
    if args.catalog_dirs:
        real_ds = CatalogTransitionDatasetV2(args.catalog_dirs, input_dim=args.input_dim, augment_noise=0.0)
        real_ds.eval()
        _, _, real_test = chronological_split(real_ds, 0.8, 0.1)
        print(f"Real test samples: {len(real_test):,}")

    if args.sim_dirs:
        sim_ds = SimTransitionDataset(args.sim_dirs, input_dim=args.input_dim, augment_noise=0.0)
        sim_ds.eval()
        _, _, sim_test = chronological_split(sim_ds, 0.8, 0.1)
        print(f"Sim  test samples: {len(sim_test):,}")

    def make_loader(subset):
        return DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_results = {}

    if args.mode == "single":
        if not args.model:
            parser.error("--model required for --mode single")
        model = NeuralPhysicsDynamics.load_checkpoint(args.model, device=device)
        print(f"\nModel: {args.model}  (input_dim={model.input_dim})")

        if real_ds is not None:
            rmse = eval_single_step(model, make_loader(real_test), device)
            print_rmse_table(rmse, "单步 RMSE — 真实车测试集", DIM_TARGETS)
            all_results["real_single_step"] = rmse

            kstep = eval_kstep_rollout(model, real_test.dataset,
                                       device, n_seeds=args.kstep_seeds)
            print_kstep_table(kstep)
            all_results["real_kstep"] = {str(k): v for k, v in kstep.items()}

        if sim_ds is not None:
            rmse = eval_single_step(model, make_loader(sim_test), device)
            print_rmse_table(rmse, "单步 RMSE — Sim 测试集", DIM_TARGETS)
            all_results["sim_single_step"] = rmse

            kstep = eval_kstep_rollout(model, sim_test.dataset,
                                       device, n_seeds=args.kstep_seeds)
            print_kstep_table(kstep)
            all_results["sim_kstep"] = {str(k): v for k, v in kstep.items()}

    elif args.mode == "cross_domain":
        if not args.model_real or not args.model_sim:
            parser.error("--model-real and --model-sim required for cross_domain mode")
        if real_ds is None or sim_ds is None:
            parser.error("--catalog-dirs and --sim-dirs both required for cross_domain mode")

        wm_real = NeuralPhysicsDynamics.load_checkpoint(args.model_real, device=device)
        wm_sim  = NeuralPhysicsDynamics.load_checkpoint(args.model_sim,  device=device)

        print(f"\nwm_real: {args.model_real}")
        print(f"wm_sim:  {args.model_sim}")

        # wm_real 在真实测试集（同域）
        rmse_rr = eval_single_step(wm_real, make_loader(real_test), device)
        print_rmse_table(rmse_rr, "wm_real → 真实数据（同域，基准）", DIM_TARGETS)

        # wm_sim 在真实测试集（sim→real gap）
        rmse_sr = eval_single_step(wm_sim, make_loader(real_test), device)
        print_rmse_table(rmse_sr, "wm_sim  → 真实数据（sim→real gap）", DIM_TARGETS)

        # wm_sim 在 sim 测试集（同域）
        rmse_ss = eval_single_step(wm_sim, make_loader(sim_test), device)
        print_rmse_table(rmse_ss, "wm_sim  → Sim 数据（同域，基准）", DIM_TARGETS)

        # wm_real 在 sim 测试集（real→sim gap）
        rmse_rs = eval_single_step(wm_real, make_loader(sim_test), device)
        print_rmse_table(rmse_rs, "wm_real → Sim 数据（real→sim gap）", DIM_TARGETS)

        # Gap 比例
        print(f"\n{'─'*60}")
        print("  Sim2Real Gap 比例（越接近 1.0 越好）")
        print(f"{'─'*60}")
        print(f"  {'维度':12s}  {'sim→real/baseline':>18}  {'real→sim/baseline':>18}")
        for name in DIM_NAMES:
            ratio_sr = rmse_sr[name] / max(rmse_rr[name], 1e-9)
            ratio_rs = rmse_rs[name] / max(rmse_ss[name], 1e-9)
            ok_sr    = "✓" if ratio_sr < 1.5 else "✗"
            ok_rs    = "✓" if ratio_rs < 1.5 else "✗"
            print(
                f"  {name:12s}  "
                f"{ratio_sr:>18.3f} {ok_sr}  "
                f"{ratio_rs:>18.3f} {ok_rs}"
            )
        print()
        print("  比例 < 1.5 → gap 小，可考虑直接混合训练 wm_mixed")
        print("  比例 ≥ 1.5 → gap 大，建议 dual-head 或 domain embedding")

        all_results = {
            "wm_real_on_real": rmse_rr,
            "wm_sim_on_real":  rmse_sr,
            "wm_sim_on_sim":   rmse_ss,
            "wm_real_on_sim":  rmse_rs,
        }

    # ─ 保存 JSON ─
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {args.output_json}")


if __name__ == "__main__":
    main()
