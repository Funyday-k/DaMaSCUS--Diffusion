"""
evaluate.py — 扩散模型生成质量的定量评估

功能：
  1. 单步散射评估：从真实数据取 condition，生成 output，对比真实 output 的分布
  2. Wasserstein-1 距离（Earth Mover's Distance）量化分布匹配
  3. 条件响应诊断：固定不同 r，看 E_in vs E_out 是否合理
  4. 边缘分布直方图（生成 vs MC 真实）
  5. 物理约束违反率统计

输出：
  - 终端打印定量指标
  - 保存 evaluation_results.png 多面板对比图

用法：
    python inference/evaluate.py
    python inference/evaluate.py --n_samples 50000 --method ddim
"""

import os
import sys
import argparse
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

from inference.sampler import DarkMatterSampler


def load_ground_truth(npz_path: str, n_samples: int = 10000,
                      seed: int = 42) -> tuple:
    """
    从真实 MC 数据中随机抽取 n_samples 对 (condition, target)。
    返回物理单位的碰撞前/后状态。
    """
    data = np.load(npz_path)
    raw_in  = data['states_in']
    raw_out = data['states_out']

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(raw_in), size=min(n_samples, len(raw_in)), replace=False)

    def to_spherical(states):
        pos = states[:, 2:5]
        vel = states[:, 5:8]
        E   = states[:, 8]
        r   = states[:, 9]
        v_rad = np.sum(pos * vel, axis=1) / r
        v_tan = np.sqrt(np.maximum(0, np.sum(vel ** 2, axis=1) - v_rad ** 2))
        return np.column_stack([r, v_rad, v_tan, E])

    cond_phys   = to_spherical(raw_in[idx])
    target_phys = to_spherical(raw_out[idx])

    return cond_phys, target_phys


def compute_wasserstein1(p: np.ndarray, q: np.ndarray) -> float:
    """
    计算一维 Wasserstein-1 距离。

    W₁(P, Q) = ∫ |F_P(x) - F_Q(x)| dx

    通过排序后的经验 CDF 差的均值计算。
    """
    p_sorted = np.sort(p)
    q_sorted = np.sort(q)[:len(p_sorted)]
    return float(np.mean(np.abs(p_sorted - q_sorted)))


def evaluate_single_step(sampler: DarkMatterSampler,
                          cond_phys: np.ndarray,
                          target_phys: np.ndarray,
                          method: str = "ddim",
                          num_steps: int = 50,
                          batch_size: int = 2048) -> dict:
    """
    单步散射评估：给定真实 condition，生成 output，对比真实 output。
    模型输出 [Δv_rad, Δv_tan] → sampler 重构为 [r, v_rad, v_tan, E(NaN)]。
    此处将 E 从物理公式补全后再评估。
    """
    n = len(cond_phys)
    generated = []

    # 分批生成以节省显存
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        cond_batch = torch.tensor(cond_phys[start:end], dtype=torch.float32)
        gen_batch = sampler.sample(cond_batch, method=method, num_steps=num_steps)
        generated.append(gen_batch.cpu().numpy())

    gen_phys = np.vstack(generated)
    # gen_phys[:,3] 是 NaN（E 待计算），这里不使用 E 做评估

    # ── 1. 各特征的 Wasserstein-1 距离（只评估 r, v_rad, v_tan）──
    feature_names = ['r', 'v_rad', 'v_tan']
    w1_distances = {}
    for i, name in enumerate(feature_names):
        w1 = compute_wasserstein1(gen_phys[:, i], target_phys[:, i])
        scale = np.std(target_phys[:, i]) + 1e-8
        w1_distances[name] = {'absolute': w1, 'relative': w1 / scale}

    # ── 2. 逐样本相对误差（r, v_rad, v_tan）──
    rel_err = np.abs(gen_phys[:, :3] - target_phys[:, :3]) / (np.abs(target_phys[:, :3]) + 1e-8)
    mean_rel_err = {name: float(rel_err[:, i].mean())
                    for i, name in enumerate(feature_names)}
    median_rel_err = {name: float(np.median(rel_err[:, i]))
                      for i, name in enumerate(feature_names)}

    # ── 3. 物理约束违反率 ──
    n_neg_vtan   = int((gen_phys[:, 2] < 0).sum())
    n_neg_r      = int((gen_phys[:, 0] < 0).sum())

    constraints = {
        'v_tan < 0':    n_neg_vtan / n * 100,
        'r < 0':        n_neg_r / n * 100,
    }

    # ── 4. 各特征统计（只评估 r, v_rad, v_tan）──
    stats = {}
    for i, name in enumerate(feature_names):
        stats[name] = {
            'gen_mean':    float(gen_phys[:, i].mean()),
            'gen_std':     float(gen_phys[:, i].std()),
            'truth_mean':  float(target_phys[:, i].mean()),
            'truth_std':   float(target_phys[:, i].std()),
        }

    return {
        'generated':      gen_phys,
        'w1_distances':   w1_distances,
        'mean_rel_err':   mean_rel_err,
        'median_rel_err': median_rel_err,
        'constraints':    constraints,
        'stats':          stats,
    }


def print_report(results: dict):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print("           扩散模型单步散射评估报告")
    print("=" * 70)

    print("\n1. Wasserstein-1 距离 (越小越好):")
    print(f"   {'特征':>8} {'W₁ 绝对':>14} {'W₁ 相对(σ)':>14}")
    print("   " + "-" * 40)
    for name, d in results['w1_distances'].items():
        print(f"   {name:>8} {d['absolute']:14.2f} {d['relative']:14.4f}")

    print("\n2. 逐样本相对误差:")
    print(f"   {'特征':>8} {'平均':>12} {'中位数':>12}")
    print("   " + "-" * 36)
    for name in ['r', 'v_rad', 'v_tan']:
        mean = results['mean_rel_err'][name]
        med  = results['median_rel_err'][name]
        print(f"   {name:>8} {mean*100:11.2f}% {med*100:11.2f}%")

    print("\n3. 物理约束违反率:")
    for desc, pct in results['constraints'].items():
        flag = "⚠️" if pct > 5 else "✓"
        print(f"   {flag} {desc}: {pct:.2f}%")

    print("\n4. 分布统计对比:")
    print(f"   {'特征':>8} {'生成均值':>12} {'真实均值':>12} {'生成σ':>12} {'真实σ':>12}")
    print("   " + "-" * 52)
    for name, s in results['stats'].items():
        print(f"   {name:>8} {s['gen_mean']:12.1f} {s['truth_mean']:12.1f} "
              f"{s['gen_std']:12.1f} {s['truth_std']:12.1f}")


def save_plots(results: dict, cond_phys: np.ndarray, target_phys: np.ndarray,
               save_path: str):
    """保存评估可视化图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib 未安装，跳过绘图)")
        return

    gen = results['generated']
    feature_names = ['r [km]', 'v_rad [km/s]', 'v_tan [km/s]']
    feature_keys  = ['r', 'v_rad', 'v_tan']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── 第一行：边缘分布直方图（r, v_rad, v_tan）──
    for i in range(3):
        ax = axes[0, i]
        all_vals = np.concatenate([gen[:, i], target_phys[:, i]])
        lo, hi = np.percentile(all_vals, [1, 99])
        bins = np.linspace(lo, hi, 80)

        ax.hist(target_phys[:, i], bins=bins, density=True,
                alpha=0.6, label='MC 真实', color='steelblue')
        ax.hist(gen[:, i], bins=bins, density=True,
                alpha=0.6, label='扩散模型', color='coral')
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('概率密度')
        ax.legend(fontsize=8)

        w1_rel = results['w1_distances'][feature_keys[i]]['relative']
        ax.set_title(f'W₁/σ = {w1_rel:.4f}')

    # ── 第二行：条件响应散点图 ──
    scatter_pairs = [
        (0, 0, 'r_in → r_out'),
        (1, 1, 'v_rad_in → v_rad_out'),
    ]
    for j, (ci, ti, title) in enumerate(scatter_pairs):
        ax = axes[1, j]
        n_plot = min(3000, len(gen))
        ax.scatter(cond_phys[:n_plot, ci], target_phys[:n_plot, ti],
                   s=1, alpha=0.3, label='MC 真实', color='steelblue')
        ax.scatter(cond_phys[:n_plot, ci], gen[:n_plot, ti],
                   s=1, alpha=0.3, label='扩散模型', color='coral')
        ax.set_xlabel(f'{feature_keys[ci]}_in')
        ax.set_ylabel(f'{feature_keys[ti]}_out')
        ax.set_title(title)
        ax.legend(fontsize=7, markerscale=5)

    # 第二行第三个：Δv_rad 残差分布
    ax = axes[1, 2]
    dv_rad_truth = target_phys[:, 1] - cond_phys[:, 1]
    dv_rad_gen   = gen[:, 1] - cond_phys[:, 1]
    lo, hi = np.percentile(np.concatenate([dv_rad_truth, dv_rad_gen]), [1, 99])
    bins = np.linspace(lo, hi, 80)
    ax.hist(dv_rad_truth, bins=bins, density=True,
            alpha=0.6, label='MC Δv_rad', color='steelblue')
    ax.hist(dv_rad_gen, bins=bins, density=True,
            alpha=0.6, label='模型 Δv_rad', color='coral')
    ax.set_xlabel('Δv_rad [km/s]')
    ax.set_title('径向速度变化分布')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\n评估图已保存 -> {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import glob

    parser = argparse.ArgumentParser(description="扩散模型评估")
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='评估样本数')
    parser.add_argument('--method', type=str, default='ddim',
                        choices=['ddim', 'ddpm'], help='采样方法')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='采样步数')
    args = parser.parse_args()

    # 找到最新模型权重（优先 checkpoints/ 子目录，兼容根目录旧版本）
    pth_files = sorted(
        glob.glob(os.path.join(ROOT, "checkpoints", "damascus_diffusion_ep*.pth")) +
        glob.glob(os.path.join(ROOT, "damascus_diffusion_ep*.pth")),
        key=lambda f: int(os.path.basename(f).split('ep')[1].split('.')[0]))
    if not pth_files:
        print("未找到模型权重文件。请先运行 training/train.py")
        sys.exit(1)

    checkpoint = pth_files[-1]
    npz_path   = os.path.join(ROOT, "parsed_transitions.npz")

    print(f"模型权重: {checkpoint}")
    print(f"采样方法: {args.method}, 步数: {args.num_steps}")
    print(f"评估样本: {args.n_samples}")

    # 加载采样器
    sampler = DarkMatterSampler(checkpoint, npz_path)

    # 准备真实数据
    cond_phys, target_phys = load_ground_truth(npz_path, args.n_samples)
    print(f"真实数据已加载: {len(cond_phys)} 条散射事件")

    # 执行评估
    results = evaluate_single_step(
        sampler, cond_phys, target_phys,
        method=args.method, num_steps=args.num_steps
    )

    # 打印报告
    print_report(results)

    # 保存图表到 outputs/ 目录
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    save_plots(results, cond_phys, target_phys,
               os.path.join(ROOT, "outputs", "evaluation_results.png"))
