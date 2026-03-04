"""
trajectory_validator.py — 端到端轨迹验证

对比 DaMaSCUS-SUN MC 轨迹与扩散模型生成轨迹的统计性质，
验证扩散模型能否在全轨迹级别替代 MC 散射采样。

验证内容：
  1. 轨迹结局分布（逃逸 / 捕获 / 超时）
  2. 每条轨迹的散射次数分布
  3. 最大穿透深度（最小半径）分布
  4. 散射位置的径向分布
  5. 能量演化统计
  6. 轨迹持续时间分布
  7. 示例轨迹 r(散射次数) 对比

方法：
  - 从原始 MC 轨迹文件中解析完整轨迹统计信息
  - 提取每条 MC 轨迹的初始条件 (r, v_rad, v_tan, E)
  - 使用相同的初始条件运行扩散模型轨迹模拟器
  - 对比两组轨迹的宏观统计性质

用法：
    # 快速测试（20 条轨迹，dt=10s）
    python inference/trajectory_validator.py --n_traj 20

    # 完整验证（100 条轨迹）
    python inference/trajectory_validator.py --n_traj 100

    # 自定义参数
    python inference/trajectory_validator.py --n_traj 50 --max_scatter 500 --dt 10
"""

import os
import sys
import glob
import argparse
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sde_physics.solar_model import R_SUN_KM
from inference.trajectory_simulator import TrajectorySimulator, EV_PER_GEV_KM2S2


# ═══════════════════════════════════════════════════════════
# 第一部分：MC 轨迹解析
# ═══════════════════════════════════════════════════════════

COLUMNS = ['index', 't', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'E', 'r']


def parse_mc_trajectory(filepath: str, energy_tol: float = 1e-3) -> dict:
    """
    解析单条 MC 轨迹文件，提取轨迹统计信息。

    参数：
        filepath:   轨迹文件路径
        energy_tol: 判定散射事件的能量跳变阈值 [eV]

    返回：
        dict with keys:
          initial_state: [r, v_rad, v_tan, E] 初始状态
          n_scatter:     散射次数
          r_min:         最小穿透半径 [km]
          E_init, E_final: 初始和最终能量 [eV]
          duration:      总时间 [s]
          outcome:       'escaped' / 'captured' / 'unknown'
          scatter_radii: 各次散射的半径 [km]
          scatter_energies: 各次散射后的能量 [eV]
          trajectory_r:  轨迹中所有散射点的 r 序列
    """
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=COLUMNS)

    if len(df) < 2:
        return None

    # ── 初始状态（球坐标）──
    row0 = df.iloc[0]
    pos0 = np.array([row0.x, row0.y, row0.z])
    vel0 = np.array([row0.vx, row0.vy, row0.vz])
    r0 = row0.r
    v_rad0 = np.dot(pos0, vel0) / max(r0, 1.0)
    v_sq = np.sum(vel0 ** 2)
    v_tan0 = np.sqrt(max(0, v_sq - v_rad0 ** 2))
    E0 = row0.E

    # ── 识别散射事件（能量跳变）──
    dE = df['E'].diff().abs()
    scatter_mask = dE > energy_tol
    scatter_idx = scatter_mask[scatter_mask].index.tolist()
    n_scatter = len(scatter_idx)

    # 散射坐标
    scatter_radii = df.loc[scatter_idx, 'r'].values if scatter_idx else np.array([])
    scatter_energies = df.loc[scatter_idx, 'E'].values if scatter_idx else np.array([])

    # ── 最小半径 ──
    r_min = df['r'].min()

    # ── 最终状态 ──
    row_f = df.iloc[-1]
    r_final = row_f.r
    pos_f = np.array([row_f.x, row_f.y, row_f.z])
    vel_f = np.array([row_f.vx, row_f.vy, row_f.vz])
    v_rad_f = np.dot(pos_f, vel_f) / max(r_final, 1.0)
    E_final = row_f.E

    # ── 持续时间 ──
    duration = df['t'].iloc[-1] - df['t'].iloc[0]

    # ── 判定结局 ──
    # 逃逸：最终位置在太阳外且径向速度向外
    if r_final > 1.5 * R_SUN_KM and v_rad_f > 0:
        outcome = 'escaped'
    # 捕获的典型标志：散射次数多、始终在太阳内部、能量大幅降低
    elif n_scatter > 20 and r_final < R_SUN_KM and E_final < E0 * 0.5:
        outcome = 'captured'
    elif n_scatter > 50 and r_final < R_SUN_KM:
        outcome = 'captured'
    else:
        outcome = 'unknown'

    # ── 散射点 r 序列（用于轨迹绘图）──
    trajectory_r = np.array([r0] + list(scatter_radii))

    return {
        'initial_state':    [r0, v_rad0, v_tan0, E0],
        'n_scatter':        n_scatter,
        'r_min':            r_min,
        'E_init':           E0,
        'E_final':          E_final,
        'duration':         duration,
        'outcome':          outcome,
        'scatter_radii':    scatter_radii,
        'scatter_energies': scatter_energies,
        'trajectory_r':     trajectory_r,
    }


def parse_all_mc_trajectories(data_dir: str,
                               max_files: int = None,
                               max_mc_scatter: int = None,
                               min_mc_scatter: int = 2,
                               energy_tol: float = 1e-3) -> list:
    """
    解析目录下所有 MC 轨迹文件。

    参数：
        data_dir:         轨迹文件目录
        max_files:        最多解析的文件数（None = 全部）
        max_mc_scatter:   过滤掉散射次数超过此值的轨迹（None = 不过滤）
        min_mc_scatter:   过滤掉散射次数低于此值的轨迹（默认 2）
        energy_tol:       散射判定能量阈值 [eV]

    返回：list[dict]，每个 dict 是一条轨迹的统计信息
    """
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True))

    # 先打乱以确保多样性（否则同一 trajectory_N 的 task 聚在一起）
    rng = np.random.default_rng(42)
    rng.shuffle(files)

    results = []
    n_parsed = 0
    for fp in tqdm(files, desc="解析 MC 轨迹"):
        try:
            stats = parse_mc_trajectory(fp, energy_tol=energy_tol)
            if stats is None:
                continue
            # 按散射次数过滤
            if min_mc_scatter is not None and stats['n_scatter'] < min_mc_scatter:
                continue
            if max_mc_scatter is not None and stats['n_scatter'] > max_mc_scatter:
                continue
            results.append(stats)
            n_parsed += 1
            if max_files is not None and n_parsed >= max_files:
                break
        except Exception as e:
            print(f"  跳过 {os.path.basename(fp)}: {e}")

    return results


# ═══════════════════════════════════════════════════════════
# 第二部分：扩散模型轨迹模拟
# ═══════════════════════════════════════════════════════════

def run_diffusion_trajectories(simulator: TrajectorySimulator,
                                initial_conditions: list,
                                max_scatterings: int = 500,
                                max_time: float = 1e6) -> list:
    """
    使用扩散模型模拟器运行轨迹，初始条件取自 MC 数据。

    参数：
        simulator:          TrajectorySimulator 实例
        initial_conditions: list of [r, v_rad, v_tan, E]
        max_scatterings:    每条轨迹的最大散射次数
        max_time:           最大模拟时间 [s]

    返回：
        list[dict]，每个 dict 是一条轨迹的结果（与 MC 格式一致）
    """
    results = []
    rng = np.random.default_rng(2024)

    for i, ic in enumerate(tqdm(initial_conditions, desc="扩散模型模拟")):
        r_init, vr_init, vt_init, E_init = ic

        t_start = time.time()
        traj = simulator.simulate_single(
            r_init=r_init,
            v_rad_init=vr_init,
            v_tan_init=vt_init,
            E_init=E_init,
            max_scatterings=max_scatterings,
            max_time=max_time,
            rng=rng,
        )
        elapsed = time.time() - t_start

        # 提取与 MC 对齐的统计信息
        traj_pts = traj['trajectory']

        # 散射点的 r 和 E 序列
        scatter_radii = traj_pts[1:, 0] if len(traj_pts) > 1 else np.array([])
        scatter_energies = traj_pts[1:, 3] if len(traj_pts) > 1 else np.array([])

        result = {
            'initial_state':    list(ic),
            'n_scatter':        traj['n_scatter'],
            'r_min':            float(traj_pts[:, 0].min()),
            'E_init':           E_init,
            'E_final':          float(traj_pts[-1, 3]),
            'duration':         traj['total_time'],
            'outcome':          traj['outcome'],
            'scatter_radii':    scatter_radii,
            'scatter_energies': scatter_energies,
            'trajectory_r':     traj_pts[:, 0],
            'wall_time':        elapsed,
        }
        results.append(result)

        # 定期打印进度
        if (i + 1) % 5 == 0 or i == 0:
            n_esc = sum(1 for r in results if r['outcome'] == 'escaped')
            n_cap = sum(1 for r in results if r['outcome'] == 'captured')
            avg_time = np.mean([r['wall_time'] for r in results])
            print(f"  [{i+1}/{len(initial_conditions)}] "
                  f"逃逸={n_esc} 捕获={n_cap} "
                  f"平均耗时={avg_time:.1f}s/条轨迹")

    return results


# ═══════════════════════════════════════════════════════════
# 第三部分：统计对比与可视化
# ═══════════════════════════════════════════════════════════

def compute_comparison_stats(mc_results: list, diff_results: list) -> dict:
    """
    计算 MC 和扩散模型轨迹的统计对比指标。
    """
    def outcome_counts(results):
        outcomes = [r['outcome'] for r in results]
        return {
            'escaped':    outcomes.count('escaped'),
            'captured':   outcomes.count('captured'),
            'max_reached': outcomes.count('max_reached'),
            'unknown':    outcomes.count('unknown'),
            'total':      len(outcomes),
        }

    mc_outcomes = outcome_counts(mc_results)
    diff_outcomes = outcome_counts(diff_results)

    mc_n_scatter = [r['n_scatter'] for r in mc_results]
    diff_n_scatter = [r['n_scatter'] for r in diff_results]

    mc_r_min = [r['r_min'] for r in mc_results]
    diff_r_min = [r['r_min'] for r in diff_results]

    mc_duration = [r['duration'] for r in mc_results]
    diff_duration = [r['duration'] for r in diff_results]

    mc_dE = [r['E_final'] - r['E_init'] for r in mc_results]
    diff_dE = [r['E_final'] - r['E_init'] for r in diff_results]

    return {
        'mc_outcomes':    mc_outcomes,
        'diff_outcomes':  diff_outcomes,
        'mc_n_scatter':   np.array(mc_n_scatter),
        'diff_n_scatter': np.array(diff_n_scatter),
        'mc_r_min':       np.array(mc_r_min),
        'diff_r_min':     np.array(diff_r_min),
        'mc_duration':    np.array(mc_duration),
        'diff_duration':  np.array(diff_duration),
        'mc_dE':          np.array(mc_dE),
        'diff_dE':        np.array(diff_dE),
    }


def print_comparison_report(stats: dict, mc_results: list, diff_results: list):
    """打印端到端验证报告"""
    print("\n" + "=" * 70)
    print("         端到端轨迹验证报告")
    print("=" * 70)

    # ── 1. 结局分布 ──
    mc_o = stats['mc_outcomes']
    df_o = stats['diff_outcomes']
    print("\n1. 轨迹结局分布:")
    print(f"   {'结局':>12} {'MC':>8} {'MC%':>8} {'扩散':>8} {'扩散%':>8}")
    print("   " + "-" * 48)
    for key in ['escaped', 'captured', 'max_reached', 'unknown']:
        mc_n = mc_o.get(key, 0)
        df_n = df_o.get(key, 0)
        mc_pct = mc_n / max(mc_o['total'], 1) * 100
        df_pct = df_n / max(df_o['total'], 1) * 100
        print(f"   {key:>12} {mc_n:8d} {mc_pct:7.1f}% {df_n:8d} {df_pct:7.1f}%")

    # ── 2. 散射次数 ──
    print("\n2. 散射次数统计:")
    for label, arr in [("MC", stats['mc_n_scatter']), ("扩散", stats['diff_n_scatter'])]:
        if len(arr) > 0:
            print(f"   {label:>6}: 均值={arr.mean():.1f} 中位={np.median(arr):.1f} "
                  f"标准差={arr.std():.1f} 范围=[{arr.min()}, {arr.max()}]")

    # ── 3. 穿透深度 ──
    print("\n3. 最小穿透半径 [R_sun]:")
    for label, arr in [("MC", stats['mc_r_min']), ("扩散", stats['diff_r_min'])]:
        arr_rs = arr / R_SUN_KM
        if len(arr_rs) > 0:
            print(f"   {label:>6}: 均值={arr_rs.mean():.3f} 中位={np.median(arr_rs):.3f} "
                  f"最深={arr_rs.min():.4f}")

    # ── 4. 持续时间 ──
    print("\n4. 轨迹持续时间 [s]:")
    for label, arr in [("MC", stats['mc_duration']), ("扩散", stats['diff_duration'])]:
        if len(arr) > 0:
            print(f"   {label:>6}: 均值={arr.mean():.0f} 中位={np.median(arr):.0f} "
                  f"标准差={arr.std():.0f}")

    # ── 5. 能量变化 ──
    print("\n5. 总能量变化 ΔE [eV]:")
    for label, arr in [("MC", stats['mc_dE']), ("扩散", stats['diff_dE'])]:
        if len(arr) > 0:
            print(f"   {label:>6}: 均值={arr.mean():.1f} 中位={np.median(arr):.1f} "
                  f"标准差={arr.std():.1f}")

    # ── 6. 性能统计 ──
    wall_times = [r.get('wall_time', 0) for r in diff_results if 'wall_time' in r]
    if wall_times:
        print(f"\n6. 扩散模型模拟性能:")
        print(f"   平均每条轨迹: {np.mean(wall_times):.2f} s")
        print(f"   总计: {sum(wall_times):.1f} s ({len(wall_times)} 条轨迹)")


def save_comparison_plots(stats: dict, mc_results: list, diff_results: list,
                          save_path: str):
    """
    生成端到端验证的多面板对比图。

    布局 (3 × 3):
      [0,0] 结局分布          [0,1] 散射次数分布       [0,2] 穿透深度分布
      [1,0] 持续时间分布      [1,1] 能量变化分布       [1,2] 散射位置径向分布
      [2,0] 示例轨迹 (MC)    [2,1] 示例轨迹 (扩散)    [2,2] 散射能量演化
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        # 尝试使用中文字体
        rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("(matplotlib 未安装，跳过绘图)")
        return

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # ── [0,0] 结局分布柱状图 ──
    ax = axes[0, 0]
    categories = ['escaped', 'captured', 'max_reached', 'unknown']
    cat_labels = ['Escaped', 'Captured', 'Max Reached', 'Unknown']
    mc_counts = [stats['mc_outcomes'].get(c, 0) for c in categories]
    df_counts = [stats['diff_outcomes'].get(c, 0) for c in categories]
    # 转为百分比
    mc_total = max(sum(mc_counts), 1)
    df_total = max(sum(df_counts), 1)
    mc_pct = [c / mc_total * 100 for c in mc_counts]
    df_pct = [c / df_total * 100 for c in df_counts]

    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w / 2, mc_pct, w, label='MC', color='steelblue', alpha=0.8)
    ax.bar(x + w / 2, df_pct, w, label='Diffusion', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylabel('%')
    ax.set_title('Trajectory Outcomes')
    ax.legend()

    # ── [0,1] 散射次数分布 ──
    ax = axes[0, 1]
    _plot_hist_comparison(ax, stats['mc_n_scatter'], stats['diff_n_scatter'],
                          'MC', 'Diffusion',
                          xlabel='# Scatterings', title='Scattering Count Distribution')

    # ── [0,2] 穿透深度分布 ──
    ax = axes[0, 2]
    _plot_hist_comparison(ax, stats['mc_r_min'] / R_SUN_KM, stats['diff_r_min'] / R_SUN_KM,
                          'MC', 'Diffusion',
                          xlabel='$r_{min}$ / $R_\\odot$', title='Penetration Depth')

    # ── [1,0] 持续时间分布 ──
    ax = axes[1, 0]
    # 对持续时间取 log10 以便可视化
    mc_log_dur = np.log10(stats['mc_duration'].clip(min=1))
    df_log_dur = np.log10(stats['diff_duration'].clip(min=1))
    _plot_hist_comparison(ax, mc_log_dur, df_log_dur,
                          'MC', 'Diffusion',
                          xlabel='$\\log_{10}$ Duration [s]',
                          title='Trajectory Duration')

    # ── [1,1] 能量变化分布 ──
    ax = axes[1, 1]
    _plot_hist_comparison(ax, stats['mc_dE'], stats['diff_dE'],
                          'MC', 'Diffusion',
                          xlabel='$\\Delta E$ [eV]', title='Total Energy Change')

    # ── [1,2] 散射位置径向分布 ──
    ax = axes[1, 2]
    mc_all_r = np.concatenate([r['scatter_radii'] for r in mc_results
                                if len(r['scatter_radii']) > 0])
    diff_all_r = np.concatenate([r['scatter_radii'] for r in diff_results
                                  if len(r['scatter_radii']) > 0])
    if len(mc_all_r) > 0 and len(diff_all_r) > 0:
        _plot_hist_comparison(ax, mc_all_r / R_SUN_KM, diff_all_r / R_SUN_KM,
                              'MC', 'Diffusion',
                              xlabel='$r_{scatter}$ / $R_\\odot$',
                              title='Scattering Location Distribution')
    else:
        ax.text(0.5, 0.5, 'No scattering data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Scattering Location Distribution')

    # ── [2,0] 示例轨迹 (MC) ──
    ax = axes[2, 0]
    _plot_example_trajectories(ax, mc_results, 'MC Trajectories (r vs scatter #)',
                                max_traj=8, color='steelblue')

    # ── [2,1] 示例轨迹 (扩散) ──
    ax = axes[2, 1]
    _plot_example_trajectories(ax, diff_results, 'Diffusion Trajectories (r vs scatter #)',
                                max_traj=8, color='coral')

    # ── [2,2] 散射能量演化 ──
    ax = axes[2, 2]
    _plot_energy_evolution(ax, mc_results, diff_results,
                           title='Energy at Scattering Events', max_traj=5)

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n验证对比图已保存 → {save_path}")
    plt.close()


def _plot_hist_comparison(ax, data_mc, data_diff, label_mc, label_diff,
                          xlabel='', title='', n_bins=40):
    """辅助函数：双直方图对比"""
    if len(data_mc) == 0 and len(data_diff) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title)
        return

    all_data = np.concatenate([data_mc, data_diff])
    lo, hi = np.percentile(all_data, [1, 99])
    bins = np.linspace(lo, hi, n_bins)

    if len(data_mc) > 0:
        ax.hist(data_mc, bins=bins, density=True, alpha=0.6,
                label=label_mc, color='steelblue')
    if len(data_diff) > 0:
        ax.hist(data_diff, bins=bins, density=True, alpha=0.6,
                label=label_diff, color='coral')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(fontsize=8)


def _plot_example_trajectories(ax, results, title, max_traj=8, color='steelblue'):
    """辅助函数：绘制示例轨迹 r(散射次序)"""
    # 选择散射次数 > 0 的轨迹
    valid = [r for r in results if r['n_scatter'] > 0]
    if not valid:
        ax.text(0.5, 0.5, 'No scattering trajectories', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title)
        return

    # 排序并取散射最多的几条（更有展示价值）
    valid = sorted(valid, key=lambda x: x['n_scatter'], reverse=True)[:max_traj]

    for traj in valid:
        r_seq = traj['trajectory_r'] / R_SUN_KM
        ax.plot(range(len(r_seq)), r_seq, alpha=0.7, linewidth=0.8, color=color)

    ax.axhline(y=1.0, color='gold', linestyle='--', linewidth=1, label='$R_\\odot$')
    ax.set_xlabel('Scattering #')
    ax.set_ylabel('$r$ / $R_\\odot$')
    ax.set_title(title)
    ax.legend(fontsize=7)


def _plot_energy_evolution(ax, mc_results, diff_results, title, max_traj=5):
    """辅助函数：绘制能量随散射的演化"""
    def _draw(results, color, label_prefix, ls='-'):
        valid = [r for r in results if len(r['scatter_energies']) > 0]
        valid = sorted(valid, key=lambda x: x['n_scatter'], reverse=True)[:max_traj]
        for j, traj in enumerate(valid):
            E_seq = np.concatenate([[traj['E_init']], traj['scatter_energies']])
            lbl = f'{label_prefix}' if j == 0 else None
            ax.plot(range(len(E_seq)), E_seq, alpha=0.6, linewidth=0.8,
                    color=color, linestyle=ls, label=lbl)

    _draw(mc_results, 'steelblue', 'MC')
    _draw(diff_results, 'coral', 'Diffusion', ls='--')
    ax.set_xlabel('Scattering #')
    ax.set_ylabel('E [eV]')
    ax.set_title(title)
    ax.legend(fontsize=8)


# ═══════════════════════════════════════════════════════════
# 第四部分：主入口
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="端到端轨迹验证：MC vs 扩散模型")
    parser.add_argument('--n_traj', type=int, default=30,
                        help='验证轨迹数（从 MC 数据中选取，默认 30）')
    parser.add_argument('--max_scatter', type=int, default=500,
                        help='扩散模拟每条轨迹的最大散射次数（默认 500）')
    parser.add_argument('--max_time', type=float, default=1e6,
                        help='扩散模拟每条轨迹的最大时间 [s]（默认 1e6）')
    parser.add_argument('--dt', type=float, default=10.0,
                        help='自由传播时间步长 [s]（默认 10）')
    parser.add_argument('--max_mc_scatter', type=int, default=1000,
                        help='MC 轨迹散射次数上限过滤（默认 1000，过滤超长轨迹以提高验证效率）')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='MC 轨迹数据目录（默认自动搜索 data/results_*/）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型权重路径（默认使用最新的）')
    parser.add_argument('--output', type=str, default='outputs/trajectory_validation.png',
                        help='输出图片路径（相对于项目根目录，默认 outputs/trajectory_validation.png）')
    args = parser.parse_args()

    # ── 0. 定位文件 ──
    npz_path = os.path.join(ROOT, "parsed_transitions.npz")
    solar_path = os.path.join(ROOT, "data", "model_agss09.dat")

    # 自动搜索 MC 轨迹目录
    if args.data_dir is None:
        result_dirs = sorted(glob.glob(os.path.join(ROOT, "data", "results_*")))
        if not result_dirs:
            print("错误：未找到 MC 轨迹数据目录 data/results_*/")
            sys.exit(1)
        data_dir = result_dirs[0]
    else:
        data_dir = args.data_dir

    # 自动搜索最新权重（优先 checkpoints/ 子目录，兼容根目录旧版本）
    if args.checkpoint is None:
        pth_files = sorted(
            glob.glob(os.path.join(ROOT, "checkpoints", "damascus_diffusion_ep*.pth")) +
            glob.glob(os.path.join(ROOT, "damascus_diffusion_ep*.pth")),
            key=lambda f: int(os.path.basename(f).split('ep')[1].split('.')[0]))
        if not pth_files:
            print("错误：未找到模型权重文件 checkpoints/damascus_diffusion_ep*.pth")
            sys.exit(1)
        checkpoint = pth_files[-1]
    else:
        checkpoint = args.checkpoint

    # 检查必要文件
    for path, desc in [(npz_path, "parsed_transitions.npz"),
                       (solar_path, "model_agss09.dat"),
                       (checkpoint, "模型权重")]:
        if not os.path.exists(path):
            print(f"错误：未找到 {desc}: {path}")
            sys.exit(1)

    print(f"MC 轨迹目录:   {data_dir}")
    print(f"模型权重:       {checkpoint}")
    print(f"验证轨迹数:     {args.n_traj}")
    print(f"最大散射次数:   {args.max_scatter}")
    print(f"dt_step:        {args.dt} s")

    # ── 1. 解析 MC 轨迹 ──
    print("\n" + "─" * 50)
    print("第 1 步：解析 MC 轨迹数据")
    print("─" * 50)
    mc_results = parse_all_mc_trajectories(
        data_dir, max_files=args.n_traj,
        max_mc_scatter=args.max_mc_scatter, min_mc_scatter=2
    )
    print(f"成功解析 {len(mc_results)} 条 MC 轨迹")

    if len(mc_results) == 0:
        print("错误：没有解析到有效的 MC 轨迹数据。")
        sys.exit(1)

    # 提取 MC 初始条件
    initial_conditions = [r['initial_state'] for r in mc_results]

    # ── 2. 初始化扩散模型模拟器 ──
    print("\n" + "─" * 50)
    print("第 2 步：初始化扩散模型轨迹模拟器")
    print("─" * 50)

    # 从目录名解析物理参数（如 results_0.000000_-35.000000 → m_chi=1 GeV, sigma=1e-35）
    dir_name = os.path.basename(data_dir)
    try:
        parts = dir_name.replace("results_", "").split("_")
        log_m = float(parts[0])
        log_sigma = float(parts[1])
        m_chi_GeV = 10 ** log_m
        sigma_p_cm2 = 10 ** log_sigma
    except (ValueError, IndexError):
        m_chi_GeV = 1.0
        sigma_p_cm2 = 1e-35
        print(f"  无法从目录名解析参数，使用默认值: m_χ={m_chi_GeV} GeV, σ_p={sigma_p_cm2:.1e} cm²")

    print(f"  物理参数: m_χ = {m_chi_GeV} GeV, σ_p = {sigma_p_cm2:.1e} cm²")

    simulator = TrajectorySimulator(
        model_checkpoint=checkpoint,
        npz_path=npz_path,
        solar_model_path=solar_path,
        m_chi_GeV=m_chi_GeV,
        sigma_p_cm2=sigma_p_cm2,
        dt_step=args.dt,
    )

    # ── 3. 运行扩散模型轨迹 ──
    print("\n" + "─" * 50)
    print(f"第 3 步：运行扩散模型轨迹模拟 ({len(initial_conditions)} 条)")
    print("─" * 50)
    t0 = time.time()
    diff_results = run_diffusion_trajectories(
        simulator, initial_conditions,
        max_scatterings=args.max_scatter,
        max_time=args.max_time,
    )
    total_sim_time = time.time() - t0
    print(f"扩散模拟完成，总耗时: {total_sim_time:.1f} s")

    # ── 4. 统计对比 ──
    print("\n" + "─" * 50)
    print("第 4 步：统计对比与可视化")
    print("─" * 50)
    comp_stats = compute_comparison_stats(mc_results, diff_results)
    print_comparison_report(comp_stats, mc_results, diff_results)

    # ── 5. 保存图表到 outputs/ ──
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    output_path = os.path.join(ROOT, args.output)
    save_comparison_plots(comp_stats, mc_results, diff_results, output_path)

    # ── 6. 保存数值结果到 outputs/ ──
    results_npz = os.path.join(ROOT, "outputs", "trajectory_validation_results.npz")
    np.savez_compressed(
        results_npz,
        mc_n_scatter=comp_stats['mc_n_scatter'],
        diff_n_scatter=comp_stats['diff_n_scatter'],
        mc_r_min=comp_stats['mc_r_min'],
        diff_r_min=comp_stats['diff_r_min'],
        mc_duration=comp_stats['mc_duration'],
        diff_duration=comp_stats['diff_duration'],
        mc_dE=comp_stats['mc_dE'],
        diff_dE=comp_stats['diff_dE'],
    )
    print(f"数值结果已保存 → {results_npz}")

    print("\n" + "=" * 70)
    print("  验证完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
