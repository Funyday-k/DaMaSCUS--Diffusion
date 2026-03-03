"""
trajectory_simulator.py — 混合物理/ML的暗物质轨迹模拟器

这是项目的核心成果：用扩散模型替代蒙特卡罗散射采样，
同时保留精确的物理（引力传播和散射率）。

工作流程：
  ┌──────────────────────────────────────────────────┐
  │  1. 初始化：DM 粒子从太阳表面进入                   │
  │  2. 自由传播：在引力势 Φ(r) 中积分轨道               │
  │     (sde_physics/free_streaming.py)                │
  │  3. 散射判定：累积光学深度 τ 达到阈值 → 发生散射       │
  │     (sde_physics/scattering_rate.py)               │
  │  4. 散射结果：扩散模型生成碰撞后状态                   │
  │     (inference/sampler.py)  ← ML 替代 MC            │
  │  5. 回到 2，重复直到粒子逃逸或被捕获                   │
  └──────────────────────────────────────────────────┘

对比原始 DaMaSCUS-SUN Monte Carlo：
  - 步骤 2, 3 完全相同（精确物理）
  - 步骤 4 用训练好的扩散模型替代蒙特卡罗散射运动学
  - 质量优势：一次训练后可超快速生成任意数量的轨迹

单位：
  - r     [km]
  - v     [km/s]
  - E     [eV]（DaMaSCUS 约定）
  - t     [s]
"""

import os
import sys
import numpy as np
import torch

# 项目根目录
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sde_physics"))
sys.path.insert(0, os.path.join(ROOT, "training"))

from sde_physics.solar_model import SolarModel, R_SUN_KM, GM_SUN
from sde_physics.free_streaming import FreeStreamer
from sde_physics.scattering_rate import ScatteringPhysics
from inference.sampler import DarkMatterSampler


class TrajectorySimulator:
    """
    混合物理/ML 暗物质轨迹模拟器。

    用法：
        sim = TrajectorySimulator(
            model_checkpoint="damascus_diffusion_ep300.pth",
            npz_path="parsed_transitions.npz",
            solar_model_path="data/model_agss09.dat",
            m_chi_GeV=1.0,
            sigma_p_cm2=1e-35,
        )

        # 模拟一条轨迹
        traj = sim.simulate_single(
            r_init=R_SUN_KM,     # 从太阳表面进入
            v_rad_init=-600.0,   # 向内
            v_tan_init=400.0,    # 切向分量
            E_init=1500.0,       # 初始能量 [eV]
        )

        # 批量模拟
        results = sim.simulate_batch(n_trajectories=1000)
    """

    def __init__(self,
                 model_checkpoint: str,
                 npz_path: str,
                 solar_model_path: str,
                 m_chi_GeV: float = 1.0,
                 sigma_p_cm2: float = 1e-35,
                 dt_step: float = 1.0):
        """
        参数：
            model_checkpoint:  扩散模型权重路径
            npz_path:          parsed_transitions.npz 路径
            solar_model_path:  AGSS09 太阳模型路径
            m_chi_GeV:         暗物质质量 [GeV/c²]
            sigma_p_cm2:       DM-质子散射截面 [cm²]
            dt_step:           自由传播时间步长 [s]
        """
        print("初始化轨迹模拟器...")

        # 1. 加载太阳模型
        self.sun = SolarModel(solar_model_path)

        # 2. 物理模块
        self.streamer = FreeStreamer(self.sun)
        self.scattering = ScatteringPhysics(self.sun, m_chi_GeV, sigma_p_cm2)

        # 3. 扩散模型采样器
        self.sampler = DarkMatterSampler(model_checkpoint, npz_path)

        # 参数
        self.m_chi      = m_chi_GeV
        self.sigma_p    = sigma_p_cm2
        self.dt_step    = dt_step

        print(f"模拟器就绪: m_χ={m_chi_GeV} GeV, σ_p={sigma_p_cm2:.1e} cm²")

    def simulate_single(self,
                        r_init: float,
                        v_rad_init: float,
                        v_tan_init: float,
                        E_init: float,
                        max_scatterings: int = 5000,
                        max_time: float = 1e7,
                        rng: np.random.Generator = None) -> dict:
        """
        模拟单条暗物质粒子轨迹。

        参数：
            r_init, v_rad_init, v_tan_init, E_init: 初始状态
            max_scatterings: 最大散射次数
            max_time:        最大模拟时间 [s]
            rng:             随机数生成器

        返回：
            dict with keys:
                trajectory:  list of [r, v_rad, v_tan, E, t]
                n_scatter:   散射次数
                outcome:     'captured' | 'escaped' | 'max_reached'
                total_time:  总时间 [s]
        """
        if rng is None:
            rng = np.random.default_rng()

        trajectory = []
        r, vr, vt, E = r_init, v_rad_init, v_tan_init, E_init
        t_total = 0.0

        trajectory.append([r, vr, vt, E, 0.0])

        for i_scat in range(max_scatterings):
            # ─── 步骤 1: 采样光学深度 ───
            tau_target = self.scattering.sample_optical_depth(rng)

            # ─── 步骤 2: 自由传播直到达到光学深度 ───
            tau_accum = 0.0
            free_steps = 0

            while tau_accum < tau_target and t_total < max_time:
                # 当前速度
                v_mag = np.sqrt(vr ** 2 + vt ** 2)

                # 一步传播
                r_new, vr_new, vt_new = self.streamer.propagate(
                    r, vr, vt, self.dt_step
                )
                t_total += self.dt_step
                free_steps += 1

                # 检查逃逸（r > 2R_sun 且向外运动）
                if r_new > 2 * R_SUN_KM and vr_new > 0:
                    trajectory.append([r_new, vr_new, vt_new, E, t_total])
                    return {
                        'trajectory': np.array(trajectory),
                        'n_scatter':  i_scat,
                        'outcome':    'escaped',
                        'total_time': t_total,
                    }

                # 如果在太阳内部，累积光学深度
                if r_new < R_SUN_KM:
                    v_cm_s = v_mag * 1e5
                    n_total = float(self.sun.total_number_density(r_new))
                    dtau = n_total * self.sigma_p * v_cm_s * self.dt_step
                    tau_accum += dtau

                r, vr, vt = r_new, vr_new, vt_new

            if t_total >= max_time:
                trajectory.append([r, vr, vt, E, t_total])
                return {
                    'trajectory': np.array(trajectory),
                    'n_scatter':  i_scat,
                    'outcome':    'max_reached',
                    'total_time': t_total,
                }

            # ─── 步骤 3: 散射！用扩散模型生成碰撞后状态 ───
            condition = torch.tensor(
                [[r, vr, vt, E]], dtype=torch.float32
            )
            result = self.sampler.sample(condition, method="ddim", num_steps=50)
            result_np = result.cpu().numpy()[0]

            # 更新状态
            r  = max(float(result_np[0]), 1.0)  # r ≥ 0
            vr = float(result_np[1])
            vt = max(float(result_np[2]), 0.0)  # v_tan ≥ 0
            E  = float(result_np[3])

            trajectory.append([r, vr, vt, E, t_total])

            # 检查捕获（能量足够低→粒子被束缚）
            v_esc_local = float(self.sun.escape_velocity(r))
            v_total = np.sqrt(vr ** 2 + vt ** 2)
            if v_total < 0.01 * v_esc_local:
                return {
                    'trajectory': np.array(trajectory),
                    'n_scatter':  i_scat + 1,
                    'outcome':    'captured',
                    'total_time': t_total,
                }

        trajectory.append([r, vr, vt, E, t_total])
        return {
            'trajectory': np.array(trajectory),
            'n_scatter':  max_scatterings,
            'outcome':    'max_reached',
            'total_time': t_total,
        }

    def simulate_batch(self, n_trajectories: int = 100,
                       v_halo: float = 220.0,
                       v_esc_halo: float = 544.0,
                       seed: int = 42) -> list:
        """
        批量模拟暗物质轨迹。

        初始条件：
            - 粒子从太阳表面（r = R_sun）进入
            - 速度从 Maxwell-Boltzmann 分布采样（截断于银河逃逸速度）
            - 随机入射方向

        参数：
            n_trajectories: 模拟轨迹数
            v_halo:         银河晕中的 DM 速度弥散 [km/s]
            v_esc_halo:     银河逃逸速度 [km/s]
            seed:           随机种子

        返回：
            list[dict] — 每条轨迹的结果字典
        """
        rng = np.random.default_rng(seed)
        results = []

        for i in range(n_trajectories):
            # 从截断 Maxwell 分布采样入射速度
            while True:
                v = rng.rayleigh(v_halo)
                if v < v_esc_halo:
                    break

            # 随机入射角
            cos_theta = rng.uniform(-1, 0)  # 向内入射
            sin_theta = np.sqrt(1 - cos_theta ** 2)

            v_rad = v * cos_theta  # 负值（向内）
            v_tan = v * sin_theta

            # 太阳表面入射
            r_init = R_SUN_KM

            # 初始能量（DaMaSCUS 约定，由训练数据决定）
            # 这里用 ½v² 作为代理，实际单位换算由模型学习
            E_init = 0.5 * v ** 2

            traj = self.simulate_single(
                r_init=r_init,
                v_rad_init=v_rad,
                v_tan_init=v_tan,
                E_init=E_init,
                max_scatterings=500,
                max_time=1e6,
                rng=rng,
            )

            results.append(traj)

            if (i + 1) % 10 == 0 or i == 0:
                n_esc = sum(1 for r in results if r['outcome'] == 'escaped')
                n_cap = sum(1 for r in results if r['outcome'] == 'captured')
                print(f"轨迹 {i+1}/{n_trajectories}: "
                      f"逃逸={n_esc} 捕获={n_cap} 未完成={len(results)-n_esc-n_cap}")

        return results


# ─────────────────────────────────────────────
# 命令行测试
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import glob

    # 寻找最新模型权重
    pth_files = sorted(glob.glob(os.path.join(ROOT, "damascus_diffusion_ep*.pth")))
    if not pth_files:
        print("未找到模型权重文件。请先运行 training/train.py")
        sys.exit(1)

    checkpoint = pth_files[-1]
    npz_path   = os.path.join(ROOT, "parsed_transitions.npz")
    solar_path = os.path.join(ROOT, "data", "model_agss09.dat")

    sim = TrajectorySimulator(
        model_checkpoint=checkpoint,
        npz_path=npz_path,
        solar_model_path=solar_path,
        m_chi_GeV=1.0,
        sigma_p_cm2=1e-35,
    )

    # 模拟 5 条测试轨迹
    print("\n开始模拟测试轨迹...")
    results = sim.simulate_batch(n_trajectories=5, seed=2024)

    for i, res in enumerate(results):
        traj = res['trajectory']
        print(f"\n轨迹 {i+1}: {res['outcome']} | "
              f"散射 {res['n_scatter']} 次 | "
              f"时间 {res['total_time']:.1f}s | "
              f"轨迹点 {len(traj)}")
        if len(traj) > 1:
            print(f"  起始: r={traj[0,0]:.0f} km, E={traj[0,3]:.1f}")
            print(f"  终止: r={traj[-1,0]:.0f} km, E={traj[-1,3]:.1f}")
