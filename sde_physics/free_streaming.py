"""
free_streaming.py — 暗物质粒子在太阳引力场中的自由传播

物理：
  在两次散射之间，暗物质粒子在太阳的引力势 Φ(r) 中沿确定性轨道运动。
  球对称情况下，角动量 L = r × v_tan 守恒，轨道由径向方程描述：

    dr/dt    = v_r
    dv_r/dt  = -dΦ/dr + L² / r³

  等价地，总比能量也守恒：
    E_specific = ½(v_r² + v_tan²) + Φ(r)  = const

  我们使用 RK45 积分器 (scipy.integrate.solve_ivp) 求解径向轨道。

单位：
  - r      [km]
  - v      [km/s]
  - t      [s]
  - Φ      [km²/s²]
"""

import numpy as np
from scipy.integrate import solve_ivp

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from solar_model import SolarModel, GM_SUN, R_SUN_KM


class FreeStreamer:
    """
    在太阳引力场中传播暗物质粒子。

    用法：
        sun = SolarModel("data/model_agss09.dat")
        streamer = FreeStreamer(sun)

        # 从状态 [r, v_rad, v_tan] 自由传播 Δt 秒
        r_new, vr_new, vt_new = streamer.propagate(
            r=3.5e5, v_rad=-600.0, v_tan=400.0, dt=100.0
        )

        # 或者传播到下一次散射位置（给定光学深度）
        result = streamer.propagate_to_scatter(
            r=3.5e5, v_rad=-600.0, v_tan=400.0,
            optical_depth=1.0, sigma_cm2=1e-35, m_chi_GeV=1.0
        )
    """

    def __init__(self, solar_model: SolarModel):
        self.sun = solar_model

    def _grav_accel(self, r_km: float) -> float:
        """
        径向引力加速度 g(r) = -GM(r)/r² [km/s²]

        对太阳外部 (r > R_sun)，用点质量近似。
        """
        if r_km < self.sun.r_km[0]:
            r_km = self.sun.r_km[0]

        if r_km > R_SUN_KM:
            M_r = self.sun.enclosed_mass[-1]  # M_sun
        else:
            M_r = float(self.sun.enclosed_mass_at(np.atleast_1d(r_km))[0])

        return -6.674e-20 * M_r / (r_km ** 2)  # G [km³/(kg·s²)]

    def _ode_rhs(self, t, y):
        """
        径向轨道的 ODE 右端项。

        y = [r, v_r]
        L = r * v_tan (角动量守恒) 作为常数存储在 self._L_current

        dr/dt   = v_r
        dv_r/dt = -GM(r)/r² + L²/r³
        """
        r, vr = y

        # 防止 r 跌到 0
        r = max(r, 1.0)

        g = self._grav_accel(r)  # 引力加速度（负值，朝向心）
        centrifugal = self._L_current ** 2 / r ** 3  # 离心力项

        dr_dt  = vr
        dvr_dt = g + centrifugal

        return [dr_dt, dvr_dt]

    def propagate(self, r: float, v_rad: float, v_tan: float,
                  dt: float) -> tuple:
        """
        自由传播固定时间 dt [s]。

        参数：
            r:     当前半径 [km]
            v_rad: 径向速度 [km/s]（负 = 朝中心）
            v_tan: 切向速度 [km/s]（≥ 0）
            dt:    传播时间 [s]

        返回：
            (r_new, v_rad_new, v_tan_new) [km, km/s, km/s]
        """
        # 角动量守恒
        L = r * abs(v_tan)
        self._L_current = L

        sol = solve_ivp(
            self._ode_rhs,
            t_span=[0, dt],
            y0=[r, v_rad],
            method='RK45',
            rtol=1e-9,
            atol=1e-9,
            max_step=dt / 10,
        )

        r_new  = max(sol.y[0, -1], 1.0)
        vr_new = sol.y[1, -1]
        vt_new = L / r_new  # v_tan = L / r

        return r_new, vr_new, vt_new

    def propagate_to_scatter(self, r: float, v_rad: float, v_tan: float,
                             optical_depth: float,
                             sigma_cm2: float, m_chi_GeV: float,
                             max_time: float = 1e6,
                             dt_step: float = 1.0) -> dict:
        """
        传播粒子直到累积光学深度达到指定值（= 发生下一次散射）。

        光学深度积分：
            τ = ∫ n(r) × σ × |v| × dt

        当 τ 达到采样的 optical_depth（通常从指数分布采样 τ ~ Exp(1)），
        即为下一次散射位置。

        参数：
            r, v_rad, v_tan: 当前状态
            optical_depth: 需要累积的光学深度（通常 = -ln(U), U ~ Uniform(0,1)）
            sigma_cm2:      DM-核散射截面 [cm²]
            m_chi_GeV:      DM 质量 [GeV]（影响约化质量，此处简化为与最丰富核的散射）
            max_time:       最大传播时间 [s]
            dt_step:        时间步长 [s]

        返回：
            dict with keys: r, v_rad, v_tan, t_elapsed, tau_accumulated, escaped
        """
        L = r * abs(v_tan)
        self._L_current = L

        current_r  = r
        current_vr = v_rad
        tau = 0.0
        t_total = 0.0

        while tau < optical_depth and t_total < max_time:
            # 一步传播
            v_mag = np.sqrt(current_vr ** 2 + (L / max(current_r, 1.0)) ** 2)

            sol = solve_ivp(
                self._ode_rhs,
                t_span=[0, dt_step],
                y0=[current_r, current_vr],
                method='RK45',
                rtol=1e-8,
                atol=1e-8,
            )

            current_r  = max(sol.y[0, -1], 1.0)
            current_vr = sol.y[1, -1]
            t_total += dt_step

            # 检查是否逃逸（r > 2 R_sun 且径向速度向外）
            if current_r > 2 * R_SUN_KM and current_vr > 0:
                return {
                    'r': current_r,
                    'v_rad': current_vr,
                    'v_tan': L / current_r,
                    't_elapsed': t_total,
                    'tau_accumulated': tau,
                    'escaped': True
                }

            # 如果在太阳内部，累积光学深度
            if current_r < R_SUN_KM:
                n_total = float(self.sun.total_number_density(current_r))
                # τ 增量 = n × σ × v × dt
                dtau = n_total * sigma_cm2 * (v_mag * 1e5) * dt_step
                # v_mag [km/s] → [cm/s] = ×1e5
                tau += dtau

        current_vt = L / max(current_r, 1.0)

        return {
            'r': current_r,
            'v_rad': current_vr,
            'v_tan': current_vt,
            't_elapsed': t_total,
            'tau_accumulated': tau,
            'escaped': False
        }


# ─────────────────────────────────────────────
# 命令行测试
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(ROOT, "data", "model_agss09.dat")

    sun = SolarModel(model_path)
    streamer = FreeStreamer(sun)

    # 测试简单传播：从 r = 500000 km 向内
    r0, vr0, vt0 = 5e5, -600.0, 400.0
    print(f"\n初始状态: r={r0:.0f} km, v_rad={vr0:.1f} km/s, v_tan={vt0:.1f} km/s")

    for dt in [10, 50, 100, 500]:
        r1, vr1, vt1 = streamer.propagate(r0, vr0, vt0, dt)
        E0 = 0.5 * (vr0 ** 2 + vt0 ** 2) + float(sun.grav_potential(np.array([r0]))[0])
        E1 = 0.5 * (vr1 ** 2 + vt1 ** 2) + float(sun.grav_potential(np.array([r1]))[0])
        print(f"  dt={dt:4d}s → r={r1:.0f} km, v_rad={vr1:.1f}, v_tan={vt1:.1f}  "
              f"|ΔE/E| = {abs(E1-E0)/abs(E0):.2e}")
