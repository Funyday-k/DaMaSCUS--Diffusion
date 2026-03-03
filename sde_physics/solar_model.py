"""
solar_model.py — AGSS09 标准太阳模型加载器与物理插值器

功能：
  - 解析 model_agss09.dat 文件
  - 提供根据半径 r 插值查询的太阳内部物理量：
    温度 T(r)、密度 ρ(r)、压强 P(r)、各元素丰度 X_i(r)
  - 计算引力势能 Φ(r) 和包裹质量 M(<r)
  - 提供暗物质-核散射所需的核数密度 n_i(r)

单位约定：
  - 半径          r     [km]
  - 温度          T     [K]
  - 密度          ρ     [g/cm³]
  - 压强          P     [dyn/cm²]
  - 引力势        Φ     [km²/s²] (比引力势，每单位质量)
  - 包裹质量      M(r)  [kg]
  - 核数密度      n_i   [1/cm³]

参考：
  Serenelli, Basu & Ferguson (2009), arXiv:0909.2668
"""

import os
import numpy as np
from scipy.interpolate import CubicSpline

# ─────────────────────────────────────────────
# 物理常数
# ─────────────────────────────────────────────
R_SUN_KM   = 6.957e5           # 太阳半径 [km]
M_SUN_KG   = 1.989e30          # 太阳质量 [kg]
G_KM       = 6.674e-20         # 引力常数 [km³/(kg·s²)]
GM_SUN     = G_KM * M_SUN_KG   # GM_sun [km³/s²] ≈ 1.327e11

# 各元素原子质量数（对应 AGSS09 的列 7-35）
ELEMENT_NAMES = [
    'H1', 'He4', 'He3', 'C12', 'C13', 'N14', 'N15',
    'O16', 'O17', 'O18', 'Ne', 'Na', 'Mg', 'Al', 'Si',
    'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti',
    'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
]

ELEMENT_A = np.array([
    1, 4, 3, 12, 13, 14, 15,
    16, 17, 18, 20, 23, 24, 27, 28,
    31, 32, 35, 40, 39, 40, 45, 48,
    51, 52, 55, 56, 59, 59,
], dtype=np.float64)

ELEMENT_Z = np.array([
    1, 2, 2, 6, 6, 7, 7,
    8, 8, 8, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28,
], dtype=np.float64)

AMU_G = 1.66054e-24  # 原子质量单位 [g]


class SolarModel:
    """
    AGSS09 标准太阳模型。

    用法示例：
        sun = SolarModel("data/model_agss09.dat")
        T    = sun.temperature(3.5e5)          # r=350000 km 处的温度 [K]
        rho  = sun.density(3.5e5)              # 密度 [g/cm³]
        phi  = sun.grav_potential(3.5e5)       # 引力势 [km²/s²]
        n_H  = sun.number_density(3.5e5, 'H1') # 氢核数密度 [1/cm³]
        v_esc = sun.escape_velocity(3.5e5)     # 局部逃逸速度 [km/s]
    """

    def __init__(self, model_file: str):
        self._load(model_file)
        self._build_interpolators()
        self._compute_gravitational_potential()

    # ─────────────────────────────────────────
    # 数据加载
    # ─────────────────────────────────────────

    def _load(self, path: str):
        """解析 AGSS09 .dat 文件"""
        rows = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or not line[0].isdigit():
                    continue
                vals = line.split()
                if len(vals) >= 35:
                    rows.append([float(v) for v in vals[:35]])

        data = np.array(rows)
        # 列含义
        self.mass_frac = data[:, 0]               # M/M_sun
        self.r_frac    = data[:, 1]               # r/R_sun
        self.r_km      = self.r_frac * R_SUN_KM   # r [km]
        self.temp      = data[:, 2]               # T [K]
        self.rho       = data[:, 3]               # ρ [g/cm³]
        self.pres      = data[:, 4]               # P [dyn/cm²]
        self.lumi_frac = data[:, 5]               # L/L_sun

        # 元素质量丰度 X_i（29种元素）
        self.abundances = data[:, 6:35]            # shape (N, 29)

        # 包裹质量 [kg]
        self.enclosed_mass = self.mass_frac * M_SUN_KG

        self._n_shells = len(self.r_km)
        print(f"太阳模型已加载: {self._n_shells} 个径向壳层, "
              f"r = [{self.r_km[0]:.0f}, {self.r_km[-1]:.0f}] km  "
              f"(r/R_sun = [{self.r_frac[0]:.5f}, {self.r_frac[-1]:.5f}])")

    # ─────────────────────────────────────────
    # 构建三次样条插值器
    # ─────────────────────────────────────────

    def _build_interpolators(self):
        """构建所有物理量的三次样条插值"""
        r = self.r_km

        # 核心热力学量
        self._cs_temp  = CubicSpline(r, np.log(self.temp), extrapolate=True)
        self._cs_rho   = CubicSpline(r, np.log(self.rho),  extrapolate=True)
        self._cs_pres  = CubicSpline(r, np.log(self.pres), extrapolate=True)

        # 包裹质量 M(r) — 用于引力势
        self._cs_mass  = CubicSpline(r, self.enclosed_mass, extrapolate=True)

        # 各元素丰度
        self._cs_abund = []
        for i in range(self.abundances.shape[1]):
            xi = self.abundances[:, i]
            # 丰度可能为零，取 log 不安全，用线性插值
            self._cs_abund.append(CubicSpline(r, xi, extrapolate=True))

    # ─────────────────────────────────────────
    # 引力势计算
    # ─────────────────────────────────────────

    def _compute_gravitational_potential(self):
        """
        计算比引力势 Φ(r) [km²/s²]（每单位质量）。

          Φ(r) = - ∫_r^∞ GM(r')/r'² dr'

        对 r < R_sun，使用太阳模型的 M(r)；
        对 r > R_sun，M(r) = M_sun（点质量近似）。
        """
        # 在表格径向网格上计算
        r = self.r_km
        M = self.enclosed_mass

        # 从外向内数值积分：Φ(r_max) = -GM_sun/r_max
        phi = np.zeros_like(r)
        phi[-1] = -GM_SUN / r[-1]

        for i in range(len(r) - 2, -1, -1):
            dr = r[i + 1] - r[i]
            # 梯形积分 GM(r)/r²
            integrand_i   = G_KM * M[i]   / r[i] ** 2
            integrand_ip1 = G_KM * M[i+1] / r[i+1] ** 2
            phi[i] = phi[i + 1] - 0.5 * (integrand_i + integrand_ip1) * dr

        self._phi_grid = phi
        self._cs_phi = CubicSpline(r, phi, extrapolate=True)

    # ─────────────────────────────────────────
    # 公开查询接口
    # ─────────────────────────────────────────

    def temperature(self, r_km: np.ndarray) -> np.ndarray:
        """温度 T(r) [K]"""
        r_km = np.atleast_1d(np.asarray(r_km, dtype=np.float64))
        return np.exp(self._cs_temp(np.clip(r_km, self.r_km[0], self.r_km[-1])))

    def density(self, r_km: np.ndarray) -> np.ndarray:
        """密度 ρ(r) [g/cm³]"""
        r_km = np.atleast_1d(np.asarray(r_km, dtype=np.float64))
        return np.exp(self._cs_rho(np.clip(r_km, self.r_km[0], self.r_km[-1])))

    def pressure(self, r_km: np.ndarray) -> np.ndarray:
        """压强 P(r) [dyn/cm²]"""
        r_km = np.atleast_1d(np.asarray(r_km, dtype=np.float64))
        return np.exp(self._cs_pres(np.clip(r_km, self.r_km[0], self.r_km[-1])))

    def enclosed_mass_at(self, r_km: np.ndarray) -> np.ndarray:
        """包裹质量 M(<r) [kg]"""
        r_km = np.atleast_1d(np.asarray(r_km, dtype=np.float64))
        result = np.where(
            r_km > self.r_km[-1],
            M_SUN_KG,
            self._cs_mass(np.clip(r_km, self.r_km[0], self.r_km[-1]))
        )
        return np.clip(result, 0, M_SUN_KG)

    def grav_potential(self, r_km: np.ndarray) -> np.ndarray:
        """比引力势 Φ(r) [km²/s²]（负值，越深越负）"""
        r_km = np.atleast_1d(np.asarray(r_km, dtype=np.float64))
        # 太阳外部：点质量近似
        inside = r_km <= self.r_km[-1]
        phi = np.where(inside,
                       self._cs_phi(np.clip(r_km, self.r_km[0], self.r_km[-1])),
                       -GM_SUN / r_km)
        return phi

    def escape_velocity(self, r_km: np.ndarray) -> np.ndarray:
        """局部逃逸速度 v_esc(r) [km/s]"""
        phi = self.grav_potential(r_km)
        return np.sqrt(-2.0 * phi)

    def element_abundance(self, r_km: np.ndarray, element: str) -> np.ndarray:
        """
        单个元素的质量丰度 X_i(r)（无量纲）。
        element: 'H1', 'He4', 'C12', 'O16', 'Fe' 等 (见 ELEMENT_NAMES)
        """
        idx = ELEMENT_NAMES.index(element)
        r_km = np.atleast_1d(np.asarray(r_km, dtype=np.float64))
        val = self._cs_abund[idx](np.clip(r_km, self.r_km[0], self.r_km[-1]))
        return np.clip(val, 0, 1)

    def number_density(self, r_km: np.ndarray, element: str) -> np.ndarray:
        """
        核数密度 n_i(r) [1/cm³]。

        n_i = ρ × X_i / (A_i × m_u)

        这决定了暗物质散射率：Γ = n_i × σ × v
        """
        idx = ELEMENT_NAMES.index(element)
        rho = self.density(r_km)
        Xi  = self.element_abundance(r_km, element)
        Ai  = ELEMENT_A[idx]
        return rho * Xi / (Ai * AMU_G)

    def total_number_density(self, r_km: np.ndarray) -> np.ndarray:
        """所有核素的总数密度 n_total(r) [1/cm³]"""
        r_km = np.atleast_1d(np.asarray(r_km, dtype=np.float64))
        n_total = np.zeros_like(r_km, dtype=np.float64)
        for i, name in enumerate(ELEMENT_NAMES):
            n_total += self.number_density(r_km, name)
        return n_total

    def mean_molecular_weight(self, r_km: np.ndarray) -> np.ndarray:
        """平均核质量 <A> (以 AMU 为单位) — 用于散射运动学"""
        r_km = np.atleast_1d(np.asarray(r_km, dtype=np.float64))
        rho = self.density(r_km)
        n_total = self.total_number_density(r_km)
        # ρ = n_total × <A> × m_u  →  <A> = ρ / (n_total × m_u)
        return rho / (n_total * AMU_G)

    # ─────────────────────────────────────────
    # 汇总信息
    # ─────────────────────────────────────────

    def summary(self):
        """打印太阳模型关键参数"""
        r_test = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) * R_SUN_KM
        print(f"\n{'r/R_sun':>8} {'T [K]':>12} {'ρ [g/cm³]':>12} {'Φ [km²/s²]':>14} "
              f"{'v_esc [km/s]':>14} {'n_H [1/cm³]':>14}")
        print("-" * 80)
        for r in r_test:
            T = self.temperature(r)[0]
            rho = self.density(r)[0]
            phi = self.grav_potential(r)[0]
            vesc = self.escape_velocity(r)[0]
            nH = self.number_density(r, 'H1')[0]
            print(f"{r/R_SUN_KM:8.2f} {T:12.3e} {rho:12.3e} {phi:14.1f} "
                  f"{vesc:14.1f} {nH:14.3e}")


# ─────────────────────────────────────────────
# 命令行测试
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(ROOT, "data", "model_agss09.dat")

    sun = SolarModel(model_path)
    sun.summary()

    # 绘制径向剖面（如果有 matplotlib）
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        r_plot = np.linspace(sun.r_km[1], sun.r_km[-1], 500)
        r_frac = r_plot / R_SUN_KM

        axes[0, 0].semilogy(r_frac, sun.temperature(r_plot))
        axes[0, 0].set_ylabel("T [K]")
        axes[0, 0].set_title("温度剖面")

        axes[0, 1].semilogy(r_frac, sun.density(r_plot))
        axes[0, 1].set_ylabel("ρ [g/cm³]")
        axes[0, 1].set_title("密度剖面")

        axes[1, 0].plot(r_frac, sun.grav_potential(r_plot))
        axes[1, 0].set_ylabel("Φ [km²/s²]")
        axes[1, 0].set_title("引力势")

        axes[1, 1].plot(r_frac, sun.escape_velocity(r_plot))
        axes[1, 1].set_ylabel("v_esc [km/s]")
        axes[1, 1].set_title("逃逸速度")

        for ax in axes.flat:
            ax.set_xlabel("r / R☉")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(ROOT, "solar_model_profiles.png"), dpi=150)
        print("\n径向剖面图已保存 -> solar_model_profiles.png")
    except ImportError:
        print("\n(matplotlib 未安装，跳过绘图)")
