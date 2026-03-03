"""
scattering_rate.py — 暗物质在太阳内部的散射率和光学深度采样

物理：
  暗物质粒子在太阳内部与原子核发生自旋无关 (SI) 散射。
  散射率由局部核数密度、截面和相对速度决定：

    Γ(r) = Σ_i n_i(r) × σ_i × v_rel

  其中 σ_i 是与第 i 种核的散射截面，按照标准的相干增强：
    σ_i = σ_p × (A_i)² × (μ_i/μ_p)²

    μ_i = m_χ × m_i / (m_χ + m_i)   约化质量
    μ_p = m_χ × m_p / (m_χ + m_p)   DM-质子约化质量
    A_i: 原子质量数

  散射间隔由泊松过程决定：
    τ ~ Exp(1) → 光学深度
    ∫ Σ_i n_i σ_i v dl = τ  → 散射位置

单位：
  - σ_p     [cm²]      DM-质子散射截面（基准参数）
  - m_χ     [GeV/c²]   暗物质质量
  - n_i     [1/cm³]    核数密度
  - v       [km/s] → [cm/s] 需要 ×1e5
  - Γ       [1/s]      散射率
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from solar_model import (
    SolarModel, ELEMENT_NAMES, ELEMENT_A, ELEMENT_Z, AMU_G, R_SUN_KM
)

M_PROTON_GEV = 0.9383  # 质子质量 [GeV/c²]


class ScatteringPhysics:
    """
    计算暗物质粒子在太阳内部的散射率。

    用法：
        sun = SolarModel("data/model_agss09.dat")
        scat = ScatteringPhysics(sun, m_chi_GeV=1.0, sigma_p_cm2=1e-35)

        # 在 r=350000 km, v=600 km/s 处的散射率
        rate = scat.scattering_rate(r_km=3.5e5, v_km_s=600.0)

        # 平均自由程
        mfp = scat.mean_free_path(r_km=3.5e5, v_km_s=600.0)

        # 采样下一次散射的光学深度
        tau = scat.sample_optical_depth()
    """

    def __init__(self, solar_model: SolarModel,
                 m_chi_GeV: float = 1.0,
                 sigma_p_cm2: float = 1e-35):
        """
        参数：
            solar_model: 标准太阳模型实例
            m_chi_GeV:   暗物质粒子质量 [GeV/c²]
            sigma_p_cm2: DM-质子 SI 散射截面 [cm²]
        """
        self.sun         = solar_model
        self.m_chi       = m_chi_GeV
        self.sigma_p     = sigma_p_cm2

        # 预计算各元素的增强截面 σ_i = σ_p × A² × (μ_i/μ_p)²
        self._sigma_i = self._compute_element_cross_sections()

    def _compute_element_cross_sections(self) -> np.ndarray:
        """
        计算各元素的散射截面增强因子。

        σ_i = σ_p × A_i² × (μ_i / μ_p)²

        SI 散射的相干增强使得重核（如 Fe, O）贡献显著，
        即使丰度远小于 H。
        """
        m_chi = self.m_chi

        # 各元素核质量 [GeV]
        m_nuc = ELEMENT_A * M_PROTON_GEV  # 近似用 A × m_p

        # 约化质量
        mu_i = m_chi * m_nuc / (m_chi + m_nuc)     # DM-核
        mu_p = m_chi * M_PROTON_GEV / (m_chi + M_PROTON_GEV)  # DM-质子

        # 截面增强
        sigma_i = self.sigma_p * ELEMENT_A ** 2 * (mu_i / mu_p) ** 2

        return sigma_i  # shape (29,)

    def element_scattering_rate(self, r_km: np.ndarray,
                                 v_km_s: float) -> np.ndarray:
        """
        各元素的散射率 Γ_i(r) [1/s]。

        返回 shape (N_elements,) 如果 r_km 是标量，
        或 (N_radius, N_elements) 如果 r_km 是数组。
        """
        r_km = np.atleast_1d(np.asarray(r_km, dtype=np.float64))
        v_cm_s = v_km_s * 1e5  # km/s → cm/s

        rates = np.zeros((len(r_km), len(ELEMENT_NAMES)))
        for i, name in enumerate(ELEMENT_NAMES):
            n_i = self.sun.number_density(r_km, name)  # [1/cm³]
            rates[:, i] = n_i * self._sigma_i[i] * v_cm_s  # [1/s]

        return rates.squeeze()

    def scattering_rate(self, r_km, v_km_s: float) -> np.ndarray:
        """总散射率 Γ(r) = Σ_i Γ_i(r) [1/s]"""
        rates = self.element_scattering_rate(r_km, v_km_s)
        if rates.ndim == 1:
            return rates.sum()
        return rates.sum(axis=-1)

    def mean_free_path(self, r_km, v_km_s: float) -> np.ndarray:
        """平均自由程 λ = v / Γ [km]"""
        rate = self.scattering_rate(r_km, v_km_s)
        v_km_s = max(v_km_s, 1e-10)
        return v_km_s / np.maximum(rate, 1e-30)

    def dominant_target(self, r_km: float) -> str:
        """在给定半径处，散射率最高的靶核"""
        rates = self.element_scattering_rate(r_km, v_km_s=600.0)
        idx = np.argmax(rates)
        return ELEMENT_NAMES[idx]

    @staticmethod
    def sample_optical_depth(rng: np.random.Generator = None) -> float:
        """
        从指数分布采样光学深度 τ ~ Exp(1)。
        等价于 τ = -ln(U), U ~ Uniform(0, 1)。
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.exponential(1.0)


# ─────────────────────────────────────────────
# 命令行测试
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(ROOT, "data", "model_agss09.dat")

    sun  = SolarModel(model_path)
    scat = ScatteringPhysics(sun, m_chi_GeV=1.0, sigma_p_cm2=1e-35)

    print("\n散射物理参数:")
    print(f"  m_χ = {scat.m_chi} GeV")
    print(f"  σ_p = {scat.sigma_p:.1e} cm²")
    print(f"  σ_O16 / σ_p = {scat._sigma_i[ELEMENT_NAMES.index('O16')] / scat.sigma_p:.1f}")
    print(f"  σ_Fe  / σ_p = {scat._sigma_i[ELEMENT_NAMES.index('Fe')] / scat.sigma_p:.1f}")

    # 径向散射率剖面
    r_test = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9]) * R_SUN_KM
    v_test = 600.0  # km/s

    print(f"\n{'r/R_sun':>8} {'Γ [1/s]':>12} {'λ [km]':>14} {'主要靶核':>10}")
    print("-" * 50)
    for r in r_test:
        rate = scat.scattering_rate(r, v_test)
        mfp  = scat.mean_free_path(r, v_test)
        target = scat.dominant_target(r)
        print(f"{r/R_SUN_KM:8.2f} {rate:12.4e} {mfp:14.1f} {target:>10}")
