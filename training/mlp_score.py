"""
mlp_score.py — 基于 FiLM 条件注入的 Score Network

架构改进（相比简单拼接版本）：
  - FiLM (Feature-wise Linear Modulation)：条件信息独立编码后，
    通过预测每层的 scale (γ) 和 shift (β) 来调制全部隐藏层，
    而不仅仅拼接在输入端。这让条件信号在每一层都能直接影响特征流。
  - 时间嵌入同样参与 FiLM 调制，让时间信息渗透到所有层。
  - x_t 单独编码，保持噪声状态与条件信息的解耦。

参考：
  FiLM: Visual Reasoning with a General Conditioning Layer (Perez et al., 2018)
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# 1. 时间嵌入
# ─────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """正弦位置编码，将标量扩散时间 t 映射为高维向量"""
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, emb_dim)


# ─────────────────────────────────────────────
# 2. FiLM 残差块
# ─────────────────────────────────────────────

class FiLMResidualBlock(nn.Module):
    """
    带 FiLM 调制的残差全连接块。

    FiLM 公式：
        h_norm   = LayerNorm(h)
        γ, β     = Linear(cond_emb).chunk(2)    # 从外部条件预测 scale/shift
        h_film   = γ * h_norm + β               # 逐元素仿射变换
        h_out    = h + Linear(SiLU(h_film))      # 残差连接
    """
    def __init__(self, hidden_dim: int, film_cond_dim: int):
        """
        hidden_dim:    隐藏层维度
        film_cond_dim: FiLM 调制源的维度（时间嵌入 + 条件嵌入拼接后的维度）
        """
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.act    = nn.SiLU()

        # 从 (time_emb ‖ cond_emb) 预测 2 * hidden_dim 个 FiLM 参数
        self.film_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(film_cond_dim, 2 * hidden_dim),
        )

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        # FiLM 参数
        film = self.film_proj(cond_emb)                    # (B, 2*hidden_dim)
        gamma, beta = film.chunk(2, dim=-1)                # each (B, hidden_dim)

        # FiLM 调制 → 全连接 → 残差
        h = gamma * self.norm(x) + beta                    # (B, hidden_dim)
        h = self.linear(self.act(h))
        return x + h


# ─────────────────────────────────────────────
# 3. 条件 Score Network（FiLM 版）
# ─────────────────────────────────────────────

class ConditionalScoreNetwork(nn.Module):
    """
    条件 Score Network（FiLM 条件注入版）：

    输入流：
      x_t        → input_proj → 隐藏流 h
      time       → SinusoidalEmbedding → time_mlp → t_emb
      condition  → cond_encoder                   → c_emb
      cond_emb   = concat(t_emb, c_emb)           → FiLM 每层调制 h

    输出：
      predicted_noise (B, state_dim)
    """
    def __init__(self,
                 state_dim:    int = 4,
                 hidden_dim:   int = 256,
                 time_emb_dim: int = 128,
                 num_layers:   int = 6):
        super().__init__()

        # --- 时间编码 ---
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # --- 条件编码器（独立于时间，赋予 condition 充分的表达空间）---
        self.cond_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # FiLM 调制源维度 = t_emb + c_emb
        film_dim = time_emb_dim + hidden_dim

        # --- 噪声状态输入投影（仅编码 x_t，不混入条件） ---
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- FiLM 残差块堆栈 ---
        self.blocks = nn.ModuleList([
            FiLMResidualBlock(hidden_dim, film_dim) for _ in range(num_layers)
        ])

        # --- 输出投影 ---
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self,
                x_t:       torch.Tensor,
                time:      torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        x_t:       (B, state_dim) 加噪后的目标状态（碰撞后）
        time:      (B,)           扩散时间步 ∈ [0, 1]
        condition: (B, state_dim) 干净的条件状态（碰撞前）
        Returns:   (B, state_dim) 预测的噪声 ε
        """
        # 时间嵌入 & 条件嵌入
        t_emb = self.time_mlp(time)           # (B, time_emb_dim)
        c_emb = self.cond_encoder(condition)  # (B, hidden_dim)

        # 拼接为 FiLM 调制源
        cond_emb = torch.cat([t_emb, c_emb], dim=-1)  # (B, film_dim)

        # 噪声状态编码
        h = self.input_proj(x_t)              # (B, hidden_dim)

        # 由条件 + 时间联合调制每一个残差层
        for block in self.blocks:
            h = block(h, cond_emb)

        return self.output_proj(h)            # (B, state_dim)