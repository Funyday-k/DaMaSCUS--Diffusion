import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """正弦位置编码，将标量扩散时间 t 映射为高维向量"""
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t):
        # t: (batch_size,)
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, emb_dim)


class ResidualBlock(nn.Module):
    """带时间条件注入的残差全连接块"""
    def __init__(self, hidden_dim, time_emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 将时间嵌入投影到 hidden_dim 并通过缩放注入
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, hidden_dim),
        )

    def forward(self, x, t_emb):
        # 时间调制 + 残差连接
        return x + self.net(x + self.time_proj(t_emb))


class ConditionalScoreNetwork(nn.Module):
    """
    条件 Score Network：
    输入为噪声目标状态 x_t 与干净的条件状态 condition 的拼接，
    预测加在目标状态上的噪声 epsilon。
    """
    def __init__(self, state_dim=4, hidden_dim=256, time_emb_dim=128, num_layers=4):
        super().__init__()

        # 时间嵌入：标量 t -> 高维向量
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # 输入投影：拼接 x_t (state_dim) + condition (state_dim)
        self.input_proj = nn.Linear(state_dim * 2, hidden_dim)

        # 残差块堆栈
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_emb_dim) for _ in range(num_layers)
        ])

        # 输出投影：预测噪声，维度与目标状态相同
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x_t, time, condition):
        """
        x_t:       (B, state_dim) 加噪后的目标状态 (碰撞后)
        time:      (B,)           扩散时间步
        condition: (B, state_dim) 干净的条件状态 (碰撞前)
        Returns:   (B, state_dim) 预测的噪声
        """
        t_emb = self.time_mlp(time)  # (B, time_emb_dim)

        # 将噪声状态与物理条件在特征维度拼接
        x_input = torch.cat([x_t, condition], dim=-1)  # (B, state_dim*2)
        h = self.input_proj(x_input)  # (B, hidden_dim)

        for block in self.res_blocks:
            h = block(h, t_emb)

        return self.output_proj(h)  # (B, state_dim)