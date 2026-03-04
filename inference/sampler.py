import torch
import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.mlp_score import ConditionalScoreNetwork
from data_pipeline.transform import DamascusDataset


# ─────────────────────────────────────────────
# 噪声调度（与训练时保持完全一致）
# ─────────────────────────────────────────────
def cosine_schedule(t: torch.Tensor):
    """余弦调度：返回 (sqrt_alpha_bar, sigma)"""
    t = t.clamp(min=1e-5)
    f_t = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    f_0 = math.cos(0.008 / 1.008 * math.pi / 2) ** 2
    alpha_bar = (f_t / f_0).clamp(min=1e-8)
    return torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar)


# ─────────────────────────────────────────────
# DDPM 采样器（随机逆扩散，与训练分布完全匹配）
# ─────────────────────────────────────────────
@torch.no_grad()
def ddpm_sample(model: ConditionalScoreNetwork,
                condition: torch.Tensor,
                num_steps: int = 200,
                device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    DDPM 逆扩散采样。
    condition: (B, 4)  归一化后的碰撞前状态 [r, v_rad, v_tan, E]
    返回:      (B, 4)  归一化后的碰撞后状态（采样结果）
    """
    model.eval()
    B, D = condition.shape

    # 从纯高斯噪声出发 x_T ~ N(0, I)
    x = torch.randn(B, D, device=device)

    # 离散化时间步 t: 1 → 0
    timesteps = torch.linspace(1.0, 1e-3, num_steps, device=device)

    for i, t_now in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i < num_steps - 1 else torch.tensor(0.0, device=device)

        t_batch = t_now.expand(B)

        # 预测噪声
        eps_pred = model(x, t_batch, condition)

        # 当前时间步的调度参数
        sqrt_ab_now, sigma_now = cosine_schedule(t_now.unsqueeze(0))
        # 前一时间步的调度参数
        sqrt_ab_prev, sigma_prev = cosine_schedule(t_prev.unsqueeze(0).clamp(min=1e-5))

        # 从 x_t 估算 x_0
        x0_pred = (x - sigma_now * eps_pred) / sqrt_ab_now
        x0_pred = x0_pred.clamp(-5, 5)  # 防止极端值

        # DDPM 后验均值
        mean = sqrt_ab_prev * x0_pred + sigma_prev * eps_pred

        # 最后一步不加噪声
        if i < num_steps - 1:
            noise_scale = torch.sqrt(
                (sigma_prev ** 2 - (sigma_now / sqrt_ab_now * sqrt_ab_prev) ** 2).clamp(min=0)
            )
            x = mean + noise_scale * torch.randn_like(x)
        else:
            x = mean

    return x


# ─────────────────────────────────────────────
# DDIM 采样器（确定性，速度更快，步数可大幅减少）
# ─────────────────────────────────────────────
@torch.no_grad()
def ddim_sample(model: ConditionalScoreNetwork,
                condition: torch.Tensor,
                num_steps: int = 50,
                eta: float = 0.0,
                device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    DDIM 采样（eta=0 时完全确定性，eta=1 时退化为 DDPM）。
    condition: (B, 4)  归一化后的碰撞前状态
    返回:      (B, 4)  归一化后的碰撞后状态
    """
    model.eval()
    B, D = condition.shape
    x = torch.randn(B, D, device=device)

    timesteps = torch.linspace(1.0, 1e-3, num_steps, device=device)

    for i, t_now in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i < num_steps - 1 else torch.tensor(0.0, device=device)

        t_batch = t_now.expand(B)
        eps_pred = model(x, t_batch, condition)

        sqrt_ab_now, sigma_now = cosine_schedule(t_now.unsqueeze(0))
        sqrt_ab_prev, sigma_prev = cosine_schedule(t_prev.unsqueeze(0).clamp(min=1e-5))

        # 预测 x_0
        x0_pred = (x - sigma_now * eps_pred) / sqrt_ab_now
        x0_pred = x0_pred.clamp(-5, 5)

        # DDIM 更新
        sigma_ddim = eta * torch.sqrt(
            (sigma_prev ** 2 / sigma_now ** 2) * (1 - sqrt_ab_now ** 2 / sqrt_ab_prev ** 2)
        ).clamp(min=0)
        direction = torch.sqrt((sigma_prev ** 2 - sigma_ddim ** 2).clamp(min=0)) * eps_pred

        x = sqrt_ab_prev * x0_pred + direction
        if eta > 0 and i < num_steps - 1:
            x = x + sigma_ddim * torch.randn_like(x)

    return x


# ─────────────────────────────────────────────
# 主推理类：整合模型加载、采样、反归一化
# ─────────────────────────────────────────────
class DarkMatterSampler:
    """
    端到端推理接口：
    输入碰撞前物理状态 (原始单位)，输出模型生成的碰撞后物理状态。
    """
    def __init__(self, checkpoint_path: str, npz_path: str,
                 state_dim=4, hidden_dim=256, time_emb_dim=128, num_layers=6):

        # 设备选择
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"推理设备: {self.device}")

        # 加载模型权重
        self.model = ConditionalScoreNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers
        ).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        # 兼容旧格式（纯 state_dict）和新格式（含 ema_state_dict 的 dict）
        if isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["ema_state_dict"])
            ep = ckpt.get("epoch", "?")
            print(f"模型已加载 (EMA 权重, epoch={ep}): {checkpoint_path}")
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"模型已加载 (训练权重): {checkpoint_path}")
        else:
            self.model.load_state_dict(ckpt)  # 旧版纯 state_dict
            print(f"模型已加载 (旧格式): {checkpoint_path}")
        self.model.eval()

        # 加载归一化统计量（从数据集中计算）
        # 若 npz 文件不存在，先自动重新解析原始数据
        if not os.path.exists(npz_path):
            print(f"未找到 {npz_path}，正在从原始数据重新解析...")
            from data_pipeline.parser import DamascusParser
            data_dir = os.path.join(os.path.dirname(npz_path), "data")
            DamascusParser(data_dir=data_dir, output_file=npz_path).run()
        dataset = DamascusDataset(npz_path, normalize=True)
        stats = dataset.get_stats()
        self.X_mean = stats['X_mean'].to(self.device)
        self.X_std  = stats['X_std'].to(self.device)
        self.Y_mean = stats['Y_mean'].to(self.device)
        self.Y_std  = stats['Y_std'].to(self.device)
        print(f"归一化参数已加载（来自 {npz_path}）")

    def _normalize_input(self, x_physical: torch.Tensor) -> torch.Tensor:
        """将物理单位输入归一化"""
        return (x_physical.to(self.device) - self.X_mean) / self.X_std

    def _denormalize_output(self, y_normalized: torch.Tensor) -> torch.Tensor:
        """将归一化输出还原为物理单位"""
        return y_normalized * self.Y_std + self.Y_mean

    def sample(self, condition_physical: torch.Tensor,
               method: str = "ddim",
               num_steps: int = 50) -> torch.Tensor:
        """
        核心采样接口。
        condition_physical: (B, 4) 碰撞前状态，物理单位 [r, v_rad, v_tan, E]
        method: "ddim"（快速，默认）或 "ddpm"（随机）
        返回: (B, 4) 碰撞后状态，物理单位 [r, v_rad, v_tan, E]
        """
        condition_norm = self._normalize_input(condition_physical)

        if method == "ddim":
            y_norm = ddim_sample(self.model, condition_norm,
                                 num_steps=num_steps, eta=0.0, device=self.device)
        elif method == "ddpm":
            y_norm = ddpm_sample(self.model, condition_norm,
                                 num_steps=num_steps, device=self.device)
        else:
            raise ValueError(f"未知采样方法: {method}，请选择 'ddim' 或 'ddpm'")

        return self._denormalize_output(y_norm)


# ─────────────────────────────────────────────
# 运行示例
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import glob

    # 找到最新的模型权重文件（优先 checkpoints/ 子目录，兼容根目录旧版本）
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pth_files = sorted(
        glob.glob(os.path.join(ROOT, "checkpoints", "damascus_diffusion_ep*.pth")) +
        glob.glob(os.path.join(ROOT, "damascus_diffusion_ep*.pth")),
        key=lambda f: int(os.path.basename(f).split('ep')[1].split('.')[0]))
    if not pth_files:
        print("未找到模型权重文件，请先运行 training/train.py 完成训练。")
        exit(1)

    checkpoint = pth_files[-1]  # 使用最新的权重
    npz_path   = os.path.join(ROOT, "parsed_transitions.npz")

    sampler = DarkMatterSampler(checkpoint, npz_path)

    # 从真实数据中取一批碰撞前状态作为条件
    data = np.load(npz_path)
    raw_in = data['states_in'][:8]  # 取前 8 条
    pos = raw_in[:, 2:5]; vel = raw_in[:, 5:8]
    E   = raw_in[:, 8];   r   = raw_in[:, 9]
    v_rad = np.sum(pos * vel, axis=1) / r
    v_tan = np.sqrt(np.maximum(0, np.sum(vel**2, axis=1) - v_rad**2))
    condition_phys = torch.tensor(
        np.column_stack([r, v_rad, v_tan, E]), dtype=torch.float32
    )

    print(f"\n碰撞前状态 (物理单位) [r, v_rad, v_tan, E]:")
    print(condition_phys.numpy())

    # 真实碰撞后状态（用于对比）
    raw_out = data['states_out'][:8]
    pos_o = raw_out[:, 2:5]; vel_o = raw_out[:, 5:8]
    E_o   = raw_out[:, 8];   r_o   = raw_out[:, 9]
    v_rad_o = np.sum(pos_o * vel_o, axis=1) / r_o
    v_tan_o = np.sqrt(np.maximum(0, np.sum(vel_o**2, axis=1) - v_rad_o**2))
    truth = np.column_stack([r_o, v_rad_o, v_tan_o, E_o])

    # DDIM 快速采样
    result = sampler.sample(condition_phys, method="ddim", num_steps=50)
    pred = result.cpu().numpy()

    print(f"\n模型生成的碰撞后状态 (物理单位) [r, v_rad, v_tan, E]:")
    print(pred)

    print(f"\n真实碰撞后状态 (地面真值):")
    print(truth)

    # 逐特征相对误差
    rel_err = np.abs(pred - truth) / (np.abs(truth) + 1e-8)
    labels = ["r", "v_rad", "v_tan", "E"]
    print("\n各特征平均相对误差:")
    for i, lbl in enumerate(labels):
        print(f"  {lbl:6s}: {rel_err[:, i].mean()*100:.2f}%")

    # 物理约束检查
    n_neg_vtan = (pred[:, 2] < 0).sum()
    print(f"\n物理一致性: v_tan < 0 的样本数 = {n_neg_vtan} / {len(pred)}"
          f"  (模型应输出 v_tan ≥ 0)")
