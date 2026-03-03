import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import sys
import os

# 将项目根目录加入模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入你之前写的模块
from data_pipeline.transform import DamascusDataset
from mlp_score import ConditionalScoreNetwork

class DiffusionTrainer:
    """
    扩散模型训练引擎。
    使用连续时间 (Continuous-time) 的 Variance Preserving (VP) SDE 噪声调度。
    """
    def __init__(self, model, dataloader, device, lr=1e-4):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        
        # 使用 AdamW 优化器，对物理数据拟合更稳定
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # 学习率调度器：余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def compute_noise_schedule(self, t):
        """
        定义加噪过程 (Forward SDE)。
        这里我们使用标准的余弦调度 (Cosine Schedule)，它在物理连续变量上表现极佳。
        t 是连续的时间步，取值范围 [0, 1]
        返回: 均值衰减系数 (sqrt_alpha_bar) 和 噪声标准差 (sigma)
        """
        # 防止 t 完全为 0 导致数值不稳定
        t = t.clamp(min=1e-5) 
        
        # 余弦调度的数学定义
        f_t = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        f_0 = math.cos(0.008 / 1.008 * math.pi / 2) ** 2
        alpha_bar = f_t / f_0
        
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sigma = torch.sqrt(1.0 - alpha_bar)
        
        return sqrt_alpha_bar, sigma

    def loss_fn(self, x0, condition):
        """
        核心物理损失函数：Conditional Denoising Score Matching
        x0:        碰撞后的真实物理状态 [r, v_rad, v_tan, E]
        condition: 碰撞前的条件状态 [r, v_rad, v_tan, E]
        """
        batch_size = x0.shape[0]
        
        # 1. 随机采样扩散时间步 t ~ Uniform(0, 1)
        # 这代表粒子处于热化过程的任意一个随机阶段
        t = torch.rand(batch_size, 1, device=self.device)
        
        # 2. 计算当前时间步的噪声强度
        sqrt_alpha_bar, sigma = self.compute_noise_schedule(t)
        
        # 3. 生成与 x0 维度相同的纯高斯噪音
        noise = torch.randn_like(x0)
        
        # 4. 执行前向加噪过程 (Forward Diffusion)
        # 物理上：把粒子推向完全各向同性的热力学平衡态
        x_t = sqrt_alpha_bar * x0 + sigma * noise
        
        # 5. 让神经网络预测噪声，同时传入碰撞前状态作为条件
        predicted_noise = self.model(x_t, t.squeeze(-1), condition)
        
        # 6. 计算均方误差 (MSE)
        # 网络预测的噪声越接近真实的噪声，说明它越理解这个物理系统的微观涨落
        loss = nn.MSELoss()(predicted_noise, noise)
        return loss

    def train(self, epochs):
        """训练循环"""
        print(f"开始训练，运行设备: {self.device}")
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            
            # 使用 tqdm 包装 dataloader 以显示进度条
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            
            for batch_x, batch_y in pbar:
                # 条件扩散: batch_x 是碰撞前状态 (condition), batch_y 是碰撞后状态 (target)
                # 数据已预加载到 GPU，无需 .to(device)
                condition = batch_x
                x0 = batch_y
                
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_fn(x0, condition)
                loss.backward()
                
                # 梯度裁剪，防止物理特征波动引发梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            self.scheduler.step()
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch} | Average Loss: {avg_loss:.6f} | LR: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # 定期保存模型权重
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"damascus_diffusion_ep{epoch}.pth")
                print(f"模型已保存 -> damascus_diffusion_ep{epoch}.pth")

# ==========================================
# 执行主程序
# ==========================================
if __name__ == "__main__":
    # 1. 设定计算设备 (优先 CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # 2. 准备数据管道 (使用我们之前写的 transform.py)
    print("正在加载并处理物理特征...")
    dataset = DamascusDataset("parsed_transitions.npz", normalize=True)
    # 将全部数据预加载到 GPU，消灭每 batch 的 CPU->GPU 传输瓶颈 (~160MB)
    dataset.to_device(device)
    print(f"全部数据已预加载到 {device} (约 {len(dataset)*4*2*4/1024**2:.0f} MB)")
    dataloader = DataLoader(
        dataset, batch_size=8192, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=False
    )
    
    # 3. 初始化条件 Score Network（FiLM 架构，num_layers=6）
    model = ConditionalScoreNetwork(state_dim=4, hidden_dim=256, time_emb_dim=128, num_layers=6)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")

    # 4. 启动训练引擎
    trainer = DiffusionTrainer(model=model, dataloader=dataloader, device=device, lr=1e-3)

    # FiLM 架构需要更多轮次收敛，建议 300 epochs
    trainer.train(epochs=300)