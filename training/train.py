import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# 导入你之前写的模块 (假设它们在同一目录下)
from data_pipeline.transform import DamascusDataset
from mlp_score import ScoreNetwork

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

    def loss_fn(self, x0):
        """
        核心物理损失函数：Denoising Score Matching
        x0: 我们从数据集中取出的“真实物理状态” [r, v_rad, v_tan, E]
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
        
        # 5. 让神经网络预测被加入的噪声
        # 我们把时间步 t 展平为 (batch_size,) 喂给模型
        predicted_noise = self.model(x_t, t.squeeze(-1))
        
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
                # 在扩散模型中，我们通常是对目标状态 (batch_y) 进行建模
                # 如果做条件扩散 (Conditional Diffusion)，batch_x 会作为额外条件传入网络
                # 这里我们先展示无条件学习 batch_y 的物理分布
                x0 = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                loss = self.loss_fn(x0)
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
    # 1. 设定计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 准备数据管道 (使用我们之前写的 transform.py)
    print("正在加载并处理物理特征...")
    dataset = DamascusDataset("parsed_transitions.npz", normalize=True)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    
    # 3. 初始化 Score Network (4维物理相空间: r, v_rad, v_tan, E)
    model = ScoreNetwork(state_dim=4, hidden_dim=256, time_emb_dim=128, num_layers=4)
    
    # 4. 启动训练引擎
    trainer = DiffusionTrainer(model=model, dataloader=dataloader, device=device, lr=2e-4)
    
    # 开始训练 100 轮 (根据你的数据量调整)
    trainer.train(epochs=100)