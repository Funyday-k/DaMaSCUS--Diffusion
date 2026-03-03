import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DamascusDataset(Dataset):
    """
    轻量级 PyTorch 数据集：
    直接读取 .npz 文件，并利用矩阵运算极速完成球坐标降维和 Z-score 归一化。
    """
    def __init__(self, npz_file="parsed_transitions.npz", normalize=True):
        super().__init__()
        # 1. 毫秒级加载压缩数据
        data = np.load(npz_file)
        raw_in = data['states_in']
        raw_out = data['states_out']
        
        # 2. 向量化物理坐标转换 (Cartesian -> Spherical)
        features_in = self._to_spherical(raw_in)
        features_out = self._to_spherical(raw_out)
        
        # 将 Numpy 转换为 PyTorch Tensor
        self.X = torch.tensor(features_in, dtype=torch.float32)
        self.Y = torch.tensor(features_out, dtype=torch.float32)
        
        # 3. Z-score 归一化
        self.normalize = normalize
        if self.normalize:
            self._apply_normalization()

    def _to_spherical(self, states):
        """
        利用 Numpy 矩阵广播，将 Nx10 的原始数据极速转化为 Nx4 的球对称物理特征。
        输入列索引: 2,3,4(x,y,z), 5,6,7(vx,vy,vz), 8(E), 9(r)
        """
        pos = states[:, 2:5]
        vel = states[:, 5:8]
        E = states[:, 8]
        r = states[:, 9] # 直接使用你数据里的 r 列
        
        # 径向速度: v_rad = (x*vx + y*vy + z*vz) / r
        v_rad = np.sum(pos * vel, axis=1) / r
        
        # 速度平方: v_mag^2 = vx^2 + vy^2 + vz^2
        v_mag_sq = np.sum(vel**2, axis=1)
        
        # 切向速度: v_tan = sqrt(v_mag^2 - v_rad^2)
        v_tan = np.sqrt(np.maximum(0, v_mag_sq - v_rad**2))
        
        # 拼装为特征矩阵 [r, v_rad, v_tan, E]
        return np.column_stack((r, v_rad, v_tan, E))

    def _apply_normalization(self):
        """缩放特征到以 0 为中心，防止梯度爆炸"""
        self.X_mean = self.X.mean(dim=0)
        self.X_std = self.X.std(dim=0) + 1e-8
        self.Y_mean = self.Y.mean(dim=0)
        self.Y_std = self.Y.std(dim=0) + 1e-8
        
        self.X = (self.X - self.X_mean) / self.X_std
        self.Y = (self.Y - self.Y_mean) / self.Y_std

    def get_stats(self):
        """返回归一化参数，供未来生成(Inference)时还原物理单位"""
        return {
            'X_mean': self.X_mean, 'X_std': self.X_std,
            'Y_mean': self.Y_mean, 'Y_std': self.Y_std
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 测试数据管道
if __name__ == "__main__":
    dataset = DamascusDataset("parsed_transitions.npz")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    x_batch, y_batch = next(iter(dataloader))
    print(f"数据管道构建成功！")
    print(f"输入 Batch Shape: {x_batch.shape} (r, v_rad, v_tan, E)")
    print(f"输出 Batch Shape: {y_batch.shape}")