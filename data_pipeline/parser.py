import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

class DamascusParser:
    """
    一次性解析器：遍历所有 DaMaSCUS-SUN 的 txt 轨迹，
    提取发生物理能量跃迁 (Scattering) 的前后状态，并打包为 .npz
    """
    def __init__(self, data_dir, output_file="parsed_transitions.npz", energy_tol=1e-5):
        self.data_dir = data_dir
        self.output_file = output_file
        self.energy_tol = energy_tol
        # 对应你数据的列：序号 时间 x y z vx vy vz E r
        self.columns = ['index', 't', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'E', 'r']

    def parse_single_file(self, file_path):
        try:
            # 读取原始纯文本数据
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=self.columns)
            
            # 计算能量差，定位碰撞点
            delta_E = df['E'].diff().abs()
            scattering_indices = delta_E[delta_E > self.energy_tol].index
            
            states_in = []
            states_out = []
            
            # 提取马尔可夫演化对 (碰撞前状态 -> 碰撞后状态)
            for idx in scattering_indices:
                if idx == 0: continue
                states_in.append(df.iloc[idx - 1].values)
                states_out.append(df.iloc[idx].values)
                
            return np.array(states_in), np.array(states_out)
        except Exception as e:
            print(f"解析 {file_path} 失败: {e}")
            return None, None

    def run(self, num_workers=None):
        # 递归搜索所有子目录下的 txt 文件
        file_list = glob.glob(os.path.join(self.data_dir, "**", "*.txt"), recursive=True)
        if not file_list:
            print(f"在 {self.data_dir} 未找到 txt 文件。")
            return

        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), len(file_list))

        print(f"找到 {len(file_list)} 个文件，使用 {num_workers} 个进程并行解析...")
        all_in, all_out = [], []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.parse_single_file, f): f for f in file_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                s_in, s_out = future.result()
                if s_in is not None and len(s_in) > 0:
                    all_in.append(s_in)
                    all_out.append(s_out)
                
        if all_in:
            final_in = np.vstack(all_in)
            final_out = np.vstack(all_out)
            # 极速压缩保存
            np.savez_compressed(self.output_file, states_in=final_in, states_out=final_out)
            print(f"\n提取成功！共 {len(final_in)} 个物理散射事件。保存至: {self.output_file}")
        else:
            print("未提取到有效数据。")

if __name__ == "__main__":
    # 假设你的 txt 数据在一个叫做 data 的文件夹里
    # 运行一次，生成 parsed_transitions.npz
    parser = DamascusParser(data_dir="./data", output_file="parsed_transitions.npz")
    parser.run()