class ConditionalScoreNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=256, time_emb_dim=128, num_layers=4):
        super().__init__()
        # ... (时间嵌入模块不变) ...
        
        # 【修改点】输入维度变成 state_dim * 2，因为要拼接 x_t 和 condition
        self.input_proj = nn.Linear(state_dim * 2, hidden_dim)
        
        # ... (中间的残差块不变) ...
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, x_t, time, condition):
        """
        x_t: 加噪后的目标状态 (碰撞后)
        time: 扩散时间步
        condition: 干净的条件状态 (碰撞前，即你的 batch_x)
        """
        t_emb = self.time_mlp(time)
        
        # 【修改点】将噪声状态与物理条件在特征维度拼接
        x_input = torch.cat([x_t, condition], dim=-1)
        h = self.input_proj(x_input)
        
        for block in self.res_blocks:
            h = block(h, t_emb)
            
        return self.output_proj(h)