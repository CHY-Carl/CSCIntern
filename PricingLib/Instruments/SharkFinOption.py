import numpy as np
from ..Base.BaseLayer import Instrument

class UpAndOutCall(Instrument):
    """
    向上敲出看涨期权 (Up-and-Out Call / Shark Fin).
    条款:
    - 如果在有效期内 S_t >= Barrier，期权敲出，获得 Rebate。
    - 如果从未敲出，到期获得 max(S_T - Strike, 0)。
    """
    def __init__(self, K: float, H: float, T: float, rebate: float = 0.0):
        super().__init__(T)
        self.K = K
        self.H = H
        self.rebate = rebate
        
    def is_path_dependent(self) -> bool:
        return True  # 必须告诉 Engine 我需要路径！

    def payoff(self, prices: np.ndarray) -> np.ndarray:
        """
        计算 Payoff。
        Input prices: 
          - 如果是路径依赖 (MC): shape (n_sims, n_steps + 1)
          - 如果是路径无关 (FDM/BS): shape (n_grid, ) 这里的输入含义由 Engine 决定
        """
        # --- 场景 A: Monte Carlo (输入是路径矩阵) ---
        if prices.ndim == 2:
            # prices shape: (sims, steps)
            # 1. 检查是否敲出: 沿时间轴(axis=1)看最大值
            S_max = np.max(prices, axis=1)
            
            # 敲出标记 (布尔向量)
            knocked_out = S_max >= self.H
            
            # 2. 计算未敲出的 Payoff (欧式 Call)
            # 取路径终点 S_T = prices[:, -1]
            S_T = prices[:, -1]
            vanilla_payoff = np.maximum(S_T - self.K, 0.0)
            
            # 3. 组合: 敲出给 rebate，没敲出给 vanilla
            # result = where(knocked_out, rebate, vanilla_payoff)
            final_payoff = np.where(knocked_out, self.rebate, vanilla_payoff)
            return final_payoff

        # --- 场景 B: FDM / BS (输入是价格网格 S) ---
        # 对于 FDM，Engine 会在网格上求解 PDE。
        # Product 的 payoff 方法在这里通常只定义 "终值条件" (t=T)
        # 敲出逻辑由 FDM Engine 的边界条件处理，而不是在这里。
        # 所以对于 FDM，我们只需要返回 t=T 时刻的收益图。
        # 但要注意: 对于 S >= H 的部分，理论上价值就是 Rebate (或0，取决于是否立即终止)
        # FDM Engine 会截断网格到 H，所以这里只处理 S < H 的情况即可。
        else:
            # 输入是 S_vec (1D array)
            # 这是一个普通的 Call Payoff，但在 S >= H 时被截断
            payoff = np.maximum(prices - self.K, 0.0)
            # 强制 S >= H 的部分为 Rebate (虽然 FDM 网格可能只到 H)
            payoff = np.where(prices < self.H, payoff, self.rebate)
            return payoff
    
    # --- 接口扩展：为 FDM 提供边界条件 ---
    def get_boundary_values(self, S_vec, t_rem, r):
        """
        FDM 专用：返回下边界和上边界的价值。
        下边界 (S=0): 0
        上边界 (S=H): Rebate * df (如果是到期支付) 或 Rebate (如果是立即支付)
        通常假设 Rebate 是到期支付的固定金额。
        """
        lower = 0.0
        # 如果 Rebate 是到期支付: Rebate * exp(-r * t_rem)
        # 如果 Rebate 是立即支付: Rebate
        # 这里假设最常见的：敲出后合约终止，Rebate 也是到期结算 (或者 rebate 本身就是现值)
        # 为了通用性，假设 Rebate 是固定金额，到期支付
        upper = self.rebate * np.exp(-r * t_rem)
        return lower, upper
    
    # 增加一个属性让 FDM 知道这有个障碍
    @property
    def barrier(self):
        return self.H