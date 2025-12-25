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
        self.K = K
        self.H = H
        self.rebate = rebate
        self.T_init = T 
        
        # --- 核心状态变量 ---
        self.is_active = True   # 期权是否依然存活
        self.triggered = False  # 障碍是否曾被触发
        
    def is_path_dependent(self) -> bool:
        return True  
    
    def update_status(self, S: float) -> bool:
        """
        检查当前价格 S 是否导致敲出。
        如果已经敲出，保持死亡状态。
        """
        if self.is_active and S >= self.H:
            self.is_active = False
            self.triggered = True

        return self.is_active

    def get_residual_value(self, S, T_rem, r):
        """
        敲出后的残值。
        如果是到期支付 Rebate，残值 = Rebate * exp(-r * T_rem)
        """
        if self.triggered:
            return self.rebate * np.exp(-r * T_rem)
        return 0.0
    
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
            
            knocked_out = S_max >= self.H
            
            # 2. 计算未敲出的 Payoff (欧式 Call)
            S_T = prices[:, -1]
            vanilla_payoff = np.maximum(S_T - self.K, 0.0)
            
            # 3. 组合: 敲出给 rebate，没敲出给 vanilla
            final_payoff = np.where(knocked_out, self.rebate, vanilla_payoff)
            return final_payoff

        # --- 场景 B: FDM / BS (输入是价格网格 S) ---
        else:
            # 输入是 S_vec (1D array)
            payoff = np.maximum(prices - self.K, 0.0)

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

        upper = self.rebate * np.exp(-r * t_rem)
        return lower, upper
    
    @property
    def barrier(self):
        return self.H









class DoubleSharkFin(Instrument):
    """
    双向鲨鱼鳍期权 (Double Knock-out Call/Put Straddle).
    条款:
    - 下障碍 H_L, 上障碍 H_U.
    - 行权区间 K_L (Put型), K_U (Call型).
    - 敲出补偿 R_L, R_U.
    """
    def __init__(self, K_L: float, K_U: float, H_L: float, H_U: float, 
                 T: float, R_L: float = 0.0, R_U: float = 0.0):
        super().__init__(T)
        self.K_L = K_L
        self.K_U = K_U
        self.H_L = H_L
        self.H_U = H_U
        self.R_L = R_L
        self.R_U = R_U

    def is_path_dependent(self) -> bool:
        return True

    def payoff(self, prices: np.ndarray) -> np.ndarray:
        """
        Input prices:
        - MC: (n_sims, n_steps + 1) -> 完整路径
        - FDM: (n_grid, ) -> 到期时刻的价格网格
        """
        # --- 情况 A: Monte Carlo (处理完整路径) ---
        if prices.ndim == 2:
            S_max = np.max(prices, axis=1)
            S_min = np.min(prices, axis=1)
            S_T = prices[:, -1]
            
            # 判断敲出状态
            ko_up = S_max >= self.H_U
            ko_dn = S_min <= self.H_L
            
            # 计算未敲出时的收益 (跨式组合)
            vanilla_payoff = np.maximum(S_T - self.K_U, 0.0) + np.maximum(self.K_L - S_T, 0.0)
            
            # 组合逻辑: 
            # 1. 优先检查向上敲出
            # 2. 再检查向下敲出
            # 3. 最后给存活收益
            final_payoff = np.where(ko_up, self.R_U, 
                                    np.where(ko_dn, self.R_L, vanilla_payoff))
            return final_payoff

        # --- 情况 B: FDM (处理到期时刻的网格点) ---
        else:
            # prices = S_vec
            # FDM网格通常被限制在 [H_L, H_U] 之间

            #!边界条件的确定，确实是FDM方法中特别需要注意的一个点
            vanilla = np.maximum(prices - self.K_U, 0.0) + np.maximum(self.K_L - prices, 0.0)
            payoff = np.where(prices <= self.H_L, self.R_L,
                              np.where(prices >= self.H_U, self.R_U, vanilla))
            
            return payoff

    def get_boundary_values(self, S_vec, t_rem, r):
        """为 FDM 提供双侧 Dirichlet 边界值"""
        lower = self.R_L * np.exp(-r * t_rem)
        upper = self.R_U * np.exp(-r * t_rem)
        return lower, upper

    @property
    def barrier_low(self): 
        return self.H_L
    
    @property
    def barrier_high(self): 
        return self.H_U