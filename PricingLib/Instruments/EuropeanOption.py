import numpy as np
from PricingLib.Base.BaseLayer import Instrument
from typing import Union

class EuropeanOption(Instrument):
    """
    最基础的欧式香草期权。
    """
    def __init__(self, K: float, T: float, option_type: str = 'call'):
        super().__init__(T)
        self.K = K
        # 统一转为小写，防止用户输入 'Call' 或 'CALL' 出错
        self.option_type = option_type.lower()
        
        if self.option_type not in ['call', 'put']:
            raise ValueError(f"Unknown option type: {option_type}")
        
    def payoff(self, prices: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Payoff = max(S - K, 0) for Call
        Payoff = max(K - S, 0) for Put
        支持标量和数组输入。
        """
        # 确保 K 能够正确广播 (即使 K 是数组，prices 是矩阵)
        # prices: (sims,) 或 (sims, steps) 或 scalar
        # self.K: scalar 或 (num_K,)
        
        # 简单情况：K 是标量
        if np.isscalar(self.K):
            if self.option_type == 'call':
                return np.maximum(prices - self.K, 0.0)
            else:
                return np.maximum(self.K - prices, 0.0)
        
        # 复杂情况：K 是数组 (Series Option)，prices 是路径终值向量
        # 我们需要利用 Broadcasting 生成矩阵
        # prices shape: (M,) -> (M, 1)
        # K shape: (N,) -> (1, N)
        # Result: (M, N)
        else:
            prices_col = np.asarray(prices)
            if prices_col.ndim == 1:
                prices_col = prices_col[:, np.newaxis]
            
            if self.option_type == 'call':
                return np.maximum(prices_col - self.K, 0.0)
            else:
                return np.maximum(self.K - prices_col, 0.0)
            
    def is_path_dependent(self) -> bool:
        return False # 普通期权只看终值 S_T
    
    def get_boundary_values(self, S_vec, t_rem, r):
        # S_vec[0] is lower, S_vec[-1] is upper
        if self.option_type == 'call':
            # Lower: S=0 -> 0
            # Upper: S=Smax -> S - K*exp(-rt)
            lower = 0.0
            upper = S_vec[-1] - self.K * np.exp(-r * t_rem)
        else:
            # Lower: S=0 -> K*exp(-rt)
            # Upper: S=Smax -> 0
            lower = self.K * np.exp(-r * t_rem)
            upper = 0.0
        return lower, upper