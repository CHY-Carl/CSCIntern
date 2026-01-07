import numpy as np
from ..Base.BaseLayer import StochasticProcess, MarketEnvironment
#TODO  加入分红率q 
class GeometricBrownianMotion(StochasticProcess):
    """
    SDE: dS = r*S*dt + sigma*S*dW
    这是一个无状态的公式集。
    """
    def __init__(self):
        pass
        
    def drift(self, market: MarketEnvironment, t: float, S: np.ndarray):
        return market.r * S
        
    def diffusion(self, market: MarketEnvironment, t: float, S: np.ndarray):
        return market.sigma * S
    
    def diffusion_prime(self, market: MarketEnvironment, t: float, S: np.ndarray):
        # d(sigma*S)/dS = sigma
        return market.sigma * np.ones_like(S)
    
    def pde_coefficients(self, market: MarketEnvironment, t: float, S_vec: np.ndarray):
        """
        为 Black-Scholes PDE 提供通用系数。
        BS PDE: dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0
        """
        r = market.r
        sigma = market.sigma
        
        # 对应 d2V/dS2 的系数
        alpha = 0.5 * (sigma**2) * (S_vec**2)
        
        # 对应 dV/dS 的系数
        beta = r * S_vec
        
        # 对应 V 的系数
        gamma = -r
        
        # 确保 gamma 也是一个与 S_vec 维度匹配的向量
        gamma_vec = np.full_like(S_vec, gamma)
        
        return (alpha, beta, gamma_vec)
    
    # [覆盖] 实现高效的完整路径生成
    def generate_full_paths(self, S0: float, market: MarketEnvironment, n_steps: int, Z: np.ndarray, method: str = 'exact') -> np.ndarray:
        T, r, sigma = market.T, market.r, market.sigma
        dt = T / n_steps
        
        if method.lower() == 'exact':
            # --- 方法 A: 解析解 (Log-Euler) ---
            # S_{t+1} = S_t * exp( (r - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z )
            # 优点：精度最高，永远非负，无离散化误差
            drift_term = (r - 0.5 * sigma**2) * dt
            diffusion_term = sigma * np.sqrt(dt) * Z
            increments = np.exp(drift_term + diffusion_term)
        elif method.lower() == 'euler':
            # --- 方法 B: Euler-Maruyama ---
            # dS = r*S*dt + sigma*S*dW
            # S_{t+1} = S_t + r*S_t*dt + sigma*S_t*dW
            #         = S_t * (1 + r*dt + sigma*sqrt(dt)*Z)
            # 优点：符合直觉，是通用 Euler 法在 GBM 上的体现
            # 缺点：一阶精度，极端情况下可能导致负价格 (虽然乘性结构实际上避免了显式负数，但 increments 可能为负)
            
            simple_return = r * dt + sigma * np.sqrt(dt) * Z
            increments = 1.0 + simple_return
            
            # [安全补丁] 防止极端波动导致价格为负
            increments = np.maximum(increments, 0)
        else:
            # 如果是其他方法（如 Milstein），GBM 没有特殊的向量化技巧，
            return None
        
        path_matrix = S0 * np.cumprod(increments, axis=1)
        
        S0_col = np.full((Z.shape[0], 1), S0)
        return np.hstack([S0_col, path_matrix])