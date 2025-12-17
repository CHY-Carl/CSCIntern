import numpy as np
from ..Base.BaseLayer import StochasticProcess, MarketEnvironment

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
        # 直接从 market 对象访问属性，无需 get_xxxx 方法
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
    def generate_full_paths(self, S0: float, market: MarketEnvironment, n_steps: int, Z: np.ndarray) -> np.ndarray:
        T, r, sigma = market.T, market.r, market.sigma
        dt = T / n_steps
        
        drift = (r - 0.5 * sigma**2) * dt
        diffusion_term = sigma * np.sqrt(dt) * Z # Z shape: (sims, steps)
        
        increments = np.exp(drift + diffusion_term)
        path_matrix = S0 * np.cumprod(increments, axis=1)
        
        S0_col = np.full((Z.shape[0], 1), S0)
        return np.hstack([S0_col, path_matrix])