import sys
sys.path.append('../../')
from PricingLib.Base.BaseLayer import MarketEnvironment, PricingEngine, Instrument
from PricingLib.Base.Utils import StandardRNG, SobolRNG

import numpy as np
from typing import Dict, Any



class MonteCarloEngine(PricingEngine):
    """
    通用蒙特卡洛引擎。
    职责：
    1. 调用 RNG Utility 生成随机数。
    2. 生成资产价格路径 (目前仅支持几何布朗运动)。
    3. 调用 Option.payoff 计算回报。
    4. 折现求均值。
    """
    
    def __init__(self, n_sims=100000, n_steps=1, rng_type='standard'):
        self.n_sims = n_sims
        self.n_steps = n_steps
        
        # 简单工厂：选择 RNG
        if rng_type.lower() == 'sobol':
            self.rng = SobolRNG()
        else:
            self.rng = StandardRNG()
            
    def calculate(self, option: Instrument, market: MarketEnvironment) -> Dict[str, Any]:
        """
        核心计算逻辑。
        """
        S, T, r, sigma = market.S, market.T, market.r, market.sigma
        
        # 边界检查
        if T <= 0:
            # 到期了，直接算内在价值
            # 构造一个 shape 兼容的输入，或者直接传 S
            return {'price': np.mean(option.payoff(S))}

        # 1. 获取随机数矩阵 Z (Shape: sims x steps)
        # 注意：Sobol RNG 可能返回 2^m 个数，我们截取前 n_sims 个? 
        # 或者为了对偶性保留全部。这里简单起见，我们信任 RNG 返回合适的数量。
        Z = self.rng.get_gaussian_matrix(self.n_sims, self.n_steps)
        n_actual_sims = Z.shape[0]
        
        # 确保 Z 的形状正确 (sims, steps)
        if Z.ndim == 1: Z = Z[:, np.newaxis]
        
        # 2. 路径生成 (Path Generation) - Geometric Brownian Motion
        dt = T / self.n_steps
        
        # 优化：如果是路径无关期权且只有一步，直接算终值，不用 cumprod
        if not option.is_path_dependent() and self.n_steps == 1:
            # S_T = S * exp(...)
            # Z[:, 0] 取第一列随机数
            ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z[:, 0])
            
            # 3. 计算 Payoff
            # 多态调用！option.payoff 会自动处理标量 K 或数组 K
            payoffs = option.payoff(ST) 
            
        else:
            # 路径依赖模式 (为未来的鲨鱼鳍/亚式做准备)
            # 生成完整路径 S_t (S0 -> S1 -> ... -> ST)
            # S_t = S_{t-1} * exp(...)
            drift = (r - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * Z
            
            # 增量矩阵
            increments = np.exp(drift + diffusion)
            # 累积乘积得到路径 (sims, steps)
            path_matrix = S * np.cumprod(increments, axis=1)
            
            # 有些产品 (如亚式) 可能需要 S0。
            # 为了通用性，我们在最前面拼上 S0 列
            S0_col = np.full((n_actual_sims, 1), S)
            full_path = np.hstack([S0_col, path_matrix])
            
            # 调用 Product 的 payoff，传入完整路径
            payoffs = option.payoff(full_path)

        # 4. 折现求均值
        # 兼容 K 是数组的情况 (Series Option)
        # 如果 payoffs 是 (M, N)，axis=0 会对 M 个模拟求均值，得到 (N,) 的价格向量
        if payoffs.ndim > 1 and payoffs.shape[1] > 1:
             # Series Option: 每一列是一个 K 的结果
             price = np.mean(payoffs, axis=0) * np.exp(-r * T)
        else:
             # Single Option
             price = np.mean(payoffs) * np.exp(-r * T)
        
        return {'price': price}