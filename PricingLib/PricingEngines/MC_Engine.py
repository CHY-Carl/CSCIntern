import sys
sys.path.append('../../')
from PricingLib.Base.BaseLayer import MarketEnvironment, PricingEngine, Instrument, StochasticProcess
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
    
    def __init__(self, process: StochasticProcess, n_sims=100000, n_steps=1, rng_type='standard', path_gen_method='euler'):
        self.process = process
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.path_gen_method = path_gen_method.lower()
        
        # 简单工厂：选择 RNG
        if rng_type.lower() == 'sobol':
            self.rng = SobolRNG()
        else:
            self.rng = StandardRNG()
    
    def _generate_paths(self, market: MarketEnvironment, n_sims: int, n_steps: int) -> np.ndarray:
        """
        通用路径生成器。
        优先使用 process 提供的快速向量化方法，否则回退到迭代法。
        """
        S0, T = market.S, market.T
        Z = self.rng.get_gaussian_matrix(n_sims, n_steps)
        n_actual_sims = Z.shape[0]
        
        # --- [关键] 尝试调用 process 提供的快速通道 ---
        fast_path = self.process.generate_full_paths(S0, market, n_steps, Z)
        
        if fast_path is not None:
            return fast_path
        
        # --- [回退] 通用迭代路径 (Euler/Milstein) ---
        paths = np.zeros((n_actual_sims, n_steps + 1))
        paths[:, 0] = S0
        dt = T / n_steps
        
        for t_step in range(1, n_steps + 1):
            St = paths[:, t_step - 1]
            dW = np.sqrt(dt) * Z[:, t_step - 1]
            
            # 多态调用，并传入 market！
            if self.path_gen_method == 'milstein':
                paths[:, t_step] = self.process.milstein_step(market, 0, St, dt, dW)
            else: # 默认为 Euler
                paths[:, t_step] = self.process.euler_step(market, 0, St, dt, dW)
                
            paths[:, t_step] = np.maximum(paths[:, t_step], 0)

        return paths
            
    def calculate(self, option: Instrument, market: MarketEnvironment) -> Dict[str, Any]:
        S, T, r = market.S, market.T, market.r
        
        if T <= 0:
            price = np.mean(option.payoff(S))
            return {'price': float(price) if np.isscalar(price) else price.flatten()}

        # 1. 确定步数
        n_steps_to_run = self.n_steps if option.is_path_dependent() else 1
        
        # 2. 生成路径 (传入 market)
        full_path = self._generate_paths(market, self.n_sims, n_steps_to_run)
        
        # 3. 计算 Payoff
        if not option.is_path_dependent() and n_steps_to_run == 1:
            payoffs = option.payoff(full_path[:, -1])
        else:
            payoffs = option.payoff(full_path)
            
        # 4. 折现求均值
        price = np.mean(payoffs, axis=0) * np.exp(-r * T)
        
        # 5. 格式化返回
        if hasattr(option, 'K') and np.isscalar(option.K):
            return {'price': float(price)}
        else:
            return {'price': price.flatten()}