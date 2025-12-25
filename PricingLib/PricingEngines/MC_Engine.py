import sys
sys.path.append('../../')
from PricingLib.Base.BaseLayer import MarketEnvironment, PricingEngine, Instrument, StochasticProcess
from PricingLib.Base.Utils import StandardRNG, SobolRNG

import numpy as np
from typing import Dict, Any, Optional



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
        
        if rng_type.lower() == 'sobol':
            self.rng = SobolRNG()
        else:
            self.rng = StandardRNG()
        
        self._cached_seed: Optional[int] = None
        self._cached_Z: Optional[np.ndarray] = None
        self._active_seed: Optional[int] = None # 当前计算周期使用的种子
    
    def _get_random_matrix(self, n_steps) -> np.ndarray:
        """
        内部方法：智能获取随机矩阵。
        逻辑：
        1. 检查 self._active_seed (当前指令的种子)。
        2. 如果它与缓存的种子一致，直接返回缓存的 Z。
        3. 否则，生成新的 Z 并更新缓存。
        """
        seed = self._active_seed
        
        if seed is None:
            return self.rng.get_gaussian_matrix(self.n_sims, n_steps, seed=None)
        
        if seed == self._cached_seed and self._cached_Z is not None:
            return self._cached_Z
        
        Z = self.rng.get_gaussian_matrix(self.n_sims, n_steps, seed=seed)
        self._cached_seed = seed
        self._cached_Z = Z
        return Z
    
    def _generate_paths(self, market: MarketEnvironment, n_sims: int, n_steps: int) -> np.ndarray:
        """
        通用路径生成器。
        优先使用 process 提供的快速向量化方法，否则回退到迭代法。
        """
        S0, T = market.S, market.T
        Z_full = self._get_random_matrix(n_steps) 
        
        # 必须确保 Z 的列数与本次需要的 n_steps 一致
        Z = Z_full[:, :n_steps] 
        n_actual_sims = Z.shape[0]
        
        # --- [关键] 尝试调用 process 提供的快速通道 ---
        fast_path = self.process.generate_full_paths(S0, market, n_steps, Z, method=self.path_gen_method)
        
        if fast_path is not None:
            return fast_path
        
        # --- [回退] 通用迭代路径 (Euler/Milstein) ---
        paths = np.zeros((n_actual_sims, n_steps + 1))
        paths[:, 0] = S0
        dt = T / n_steps
        
        for t_step in range(1, n_steps + 1):
            St = paths[:, t_step - 1]
            dW = np.sqrt(dt) * Z[:, t_step - 1]
            
            if self.path_gen_method == 'milstein':
                paths[:, t_step] = self.process.milstein_step(market, 0, St, dt, dW)
            else:
                paths[:, t_step] = self.process.euler_step(market, 0, St, dt, dW)
                
            paths[:, t_step] = np.maximum(paths[:, t_step], 0)

        return paths
            
    def calculate(self, option: Instrument, market: MarketEnvironment, **kwargs) -> Dict[str, Any]:
        S, T, r = market.S, market.T, market.r
        
        if T <= 0:
            price = np.mean(option.payoff(S))
            return {'price': float(price) if np.ndim(price) == 0 else price.flatten()}
        # 1. 决定路径步数
        if self.path_gen_method == 'exact' and not option.is_path_dependent():
             n_steps_to_run = 1
        else:
            n_steps_to_run = self.n_steps
        
        # 2. 生成路径
        full_path = self._generate_paths(market, self.n_sims, n_steps_to_run)
        
        # 3. 计算 Payoff
        if option.is_path_dependent():
            payoffs = option.payoff(full_path)
        else:
            payoffs = option.payoff(full_path[:, -1])
            
        # 4. 折现求均值
        price = np.mean(payoffs, axis=0) * np.exp(-r * T)
        
        # 5. 格式化返回
        if np.ndim(price) == 0:
            return {'price': float(price)}
        else:
            return {'price': price.flatten()}
    
    def get_price(self, option, market, **kwargs):
        self._active_seed = kwargs.get('seed', None)
        return super().get_price(option, market, **kwargs)
    
    def get_delta(self, option, market, **kwargs):
        self._active_seed = kwargs.get('seed', None)
        return super().get_delta(option, market, **kwargs)

    def get_gamma(self, option, market, **kwargs):
        self._active_seed = kwargs.get('seed', None)
        return super().get_gamma(option, market, **kwargs)

    def get_vega(self, option, market, **kwargs):
        self._active_seed = kwargs.get('seed', None)
        return super().get_vega(option, market, **kwargs)
    
    def get_theta(self, option, market, **kwargs):
        self._active_seed = kwargs.get('seed', None)
        return super().get_theta(option, market, **kwargs)
    
    def get_rho(self, option, market, **kwargs):
        self._active_seed = kwargs.get('seed', None)
        return super().get_rho(option, market, **kwargs)
    
    def get_vanna(self, option, market, **kwargs):
        self._active_seed = kwargs.get('seed', None)
        return super().get_vanna(option, market, **kwargs)
    
    def get_volga(self, option, market, **kwargs):
        self._active_seed = kwargs.get('seed', None)
        return super().get_volga(option, market, **kwargs)
    