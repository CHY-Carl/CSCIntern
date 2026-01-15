import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Union, Callable, List
from abc import ABC, abstractmethod

# ==========================================================
# ======= 4. Market Environment (市场环境) =================
# ==========================================================
@dataclass(frozen=True)
class MarketEnvironment:
    """
    一个不可变的、可哈希的市场环境数据类。
    `frozen=True` 会自动生成 __hash__ 和 __eq__ 方法。
    """
    S: float
    r: float
    sigma: float
    T: float
    
    def clone(self, **kwargs):
        """
        创建一个新的、修改了部分属性的市场环境实例。
        """
        from dataclasses import replace
        return replace(self, **kwargs)

# ==========================================================
# ======= 5. Instrument (金融产品基类) =====================
# ==========================================================
class Instrument(ABC):
    """
    所有金融产品的父类。
    它只定义合约条款 (Payoff)，不知道如何定价。
    """
    def __init__(self, T: float):
        self.T = T
        self.is_active = True

    def update_status(self, S: float) -> bool:
        """更新存活状态。默认永远活着。"""
        return self.is_active

    def get_residual_value(self, S, T_rem, r):
        """敲出后的残值 (Rebate PV)"""
        return 0.0
        
    @abstractmethod
    def payoff(self, prices: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        给定终值(或路径)，计算 Payoff。
        必须支持向量化输入 (NumPy Broadcasting)。
        """
        pass

    @abstractmethod
    def is_path_dependent(self) -> bool:
        """是否路径依赖 (决定 MC 是否需要存储完整路径)"""
        pass

    @abstractmethod
    def get_critical_points(self) -> List[float]:
        """
        [新增] 返回需要加密网格的关键价格点 (Strike, Barrier)。
        """
        pass

    def get_boundary_values(self, S_vec, t_rem, r):
        """
        返回下边界和上边界的价值。
        用于 FDM 边界条件处理。
        返回: (lower_value, upper_value)
        """
        return 0.0, 0.0
    
        # --- [新增] FDM 离散事件接口 (虚方法) ---
    
    def get_event_dates(self, T_total: float) -> List[float]:
        """
        [虚方法] 如果产品有离散事件，子类应重写此方法。
        返回所有事件发生的【剩余时间点】(T_rem)
        默认实现返回一个空列表，表示没有离散事件。
        Args:
            T_total (float): 产品的总期限（年）。
        Returns:
            List[float]: 事件日期列表 (年化剩余时间)。
        """
        return []

    def get_event_handler(self) -> Callable:
        """
        [虚方法] 如果产品有离散事件，子类应重写此方法。
        返回一个事件处理器函数。此函数将在每个事件日被 EventFDMEngine 调用，
        用于修正当前的价值向量。
        Handler Function Signature:
            handler(V_current: np.ndarray, S_vec: np.ndarray, t_event_rem: float) -> np.ndarray
        默认实现返回一个“什么都不做”的函数，直接原样返回价值向量。
        Returns:
            Callable: 事件处理器函数。
        """
        return lambda V, S_vec, t_event_rem: V



class ProductPortfolio:
    """
    一个简单的容器类，用于表示由多个金融工具组成的投资组合。
    Engine 会识别这个类型，并对其组件分别进行定价，然后汇总结果。
    """
    def __init__(self, components: List[Instrument]):
        if not components:
            raise ValueError("Component list cannot be empty.")
        self.components = components

    def set_market_environment(self, market: MarketEnvironment):
        """
        将市场环境广播给所有需要它的子组件。
        """
        for comp in self.components:
            if hasattr(comp, 'set_market_environment'):
                comp.set_market_environment(market)

# ==========================================================
# ======= 6. Pricing Engine (定价引擎基类) =================
# ==========================================================
class PricingEngine(ABC):
    """
    定价引擎父类。
    定义统一的计算接口，并提供通用的数值 Greeks 计算逻辑。
    """
    
    @abstractmethod
    def calculate(self, option: Instrument, market: MarketEnvironment, **kwargs) -> Dict[str, Any]:
        """
        核心计算方法。
        返回字典: {'price': float, 'greeks': dict, ...}
        """
        pass

        # --- 公共接口 (Public Interface) ---
    # 外部只调用这些方法，不关心内部是解析解还是数值解
    
    def get_price(self, option, market, **kwargs):
        return self.calculate(option, market, **kwargs)['price']

    def get_delta(self, option, market, **kwargs):
        return self._calculate_numerical_delta(option, market, **kwargs)

    def get_gamma(self, option, market, **kwargs):
        return self._calculate_numerical_gamma(option, market, **kwargs)

    def get_vega(self, option, market, **kwargs):
        return self._calculate_numerical_vega(option, market, **kwargs)
    
    def get_theta(self, option, market, **kwargs):
        return self._calculate_numerical_theta(option, market, **kwargs)
    
    def get_rho(self, option, market, **kwargs):
        return self._calculate_numerical_rho(option, market, **kwargs)
    
    def get_vanna(self, option, market, **kwargs):
        return self._calculate_numerical_vanna(option, market, **kwargs)
    
    def get_volga(self, option, market, **kwargs):
        return self._calculate_numerical_volga(option, market, **kwargs)
    


    # --- 通用数值 Greeks 计算 (Template Method) ---
    # 所有子类 Engine (MC, FDM) 都可以直接使用这些方法，无需重写
    
    def _calculate_numerical_delta(self, option, market, **kwargs):
        dS = market.S * 0.01
        m_up = market.clone(S=market.S + dS)
        m_dn = market.clone(S=market.S - dS)
        return (self.get_price(option, m_up, **kwargs) - self.get_price(option, m_dn, **kwargs)) / (2 * dS)

    def _calculate_numerical_gamma(self, option, market, **kwargs):
        dS = market.S * 0.01
        m_up = market.clone(S=market.S + dS)
        m_dn = market.clone(S=market.S - dS)
        p_up = self.get_price(option, m_up, **kwargs)
        p_mid = self.get_price(option, market, **kwargs)
        p_dn = self.get_price(option, m_dn, **kwargs)
        return (p_up - 2 * p_mid + p_dn) / (dS ** 2)

    def _calculate_numerical_vega(self, option, market, **kwargs):
        dVol = 0.01
        m_up = market.clone(sigma=market.sigma + dVol)
        m_dn = market.clone(sigma=market.sigma - dVol)
        p_up = self.get_price(option, m_up, **kwargs)
        p_dn = self.get_price(option, m_dn, **kwargs)
        return (p_up - p_dn) / 2 # per 1% vol

    def _calculate_numerical_theta(self, option, market, **kwargs):
        dT = 1/365
        # Time decay: T decreases
        m_fut = market.clone(T=market.T - dT)
        m_pst = market.clone(T=market.T + dT)
        p_fut = self.get_price(option, m_fut, **kwargs)
        p_pst = self.get_price(option, m_pst, **kwargs)
        return (p_fut - p_pst) / 2

    def _calculate_numerical_rho(self, option, market, **kwargs):
        dR = 0.01
        m_up = market.clone(r=market.r + dR)
        m_dn = market.clone(r=market.r - dR)
        p_up = self.get_price(option, m_up, **kwargs)
        p_dn = self.get_price(option, m_dn, **kwargs)
        return (p_up - p_dn) / 2

    def _calculate_numerical_volga(self, option, market, **kwargs):
        """
        Volga = d2V / dSigma2
        单位: Vega 变化 per 1% volatility
        """
        dVol = 0.01
        m_up = market.clone(sigma=market.sigma + dVol)
        m_dn = market.clone(sigma=market.sigma - dVol)
        
        p_up = self.get_price(option, m_up, **kwargs)
        p_mid = self.get_price(option, market, **kwargs)
        p_dn = self.get_price(option, m_dn, **kwargs)
        
        return (p_up - 2 * p_mid + p_dn) 

    def _calculate_numerical_vanna(self, option, market, **kwargs):
        """
        Vanna = d2V / dS dSigma (Delta 对 Sigma 的敏感度)
        单位: Delta 变化 per 1% volatility
        算法: 交叉中心差分 (Finite Difference on Cross Derivatives)
        """
        dS = market.S * 0.01
        dVol = 0.01
        
        # 我们需要在 (sigma + dVol) 和 (sigma - dVol) 两个平行宇宙里分别计算 Delta
        
        # 1. 宇宙 A: Sigma Up
        m_up_vol = market.clone(sigma=market.sigma + dVol)
        m_up_vol_up_S = m_up_vol.clone(S=m_up_vol.S + dS)
        m_up_vol_dn_S = m_up_vol.clone(S=m_up_vol.S - dS)
        
        p_up_vol_up_S = self.get_price(option, m_up_vol_up_S, **kwargs)
        p_up_vol_dn_S = self.get_price(option, m_up_vol_dn_S, **kwargs)
        
        delta_up_vol = (p_up_vol_up_S - p_up_vol_dn_S) / (2 * dS)
        
        # 2. 宇宙 B: Sigma Down
        m_dn_vol = market.clone(sigma=market.sigma - dVol)
        m_dn_vol_up_S = m_dn_vol.clone(S=m_dn_vol.S + dS)
        m_dn_vol_dn_S = m_dn_vol.clone(S=m_dn_vol.S - dS)
        
        p_dn_vol_up_S = self.get_price(option, m_dn_vol_up_S, **kwargs)
        p_dn_vol_dn_S = self.get_price(option, m_dn_vol_dn_S, **kwargs)
        
        delta_dn_vol = (p_dn_vol_up_S - p_dn_vol_dn_S) / (2 * dS)
        
        # 3. Vanna = (Delta_Up - Delta_Dn) / (2 * dVol)
        # 注意: 这里的 2 * dVol 是因为我们做了 Sigma 的中心差分
        # 如果我们想要 per 1% 单位，且 dVol=0.01，则除以 2 即可
        return (delta_up_vol - delta_dn_vol) / 2



# ==========================================================
# ======= 7. Stochastic Process (随机过程) =================
# ==========================================================
class StochasticProcess(ABC):
    """
    随机过程的抽象基类 (无状态版)。
    它是一个纯粹的数学公式集，不存储任何市场数据。
    """
    @abstractmethod
    def drift(self, market: MarketEnvironment, t: float, S: np.ndarray) -> np.ndarray:
        """返回 SDE 中的漂移项: mu(S,t) * S"""
        pass
        
    @abstractmethod
    def diffusion(self, market: MarketEnvironment, t: float, S: np.ndarray) -> np.ndarray:
        """返回 SDE 中的扩散项: sigma(S,t) * S"""
        pass    
    
    @abstractmethod
    def diffusion_prime(self, market: MarketEnvironment, t: float, S: np.ndarray) -> np.ndarray:
        """返回扩散项对 S 的一阶导数: d(diffusion)/dS"""
        pass

    @abstractmethod
    def pde_coefficients(self, market: MarketEnvironment, t: float, S_vec: np.ndarray) -> tuple:
        """返回 PDE 系数 (mu, sigma_sq, rate)"""
        pass

    def euler_step(self, market: MarketEnvironment, t: float, S: np.ndarray, dt: float, dW: np.ndarray) -> np.ndarray:
        """提供一个默认的 Euler 离散化步进逻辑"""
        return S + self.drift(market, t, S) * dt + self.diffusion(market, t, S) * dW

    def milstein_step(self, market: MarketEnvironment, t: float, S: np.ndarray, dt: float, dW: np.ndarray) -> np.ndarray:
        """提供一个默认的 Milstein 离散化步进逻辑"""
        g = self.diffusion(market, t, S)
        g_prime = self.diffusion_prime(market, t, S)
        correction_term = 0.5 * g * g_prime * (dW**2 - dt)
        
        return self.euler_step(market, t, S, dt, dW) + correction_term

    def generate_full_paths(self, S0: float, market: MarketEnvironment, n_steps: int, Z: np.ndarray, **kwargs) -> np.ndarray:
        """
        [可选] 提供一个高效的、一次性生成完整路径的向量化方法。
        如果子类不实现，将返回 None，迫使 MCEngine 使用迭代法。
        """
        return None