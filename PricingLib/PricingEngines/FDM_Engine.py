import sys
sys.path.append('../../')
from PricingLib.Base.BaseLayer import PricingEngine, Instrument, MarketEnvironment, StochasticProcess, ProductPortfolio
from PricingLib.Base.Utils import MathUtils, MatrixUtils


from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Union



class FDMCoefficients(ABC):
    """策略接口：只负责计算矩阵系数 a, b, c"""
    @abstractmethod
    def get_coefficients(self, process: StochasticProcess, dt: float, dS: float, S_vec: np.ndarray):
        """返回 LHS(左边) 和 RHS(右边) 的系数"""
        pass

class CNScheme(FDMCoefficients):
    """
    通用的 Crank-Nicolson 有限差分格式。
    """
    def get_coefficients(self, 
                         process: StochasticProcess, 
                         market: MarketEnvironment, 
                         dt: float, 
                         dS: float, 
                         S_vec: np.ndarray) -> tuple[tuple, tuple]:
        """
        根据通用的 PDE 系数计算 Crank-Nicolson 格式的系数向量。
        
        求解的 PDE 形式: 
        ∂V/∂t + α(S,t) * ∂²V/∂S² + β(S,t) * ∂V/∂S + γ(S,t) * V = 0
        
        Args:
            process: 提供了 PDE 系数的随机过程对象。
            market: 包含了市场参数的对象。
            dt: 时间步长。
            dS: 资产价格步长。
            S_vec: 完整的资产价格网格向量。
            
        Returns:
            一个元组，包含两个子元组：
            ((l_lhs, d_lhs, u_lhs), (l_rhs, d_rhs, u_rhs))
            分别代表 LHS 和 RHS 矩阵的三对角线系数。
        """
        # 1. 从 process 获取内部网格点上的通用 PDE 系数
        inner_S_vec = S_vec[1:-1]
        alpha_vec, beta_vec, gamma_vec = process.pde_coefficients(market, 0, inner_S_vec)
        
        # 2. 计算空间算子 L 离散化后的通用三对角系数 (a, b, c)
        # LV_i ≈ a_i * V_{i-1} + b_i * V_i + c_i * V_{i+1}
        
        # 对应 V_{i-1} 的系数
        a_vec = alpha_vec / dS**2 - beta_vec / (2 * dS)
        
        # 对应 V_{i} 的系数
        b_vec = -2 * alpha_vec / dS**2 + gamma_vec
        
        # 对应 V_{i+1} 的系数
        c_vec = alpha_vec / dS**2 + beta_vec / (2 * dS)
        
        # 3. 构建 Crank-Nicolson 格式的 LHS 和 RHS 系数
        # 方程: (I - 0.5*dt*L)V_new = (I + 0.5*dt*L)V_old
        
        # LHS: (I - 0.5*dt*L) 的系数
        l_lhs = -0.5 * dt * a_vec  # 下对角线
        d_lhs = 1.0 - 0.5 * dt * b_vec  # 主对角线
        u_lhs = -0.5 * dt * c_vec  # 上对角线
        
        # RHS: (I + 0.5*dt*L) 的系数
        l_rhs = 0.5 * dt * a_vec   # 下对角线
        d_rhs = 1.0 + 0.5 * dt * b_vec   # 主对角线
        u_rhs = 0.5 * dt * c_vec   # 上对角线
        
        return (l_lhs, d_lhs, u_lhs), (l_rhs, d_rhs, u_rhs)

class ImplicitScheme(FDMCoefficients):
    """
    完全隐式有限差分格式 (Fully Implicit / Backward Euler)。
    特点：无条件稳定，具有单调性（不会产生 C-N 那样的数值振荡），
    但精度为 O(dt + dS^2)，比 C-N 的 O(dt^2 + dS^2) 低。
    """
    def get_coefficients(self, 
                         process: StochasticProcess, 
                         market: MarketEnvironment, 
                         dt: float, 
                         dS: float, 
                         S_vec: np.ndarray) -> tuple[tuple, tuple]:
        
        # 1. 从 process 获取内部网格点上的通用 PDE 系数
        inner_S_vec = S_vec[1:-1]
        alpha_vec, beta_vec, gamma_vec = process.pde_coefficients(market, 0, inner_S_vec)
        
        # 2. 计算空间算子 L 的通用三对角系数 (a, b, c)
        # 这些系数与 C-N 中完全一致，代表 L 的离散化
        
        # 对应 V_{i-1}
        a_vec = alpha_vec / dS**2 - beta_vec / (2 * dS)
        
        # 对应 V_{i}
        b_vec = -2 * alpha_vec / dS**2 + gamma_vec
        
        # 对应 V_{i+1}
        c_vec = alpha_vec / dS**2 + beta_vec / (2 * dS)
        
        # 3. 构建完全隐式格式的 LHS 和 RHS 系数
        # 方程: (I - dt*L)V_new = V_old
        
        # LHS: (I - dt*L) 的系数
        # 注意这里乘的是 1.0 * dt，而不是 0.5 * dt
        l_lhs = -1.0 * dt * a_vec  # 下对角线
        d_lhs = 1.0 - 1.0 * dt * b_vec  # 主对角线
        u_lhs = -1.0 * dt * c_vec  # 上对角线
        
        # RHS: Identity (单位矩阵)
        # V_new = ... * V_old
        # 所以 RHS 对应的矩阵就是 I。
        # 对角线为 1，上下对角线为 0。
        
        # 构造全 0 的上下对角线
        zeros = np.zeros_like(a_vec)
        # 构造全 1 的主对角线
        ones = np.ones_like(b_vec)
        
        l_rhs = zeros
        d_rhs = ones
        u_rhs = zeros
        
        return (l_lhs, d_lhs, u_lhs), (l_rhs, d_rhs, u_rhs)


class FDMEngine(PricingEngine):
    def __init__(self, process: StochasticProcess, M_space=100, N_time=100, std_mult=5.0, scheme_type='CN'):
        self.process = process
        self.M = int(M_space)
        self.N = int(N_time)
        self.std_mult = std_mult
        self.scheme = ImplicitScheme() if scheme_type == 'Implicit' else CNScheme()

    def _generate_grid(self, option: Instrument, market: MarketEnvironment):
        S, T, r, sigma = market.S, market.T, market.r, market.sigma
        
        if hasattr(option, 'barrier'):
            S_min = 0.0
            S_max = option.barrier
        elif hasattr(option, 'barrier_high') and hasattr(option, 'barrier_low'):
            S_min = option.barrier_low
            S_max = option.barrier_high
        elif hasattr(option, 'barrier_low'):
            S_min = option.barrier_low
            log_S0 = np.log(S)
            drift = (r - 0.5 * sigma**2) * T
            vol_T = sigma * np.sqrt(T)
            S_max = np.exp(log_S0 + drift + self.std_mult * vol_T)
        else:
            S_min = 0.0
            #TODO: 其他SDE条件下的统计 S_max 计算
            log_S0 = np.log(S) if S > 0 else 0.0
            drift = (r - 0.5 * sigma**2) * T
            vol_T = sigma * np.sqrt(T)
            S_stat_max = np.exp(log_S0 + drift + self.std_mult * vol_T)
            
            S_max = S_stat_max

            K_val = 0
            if hasattr(option, 'K'):
                K_val = np.max(option.K) if not np.isscalar(option.K) else option.K
                S_max = max(S_stat_max, K_val * 1.1)
            if hasattr(option, 'S0'):
                S_max = max(S_stat_max, option.S0 * 2)
            
        S_vec = np.linspace(S_min, S_max, self.M + 1)
        dt = T / self.N
    
        return S_vec, dt

    def _solve_on_grid(self, option: Instrument, market: MarketEnvironment, S_vec: np.ndarray, dt: float):
        S, T, r, sigma = market.S, market.T, market.r, market.sigma
        
        dS = S_vec[1] - S_vec[0]
        (l_lhs, d_lhs, u_lhs), (l_rhs, d_rhs, u_rhs) = self.scheme.get_coefficients(self.process, market, dt, dS, S_vec)
        
        V = option.payoff(S_vec) 
        if V.ndim == 1: V = V[:, np.newaxis]
        
        for i in range(self.N):
            t_rem = T - (i + 1) * dt
            t_old = t_rem + dt
            
            # 显式更新边界值
            bc_l_old, bc_u_old = option.get_boundary_values(S_vec, t_old, r)
            V[0, :] = bc_l_old
            V[-1, :] = bc_u_old
            
            rhs_vec = l_rhs[:, None] * V[:-2, :] + \
                      d_rhs[:, None] * V[1:-1, :] + \
                      u_rhs[:, None] * V[2:, :]
            
            bc_l_new, bc_u_new = option.get_boundary_values(S_vec, t_rem, r)
            
            # 边界移项 (注意 l_lhs[0] 和 u_lhs[-1] 包含边界系数)
            rhs_vec[0, :] -= l_lhs[0] * bc_l_new
            rhs_vec[-1, :] -= u_lhs[-1] * bc_u_new
            
            # 求解
            V[1:-1, :] = MatrixUtils.solve_tridiagonal(u_lhs, d_lhs, l_lhs, rhs_vec)
            
        bc_l, bc_u = option.get_boundary_values(S_vec, 0, r)
        V[0, :] = bc_l; V[-1, :] = bc_u
        return V

    def calculate(self, option: Instrument, market: MarketEnvironment, **kwargs) -> Dict[str, Any]:
        S_vec, dt = self._generate_grid(option, market)
        V_final = self._solve_on_grid(option, market, S_vec, dt)
        price = MathUtils.linear_interp(market.S, S_vec, V_final)
    
        # 如果当前 S 已经在边界外，直接处理（安全保护）
        #TODO 安全保护的向量期权情况处理逻辑
        if market.S <= S_vec[0] or market.S >= S_vec[-1]:
            bc_l, bc_u = option.get_boundary_values(S_vec, market.T, market.r)
            price = bc_l if market.S <= S_vec[0] else bc_u
        
        if isinstance(price, np.ndarray) and price.size == 1:
            price = price.item()
        
        if np.ndim(price) == 0:
            return {'price': float(price)}
        else:
            return {'price': price.flatten()}

    
    def get_delta(self, option, market, **kwargs):
        S_vec, dt = self._generate_grid(option, market)

        if hasattr(option, 'barrier_high') and market.S >= option.barrier_high:
            return 0.0
        if hasattr(option, 'barrier_low') and market.S <= option.barrier_low:
            return 0.0
        if hasattr(option, 'barrier') and market.S >= option.barrier: 
            return 0.0
        if hasattr(option, 'ko_barrier') and market.S >= option.ko_barrier:
            return 0.0
        
        V = self._solve_on_grid(option, market, S_vec, dt)
        dS_grid = S_vec[1] - S_vec[0]
        D_grid = np.gradient(V, dS_grid, axis=0, edge_order=2)

        delta_val = MathUtils.linear_interp(market.S, S_vec, D_grid)

        if isinstance(delta_val, np.ndarray) and delta_val.size == 1:
            return delta_val.item() 
        
        return delta_val

    def get_gamma(self, option, market, **kwargs):
        S_vec, dt = self._generate_grid(option, market)

        if hasattr(option, 'barrier_high') and market.S >= option.barrier_high:
            return 0.0
        if hasattr(option, 'barrier_low') and market.S <= option.barrier_low:
            return 0.0
        if hasattr(option, 'barrier') and market.S >= option.barrier: 
            return 0.0
        if hasattr(option, 'ko_barrier') and market.S >= option.ko_barrier:
            return 0.0
        
        V = self._solve_on_grid(option, market, S_vec, dt)
        dS_grid = S_vec[1] - S_vec[0]
        D_grid = np.gradient(V, dS_grid, axis=0, edge_order=2)
        G_grid = np.gradient(D_grid, dS_grid, axis=0, edge_order=2)

        Gamma_val = MathUtils.linear_interp(market.S, S_vec, G_grid)

        if isinstance(Gamma_val, np.ndarray) and Gamma_val.size == 1:
            return Gamma_val.item() 
        
        return Gamma_val

    def get_vega(self, option, market, **kwargs):
        S_vec, dt = self._generate_grid(option, market)

        if hasattr(option, 'barrier_high') and market.S >= option.barrier_high:
            return 0.0
        if hasattr(option, 'barrier_low') and market.S <= option.barrier_low:
            return 0.0
        if hasattr(option, 'barrier') and market.S >= option.barrier: 
            return 0.0
        if hasattr(option, 'ko_barrier') and market.S >= option.ko_barrier:
            return 0.0
        
        dVol = 0.01
        m_up = market.clone(sigma=market.sigma + dVol)
        m_dn = market.clone(sigma=market.sigma - dVol)
        p_up = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_up, S_vec, dt))
        p_dn = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_dn, S_vec, dt))
        Vega_val = (p_up - p_dn) / 2

        if isinstance(Vega_val, np.ndarray) and Vega_val.size == 1:
            return Vega_val.item()
        return Vega_val

    def get_theta(self, option, market, **kwargs):
        S_vec, dt = self._generate_grid(option, market)

        if hasattr(option, 'barrier_high') and market.S >= option.barrier_high:
            return 0.0
        if hasattr(option, 'barrier_low') and market.S <= option.barrier_low:
            return 0.0
        if hasattr(option, 'barrier') and market.S >= option.barrier: 
            return 0.0
        if hasattr(option, 'ko_barrier') and market.S >= option.ko_barrier:
            return 0.0
        
        dT = 1/365
        # 锁定 dt，只改变 T
        m_fut = market.clone(T=market.T - dT)
        m_pst = market.clone(T=market.T + dT)
        
        # 为了避免时间轴错位，这里可以微调 dt 以匹配 N
        dt_fut = m_fut.T / self.N
        dt_pst = m_pst.T / self.N
        
        p_fut = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_fut, S_vec, dt_fut))
        p_pst = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_pst, S_vec, dt_pst))
        theta_val = (p_fut - p_pst) / 2

        if isinstance(theta_val, np.ndarray) and theta_val.size == 1:
            return theta_val.item()
        
        return theta_val

    def get_rho(self, option, market, **kwargs):
        S_vec, dt = self._generate_grid(option, market)

        if hasattr(option, 'barrier_high') and market.S >= option.barrier_high:
            return 0.0
        if hasattr(option, 'barrier_low') and market.S <= option.barrier_low:
            return 0.0
        if hasattr(option, 'barrier') and market.S >= option.barrier: 
            return 0.0
        if hasattr(option, 'ko_barrier') and market.S >= option.ko_barrier:
            return 0.0
        
        dR = 0.01
        m_up = market.clone(r=market.r + dR)
        m_dn = market.clone(r=market.r - dR)
        p_up = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_up, S_vec, dt))
        p_dn = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_dn, S_vec, dt))
        rho_val = (p_up - p_dn) / 2

        if isinstance(rho_val, np.ndarray) and rho_val.size == 1:
            return rho_val.item()
        
        return rho_val


    def get_volga(self, option, market, **kwargs):
        S_vec, dt = self._generate_grid(option, market)

        if hasattr(option, 'barrier_high') and market.S >= option.barrier_high:
            return 0.0
        if hasattr(option, 'barrier_low') and market.S <= option.barrier_low:
            return 0.0
        if hasattr(option, 'barrier') and market.S >= option.barrier: 
            return 0.0
        if hasattr(option, 'ko_barrier') and market.S >= option.ko_barrier:
            return 0.0
        
        dVol = 0.01
        m_up = market.clone(sigma=market.sigma + dVol)
        m_dn = market.clone(sigma=market.sigma - dVol)
        p_up = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_up, S_vec, dt))
        p_mid = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, market, S_vec, dt))
        p_dn = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_dn, S_vec, dt))
        volga_val = (p_up - 2 * p_mid + p_dn)

        if isinstance(volga_val, np.ndarray) and volga_val.size == 1:
            return volga_val.item()
        
        return volga_val

    def get_vanna(self, option, market, **kwargs):
        S_vec, dt = self._generate_grid(option, market)

        if hasattr(option, 'barrier_high') and market.S >= option.barrier_high:
            return 0.0
        if hasattr(option, 'barrier_low') and market.S <= option.barrier_low:
            return 0.0
        if hasattr(option, 'barrier') and market.S >= option.barrier: 
            return 0.0
        if hasattr(option, 'ko_barrier') and market.S >= option.ko_barrier:
            return 0.0
        
        dS_grid = S_vec[1] - S_vec[0]
        dVol = 0.01
        m_up = market.clone(sigma=market.sigma + dVol)
        m_dn = market.clone(sigma=market.sigma - dVol)
        V_up = self._solve_on_grid(option, m_up, S_vec, dt)
        V_dn = self._solve_on_grid(option, m_dn, S_vec, dt)
        D_up = np.gradient(V_up, dS_grid, axis=0, edge_order=2)
        D_dn = np.gradient(V_dn, dS_grid, axis=0, edge_order=2)
        d_up_val = MathUtils.linear_interp(market.S, S_vec, D_up)
        d_dn_val = MathUtils.linear_interp(market.S, S_vec, D_dn)
        vanna_val = (d_up_val - d_dn_val) / 2

        if isinstance(vanna_val, np.ndarray) and vanna_val.size == 1:
            return vanna_val.item()
        
        return vanna_val
    







class EventFDMEngine(FDMEngine):
    """
    事件驱动的有限差分引擎。
    继承自 FDMEngine，增加了处理离散事件（如敲出观察）的能力。
    """
    def _solve_on_grid(self, option: Instrument, market: MarketEnvironment, S_vec: np.ndarray, dt: float, is_unified_model: bool) -> np.ndarray:
        """
        修正版 FDM 网格求解器 - 兼容双轨并行 (Dual-Grid) 模式
        逻辑：
        - Unified Model: 同时求解 Active 和 Shadow 两个 PDE，并在 S <= B_in 处耦合。
        - Old Model: 单一 PDE 求解，应用静态边界。
        """
        S, T = market.S, market.T
        dS = S_vec[1] - S_vec[0]
        
        # 获取系数矩阵
        (l_lhs, d_lhs, u_lhs), (l_rhs, d_rhs, u_rhs) = self.scheme.get_coefficients(
            self.process, market, dt, dS, S_vec
        )
        
        # 获取事件信息
        event_dates_rem = option.get_event_dates(T)
        event_handler = option.get_event_handler()
        
        # 初始化payoff
        V = option.payoff(S_vec)
        if V.ndim == 1: V = V[:, np.newaxis]

        # 初始化影子网格---
        V_shadow = None
        if is_unified_model and hasattr(option, 'payoff_shadow'):
            V_shadow = option.payoff_shadow(S_vec)
            if V_shadow.ndim == 1: V_shadow = V_shadow[:, np.newaxis]

        # 注入市场环境
        if hasattr(option, 'set_market_environment'):
            option.set_market_environment(market)
            
        epsilon = 1e-9

        # 时间回溯主循环
        for i in range(self.N):
            tau_old = i * dt
            tau_new = (i + 1) * dt
            
            # -----------------------------------------------------------
            # 标准 PDE 步进
            # -----------------------------------------------------------
            rhs_vec = l_rhs[:, None] * V[:-2, :] + \
                      d_rhs[:, None] * V[1:-1, :] + \
                      u_rhs[:, None] * V[2:, :]
            
            # --- 根据模型类型决定处理方式 ---
            if is_unified_model:
                rhs_shadow = l_rhs[:, None] * V_shadow[:-2, :] + \
                             d_rhs[:, None] * V_shadow[1:-1, :] + \
                             u_rhs[:, None] * V_shadow[2:, :]
    
                bc_l, bc_u = option.get_boundary_values(S_vec, tau_new, market.r)
                
                # 将边界条件应用到两个 RHS 向量
                #TODO 后续的结构可能有物理边界差异化需求
                rhs_vec[0, :] -= l_lhs[0] * bc_l
                rhs_vec[-1, :] -= u_lhs[-1] * bc_u
                
                rhs_shadow[0, :] -= l_lhs[0] * bc_l
                rhs_shadow[-1, :] -= u_lhs[-1] * bc_u
                
                # 求解三对角矩阵
                V[1:-1, :] = MatrixUtils.solve_tridiagonal(u_lhs, d_lhs, l_lhs, rhs_vec)
                V_shadow[1:-1, :] = MatrixUtils.solve_tridiagonal(u_lhs, d_lhs, l_lhs, rhs_shadow)
                
                # 更新边界点
                V[0, :] = bc_l; V[-1, :] = bc_u
                V_shadow[0, :] = bc_l; V_shadow[-1, :] = bc_u

            else:
                # 拆分组件模型
                bc_l_new, bc_u_new = option.get_boundary_values(S_vec, tau_new, market.r)
                rhs_vec[0, :] -= l_lhs[0] * bc_l_new
                rhs_vec[-1, :] -= u_lhs[-1] * bc_u_new
                V[1:-1, :] = MatrixUtils.solve_tridiagonal(u_lhs, d_lhs, l_lhs, rhs_vec)
                V[0, :] = bc_l_new
                V[-1, :] = bc_u_new

            # -----------------------------------------------------------
            # 耦合 (Coupling) - 仅统一模型
            # -----------------------------------------------------------
            if is_unified_model:
                if hasattr(option, 'B_in'):
                    ki_mask = S_vec <= option.B_in
                    V[ki_mask] = V_shadow[ki_mask]

            # -----------------------------------------------------------
            # 事件处理 (Event Handling / Knock-out)
            # -----------------------------------------------------------
            triggered_events = [t for t in event_dates_rem if tau_old + epsilon < t <= tau_new + epsilon]
            
            if triggered_events:
                event_time_rem_from_T = triggered_events[-1]
                
                # 处理主网格事件
                V_1d = V.flatten()
                V_modified_1d = event_handler(V_1d, S_vec, event_time_rem_from_T)
                V = V_modified_1d[:, np.newaxis]
                
                # 如果存在影子网格，也应用敲出事件
                if is_unified_model and V_shadow is not None:
                    V_shadow_1d = V_shadow.flatten()
                    V_shadow_modified_1d = event_handler(V_shadow_1d, S_vec, event_time_rem_from_T)
                    V_shadow = V_shadow_modified_1d[:, np.newaxis]

        return V.flatten()
    

    def calculate(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, **kwargs) -> Dict[str, Any]:
        """
        - 如果是单个 Instrument，直接为其定价。
        - 如果是 ProductPortfolio，则分别为其每个组件定价，然后加总。
        """
        if isinstance(option, ProductPortfolio):
            option.set_market_environment(market)
            
            total_price = 0.0
            component_prices = {}
            
            for component in option.components:
                comp_result = self._calculate_single_instrument_price(component, market)
                comp_price = comp_result['price']
                total_price += comp_price
                component_prices[component.__class__.__name__] = comp_price

            return {'price': float(total_price), 'component_prices': component_prices}

        elif isinstance(option, Instrument):
            return self._calculate_single_instrument_price(option, market)
        
        else:
            raise TypeError("Input 'option' must be of type Instrument or ProductPortfolio.")


    def _calculate_single_instrument_price(self, instrument: Instrument, market: MarketEnvironment) -> Dict[str, Any]:
        """
        负责为[单个]Instrument 完成定价。
        """
        is_unified_model = hasattr(instrument, 'payoff_shadow')

        if hasattr(instrument, 'check_immediate_termination'):
            is_terminated, term_value = instrument.check_immediate_termination(market.S, market.T)
            if is_terminated:
                return {'price': float(term_value)}


        S_vec, dt = self._generate_grid(instrument, market)
        
        if not is_unified_model:
            if market.S < S_vec[0]:
                if hasattr(instrument, 'get_shadow_instrument'):
                    shadow_opt = instrument.get_shadow_instrument()
                    return self._calculate_single_instrument_price(shadow_opt, market)
                else:
                    bc_l, _ = instrument.get_boundary_values(S_vec, market.T, market.r)
                    return {'price': float(bc_l)}

            if market.S > S_vec[-1]:
                 _, bc_u = instrument.get_boundary_values(S_vec, market.T, market.r)
                 return {'price': float(bc_u)}
        
        V_final = self._solve_on_grid(instrument, market, S_vec, dt, is_unified_model)
        
        price = MathUtils.linear_interp(market.S, S_vec, V_final)
        
        if isinstance(price, np.ndarray) and price.size == 1:
            price = price.item()
            
        return {'price': float(price)}



    def _get_perturbed_price(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, perturbations: Dict[str, float]) -> float:
        """
        计算在对市场参数进行一组微扰后，产品的价格。
        Args:
            option: 要定价的产品或组合。
            market: 原始市场环境。
            perturbations: 一个字典，描述要进行的微扰。
                        例如: {'S': dS}, {'sigma': -d_sigma}, {'S': dS, 'T': -dT}              
        Returns:
            微扰后的产品价格。
        """
        if not perturbations:
            return self.calculate(option, market)['price']

        market_copy = market.clone()
        
        for param_name, d_param in perturbations.items():
            original_value = getattr(market_copy, param_name)
            setattr(market_copy, param_name, original_value + d_param)
            
        return self.calculate(option, market_copy)['price']
    



    def get_delta(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, dS_ratio=0.01) -> float:
        """
        - 如果传入的是 ProductPortfolio，强制使用整体微扰法，以保证数学正确性。
        - 如果传入的是单个 Instrument，使用高效的、基于网格梯度的“查表法”。
        """
        if isinstance(option, ProductPortfolio):
            dS = market.S * dS_ratio
            return self._get_delta_by_perturbation(option, market, dS)
        
        elif isinstance(option, Instrument):
            return self._get_delta_from_grid(option, market)
        
        else:
            raise TypeError(f"Unsupported option type for get_delta: {type(option)}")


    def _get_delta_by_perturbation(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, dS: float) -> float:
        """
        使用通用微扰框架计算 Delta。
        """
        if abs(dS) < 1e-9:
            return 0.0
        
        price_up = self._get_perturbed_price(option, market, {'S': dS})
        price_down = self._get_perturbed_price(option, market, {'S': -dS})
        
        return (price_up - price_down) / (2 * dS)


    def _get_delta_from_grid(self, instrument: Instrument, market: MarketEnvironment) -> float:
        """
        基于网格的差分法（查表法），仅适用于单个 Instrument
        """
        is_unified_model = hasattr(instrument, 'payoff_shadow')

        if hasattr(instrument, 'check_immediate_termination'):
            is_terminated, _ = instrument.check_immediate_termination(market.S, market.T)
            if is_terminated:
                return 0.0
            
        if not is_unified_model:
            # [边界外检查 - 下界/影子合约]
            if hasattr(instrument, 'barrier_low') and instrument.barrier_low is not None and market.S < instrument.barrier_low:
                if hasattr(instrument, 'get_shadow_instrument'):
                    shadow_opt = instrument.get_shadow_instrument()
                    # 递归调用
                    return self.get_delta(shadow_opt, market)
                else:
                    return 0.0
        
        S_vec, dt = self._generate_grid(instrument, market)
    
        if market.S < S_vec[0] or market.S > S_vec[-1]:
            return 0.0

        V = self._solve_on_grid(instrument, market, S_vec, dt, is_unified_model)
        
        dS_grid = S_vec[1] - S_vec[0]
        D_grid = np.gradient(V, dS_grid, axis=0, edge_order=2)
        
        delta_val = MathUtils.linear_interp(market.S, S_vec, D_grid)

        if isinstance(delta_val, np.ndarray) and delta_val.size == 1:
            return delta_val.item() 
        
        return delta_val
    


    def get_gamma(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, dS_ratio=0.01) -> float:
        """
        - 如果传入的是 ProductPortfolio，强制使用整体微扰法（3点差分）。
        - 如果传入的是单个 Instrument，使用您提供的、基于网格梯度的“查表法”。
        """
        if isinstance(option, ProductPortfolio):
            dS = market.S * dS_ratio
            return self._get_gamma_by_perturbation(option, market, dS)
        
        elif isinstance(option, Instrument):
            return self._get_gamma_from_grid(option, market)
        
        else:
            raise TypeError(f"Unsupported option type for get_gamma: {type(option)}")


    def _get_gamma_by_perturbation(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, dS: float) -> float:
        """
        使用通用微扰框架计算 Gamma
        使用标准的中心差分二阶导数公式: (V(S+dS) - 2V(S) + V(S-dS)) / dS^2
        """
        if abs(dS) < 1e-9:
            return 0.0

        price_up = self._get_perturbed_price(option, market, {'S': dS})
        price_down = self._get_perturbed_price(option, market, {'S': -dS})
        price_now = self.calculate(option, market)['price']
        
        return (price_up - 2 * price_now + price_down) / (dS ** 2)


    def _get_gamma_from_grid(self, instrument: Instrument, market: MarketEnvironment) -> float:
        """
        基于网格的 Gamma 计算（查表法），仅适用于单个 Instrument。
        """
        is_unified_model = hasattr(instrument, 'payoff_shadow')

        if hasattr(instrument, 'check_immediate_termination'):
            is_terminated, _ = instrument.check_immediate_termination(market.S, market.T)
            if is_terminated:
                return 0.0

        if not is_unified_model:
            if hasattr(instrument, 'barrier_low') and instrument.barrier_low is not None and market.S < instrument.barrier_low:
                if hasattr(instrument, 'get_shadow_instrument'):
                    shadow_opt = instrument.get_shadow_instrument()
                    return self.get_gamma(shadow_opt, market)
                else:
                    return 0.0
            
        S_vec, dt = self._generate_grid(instrument, market)
        
        V = self._solve_on_grid(instrument, market, S_vec, dt, is_unified_model)
        dS_grid = S_vec[1] - S_vec[0]
        D_grid = np.gradient(V, dS_grid, axis=0, edge_order=2)
        G_grid = np.gradient(D_grid, dS_grid, axis=0, edge_order=2)

        Gamma_val = MathUtils.linear_interp(market.S, S_vec, G_grid)

        if isinstance(Gamma_val, np.ndarray) and Gamma_val.size == 1:
            return Gamma_val.item() 
        
        return Gamma_val
    

    def get_vega(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, d_sigma=0.01) -> float:
        """
        - 如果传入的是 ProductPortfolio，强制使用整体微扰法，以保证数学正确性。
        """
        if isinstance(option, ProductPortfolio):
            return self._get_vega_by_perturbation(option, market, d_sigma)
        
        elif isinstance(option, Instrument):
            return self._get_vega_from_grid(option, market, d_sigma)
        
        else:
            raise TypeError(f"Unsupported option type for get_vega: {type(option)}")


    def _get_vega_by_perturbation(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, d_sigma: float) -> float:
        """
        使用通用微扰框架计算 Vega。
        """
        if abs(d_sigma) < 1e-9:
            return 0.0
        
        price_up = self._get_perturbed_price(option, market, {'sigma': d_sigma})
        price_down = self._get_perturbed_price(option, market, {'sigma': -d_sigma})
        
        return (price_up - price_down) / (2)


    def _get_vega_from_grid(self, instrument: Instrument, market: MarketEnvironment, d_sigma=0.01) -> float:
        """
        为单个 Instrument 计算 Vega
        """
        is_unified_model = hasattr(instrument, 'payoff_shadow')

        if hasattr(instrument, 'check_immediate_termination'):
            is_terminated, _ = instrument.check_immediate_termination(market.S, market.T)
            if is_terminated:
                return 0.0

        if not is_unified_model:
            if hasattr(instrument, 'barrier_low') and instrument.barrier_low is not None and market.S < instrument.barrier_low:
                if hasattr(instrument, 'get_shadow_instrument'):
                    shadow_opt = instrument.get_shadow_instrument()
                    return self.get_vega(shadow_opt, market)
                else:
                    return 0.0
            
        S_vec, dt_orig = self._generate_grid(instrument, market)
        
        market_up = market.clone(sigma=market.sigma + d_sigma)
        market_down = market.clone(sigma=market.sigma - d_sigma)
        
        V_up = self._solve_on_grid(instrument, market_up, S_vec, dt_orig, is_unified_model)
        V_down = self._solve_on_grid(instrument, market_down, S_vec, dt_orig, is_unified_model)

        price_up = MathUtils.linear_interp(market.S, S_vec, V_up)
        price_down = MathUtils.linear_interp(market.S, S_vec, V_down)
        
        if abs(d_sigma) < 1e-9:
            return 0.0
            
        vega_val = (price_up - price_down) / (2)
        
        return float(vega_val)
    



    def get_theta(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment) -> float:
        if isinstance(option, ProductPortfolio):
            return self._get_theta_by_perturbation(option, market)
        
        elif isinstance(option, Instrument):
            return self._get_theta_from_grid(option, market)
        
        else:
            raise TypeError(f"Unsupported option type for get_daily_theta_pnl: {type(option)}")

    def _get_theta_by_perturbation(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, dT: float = 1/365.0) -> float:
        price_future = self._get_perturbed_price(option, market, {'T': -dT})
        price_past = self._get_perturbed_price(option, market, {'T': dT})
       
        return (price_future - price_past) / 2.0


    def _get_theta_from_grid(self, instrument: Instrument, market: MarketEnvironment) -> float:
        is_unified_model = hasattr(instrument, 'payoff_shadow')

        if hasattr(instrument, 'check_immediate_termination'):
            is_terminated, _ = instrument.check_immediate_termination(market.S, market.T)
            if is_terminated:
                return 0.0

        if not is_unified_model:
            if hasattr(instrument, 'barrier_low') and instrument.barrier_low is not None and market.S < instrument.barrier_low:
                if hasattr(instrument, 'get_shadow_instrument'):
                    shadow_opt = instrument.get_shadow_instrument()
                    return self.get_daily_theta_pnl(shadow_opt, market)
                else:
                    return 0.0
                
        S_vec, dt = self._generate_grid(instrument, market)
        
        dT = 1/365
        m_fut = market.clone(T=market.T - dT)
        m_pst = market.clone(T=market.T + dT)
        
        dt_fut = m_fut.T / self.N
        dt_pst = m_pst.T / self.N
        
        p_fut = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(instrument, m_fut, S_vec, dt_fut, is_unified_model))
        p_pst = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(instrument, m_pst, S_vec, dt_pst, is_unified_model))
        theta_val = (p_fut - p_pst) / 2

        if isinstance(theta_val, np.ndarray) and theta_val.size == 1:
            return theta_val.item()
        
        return theta_val
    


    def get_rho(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, dR: float = 0.01) -> float:
        if isinstance(option, ProductPortfolio):
            return self._get_rho_by_perturbation(option, market, dR)
        
        elif isinstance(option, Instrument):
            return self._get_rho_from_grid(option, market, dR)
        
        else:
            raise TypeError(f"Unsupported option type for get_rho: {type(option)}")

    def _get_rho_by_perturbation(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, dR: float) -> float:
        if abs(dR) < 1e-9:
            return 0.0
        
        price_up = self._get_perturbed_price(option, market, {'r': dR})
        price_down = self._get_perturbed_price(option, market, {'r': -dR})
        
        return (price_up - price_down) / 2


    def _get_rho_from_grid(self, instrument: Instrument, market: MarketEnvironment, dR: float) -> float:
        is_unified_model = hasattr(instrument, 'payoff_shadow')

        if hasattr(instrument, 'check_immediate_termination'):
            is_terminated, _ = instrument.check_immediate_termination(market.S, market.T)
            if is_terminated:
                return 0.0

        if not is_unified_model:
            if hasattr(instrument, 'barrier_low') and instrument.barrier_low is not None and market.S < instrument.barrier_low:
                if hasattr(instrument, 'get_shadow_instrument'):
                    shadow_opt = instrument.get_shadow_instrument()
                    return self.get_rho(shadow_opt, market)
                else:
                    return 0.0
                
        S_vec, dt = self._generate_grid(instrument, market)
        
        dR = 0.01
        m_up = market.clone(r=market.r + dR)
        m_dn = market.clone(r=market.r - dR)
        p_up = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(instrument, m_up, S_vec, dt, is_unified_model))
        p_dn = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(instrument, m_dn, S_vec, dt, is_unified_model))
        rho_val = (p_up - p_dn) / 2

        if isinstance(rho_val, np.ndarray) and rho_val.size == 1:
            return rho_val.item()
        
        return rho_val
    



    def get_volga(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, d_sigma: float = 0.01) -> float:
        if isinstance(option, ProductPortfolio):
            return self._get_volga_by_perturbation(option, market, d_sigma)
        
        elif isinstance(option, Instrument):
            return self._get_volga_from_grid(option, market, d_sigma)
        
        else:
            raise TypeError(f"Unsupported option type for get_volga: {type(option)}")

    def _get_volga_by_perturbation(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, d_sigma: float) -> float:
        if abs(d_sigma) < 1e-9:
            return 0.0
        
        price_up = self._get_perturbed_price(option, market, {'sigma': d_sigma})
        price_down = self._get_perturbed_price(option, market, {'sigma': -d_sigma})
        price_now = self.calculate(option, market)['price']
        
        return (price_up - 2 * price_now + price_down)


    def _get_volga_from_grid(self, instrument: Instrument, market: MarketEnvironment, d_sigma: float) -> float:
        is_unified_model = hasattr(instrument, 'payoff_shadow')

        if hasattr(instrument, 'check_immediate_termination'):
            is_terminated, _ = instrument.check_immediate_termination(market.S, market.T)
            if is_terminated:
                return 0.0

        if not is_unified_model:
            if hasattr(instrument, 'barrier_low') and instrument.barrier_low is not None and market.S < instrument.barrier_low:
                if hasattr(instrument, 'get_shadow_instrument'):
                    shadow_opt = instrument.get_shadow_instrument()
                    return self.get_volga(shadow_opt, market)
                else:
                    return 0.0
                
        S_vec, dt = self._generate_grid(instrument, market)
        
        market_up = market.clone(sigma=market.sigma + d_sigma)
        market_down = market.clone(sigma=market.sigma - d_sigma)
        
        V_up = self._solve_on_grid(instrument, market_up, S_vec, dt, is_unified_model)
        V_mid = self._solve_on_grid(instrument, market, S_vec, dt, is_unified_model) 
        V_down = self._solve_on_grid(instrument, market_down, S_vec, dt, is_unified_model)
        
        price_up = MathUtils.linear_interp(market.S, S_vec, V_up)
        price_mid = MathUtils.linear_interp(market.S, S_vec, V_mid)
        price_down = MathUtils.linear_interp(market.S, S_vec, V_down)
        
        if abs(d_sigma) < 1e-9:
            return 0.0

        volga_val = (price_up - 2 * price_mid + price_down)

        return float(volga_val)




    def get_vanna(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, dS_ratio: float = 0.01, d_sigma: float = 0.01) -> float:
        if isinstance(option, ProductPortfolio):
            dS = market.S * dS_ratio
            return self._get_vanna_by_perturbation(option, market, dS, d_sigma)
        
        elif isinstance(option, Instrument):
            return self._get_vanna_from_grid(option, market, d_sigma)
        
        else:
            raise TypeError(f"Unsupported option type for get_vanna: {type(option)}")

    def _get_vanna_by_perturbation(self, option: Union[Instrument, ProductPortfolio], market: MarketEnvironment, dS: float, d_sigma: float) -> float:
        if abs(dS) < 1e-9 or abs(d_sigma) < 1e-9:
            return 0.0

        # 1. 在 (sigma + d_sigma) 的宇宙里计算 Delta
        market_up_vol = market.clone(sigma=market.sigma + d_sigma)
        price_up_vol_up_S = self._get_perturbed_price(option, market_up_vol, {'S': dS})
        price_up_vol_down_S = self._get_perturbed_price(option, market_up_vol, {'S': -dS})
        delta_up_vol = (price_up_vol_up_S - price_up_vol_down_S) / (2 * dS)

        # 2. 在 (sigma - d_sigma) 的宇宙里计算 Delta
        market_down_vol = market.clone(sigma=market.sigma - d_sigma)
        price_down_vol_up_S = self._get_perturbed_price(option, market_down_vol, {'S': dS})
        price_down_vol_down_S = self._get_perturbed_price(option, market_down_vol, {'S': -dS})
        delta_down_vol = (price_down_vol_up_S - price_down_vol_down_S) / (2 * dS)


        vanna = (delta_up_vol - delta_down_vol) / (2)

        return vanna


    def _get_vanna_from_grid(self, instrument: Instrument, market: MarketEnvironment, d_sigma: float) -> float:
        is_unified_model = hasattr(instrument, 'payoff_shadow')

        if hasattr(instrument, 'check_immediate_termination'):
            is_terminated, _ = instrument.check_immediate_termination(market.S, market.T)
            if is_terminated:
                return 0.0
        if not is_unified_model:
            if hasattr(instrument, 'barrier_low') and instrument.barrier_low is not None and market.S < instrument.barrier_low:
                if hasattr(instrument, 'get_shadow_instrument'):
                    shadow_opt = instrument.get_shadow_instrument()
                    return self.get_vanna(shadow_opt, market)
                else:
                    return 0.0
                
        S_vec, dt = self._generate_grid(instrument, market)
        
        dS_grid = S_vec[1] - S_vec[0]
        dVol = 0.01
        m_up = market.clone(sigma=market.sigma + dVol)
        m_dn = market.clone(sigma=market.sigma - dVol)
        V_up = self._solve_on_grid(instrument, m_up, S_vec, dt, is_unified_model)
        V_dn = self._solve_on_grid(instrument, m_dn, S_vec, dt, is_unified_model)
        D_up = np.gradient(V_up, dS_grid, axis=0, edge_order=2)
        D_dn = np.gradient(V_dn, dS_grid, axis=0, edge_order=2)
        d_up_val = MathUtils.linear_interp(market.S, S_vec, D_up)
        d_dn_val = MathUtils.linear_interp(market.S, S_vec, D_dn)
        vanna_val = (d_up_val - d_dn_val) / 2

        if isinstance(vanna_val, np.ndarray) and vanna_val.size == 1:
            return vanna_val.item()
        
        return vanna_val



