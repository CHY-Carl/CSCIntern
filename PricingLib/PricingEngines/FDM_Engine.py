import sys
sys.path.append('../../')
from PricingLib.Base.BaseLayer import PricingEngine, Instrument, MarketEnvironment, StochasticProcess
from PricingLib.Base.Utils import MathUtils, MatrixUtils


from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any



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
        else:
            S_min = 0.0
            #TODO: 其他SDE条件下的统计 S_max 计算
            log_S0 = np.log(S)
            drift = (r - 0.5 * sigma**2) * T
            vol_T = sigma * np.sqrt(T)
            S_stat_max = np.exp(log_S0 + drift + self.std_mult * vol_T)
            
            K_val = 0
            if hasattr(option, 'K'):
                K_val = np.max(option.K) if not np.isscalar(option.K) else option.K
            S_max = max(S_stat_max, K_val * 1.1)
            
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