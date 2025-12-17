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
    #TODO: 实现隐式格式
    pass


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
            S_max = option.barrier
        else:
            #TODO: 其他SDE条件下的统计 S_max 计算
            log_S0 = np.log(S)
            drift = (r - 0.5 * sigma**2) * T
            vol_T = sigma * np.sqrt(T)
            S_stat_max = np.exp(log_S0 + drift + self.std_mult * vol_T)
            
            K_val = 0
            if hasattr(option, 'K'):
                K_val = np.max(option.K) if not np.isscalar(option.K) else option.K
            S_max = max(S_stat_max, K_val * 1.1)
            
        S_vec = np.linspace(0, S_max, self.M + 1)
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
            # V[1:-1, :] = MatrixUtils.solve_tridiagonal(u_lhs[:-1], d_lhs, l_lhs[1:], rhs_vec)
            
        bc_l, bc_u = option.get_boundary_values(S_vec, 0, r)
        V[0, :] = bc_l; V[-1, :] = bc_u
        return V

    def calculate(self, option: Instrument, market: MarketEnvironment) -> Dict[str, Any]:
        S_vec, dt = self._generate_grid(option, market)

        # # --- 抓鬼代码 ---
        # print(f"!!! CRITICAL DEBUG !!!")
        # print(f"Inputs: S={market.S}, K={option.K}, T={market.T}, r={market.r}, sigma={market.sigma}")
        # print(f"Params: M={self.M}, N={self.N}, dt={dt}")
        # print(f"Grid: S_max={S_vec[-1]}, len(S_vec)={len(S_vec)}")
        # # ----------------

        V_final = self._solve_on_grid(option, market, S_vec, dt)
        price = MathUtils.linear_interp(market.S, S_vec, V_final)
        
        if np.isscalar(option.K):
            return {'price': float(price)}
        else:
            return {'price': price.flatten()}

    # --- Greeks Overrides (Grid Locking) ---
    # 确保这些方法都在类内部定义
    
    def get_delta(self, option, market):
        S_vec, dt = self._generate_grid(option, market)
        V = self._solve_on_grid(option, market, S_vec, dt)
        dS_grid = S_vec[1] - S_vec[0]
        D_grid = np.gradient(V, dS_grid, axis=0, edge_order=2)
        return MathUtils.linear_interp(market.S, S_vec, D_grid)

    def get_gamma(self, option, market):
        S_vec, dt = self._generate_grid(option, market)
        V = self._solve_on_grid(option, market, S_vec, dt)
        dS_grid = S_vec[1] - S_vec[0]
        D_grid = np.gradient(V, dS_grid, axis=0, edge_order=2)
        G_grid = np.gradient(D_grid, dS_grid, axis=0, edge_order=2)
        return MathUtils.linear_interp(market.S, S_vec, G_grid)

    def get_vega(self, option, market):
        S_vec, dt = self._generate_grid(option, market)
        dVol = 0.01
        m_up = market.clone(sigma=market.sigma + dVol)
        m_dn = market.clone(sigma=market.sigma - dVol)
        p_up = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_up, S_vec, dt))
        p_dn = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_dn, S_vec, dt))
        return (p_up - p_dn) / 2

    def get_theta(self, option, market):
        S_vec, dt = self._generate_grid(option, market)
        dT = 1/365
        # 锁定 dt，只改变 T
        m_fut = market.clone(T=market.T - dT)
        m_pst = market.clone(T=market.T + dT)
        
        # 为了避免时间轴错位，这里可以微调 dt 以匹配 N
        dt_fut = m_fut.T / self.N
        dt_pst = m_pst.T / self.N
        
        p_fut = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_fut, S_vec, dt_fut))
        p_pst = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_pst, S_vec, dt_pst))
        return (p_fut - p_pst) / 2

    def get_rho(self, option, market):
        S_vec, dt = self._generate_grid(option, market)
        dR = 0.01
        m_up = market.clone(r=market.r + dR)
        m_dn = market.clone(r=market.r - dR)
        p_up = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_up, S_vec, dt))
        p_dn = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_dn, S_vec, dt))
        return (p_up - p_dn) / 2

    def get_volga(self, option, market):
        S_vec, dt = self._generate_grid(option, market)
        dVol = 0.01
        m_up = market.clone(sigma=market.sigma + dVol)
        m_dn = market.clone(sigma=market.sigma - dVol)
        p_up = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_up, S_vec, dt))
        p_mid = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, market, S_vec, dt))
        p_dn = MathUtils.linear_interp(market.S, S_vec, self._solve_on_grid(option, m_dn, S_vec, dt))
        return (p_up - 2 * p_mid + p_dn)

    def get_vanna(self, option, market):
        S_vec, dt = self._generate_grid(option, market)
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
        return (d_up_val - d_dn_val) / 2