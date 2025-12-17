import sys
sys.path.append('../../')
from PricingLib.Base.BaseLayer import PricingEngine, Instrument, MarketEnvironment
from PricingLib.Base.Utils import MathUtils, MatrixUtils


from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any



class FDMCoefficients(ABC):
    """策略接口：只负责计算矩阵系数 a, b, c"""
    @abstractmethod
    def get_coefficients(self, dt, sigma, r, j_vec):
        """返回 LHS(左边) 和 RHS(右边) 的系数"""
        pass

class CNScheme(FDMCoefficients):
    def get_coefficients(self, dt, sigma, r, j):
        a = 0.25 * dt * (sigma**2 * j**2 - r * j)
        b = -0.5 * dt * (sigma**2 * j**2 + r)
        c = 0.25 * dt * (sigma**2 * j**2 + r * j)
        
        # LHS: 1-b, -c, -a
        lhs_coeffs = (-a, 1 - b, -c) # lower, diag, upper
        # RHS: 1+b, c, a
        rhs_coeffs = (a, 1 + b, c)   # lower, diag, upper
        return lhs_coeffs, rhs_coeffs

class ImplicitScheme(FDMCoefficients):
    """完全隐式法 (Fully Implicit)"""
    def get_coefficients(self, dt, sigma, r, j):
        # 隐式法: V_i = a V_{i-1} + b V_i + c V_{i+1} (at t) + V(t+1)
        # 系数通常是:
        a = 0.5 * dt * (r * j - sigma**2 * j**2)
        b = 1 + dt * (sigma**2 * j**2 + r)
        c = -0.5 * dt * (sigma**2 * j**2 + r * j)
        
        # LHS: b, a, c (注意符号约定，通常移项到左边)
        # 修正标准形式: -a V_{j-1} + (1+...) V_j - c V_{j+1} = V_{next}
        
        alpha = 0.5 * dt * (sigma**2 * j**2 - r * j)
        beta  = 0.5 * dt * (sigma**2 * j**2 + r * j)
        
        # LHS: -alpha, 1 + dt*sigma^2*j^2 + dt*r, -beta
        lhs_lower = -alpha
        lhs_diag  = 1 + dt * (sigma**2 * j**2 + r)
        lhs_upper = -beta
        
        # RHS: Identity (只包含 V_next 本身)
        rhs_lower = np.zeros_like(j, dtype=float)
        rhs_diag  = np.ones_like(j, dtype=float)
        rhs_upper = np.zeros_like(j, dtype=float)
        
        return (lhs_lower, lhs_diag, lhs_upper), (rhs_lower, rhs_diag, rhs_upper)


class FDMEngine(PricingEngine):
    def __init__(self, M_space=100, N_time=100, std_mult=5.0, scheme_type='CN'):
        self.M = int(M_space)
        self.N = int(N_time)
        self.std_mult = std_mult
        self.scheme = ImplicitScheme() if scheme_type == 'Implicit' else CNScheme()

    def _generate_grid(self, option: Instrument, market: MarketEnvironment):
        S, T, r, sigma = market.S, market.T, market.r, market.sigma
        
        if hasattr(option, 'barrier'):
            S_max = option.barrier
        else:
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
        
        j = np.arange(1, self.M)
        (l_lhs, d_lhs, u_lhs), (l_rhs, d_rhs, u_rhs) = self.scheme.get_coefficients(dt, sigma, r, j)
        
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