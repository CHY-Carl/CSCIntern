import numpy as np
from scipy.stats import norm, qmc
from scipy.linalg import solve_banded
from abc import ABC, abstractmethod

# ==========================================================
# ======= 1. Math Utilities (数学工具库) ===================
# ==========================================================
class MathUtils:
    """提供底层的数学函数封装，便于未来替换实现或统一管理精度"""
    
    @staticmethod
    def norm_cdf(x):
        """标准正态分布累积概率函数 N(x)"""
        return norm.cdf(x)

    @staticmethod
    def norm_pdf(x):
        """标准正态分布概率密度函数 n(x)"""
        return norm.pdf(x)
    
    @staticmethod
    def linear_interp(x_target, x_grid, y_grid):
        """
        线性插值封装 (支持向量化)
        y_grid 可以是 (M,) 或 (M, K)
        """
        if y_grid.ndim == 1:
            return np.interp(x_target, x_grid, y_grid)
        else:
            # 对每一列进行插值 (针对 Series Option 的矩阵)
            return np.apply_along_axis(lambda v: np.interp(x_target, x_grid, v), 0, y_grid)

# ==========================================================
# ======= 2. Matrix Utilities (矩阵/求解器) =================
# ==========================================================
class MatrixUtils:
    """FDM 核心求解器封装"""
    
    @staticmethod
    def solve_tridiagonal(upper, diag, lower, rhs):
        """
        求解三对角方程组 A * x = rhs (支持向量化 rhs)
        upper, diag, lower: 一维数组 (长度 N)
        """
        N = len(diag)
        ab = np.zeros((3, N))
        
        # 构造 scipy.linalg.solve_banded 需要的带状矩阵格式
        # Row 0: Upper diagonal (从第2个元素开始有效)
        ab[0, 1:] = upper[:-1]
        # Row 1: Main diagonal
        ab[1, :]  = diag
        # Row 2: Lower diagonal (前 N-1 个元素有效)
        ab[2, :-1] = lower[1:]
        
        return solve_banded((1, 1), ab, rhs, check_finite=False)

# ==========================================================
# ======= 3. RNG Utilities (随机数生成器) ===================
# ==========================================================
class RandomNumberGenerator(ABC):
    """随机数生成器接口"""
    @abstractmethod
    def get_gaussian_matrix(self, n_sims: int, n_steps: int, seed: int = None) -> np.ndarray:
        pass

class StandardRNG(RandomNumberGenerator):
    """标准伪随机数生成器"""
    def __init__(self, use_antithetic=True):
        self.use_antithetic = use_antithetic
        
    def get_gaussian_matrix(self, n_sims, n_steps, seed=None):
        rng = np.random.default_rng(seed)

        if self.use_antithetic:
            # 生成一半，取反拼接
            n_half = int(n_sims / 2)
            if n_sims % 2 != 0: 
                raise ValueError("Antithetic sampling requires even n_sims")
                
            Z_half = rng.standard_normal((n_half, n_steps))
            return np.concatenate((Z_half, -Z_half))
        else:
            return rng.standard_normal((n_sims, n_steps))

class SobolRNG(RandomNumberGenerator):
    """Sobol 序列生成器 (Quasi-MC)"""
    def __init__(self, scramble=True):
        self.scramble = scramble
        
    def get_gaussian_matrix(self, n_sims, n_steps, seed=None):
        # Sobol 序列长度建议为 2 的幂次
        m = int(np.ceil(np.log2(n_sims)))
        n_actual = 2**m
        
        sampler = qmc.Sobol(d=n_steps, scramble=self.scramble, seed=seed)
        uniforms = sampler.random(n=n_actual)
        
        # Inverse Transform Sampling
        Z = norm.ppf(uniforms)
        
        # 返回全部生成的
        return Z


# ==========================================================
# ======= 4. SharkFin Analytical Solution ===================
# ==========================================================
class SharkFinFormulas:
    """
    鲨鱼鳍期权 (Up-and-Out Call) 定价引擎。
    包含解析解及离散观察调整 (BGK Correction)。
    """

    @staticmethod
    def _n_cdf(x):
        return norm.cdf(x)

    @staticmethod
    def up_and_out_call(S, K, H, T, r, sigma, rebate=0, q=0):
        """
        [连续观察] 向上敲出看涨期权定价 (Up-and-Out Call).
        
        参数:
        S     : 标的资产当前价格
        K     : 行权价
        H     : 障碍价格 (Up Barrier)
        T     : 剩余期限 (年)
        r     : 无风险利率
        sigma : 波动率
        q     : 连续分红率 (默认0.0)
        rebate: 敲出补偿金 (Rebate), 假设为到期支付 (Payment at Maturity)
        """
        # --- 1. 边界情况处理 ---
        if S >= H:
            return rebate * np.exp(-r * T)
        
        if K >= H:
            return SharkFinFormulas._calc_rebate_only(S, H, T, r, sigma, rebate, q)

        if T <= 1e-9:
            return max(S - K, 0.0) if S < H else rebate

        # --- 2. 基础参数 ---
        sqrt_T = np.sqrt(T)
        vol = sigma * sqrt_T
        b = r - q  # Cost of carry (携带成本)
        
        # mu: 对应几何布朗运动 drift 部分的参数 (b - 0.5*sigma^2) / sigma^2
        mu = (b - 0.5 * sigma**2) / sigma**2
        
        # 3. 计算 8 个关键的 d 参数
        d1 = (np.log(S/K) + (b + 0.5*sigma**2)*T) / vol
        d2 = d1 - vol
        
        d3 = (np.log(S/H) + (b + 0.5*sigma**2)*T) / vol
        d4 = d3 - vol
        
        d5 = (np.log(H**2/(S*K)) + (b + 0.5*sigma**2)*T) / vol
        d6 = d5 - vol
        
        d7 = (np.log(H/S) + (b + 0.5*sigma**2)*T) / vol
        d8 = d7 - vol

        # 4. 组装公式
        # Term 1: 标准价差期权 (封顶看涨)
        term_vanilla_spread = (S * np.exp((b-r)*T) * (SharkFinFormulas._n_cdf(d1) - SharkFinFormulas._n_cdf(d3)) - 
                               K * np.exp(-r*T)    * (SharkFinFormulas._n_cdf(d2) - SharkFinFormulas._n_cdf(d4)))

        # Term 2: 镜像调整项
        factor_S = (H/S)**(2*(mu + 1))
        factor_K = (H/S)**(2*mu)
        
        term_reflection = (factor_S * S * np.exp((b-r)*T) * (SharkFinFormulas._n_cdf(d5) - SharkFinFormulas._n_cdf(d7)) - 
                           factor_K * K * np.exp(-r*T)    * (SharkFinFormulas._n_cdf(d6) - SharkFinFormulas._n_cdf(d8)))

        price_option = term_vanilla_spread - term_reflection

        # 5. Rebate 
        price_rebate = 0
        if rebate > 0:
            price_rebate = SharkFinFormulas._calc_rebate_only(S, H, T, r, sigma, rebate, q)

        return max(price_option + price_rebate, 0)

    @staticmethod
    def _calc_rebate_only(S, H, T, r, sigma, rebate, q=0):
        """
        计算纯 Rebate 部分的价值 (Payment at Maturity).
        即: Rebate * exp(-rT) * P(触碰 H)
        """
        if rebate <= 0: return 0
        
        b = r - q
        vt = sigma * np.sqrt(T)
        mu_nu = (b - 0.5 * sigma**2)
        dist = np.log(H/S)
        
        z1 = (-dist + mu_nu * T) / vt
        z2 = (-dist - mu_nu * T) / vt
        
        gamma = 2 * mu_nu / (sigma**2)
        
        prob_hit = SharkFinFormulas._n_cdf(z1) + (H/S)**gamma * SharkFinFormulas._n_cdf(z2)
        
        return rebate * np.exp(-r*T) * np.clip(prob_hit, 0.0, 1.0)

    @staticmethod
    def up_and_out_call_discrete(S, K, H, T, r, sigma, rebate=0, dt=1/252, q=0):
        """
        [离散观察] 使用 Broadie-Glasserman-Kohl (1997) 连续性修正。
        将障碍 H 向外移动 (Up Barrier -> 调高 H)。
        """
        # BGK 常数 beta = -zeta(0.5) / sqrt(2*pi) ≈ 0.5826
        beta = 0.5826
        
        # 调整障碍 
        H_shifted = H * np.exp(beta * sigma * np.sqrt(dt))
        
        # 如果调整后的障碍比 S 还小，则维持原状或判定敲出，这里直接代入
        return SharkFinFormulas.up_and_out_call(S, K, H_shifted, T, r, sigma, rebate, q)



# ==========================================================
# ======= 5. Date Utils ==============
# ==========================================================

import numpy as np
import pandas as pd
from datetime import datetime, date

class DateUtils:
    """
    日期处理工具类。
    职责：计算年化时间 T，处理交易日历。
    """
    
    @staticmethod
    def to_date(d):
        """将字符串或 timestamp 转换为 datetime.date 对象"""
        if isinstance(d, str):
            return pd.to_datetime(d).date()
        elif isinstance(d, pd.Timestamp):
            return d.date()
        elif isinstance(d, datetime):
            return d.date()
        return d

    @staticmethod
    def year_fraction(start_date, end_date, day_count='ACT/365'):
        """
        计算两个日期之间的年化时间 T。
        目前支持: 'ACT/365', 'ACT/252' (工作日)
        """
        start = DateUtils.to_date(start_date)
        end = DateUtils.to_date(end_date)
        
        if start >= end:
            return 0.0
            
        if day_count == 'ACT/365':
            # 实际天数 / 365 (日历日)
            delta = (end - start).days
            return delta / 365.0
            
        elif day_count == 'ACT/252':
            #TODO 后续引入交易所日历时需完善这部分逻辑
            # 交易日数 / 252
            bus_days = len(pd.bdate_range(start, end)) - 1 # 减1是因为由T日到T+1日算1天
            return max(0.0, bus_days / 252.0)
            
        else:
            raise ValueError(f"Unknown day count convention: {day_count}")


# ==========================================================
# ======= 6. Plot Utils ========================
# ==========================================================

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

class PlotUtils:
    """
    可视化工具类。
    职责：接收回测结果 DataFrame，绘制标准的 P&L 分析图表。
    """
    
    @staticmethod
    def plot_hedging_results(history_df, initial_cash, show_plot=True):
        # 设置风格
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        df = history_df.copy()

        if 'PnL' not in df.columns:
            df['PnL'] = df['NAV'] - initial_cash
        
        df['Daily_PnL'] = df['NAV'].diff().fillna(0)
            
        final_pnl = df['PnL'].iloc[-1]
        
        # 创建画布
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # 图1: 标的走势 vs 对冲仓位
        ax1 = axes[0]
        ax1.plot(df['Date'], df['Spot'], label='Spot Price', color='blue')
        ax1.set_ylabel('Index Points')
        ax1.legend(loc='upper left')
        
        ax1_r = ax1.twinx()
        ax1_r.plot(df['Date'], df['Units'], label='Hedge Position (Lots)', color='orange', linestyle='--')
        ax1_r.set_ylabel('Futures Lots')
        ax1_r.legend(loc='upper right')
        ax1.set_title('Market & Hedge Position')
        
        # 图2: 账户净值 NAV
        ax2 = axes[1]
        ax2.plot(df['Date'], df['NAV'], label='Net Asset Value', color='green')

        ax2.set_ylabel('Account Value')
        ax2.set_title('Account NAV')
        ax2.legend()

        # 图3: 累计 PnL
        ax3 = axes[2]
        ax3.fill_between(df['Date'], df['PnL_Strategy'], 0, where=(df['PnL_Strategy']>=0), facecolor='red', alpha=0.3)
        ax3.fill_between(df['Date'], df['PnL_Strategy'], 0, where=(df['PnL_Strategy']<0), facecolor='green', alpha=0.3)
        
        # 画线
        ax3.plot(df['Date'], df['PnL_Strategy'], label='Strategy P&L (Excl. Interest)', color='black', linewidth=1.5)
        ax3.plot(df['Date'], df['PnL_Total'], label='Total P&L (Incl. Interest)', color='gray', linestyle='--', alpha=0.6)
        
        ax3.set_ylabel('Profit / Loss')
        ax3.legend(loc='upper left')
        ax3.set_title(f'Cumulative P&L (Final: {final_pnl:,.0f})')

        # 图4: 每日 PnL
        ax4 = axes[3]
        ax4.bar(df['Date'], df['Daily_PnL'], color=np.where(df['Daily_PnL']>0, 'red', 'green'))
        ax4.set_ylabel('Daily P&L')
        ax4.set_title('Daily Profit / Loss')

        
        plt.tight_layout()

        if show_plot:
            plt.show()
            return None # 或者返回 fig
        else:
            return fig



