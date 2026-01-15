import numpy as np
from typing import List, Tuple, Callable
from functools import partial
from ..Base.BaseLayer import Instrument, MarketEnvironment
from ..PricingEngines.FDM_Engine import FDMEngine, EventFDMEngine
from ..Processes.GBM import GeometricBrownianMotion

import numpy as np
from typing import List, Tuple, Callable, Optional
from functools import lru_cache

from ..Base.BaseLayer import Instrument 

class SnowballOption(Instrument):
    """
    雪球结构期权产品类 (适用于 Monte Carlo 引擎)。
    
    [V5 - 存续期逻辑修正版]
    1. 修正了存续期产品的票息计算逻辑，正确区分 '剩余期限' 和 '总期限'。
    2. 参数接口优化：T_rem 为可选，默认为 None (即等于 T_total)。
    3. 内部计算基于 1.0 单位面值。
    """
    
    def __init__(self, S0: float, knock_out_barrier: float, knock_in_barrier: float, coupon_rate: float,
                 rebate_rate: float, T_total: float, knock_out_obs_indices: List[int], knock_in_obs_indices: List[int],
                 dt: float, T_rem: float = None, has_knocked_in: bool = False):
        """
        初始化雪球产品 (单位面值)。

        Args:
            S0 (float): 期初价格。
            knock_out_barrier (float): 敲出价格。
            knock_in_barrier (float): 敲入价格。
            coupon_rate_annual (float): 年化票息率 (用于敲出)。
            rebate_rate_annual (float): 年化回报率 (用于未敲入到期)。
            T_total (float): 产品原始总期限（年）。
            knock_out_obs_indices (List[int]): 敲出观察日对应的模拟步数索引列表。
            knock_in_obs_indices (List[int]): 敲入观察日对应的模拟步数索引列表。
            dt (float): 每个模拟步长代表的年化时间。
            T_rem (float, optional): 距离到期日的剩余时间。默认为 None (即新发产品，等于 T_total)。
            has_knocked_in (bool, optional): 在模拟开始前是否已经发生过敲入。默认为 False。
        """
        self.S0 = S0
        self.B_out = knock_out_barrier
        self.B_in = knock_in_barrier
        self.C_annual = coupon_rate
        self.R_annual = rebate_rate
        
        # [核心逻辑] 处理时间参数
        self.T_total = T_total
        if T_rem is None:
            self.T_rem = T_total # 新发产品
        else:
            self.T_rem = T_rem   # 存续期产品
            
        super().__init__(T=self.T_rem)
        
        self.ko_indices = knock_out_obs_indices
        self.ki_indices = knock_in_obs_indices
        self.dt = dt
        self.N = 1.0 
        
        # 历史状态
        self.is_already_knocked_in = has_knocked_in
        
        # 回测状态
        self.is_active = True
        self.knock_in_triggered = has_knocked_in
        
    def is_path_dependent(self) -> bool:
        return True

    def process_one_path(self, price_path: np.ndarray) -> Tuple[float, float]:
        """
        处理单条路径，返回【单位】支付。
        """
        
        # --- 1. 检查敲出 ---
        for t_idx in self.ko_indices:
            if price_path[t_idx] >= self.B_out:
                # settlement_time_from_now: 从【今天】算起的敲出时间 (用于折现)
                # total_holding_time: 从【起息日】算起的总持有时间 (用于算票息)
                # 计算公式：总持有时间 = (总期限 - 剩余期限) + 从今天过的的时间
                settlement_time_from_now = t_idx * self.dt
                time_elapsed_already = self.T_total - self.T_rem
                total_holding_time = time_elapsed_already + settlement_time_from_now
                
                # 支付 = 1.0 * (1 + C * 总持有时间)
                payoff = self.N * (1 + self.C_annual * total_holding_time)
                
                return payoff, settlement_time_from_now

        # --- 2. 处理到期 ---
        is_future_knocked_in = False
        if self.ki_indices:
            path_at_ki_dates = price_path[self.ki_indices]
            min_price_on_path = np.min(path_at_ki_dates)
            is_future_knocked_in = min_price_on_path <= self.B_in
        
        final_knock_in_status = self.is_already_knocked_in or is_future_knocked_in
        
        settlement_time_from_now = self.T_rem
        
        if not final_knock_in_status:
            # 未敲入: 1.0 * (1 + R * 总期限)
            # 注意：全期回报通常是基于总期限计算的
            payoff = self.N * (1 + self.R_annual * self.T_total)
        else:
            # 敲入: 1.0 * min(1, ST/S0)
            final_price = price_path[-1]
            payoff = self.N * min(1, final_price / self.S0)
            
        return payoff, settlement_time_from_now

    #TODO DeltaHedge时需要更新update_status 和 get_residual_value 方法
    def update_status(self, S: float) -> bool:
        if self.is_active and not self.knock_in_triggered:
            if S <= self.B_in:
                self.knock_in_triggered = True
        return self.is_active

    def get_residual_value(self, S: float, T_rem: float, r: float) -> float:
        return 0.0
        
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        if prices.ndim == 1:
            final_payoff, _ = self.process_one_path(prices)
            return final_payoff
        else:
            raise NotImplementedError("Use engine logic.")
    
    def get_critical_points(self) -> list:
        return [self.B_out, self.B_in]
        

class SmallSnowball(Instrument):
    """
    小雪球(Monte Carlo 版本)
    [逻辑定义]
    这是一个纯粹的障碍二元结构 (Digital Barrier / OTU)：
    1. 敲出 (Knock-Out): 在观察日 S >= B_out，立即终止，获得 (1 + Coupon * 持有时间)。
    2. 未敲出 (No KO): 如果持有到期从未敲出，支付 0.0 (一分钱没有)。
    """
    
    def __init__(self, S0: float, knock_out_barrier: float, coupon_rate: float, T_total: float,
                 knock_out_obs_indices: List[int], dt: float, T_rem: float = None):
        self.S0 = S0
        self.B_out = knock_out_barrier
        self.C_annual = coupon_rate
        self.T_total = T_total
        self.T_rem = T_total if T_rem is None else T_rem
            
        super().__init__(T=self.T_rem)
        self.ko_indices = knock_out_obs_indices
        self.dt = dt
        self.N = 1.0 
        
    def is_path_dependent(self) -> bool:
        return True

    def process_one_path(self, price_path: np.ndarray) -> Tuple[float, float]:
        # --- 1. 检查提前敲出 (Early Knock-Out) ---
        for t_idx in self.ko_indices:
            if t_idx < len(price_path) and price_path[t_idx] >= self.B_out:
                # 结算时间 (从现在起)
                settlement_time_from_now = t_idx * self.dt
                time_elapsed_already = self.T_total - self.T_rem
                total_holding_time = time_elapsed_already + settlement_time_from_now
                
                payoff = self.N * (1 + self.C_annual * total_holding_time)
                
                return payoff, settlement_time_from_now

        # --- 2. 处理到期 (Maturity Check) ---
        
        final_price = price_path[-1]
        if final_price >= self.B_out:
            settlement_time_from_now = self.T_rem
            payoff = self.N * (1 + self.C_annual * self.T_total)
            return payoff, settlement_time_from_now

        # --- 3. 彻底未敲出 (No Touch) ---
        payoff = 0.0
        settlement_time_from_now = self.T_rem
            
        return payoff, settlement_time_from_now

    # --- 辅助接口 ---
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        if prices.ndim == 1:
            final_payoff, _ = self.process_one_path(prices)
            return final_payoff
        else:
            raise NotImplementedError("Use engine logic.")
    
    def get_critical_points(self) -> list:
        return [self.B_out]
         



class KnockOutCouponComponent(Instrument):
    """
    雪球组件一：向上敲出票息 (One-touch Up / OTU)。
    
    职责：
    1. 负责计算所有敲出情形下的收益 (票息)。
    2. 求解域: [0, S_max] (全域)。
    3. 互斥逻辑: 当 S >= B_out 且为观察日时，立即兑付。
       (非观察日的高价区不终止，需计算回撤风险)
    """
    
    def __init__(self, knock_out_barrier: float, coupon_rate: float, T_total: float, observation_months: List[int],
                 T_rem: float = None):
        """
        Args:
            knock_out_barrier: 敲出障碍价 (B_out)。
            coupon_rate: 年化票息率 (C)。
            T_total: 产品原始总期限 (年)。
            observation_months: 观察月份列表 (e.g. [3, ..., 24])。
            T_rem: 剩余期限。
        """
        self.T_total = T_total
        self.T_rem = T_total if T_rem is None else T_rem
        
        super().__init__(T=self.T_rem)
        
        self.B_out = knock_out_barrier
        self.C_annual = coupon_rate
        self.obs_months = observation_months
        self.N = 1.0

    # --- 网格定义 ---
    
    @property
    def ko_barrier(self) -> float:
        return self.B_out

    def get_critical_points(self):
        return [self.B_out]
        
    def is_path_dependent(self) -> bool:
        return True

    def payoff(self, S_vec: np.ndarray) -> np.ndarray:
        """
        到期 Payoff。
        如果到期日也是观察日，且 S >= B_out，支付累积票息。
        """
        payoff_vec = np.zeros_like(S_vec)
        
        # 检查到期日 (T_total) 是否在观察月中
        last_month = int(round(self.T_total * 12))
        
        if last_month in self.obs_months:
            knock_out_mask = (S_vec >= self.B_out)
            val = 1.0 * (1 + self.C_annual * self.T_total)
            payoff_vec[knock_out_mask] = val
            
        return payoff_vec

    # --- 立即终止逻辑 (处理当前即敲出) ---

    def check_immediate_termination(self, S: float, t_rem: float) -> Tuple[bool, float]:
        """
        检查是否需要立即终止计算。
        逻辑：
        只有当 S >= B_out **且** 当前时刻 (t_rem) 恰好是观察日时，才立即返回值。
        否则 (即使 S很高但不是观察日)，必须跑 FDM 计算回撤风险。
        """
        if S >= self.B_out:
            # 检查 t_rem 是否对应某个观察日
            time_elapsed = self.T_total - t_rem
            # 允许微小的浮点误差 (例如 1天)
            epsilon = 1e-5 
            
            for m in self.obs_months:
                t_obs = m / 12.0
                if abs(time_elapsed - t_obs) < epsilon:
                    payoff_val = 1.0 * (1 + self.C_annual * t_obs)
                    return True, payoff_val
                    
        return False, 0.0

    # --- 边界条件 (处理 S_max) ---

    def get_boundary_values(self, S_vec: np.ndarray, t_rem: float, r: float) -> Tuple[float, float]:
        """
        计算 FDM 边界值。
        
        Lower (S=0): 0.0。
        Upper (S=S_max): 假设必定在【最近的下一个】观察日敲出。
        """
        lower_val = 0.0
        upper_val = 0.0
        
        # 计算当前模拟走到的绝对时刻 (从起息日开始)
        current_time_from_start = self.T_total - t_rem
        
        # 在预定义的观察月列表中，寻找最近的一个未来观察日
        next_obs_time = None
        epsilon = 1e-5  # 容差
        
        for m in self.obs_months:
            t_obs = m / 12.0
            # 找到第一个“尚未过去”的观察日 (>= 当前时刻)
            if t_obs >= current_time_from_start - epsilon:
                next_obs_time = t_obs
                break
        
        # 计算边界值
        if next_obs_time is not None:
            # 计算该观察日敲出时的绝对收益 (本金 + 累积票息)
            ko_payoff = 1.0 * (1 + self.C_annual * next_obs_time)
            
            # 计算折现时间：从观察日折现回当前模拟时刻
            dt_discount = next_obs_time - current_time_from_start
            if dt_discount > 0:
                upper_val = ko_payoff * np.exp(-r * dt_discount)
            else:
                upper_val = ko_payoff
                
        return lower_val, upper_val
    
    def get_event_dates(self, current_T_rem: float) -> List[float]:
        """
        计算观察日对应的【剩余期限 (Time to Maturity)】
        返回每个观察日发生时，距离到期日 T_total 还有多少时间。
        t_rem = T_total - t_observation_from_start
        """
        # 计算当前的绝对时刻 (从起息日开始算)
        current_abs_time = self.T_total - current_T_rem
        
        events_rem = []
        for m in self.obs_months:
            t_obs_from_start = m / 12.0
            
            # 筛选未来的观察日
            if t_obs_from_start > current_abs_time:
                
                # 转换为 "剩余时间" (Time to Maturity)
                t_maturity_rem = self.T_total - t_obs_from_start
                
                # 4. 只有当这个剩余时间在模拟窗口内，才加入事件列表
                if 0 < t_maturity_rem <= current_T_rem:
                    events_rem.append(t_maturity_rem)
        
        return sorted(events_rem)



    def get_event_handler(self) -> Callable:
        """
        观察日处理：如果 S >= B_out，支付票息。
        """
        return partial(self._knock_out_handler, 
                       barrier=self.B_out, 
                       coupon=self.C_annual,
                       T_total=self.T_total)

    @staticmethod
    def _knock_out_handler(V: np.ndarray, S: np.ndarray, t_event_rem: float, 
                           barrier: float, coupon: float, T_total: float) -> np.ndarray:
        
        knock_out_mask = (S >= barrier)
        time_from_start = T_total - t_event_rem
        payoff_value = 1.0 * (1 + coupon * time_from_start)
        V_new = V.copy()
        V_new[knock_out_mask] = payoff_value
        return V_new



class DoubleNoTouchComponent(Instrument):
    """
    雪球组件二：双不触碰 (Double No-Touch / DNT)。
    """
    
    def __init__(self, knock_out_barrier: float, knock_in_barrier: float, rebate_rate: float,
                 T_total: float, observation_months: List[int], T_rem: float = None):
        
        self.T_total = T_total
        self.T_rem = T_total if T_rem is None else T_rem
        
        super().__init__(T=self.T_rem)
        
        self.B_out = knock_out_barrier
        self.B_in = knock_in_barrier
        self.R_annual = rebate_rate
        self.obs_months = observation_months
        self.N = 1.0

    # --- 网格定义 ---
    @property
    def barrier_low(self) -> float: 
        return self.B_in
    
    @property
    def ko_barrier(self) -> float:
        return self.B_out

    def get_critical_points(self):
        return [self.B_out, self.B_in]
    
    def is_path_dependent(self) -> bool:
        return True

    def payoff(self, S_vec: np.ndarray) -> np.ndarray:
        final_payoff_value = 1.0 * (1 + self.R_annual * self.T_total)
        payoff = np.zeros_like(S_vec)
        
        valid_mask = (S_vec > self.B_in)
        last_month = int(round(self.T_total * 12))
        if last_month in self.obs_months:
            valid_mask &= (S_vec < self.B_out)
            
        payoff[valid_mask] = final_payoff_value
        return payoff

    # --- 立即终止逻辑 ---

    def check_immediate_termination(self, S: float, t_rem: float) -> Tuple[bool, float]:
        """
        1. S <= B_in: 敲入死掉 -> 0.0
        2. S >= B_out (观察日): 敲出死掉 -> 0.0
        """
        # 敲入检查
        if S <= self.B_in:
            return True, 0.0
            
        # 敲出检查 (仅观察日)
        if S >= self.B_out:
            current_elapsed = self.T_total - t_rem
            epsilon = 1e-5
            for m in self.obs_months:
                t_obs = m / 12.0
                if abs(current_elapsed - t_obs) < epsilon:
                    return True, 0.0
        
        return False, 0.0

    # --- 边界条件 ---

    def get_boundary_values(self, S_vec: np.ndarray, t_rem: float, r: float) -> Tuple[float, float]:
        # Lower (S=B_in): 0.0 (吸收壁)
        # Upper (S=S_max): 0.0 (假设敲出)
        return 0.0, 0.0

    # --- 事件逻辑 ---

    def get_event_dates(self, current_T_rem: float) -> List[float]:
        current_abs_time = self.T_total - current_T_rem
        
        events_rem = []
        for m in self.obs_months:
            t_obs_from_start = m / 12.0
            if t_obs_from_start > current_abs_time:
                
                t_maturity_rem = self.T_total - t_obs_from_start
            
                if 0 < t_maturity_rem <= current_T_rem:
                    events_rem.append(t_maturity_rem)
        return sorted(events_rem)
    
    def get_event_handler(self) -> Callable:
        return partial(self._knock_out_check_handler, barrier=self.B_out)

    @staticmethod
    def _knock_out_check_handler(V: np.ndarray, S: np.ndarray, t_rem: float, barrier: float) -> np.ndarray:
        V[S >= barrier] = 0.0 
        return V


class KnockInPutComponent(Instrument):
    """
    敲入后剩余价值组件 (KI-Put)。
    
    职责：
    1. 定义 V0 (等待敲入) 的 PDE 特性 (边界、Payoff)。
    2. 定义 V1 (已敲入) 的产品形态 (Shadow Instrument)。
    3. 定义互斥逻辑 (Termination)。
    """
    
    def __init__(self, initial_price: float, knock_in_barrier: float, knock_out_barrier: float, 
                 T_rem: float, observation_months: List[int], coupon_rate: float, T_total: float):
        
        super().__init__(T=T_rem)
        self.S0 = initial_price
        self.B_in = knock_in_barrier
        self.B_out = knock_out_barrier
        self.T_rem = T_rem
        self.obs_months = observation_months
        self.C_annual = coupon_rate
        self.T_total = T_total
        self.N = 1.0
        
        self._current_market = None 

    def set_market_environment(self, market: MarketEnvironment):
        self._current_market = market

    # --- 网格定义 ---
    @property
    def barrier_low(self): 
        """主网格仅在 [75, S_max] 生成。"""
        return self.B_in

    @property
    def ko_barrier(self):
        return self.B_out 

    def get_critical_points(self):
        return [self.B_out, self.B_in]
    
    def is_path_dependent(self) -> bool:
        return True

    def payoff(self, S_vec: np.ndarray) -> np.ndarray:
        return np.zeros_like(S_vec)

    # ---  互斥/终止逻辑 ---
    
    def check_immediate_termination(self, S: float, t_rem: float) -> Tuple[bool, float]:
        """
        如果 S >= 103，根据互斥逻辑，本组件价值强制为 0。
        (由 OTU 组件负责该区域)
        """
        if S >= self.B_out:
            current_elapsed = self.T_total - t_rem
            epsilon = 1e-5
            for m in self.obs_months:
                t_obs = m / 12.0
                if abs(current_elapsed - t_obs) < epsilon:
                    return True, 0.0
        return False, 0.0

    # --- 影子合约 ---

    def get_shadow_instrument(self) -> Instrument:
        """
        返回代表“已敲入世界”的期权。
        """
        return self._get_helper_option_instance(self.T_rem)

    # --- 边界条件 ---

    def get_boundary_values(self, S_vec: np.ndarray, t_rem: float, r: float) -> Tuple[float, float]:
        """
        计算主网格边界。
        Lower (S=75): 必须等于影子合约在 75 处的值。
        """
        if self._current_market is None:
            raise RuntimeError("Market environment not set.")
        
        lower_val = self._solve_helper_value_at_boundary(t_rem, r, self._current_market.sigma)
        upper_val = 0.0
        
        return lower_val, upper_val

    # --- 内部辅助逻辑 ---

    @lru_cache(maxsize=128)
    def _solve_helper_value_at_boundary(self, t_rem: float, r: float, sigma: float) -> float:
        """
        求解 S=75 处的影子合约价值。
        """
        if t_rem <= 1e-5: return min(1.0, self.B_in / self.S0) * self.N
        
        helper = self._get_helper_option_instance(t_rem)
        market_sub = MarketEnvironment(S=self.B_in, r=r, sigma=sigma, T=t_rem)
        
        temp_engine = EventFDMEngine(GeometricBrownianMotion(), M_space=2000, N_time=252)
        
        res = temp_engine.calculate(helper, market_sub)
        return float(res['price']) if isinstance(res, dict) else res

    def _get_helper_option_instance(self, t_rem_current: float) -> Instrument:
        """构造影子合约 (Knocked-In Option)"""
        parent = self
        
        class _KnockedInOption(Instrument):
            def __init__(self):
                super().__init__(T=t_rem_current)
                self.N = parent.N
                self.S0 = parent.S0
                self.B_out = parent.B_out
                self.T_total = parent.T_total
                self.obs_months = parent.obs_months
                
            def payoff(self, S_vec):
                return self.N * np.minimum(1.0, S_vec / self.S0)
            
            def get_boundary_values(self, S_vec, t, r):
                return 0.0, 0.0
            
            def is_path_dependent(self):
                return True
            
            @property
            def ko_barrier(self):
                return self.B_out

            def get_critical_points(self):
                return [parent.B_out]
            
            def get_event_dates(self, current_T):
                return parent.get_event_dates(current_T) # 复用

            def get_event_handler(self):
                return parent.get_event_handler() # 复用
            
            def check_immediate_termination(self, S, t):
                return parent.check_immediate_termination(S, t) # 复用
                
        return _KnockedInOption()

    # --- 事件逻辑 ---
    def get_event_dates(self, current_T_rem: float) -> List[float]:
        current_abs_time = self.T_total - current_T_rem
        
        events_rem = []
        for m in self.obs_months:
            t_obs_from_start = m / 12.0
            if t_obs_from_start > current_abs_time:
                
                t_maturity_rem = self.T_total - t_obs_from_start
            
                if 0 < t_maturity_rem <= current_T_rem:
                    events_rem.append(t_maturity_rem)
        return sorted(events_rem)

    def get_event_handler(self) -> Callable:
        return partial(self._knock_out_check_handler, barrier=self.B_out)

    @staticmethod
    def _knock_out_check_handler(V: np.ndarray, S: np.ndarray, t_rem: float, barrier: float) -> np.ndarray:
        V[S >= barrier] = 0.0 
        return V
    
class UnifiedSnowball(Instrument):
    """
     统一雪球结构 (Unified Snowball)
    此类定义了两个平行的状态空间：
    1. Active State (未敲入): 包含 DNT, KI, KO 三种逻辑。
    2. Shadow State (已敲入): 包含 Asset, KO 两种逻辑。
    
    Engine 将同时求解这两个状态，并在 S <= B_in 处进行耦合。
    """
    
    def __init__(self, initial_price: float, knock_in_barrier: float, knock_out_barrier: float, 
                 coupon_rate: float, rebate_rate: float, T_total: float, observation_months: List[int],
                 T_rem: Optional[float] = None):
        
        self.T_rem = T_total if T_rem is None else T_rem
        super().__init__(T=self.T_rem)
        
        self.S0 = initial_price
        self.B_in = knock_in_barrier
        self.B_out = knock_out_barrier
        self.C_annual = coupon_rate
        self.R_annual = rebate_rate
        self.T_total = T_total
        self.obs_months = observation_months
        self.N = 1.0  # 名义本金

    # -------------------------------------------------------------------------
    # 1. 终值定义 (Terminal Conditions at t=T)
    # -------------------------------------------------------------------------

    def payoff(self, S_vec: np.ndarray) -> np.ndarray:
        """
        [主网格/未敲入状态] 的终值定义。
        
        逻辑优先级 (即使是到期日，若触碰 KO 线也视为敲出):
        1. S >= B_out: 敲出 (获得 Coupon)
        2. S <= B_in:  敲入 (获得 Asset)
        3. 中间区域:    双不敲 (获得 Rebate)
        """
        val_ko = self.N * (1 + self.C_annual * self.T_total)
        val_dnt = self.N * (1 + self.R_annual * self.T_total)
        val_ki  = self.N * np.minimum(1.0, S_vec / self.S0)
        
        # 嵌套逻辑: KO > KI > DNT
        return np.where(
            S_vec >= self.B_out, 
            val_ko, 
            np.where(S_vec <= self.B_in, val_ki, val_dnt)
        )

    def payoff_shadow(self, S_vec: np.ndarray) -> np.ndarray:
        """
        [已敲入状态] 的终值定义。
        
        逻辑优先级:
        1. S >= B_out: 敲出 (获得 Coupon) - 即使曾敲入，最后一天敲出也算赢
        2. S < B_out:  资产价值 (Asset) - 已无 DNT 可能性
        """
        val_ko = self.N * (1 + self.C_annual * self.T_total)
        val_asset = self.N * np.minimum(1.0, S_vec / self.S0)
        
        return np.where(S_vec >= self.B_out, val_ko, val_asset)

    # -------------------------------------------------------------------------
    # 2. 离散事件 (Discrete Events / Knock-Out)
    # -------------------------------------------------------------------------

    def get_event_dates(self, current_T_rem: float) -> List[float]:
        """
        计算未来观察日距离当前时刻的剩余时间。
        """
        time_elapsed_at_pricing = self.T_total - self.T_rem
        future_events_rem = []
        
        for m in self.obs_months:
            t_obs_from_start = m / 12.0
            if t_obs_from_start > time_elapsed_at_pricing:
                t_event_rem = self.T_total - t_obs_from_start
                if 0 < t_event_rem <= current_T_rem:
                    future_events_rem.append(t_event_rem)
                
        # 返回升序列表 (对于 FDM 倒推，通常需要按时间顺序检查)
        return sorted(future_events_rem)

    def get_event_handler(self) -> Callable:
        """
        定义观察日的敲出行为。
        注意：此 Handler 对 [主网格] 和 [影子网格] 通用。
        只要 S >= B_out，无论之前处于什么状态，都立即变为敲出收益。
        """
        def _knock_out_handler(V: np.ndarray, S: np.ndarray, t_rem: float, barrier: float) -> np.ndarray:
            elapsed_time = self.T_total - t_rem
            rebate = self.N * (1 + self.C_annual * elapsed_time)
            V[S >= barrier] = rebate
            return V
        
        return partial(_knock_out_handler, barrier=self.B_out)

    # -------------------------------------------------------------------------
    # 3. 边界条件 (Boundary Conditions)
    # -------------------------------------------------------------------------

    def get_boundary_values(self, S_vec: np.ndarray, t_rem: float, r: float) -> Tuple[float, float]:
        """
        定义物理网格上下界的性质 (Dirichlet 边界)。
        只定义 S_min 和 S_max 的行为。
        """
        lower_val = 0.0
        upper_val = 0.0

        current_time_from_start = self.T_total - t_rem
        next_obs_time = None
        epsilon = 1e-5 
        
        for m in self.obs_months:
            t_obs = m / 12.0
            if t_obs >= current_time_from_start - epsilon:
                next_obs_time = t_obs
                break
        
        if next_obs_time is not None:
            ko_payoff = 1.0 * (1 + self.C_annual * next_obs_time)
            dt_discount = next_obs_time - current_time_from_start
            if dt_discount > 0:
                upper_val = ko_payoff * np.exp(-r * dt_discount)
            else:
                upper_val = ko_payoff
        return lower_val, upper_val
    
    def is_path_dependent(self) -> bool:
        return True

    def get_critical_points(self):
        return [self.B_out, self.B_in]
    



class StepDownSnowball(Instrument):
    """
    [Step-down Snowball] 阶梯降敲雪球产品类
    """
    
    def __init__(self, 
                 initial_price: float, knock_in_barrier: float, obs_dates: List[int],          
                 initial_ko_barrier: float, step_down_size: float, coupon_rate: float,
                 rebate_rate: float, T_total: float,                
                 dt: float, T_rem: float = None, has_knocked_in: bool = False,
                 ko_floor: float = None):       
        """
        Args:
            obs_dates: 观察月份列表 (整数)，例如 [3, 6, 9, 12]。
            initial_ko_barrier: 第一个观察日(列表中的第一个月)的敲出价格。
            step_down_size: 每次观察下调的数值。
            dt: Monte Carlo 时间步长。
        """
        # 1. 基础参数
        self.S0 = initial_price
        self.B_in = knock_in_barrier
        
        self.C_annual = coupon_rate
        self.R_annual = rebate_rate
        self.T_total = T_total
        self.dt = dt
        self.N = 1.0
        self.obs_dates = obs_dates
        
        self.T_rem = T_total if T_rem is None else T_rem
        super().__init__(T=self.T_rem)
        
        self.has_knocked_in = has_knocked_in
        
        self.ko_barriers_full_schedule = []
        
        for i in range(len(obs_dates)):
            current_barrier = initial_ko_barrier - (i * step_down_size)
            
            if ko_floor is not None:
                current_barrier = max(current_barrier, ko_floor)
            
            self.ko_barriers_full_schedule.append(current_barrier)
    
        
        current_time_elapsed = self.T_total - self.T_rem
        
        self.mc_ko_schedule = []    # MC: List of (step_idx, barrier)
        self.fdm_event_map = {}     # FDM: Dict {t_rem : barrier}
        self.maturity_barrier = float('inf') 
        
        epsilon = 1e-5 
        
        for month_idx, barrier in zip(obs_dates, self.ko_barriers_full_schedule):
            date_in_years = month_idx / 12.0
    
            if abs(date_in_years - self.T_total) < epsilon:
                self.maturity_barrier = barrier
            
            if date_in_years > current_time_elapsed + epsilon:
                time_from_now = date_in_years - current_time_elapsed
                step_idx = int(round(time_from_now / self.dt))
                if step_idx > 0:
                    self.mc_ko_schedule.append((step_idx, barrier))
                
                # FDM 处理 (使用剩余时间)
                t_rem_at_event = self.T_total - date_in_years
                self.fdm_event_map[round(t_rem_at_event, 6)] = barrier

    
    def is_path_dependent(self) -> bool:
        return True

    def process_one_path(self, price_path: np.ndarray) -> Tuple[float, float]:
        for step_idx, barrier in self.mc_ko_schedule:
            if step_idx < len(price_path):
                if price_path[step_idx] >= barrier:
                    settlement_time = step_idx * self.dt
                    total_time = (self.T_total - self.T_rem) + settlement_time
                    payoff = self.N * (1 + self.C_annual * total_time)
                    return payoff, settlement_time
        
        final_price = price_path[-1]
        if final_price >= self.maturity_barrier:
            return self.N * (1 + self.C_annual * self.T_total), self.T_rem
            
        is_knocked_in = self.has_knocked_in
        if not is_knocked_in:
            if np.min(price_path) <= self.B_in:
                is_knocked_in = True
        
        if is_knocked_in:
            return self.N * min(1.0, final_price / self.S0), self.T_rem
        else:
            return self.N * (1 + self.R_annual * self.T_total), self.T_rem


    def payoff(self, S_vec: np.ndarray) -> np.ndarray:
        val_ko = self.N * (1 + self.C_annual * self.T_total)
        val_dnt = self.N * (1 + self.R_annual * self.T_total)
        val_ki  = self.N * np.minimum(1.0, S_vec / self.S0)
        return np.where(S_vec >= self.maturity_barrier, val_ko, 
                        np.where(S_vec <= self.B_in, val_ki, val_dnt))

    def payoff_shadow(self, S_vec: np.ndarray) -> np.ndarray:
        val_ko = self.N * (1 + self.C_annual * self.T_total)
        val_asset = self.N * np.minimum(1.0, S_vec / self.S0)
        return np.where(S_vec >= self.maturity_barrier, val_ko, val_asset)

    def get_event_dates(self, current_T_rem: float) -> List[float]:
        events = sorted(list(self.fdm_event_map.keys()))
        valid_events = [t for t in events if 0 < t <= current_T_rem]
        return valid_events

    def get_event_handler(self) -> Callable:
        event_map = self.fdm_event_map
        coupon = self.C_annual
        T_total = self.T_total
        N = self.N
        
        def _step_down_handler(V: np.ndarray, S: np.ndarray, t_rem_current: float) -> np.ndarray:
            key_t = round(t_rem_current, 6)
            if key_t in event_map:
                current_barrier = event_map[key_t]
            else:
                all_times = np.array(list(event_map.keys()))
                idx = (np.abs(all_times - t_rem_current)).argmin()
                closest_t = all_times[idx]
                current_barrier = event_map[closest_t]
            
            knock_out_mask = (S >= current_barrier)
            elapsed_time = T_total - t_rem_current
            rebate_val = N * (1 + coupon * elapsed_time)
            
            V_new = V.copy()
            V_new[knock_out_mask] = rebate_val
            return V_new
            
        return _step_down_handler

    def get_boundary_values(self, S_vec: np.ndarray, t_rem: float, r: float) -> Tuple[float, float]:
        lower_val = self.N * np.minimum(1.0, S_vec[0] / self.S0)
        upper_val = 0.0
        
        current_time_from_start = self.T_total - t_rem
        
        next_obs_time = None
        epsilon = 1e-5 
        
        for m in self.obs_dates:
            t_obs = m / 12.0 
            if t_obs >= current_time_from_start - epsilon:
                next_obs_time = t_obs
                break
        
        if next_obs_time is not None:
            ko_payoff = self.N * (1 + self.C_annual * next_obs_time)
            dt_discount = next_obs_time - current_time_from_start
            
            if dt_discount > 0:
                upper_val = ko_payoff * np.exp(-r * dt_discount)
            else:
                upper_val = ko_payoff
        else:
            elapsed = self.T_total - t_rem
            upper_val = self.N * (1 + self.C_annual * elapsed)

        return lower_val, upper_val

    def get_critical_points(self) -> List[float]:
        points = [self.B_in]
        unique_ko_barriers = set(self.ko_barriers_full_schedule)
        points.extend(unique_ko_barriers)
        return sorted(list(points))





class PhoenixDCN(Instrument):
    """
    [Phoenix DCN] 凤凰结构 / 数字票息票据
    """
    
    def __init__(self, initial_price: float, knock_in_barrier: float, coupon_barrier: float,        
                 obs_dates: List[int], initial_ko_barrier: float, step_down_size: float,         
                 ko_start_month: int, coupon_rate_annual: float, T_total: float,
                 dt: float, T_rem: float = None,
                 ko_floor: float = None):       
        """
        Args:
            coupon_rate_annual: 年化票息率。内部会自动除以12计算单期票息金额。
            ko_start_month: 从第几个月开始观察敲出。
            obs_dates: 观察月份列表, 包含所有观察月份，例如 [1, 2, 3, ..., 12]。
        """
        self.S0 = initial_price
        self.B_in = knock_in_barrier
        self.B_coupon = coupon_barrier
        self.N = 1.0
        self.coupon_amount = self.N * coupon_rate_annual / 12.0
        
        self.T_total = T_total
        self.dt = dt
        self.T_rem = T_total if T_rem is None else T_rem
        super().__init__(T=self.T_rem)
        
        
        self.ko_schedule_full = [] 
        self.obs_dates_years = []  
        

        start_idx = obs_dates.index(ko_start_month)


        for i, m in enumerate(obs_dates):
            t_year = m / 12.0
            self.obs_dates_years.append(t_year)
            
            if m < ko_start_month:
                barrier = float('inf')
            else:
                steps = i - start_idx
                barrier = initial_ko_barrier - (steps * step_down_size)
                if ko_floor is not None:
                    barrier = max(barrier, ko_floor)
            
            self.ko_schedule_full.append(barrier)

        current_time_elapsed = self.T_total - self.T_rem
        
        self.mc_schedule = []       
        self.fdm_event_map = {}     
        self.maturity_barrier = float('inf') 
        
        epsilon = 1e-5
        
        for t_year, barrier in zip(self.obs_dates_years, self.ko_schedule_full):
            is_maturity_date = False
            if abs(t_year - self.T_total) < epsilon:
                self.maturity_barrier = barrier
                self.is_maturity_obs = True
                is_maturity_date = True
            
            if t_year > current_time_elapsed + epsilon:
                t_rem_at_event = self.T_total - t_year
                self.fdm_event_map[round(t_rem_at_event, 6)] = barrier
                
                if not is_maturity_date:
                    time_from_now = t_year - current_time_elapsed
                    step_idx = int(round(time_from_now / self.dt))
                    if step_idx > 0:
                        self.mc_schedule.append((step_idx, barrier))

    
    def is_path_dependent(self) -> bool:
        return True

    def process_one_path(self, price_path: np.ndarray) -> Tuple[float, float]:
        accumulated_payoff = 0.0
        
        for step_idx, ko_barrier in self.mc_schedule:
            if step_idx < len(price_path):
                current_S = price_path[step_idx]
                
                if current_S >= ko_barrier:
                    current_payoff = self.N + self.coupon_amount
                    total_payoff = accumulated_payoff + current_payoff
                    settlement_time = step_idx * self.dt
                    return total_payoff, settlement_time
                
                if current_S >= self.B_coupon:
                    accumulated_payoff += self.coupon_amount
        
        final_price = price_path[-1]
        final_settlement_time = self.T_rem

        if self.is_maturity_obs and final_price >= self.B_coupon:
            final_payoff = self.N + self.coupon_amount
            return accumulated_payoff + final_payoff, final_settlement_time
            
        final_principal = 0.0
        
        if final_price >= self.B_in:
            final_principal = self.N
        else:
            final_principal = self.N * (final_price / self.S0)
            
        return accumulated_payoff + final_principal, final_settlement_time


    def payoff(self, S_vec: np.ndarray) -> np.ndarray:
        val_ko = self.N + self.coupon_amount
        val_coupon_alive = self.N + self.coupon_amount
        val_protect = self.N
        val_ki = self.N * (S_vec / self.S0)
        
        res = np.where(S_vec >= self.B_coupon, 
                       val_coupon_alive,
                       np.where(S_vec >= self.B_in, val_protect, val_ki))
        res = np.where(S_vec >= self.maturity_barrier, val_ko, res)
        
        return res

    def get_event_dates(self, current_T_rem: float) -> List[float]:
        events = sorted(list(self.fdm_event_map.keys()))
        valid_events = [t for t in events if 0 < t <= current_T_rem]
        return valid_events

    def get_event_handler(self) -> Callable:
        event_map = self.fdm_event_map
        coupon_amt = self.coupon_amount
        coupon_barrier = self.B_coupon
        N = self.N
        
        def _phoenix_handler(V: np.ndarray, S: np.ndarray, t_rem_current: float) -> np.ndarray:
            key_t = round(t_rem_current, 6)
            if key_t in event_map:
                current_ko_barrier = event_map[key_t]
            else:
                all_times = np.array(list(event_map.keys()))
                idx = (np.abs(all_times - t_rem_current)).argmin()
                closest_t = all_times[idx]
                current_ko_barrier = event_map[closest_t]
            
            V_new = V.copy()
            
            ko_mask = (S >= current_ko_barrier)
            V_new[ko_mask] = N + coupon_amt
            
            alive_mask = ~ko_mask
            coupon_mask = alive_mask & (S >= coupon_barrier)
            V_new[coupon_mask] += coupon_amt
            
            return V_new
            
        return _phoenix_handler

    def get_boundary_values(self, S_vec: np.ndarray, t_rem: float, r: float) -> Tuple[float, float]:
        lower_val = self.N * (S_vec[0] / self.S0)
        upper_val = 0.0
        
        current_time_from_start = self.T_total - t_rem
        next_obs_time = None
        epsilon = 1e-5 
        
        for t_obs in self.obs_dates_years:
            if t_obs >= current_time_from_start - epsilon:
                next_obs_time = t_obs
                break
        
        target_payoff = self.N + self.coupon_amount
        
        if next_obs_time is not None:
            dt_discount = next_obs_time - current_time_from_start
            if dt_discount > 0:
                upper_val = target_payoff * np.exp(-r * dt_discount)
            else:
                upper_val = target_payoff
            
        return lower_val, upper_val


    def get_critical_points(self) -> List[float]:
        points = {self.B_in, self.B_coupon}
        for barrier in self.fdm_event_map.values():
            points.add(barrier)
        valid_points = [p for p in points if p != float('inf')]
        return sorted(valid_points)











