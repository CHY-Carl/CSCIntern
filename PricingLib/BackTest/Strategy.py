from abc import ABC, abstractmethod
from ..Base.BaseLayer import PricingEngine, Instrument, MarketEnvironment

# ==========================================================
# ======= Level 1: 顶层策略基类 ============================
# ==========================================================
class Strategy(ABC):
    """
    所有交易策略的根基类。
    """
    def __init__(self, engine: PricingEngine):
        """
        Args:
            engine: 用于计算 Greeks 的定价引擎 (如 AnalyticBSEngine, FDMEngine)
        """
        self.engine = engine

    @abstractmethod
    def get_signal(self, *args, **kwargs):
        """
        核心逻辑：根据输入生成信号。
        """
        pass

# ==========================================================
# ======= Level 2: 对冲策略基类 ============================
# ==========================================================
class HedgingStrategy(Strategy):
    """
    对冲专用策略基类。
    特点：
    1. 依赖于具体的 PricingEngine 来计算风险指标 (Delta, Gamma 等)。
    2. 目标是管理特定 Product (Instrument) 的风险。
    """
    
    def __init__(self, engine: PricingEngine):
        """
        Args:
            engine: 用于计算 Greeks 的定价引擎 (如 AnalyticBSEngine, FDMEngine)
        """
        self.engine = engine
    
    @abstractmethod
    def calculate_target_position(self, option: Instrument, market: MarketEnvironment) -> tuple[float, dict]:
        """
        子类必须实现：计算目标持仓量。
        """
        pass
    
    def get_signal(self, option, market, **kwargs):
        return self.calculate_target_position(option, market, **kwargs)


# ==========================================================
# ======= Level 3: 具体实现 - Delta 对冲 ====================
# ==========================================================
class DeltaHedgeStrategy(HedgingStrategy):
    """
    标准的 Delta Neutral 对冲策略。
    逻辑: 通过持有标的资产来抵消期权的 Delta。
    Target Position (Futures) = -1 * Option_Delta * Ratio
    """
    
    def __init__(self, engine: PricingEngine,
                 notional_amount: float,
                 hedge_instrument: str = 'Future',
                 future_multiplier: int = 200, 
                 threshold: float = 0.0):
        """
        Args:
            engine: 定价引擎
            option_multiplier: 期权合约乘数 (MO=100)
            future_multiplier: 对冲工具(期货)乘数 (IM=200)
            threshold: 调仓阈值 (Delta变动超过此值才交易，节省手续费)
        """
        super().__init__(engine)
        self.notional_amount = notional_amount
        self.hedge_instrument = hedge_instrument
        self.fut_mult = future_multiplier
        self.threshold = threshold
        
        # 状态变量：记录上一次的目标持仓，用于阈值判断
        self.last_target = 0.0 

    def calculate_target_position(self, option: Instrument, market: MarketEnvironment, **kwargs) -> tuple[float, dict]:
        delta_pct = self.engine.get_delta(option, market, **kwargs)

        current_real_spot = kwargs.get('current_real_spot')
        if current_real_spot is None:
            raise ValueError("DeltaHedgeStrategy requires 'current_real_spot' in kwargs to calculate position.")

        required_hedge_value = delta_pct * self.notional_amount

        target_shares = required_hedge_value / current_real_spot
        
        if self.hedge_instrument == 'Future':
            target_pos = target_shares / self.fut_mult
        else:
            target_pos = target_shares
        
        if self.threshold > 0:
            change = abs(target_pos - self.last_target)
            if change < self.threshold:
                return self.last_target, {'delta': float(delta_pct)}
        
        self.last_target = target_pos

        return float(target_pos), {'delta': float(delta_pct)}