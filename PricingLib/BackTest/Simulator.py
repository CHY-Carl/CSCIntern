import pandas as pd
import numpy as np
from tqdm import tqdm
from ..Base.Utils import DateUtils
from ..Base.BaseLayer import MarketEnvironment, Instrument
from .Strategy import Strategy, HedgingStrategy, DeltaHedgeStrategy
from .Account import Account

class BacktestSimulator:
    """
    通用回测模拟器 (Event Loop)。
    
    核心职责：
    1. 维护回测的时间轴 (Time Loop)。
    2. 每日构建最新的市场环境 (Market State)。
    3. 调用 Strategy 获取交易信号。
    4. 指挥 Account 执行交易。
    5. 进行每日盯市 (Mark-to-Market) 核算。
    """
    
    def __init__(self, 
                 account: Account, 
                 strategy: Strategy, 
                 market_data_df: pd.DataFrame, 
                 config: dict,
                 product: Instrument = None):
        """
        Args:
            account: 资金账户对象 (状态机，记录 Cash 和 Units)
            strategy: 策略对象 (决策机，内含 Engine，输出 Target Position)
            market_data_df: 行情数据 (必须包含 Date, Spot, Future, Rate)
            config: 配置字典 (包含 Expiry_Date, Multipliers, FeeRates 等)
            product: 具体的金融产品对象 (如 EuropeanOption, SharkFinOption)
        """
        self.account = account
        self.strategy = strategy
        self.data = market_data_df.sort_values('Date').reset_index(drop=True)
        self.cfg = config
        self.product = product

        self.hedge_inst = config.get('hedge_instrument', 'Future')
        
        # 数据完整性检查
        required_cols = ['Date', 'Spot', 'Rate']
        if self.hedge_inst == 'Future':
            required_cols = ['Date', 'Spot', 'Future', 'Rate']

        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"Market Data must contain columns: {required_cols}")

    def run(self, **kwargs) -> pd.DataFrame:
        """
        执行回测主循环。
        Returns:
            pd.DataFrame: 每日账户状态历史 (NAV, Cash, Units, P&L)
        """
        # print(f"Starting Simulation ({len(self.data)} steps)...")
        
        expiry_date = self.cfg.get('Expiry_Date')
        notional = self.cfg.get('notional_amount')
        ref_S0 = self.cfg.get('initial_spot', self.data.iloc[0]['Spot'])

        if self.hedge_inst == 'Future':
            hedge_mult = self.cfg.get('future_multiplier', 1.0)
        else:
            hedge_mult = 1.0 

        if isinstance(self.strategy, HedgingStrategy):
            T_0 = DateUtils.year_fraction(self.data.iloc[0]['Date'], expiry_date, 'ACT/365') if expiry_date else 0.0

            # 【修改点】构建初始归一化市场 (S=1.0)
            market_0 = MarketEnvironment(
                S=1.0,
                r=self.data.iloc[0]['Rate'],
                sigma=self.data.iloc[0].get('Vol', self.cfg.get('sigma', 0.2)),
                T=T_0
            )

            # 计算单位权利金 (百分比形式，例如 0.05)
            init_price_pct = self.strategy.engine.get_price(self.product, market_0)
            
            # 计算总权利金收入 (百分比 * 名义本金)
            total_premium = init_price_pct * notional
            self.account.deposit(total_premium)
            
            # print(f"DEBUG: Initial Sell. Price(%): {init_price_pct:.4f}, Premium: {total_premium:,.2f}")

        last_processed_row = None
        # --- 主循环：遍历交易日 ---
        #! tqdm 进度条在纯python脚本里用
        # for i, row in tqdm(self.data.iterrows(), total=len(self.data), mininterval=1.0):
        for i, row in self.data.iterrows():
            curr_date = row['Date']
            last_processed_row = row
            
            # [结算检查] 如果到达到期日，停止常规交易，转入结算
            if curr_date >= expiry_date:
                print(f"DEBUG: Reached Expiry Date {curr_date.date()}. Switching to Settlement.")
                break

            real_S = row['Spot']
            F = row['Future']
            r = row['Rate']
            sigma = row.get('Vol', self.cfg.get('sigma', 0.2))
    
            if self.hedge_inst == 'Future':
                current_hedge_price = row['Future']
            else:
                current_hedge_price = row['Spot']

            if i > 0:
                if self.hedge_inst == 'Future':
                    prev_hedge_price = self.data.iloc[i-1]['Future']
                    self.account.settle_daily_pnl(prev_hedge_price, current_hedge_price, hedge_mult)
                else:
                    pass
            # -----------------------------------------------------
            # 1. 市场环境更新 (Update Market)
            # -----------------------------------------------------
            norm_S = real_S / ref_S0

            T_rem = 0.0
            if expiry_date:
                T_rem = DateUtils.year_fraction(curr_date, expiry_date, 'ACT/365')
            
            market = MarketEnvironment(S=norm_S, r=r, sigma=sigma, T=T_rem)
            
            # -----------------------------------------------------
            # 2. 账户计息 (Accrue Interest)
            # -----------------------------------------------------
            if i > 0:
                prev_date = self.data.iloc[i-1]['Date']
                days_diff = (pd.to_datetime(curr_date) - pd.to_datetime(prev_date)).days
                self.account.accrue_interest(r, days_diff / 365.0)
            
            # -----------------------------------------------------
            # 3. 策略执行 (Strategy Execution)
            # -----------------------------------------------------
            # 如果期权已到期 (T < 0)，停止对冲，准备结算
            if T_rem < 0: break
            
            opt_mtm_value = 0.0
            is_alive = self.product.update_status(norm_S)

            if is_alive:
                if T_rem > 0.002: # 约半个交易日
                    target_pos, signals_to_log = self.strategy.get_signal(self.product, market, current_real_spot=real_S, seed=np.random.randint(1e6))
                else:
                    # 临近到期，强制平仓归零
                    target_pos = 0.0
                    if isinstance(self.strategy, DeltaHedgeStrategy):
                        signals_to_log = {'delta': 0.0}
                    else:
                        raise ValueError("Unsupported strategy type for final near-expiry logging.")
                if T_rem > 0:
                    try:
                        unit_price = self.strategy.engine.get_price(self.product, market)
                        opt_mtm_value = unit_price * notional
                    except:
                        opt_mtm_value = 0.0
                
            else:
                target_pos = 0.0
                residual_value = self.product.get_residual_value(norm_S, T_rem, r)
                # print(f"DEBUG: Product Inactive on {curr_date.date()}, Residual Value per Unit: {residual_value:.2f}")
                opt_mtm_value = residual_value * notional

                if isinstance(self.strategy, DeltaHedgeStrategy):
                    signals_to_log = {'delta': 0.0}
                else:
                    raise ValueError("Unsupported strategy type for final logging.")
                
            # -----------------------------------------------------
            # 4. 交易执行 (Trade Execution)
            # -----------------------------------------------------
            current_pos = self.account.units
            trade_qty = target_pos - current_pos
            
            self.account.trade(change_in_units=trade_qty, 
                               execute_price=current_hedge_price,
                               multiplier=hedge_mult,
                               fee_rate=self.cfg.get('fee_rate', 0.0))
            
            # -----------------------------------------------------
            # 5. 盯市与记录 (Mark-to-Market)
            # -----------------------------------------------------
            
            self.account.mark_to_market(
                date=curr_date,
                spot=real_S,
                hedge_price=current_hedge_price,
                option_val=opt_mtm_value, # 卖方视角下，这是负债
                rate=r,
                strategy_signals=signals_to_log
            )

        # 循环结束
        if last_processed_row is None:
            raise ValueError("No market data processed. Check data or expiry date.")

        curr_date = last_processed_row['Date']
        S_final = last_processed_row['Spot']
        r_final = last_processed_row['Rate']

        if self.hedge_inst == 'Future':
            final_hedge_close_price = last_processed_row['Future']
        else:
            final_hedge_close_price = last_processed_row['Spot']
        
        # 计算期权的最终价值 (Final Payoff)
        final_payout_total = 0.0
        S_final_norm = S_final / ref_S0

        if self.product.is_active:
            final_unit_payoff = self.product.payoff(S_final_norm)
            final_payout_total = final_unit_payoff * notional
        else:
            final_unit_payoff = self.product.get_residual_value(S_final_norm, 0, r_final)
            final_payout_total = final_unit_payoff * notional
        
        if final_payout_total > 0:
            self.account.cash -= final_payout_total
            
        # 2. 平仓期货头寸
        if self.account.units != 0:
            self.account.trade(change_in_units=-self.account.units, # 负数代表卖出平仓
                               execute_price=final_hedge_close_price,
                               multiplier=hedge_mult,
                               fee_rate=self.cfg.get('fee_rate', 0.0))
        else:
            print("DEBUG: No futures position to liquidate.")

        # 3. 记录最终状态 (After All Settlement)
        if isinstance(self.strategy, DeltaHedgeStrategy):
            strategy_signals = {'delta': 0.0}
        else:
            raise ValueError("Unsupported strategy type for final logging.")

        self.account.mark_to_market(
            date=expiry_date, 
            spot=S_final,
            hedge_price=final_hedge_close_price,
            option_val=0.0,
            rate=r_final,
            strategy_signals=strategy_signals
        )
        print("Simulation Completed.")
        return self.account.get_history_df()