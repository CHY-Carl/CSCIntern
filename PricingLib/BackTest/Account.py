import pandas as pd
import numpy as np

class Account:
    def __init__(self, initial_cash=0.0, mode='Future'):
        self.cash = initial_cash
        self.units = 0.0  # 持有的资产数量 (期货手数/现货股数)
        self.history = []
        self.mode = mode  # 'Future' or 'Spot'
        
    def accrue_interest(self, r, dt):
        """现金生息"""
        if self.cash == 0: return 0.0
        interest = self.cash * (np.exp(r * dt) - 1.0)
        self.cash += interest
        return interest
    
    def deposit(self, amount):
        """存入资金 (如收到期权权利金)"""
        self.cash += amount


    def trade(self, change_in_units, execute_price, multiplier, fee_rate):
        """
        执行交易。
        如果是 Spot 模式，买入需要扣除全额本金，卖出获得全额本金。
        如果是 Future 模式，本金不发生变动，仅记录手续费
        """
        if change_in_units == 0: return 0.0
        
        notional_value = abs(change_in_units) * execute_price * multiplier
        commission = notional_value * fee_rate
        
        if self.mode == 'Spot':
            trade_cost = change_in_units * execute_price * multiplier
            self.cash -= trade_cost
        
        elif self.mode == 'Future':
            pass

        self.cash -= commission
        self.units += change_in_units
        
        return commission
    
    def settle_daily_pnl(self, prev_price, curr_price, multiplier):
        """
        [期货模式] 每日无负债结算 (Mark-to-Market Settlement)。
        存量持仓的盈亏直接进入现金账户。
        """
        if self.units == 0: return 0.0
        
        # 盈亏 = 持仓量 * 价格变动 * 乘数
        pnl = self.units * (curr_price - prev_price) * multiplier
        self.cash += pnl
        return pnl

    def mark_to_market(self, date, spot, hedge_price, option_val, rate, strategy_signals: dict = None):
        """
        记录每日状态。
        NAV = Cash - Option_Liability
        """
        if self.mode == 'Future':
            asset_market_value = 0
        else:  
            asset_market_value = self.units * hedge_price

        net_value = self.cash + asset_market_value - option_val
        
        record = {
            'Date': date,
            'Spot': spot,
            'Hedge_Price': hedge_price,
            'Rate': rate,
            'Cash': self.cash, # 此时 Cash 包含了 初始+权利金+利息+期货累计盈亏-手续费
            'Units': self.units,
            'Asset_Val': asset_market_value,
            'Option_Val': option_val,
            'NAV': net_value
        }

        if strategy_signals is not None:
            record.update(strategy_signals)

        self.history.append(record)
        return net_value

    def get_history_df(self):
        return pd.DataFrame(self.history)