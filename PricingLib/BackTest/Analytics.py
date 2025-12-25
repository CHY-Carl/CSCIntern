import pandas as pd
import numpy as np

class BacktestAnalyzer:
    def __init__(self, history_df, initial_cash):
        self.df = history_df.copy()
        self.initial_cash = initial_cash
        self._calculate_metrics()
        
    def _calculate_metrics(self):
        # 1. 总盈亏
        self.df['PnL_Total'] = self.df['NAV'] - self.initial_cash
        self.df['PnL_Daily'] = self.df['PnL_Total'].diff().fillna(0.0)
        self.df['PnL_Daily'].iloc[0] = self.df['NAV'].iloc[0] - self.initial_cash
        
        # 2. 计算本金产生的纯利息 (Cumulative Risk-Free Interest)
        # 逻辑：每一天，Initial_Cash 都会产生利息
        
        # 计算每一行距离上一行的天数
        self.df['dt_days'] = self.df['Date'].diff().dt.days.fillna(0)
        
        # 向量化计算
        r = self.df['Rate'].values
        dt = self.df['dt_days'].values / 365.0
        
        daily_interest = self.initial_cash * (np.exp(r * dt) - 1.0)
        self.df['Cum_Interest'] = np.cumsum(daily_interest)
        
        # 3. 剥离利息后的策略盈亏
        self.df['PnL_Strategy'] = self.df['PnL_Total'] - self.df['Cum_Interest']

    def get_clean_df(self):
        return self.df