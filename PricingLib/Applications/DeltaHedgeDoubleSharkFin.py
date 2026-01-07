import xlwings as xw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('../../') 

# 导入业务模块
from PricingLib.BackTest.BacktestApp import run_backtest_logic
from PricingLib.Base.Utils import PlotUtils

def run_hedge_backtest_double_sharkfin(sheet):
    """
    Excel 按钮入口：读取双向鲨鱼鳍参数，运行回测，输出结果。
    """
    # -----------------------------------------------------
    # 从 Excel 读取双向鲨鱼鳍的配置参数
    # -----------------------------------------------------
    # 假设参数从 B2 单元格开始
    K_L = sheet.range('B2').value      # [新] Put型行权价
    K_U = sheet.range('B3').value      # [新] Call型行权价
    H_L = sheet.range('B4').value      # [新] 下障碍
    H_U = sheet.range('B5').value      # [新] 上障碍
    R_L = sheet.range('B6').value      # [新] 下返息
    R_U = sheet.range('B7').value      # [新] 上返息
    
    expiry_date_val = sheet.range('B8').value
    option_type = sheet.range('B9').value 
    initial_cash = sheet.range('B10').value
    fee_rate = sheet.range('B11').value
    threshold = sheet.range('B12').value
    engine_type = sheet.range('B13').value
    strategy_type = sheet.range('B14').value
    
    notional_amount = sheet.range('B17').value
    hedge_instrument = sheet.range('B18').value

    mc_n_steps, fdm_n_space, fdm_n_time = None, None, None
    if engine_type == 'MC':
        mc_n_steps = int(sheet.range('B15').value)
    elif engine_type == 'FDM':
        fdm_n_space = int(sheet.range('B15').value)
        fdm_n_time = int(sheet.range('B16').value)

    # -----------------------------------------------------
    # 读取行情数据并计算期权份数
    # -----------------------------------------------------
    raw_data = sheet.range('C1').options(pd.DataFrame, index=False, expand='table').value
    df = raw_data.copy()

    if len(df.columns) >= 5:
        df.columns = ['Date', 'Spot', 'Future', 'Rate', 'Vol'] + list(df.columns[5:])
    else:
        df.columns = ['Date', 'Spot', 'Future', 'Rate'] + list(df.columns[4:])
    df['Date'] = pd.to_datetime(df['Date'])
    
    S0 = df['Spot'].iloc[0]
    num_options = notional_amount / S0
    print(f"Initial Spot: {S0:.2f}, Notional: {notional_amount:,.0f}, Calculated Num Options: {num_options:,.2f}")

    # -----------------------------------------------------
    # config setup
    # -----------------------------------------------------
    config = {
        'K_L': float(K_L),
        'K_U': float(K_U),
        'H_L': float(H_L),
        'H_U': float(H_U),
        'R_L': float(R_L),
        'R_U': float(R_U),
        'Expiry_Date': pd.to_datetime(expiry_date_val),
        
        'initial_cash': float(initial_cash),
        'num_options': num_options,
        'hedge_instrument': hedge_instrument,
        'future_multiplier': 200,
        'fee_rate': fee_rate,
        'threshold': threshold,
        'engine_type': engine_type,
        'mc_n_steps': mc_n_steps,
        'fdm_n_space': fdm_n_space,
        'fdm_n_time': fdm_n_time,
    }
    
    # -----------------------------------------------------
    # 调用核心业务逻辑
    # -----------------------------------------------------
    result_df = run_backtest_logic(df, config, strategy_type=strategy_type, option_type=option_type)

    # -----------------------------------------------------
    # 数据清洗、画图、写回
    # -----------------------------------------------------
    for col in result_df.columns:
        if pd.api.types.is_numeric_dtype(result_df[col]):
            result_df[col] = result_df[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
    
    fig = PlotUtils.plot_hedging_results(result_df, config['initial_cash'], show_plot=False)
    pic_name = 'DoubleSharkFin_PnL_Chart'
    
    for pic in sheet.pictures:
        if pic.name == pic_name:
            pic.delete()
            break
            
    sheet.pictures.add(fig, name=pic_name, update=True, 
                       left=sheet.range('R1').left, 
                       top=sheet.range('R1').top)
    plt.close(fig)

    output_cell = 'I1'
    sheet.range(output_cell).expand('table').clear_contents()
    
    def get_df_to_write(df):
        exclude_columns = ['Date', 'Spot', 'Hedge_Price', 'Rate', 'Asset_Val', 'dt_days', 'Cum_Interest']
        indicators = [col for col in df.columns if col not in exclude_columns]
        return df[indicators]
        
    sheet.range(output_cell).options(index=False).value = get_df_to_write(result_df)