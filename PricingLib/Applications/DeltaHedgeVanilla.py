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

def run_hedge_backtest_vanilla(sheet):
    """
    Excel 按钮入口：读取界面参数，运行回测，输出结果。
    """
    # -----------------------------------------------------
    # 1. 从 Excel 读取配置参数 (B列)
    # -----------------------------------------------------
    K = sheet.range('B2').value
    expiry_date_val = sheet.range('B3').value
    option_type = sheet.range('B4').value
    initial_cash = sheet.range('B5').value
    fee_rate = sheet.range('B6').value
    threshold = sheet.range('B7').value
    engine_type = sheet.range('B8').value
    strategy_type = sheet.range('B9').value
    mc_n_steps = None
    fdm_n_space = None
    fdm_n_time = None

    if engine_type == 'MC':
        mc_n_steps = int(sheet.range('B10').value)
    elif engine_type == 'FDM':
        fdm_n_space = int(sheet.range('B10').value)
        fdm_n_time = int(sheet.range('B11').value)

    notional_amount = sheet.range('B12').value
    hedge_instrument = sheet.range('B13').value
    
    # -----------------------------------------------------
    # 2. 从 Excel 读取行情数据 (C列 - G列)
    # -----------------------------------------------------
    # 使用 expand('table') 自动扩展读取所有连续数据
    raw_data = sheet.range('C1').options(pd.DataFrame, index=False, expand='table').value
    
    df = raw_data.copy()
    
    if len(df.columns) >= 5:
        df.columns = ['Date', 'Spot', 'Future', 'Rate', 'Vol'] + list(df.columns[5:])
    else:
        df.columns = ['Date', 'Spot', 'Future', 'Rate'] + list(df.columns[4:])
        
    df['Date'] = pd.to_datetime(df['Date'])


    initial_real_spot = df['Spot'].iloc[0]
    
    # 3. 准备 Config
    config = {
        'K': K, 
        'Expiry_Date': pd.to_datetime(expiry_date_val),
        'initial_cash': float(initial_cash),
        'future_multiplier': 200,
        'notional_amount': float(notional_amount), 
        'initial_spot': initial_real_spot,

        'hedge_instrument': hedge_instrument, 
        'fee_rate': fee_rate,
        'threshold': threshold, 

        'engine_type': engine_type,
        'mc_n_steps': mc_n_steps,
        'fdm_n_space': fdm_n_space,
        'fdm_n_time': fdm_n_time
    }
    # -----------------------------------------------------
    # 3. 调用业务逻辑
    # -----------------------------------------------------
    result_df = run_backtest_logic(df, config, strategy_type=strategy_type, option_type=option_type)
    result_df = result_df.astype(object) # 先转为 object 以便处理混合类型
    
    def clean_cell(x):
        if isinstance(x, np.ndarray):
            return x.item()
        if hasattr(x, 'item'):
            return x.item()
        return x
        
    # 对所有列应用清洗
    for col in result_df.columns:
        result_df[col] = result_df[col].apply(clean_cell)
    

    # -----------------------------------------------------
    # 4. 画图并嵌入 Excel
    # -----------------------------------------------------
    fig = PlotUtils.plot_hedging_results(result_df, config['initial_cash'], show_plot=False)
    pic_name = 'Hedging_PnL_Chart'
    
    for pic in sheet.pictures:
        if pic.name == pic_name:
            pic.delete()
            break
            
    # 将图片插入到指定位置 left, top 决定图片左上角的位置
    sheet.pictures.add(fig, name=pic_name, update=True, 
                       left=sheet.range('R1').left, 
                       top=sheet.range('R1').top)
    
    plt.close(fig)
    # -----------------------------------------------------
    # 5. 结果写回当前 Sheet
    # -----------------------------------------------------
    output_cell = 'I1'
    
    # 清除旧数据 (防止数据量变少时残留旧数据)
    # expand('table') 会清除整个连续区域，小心使用，或者只清除特定列
    sheet.range(output_cell).expand('table').clear_contents()
    
    def get_df_to_write(df):
        exclude_columns = ['Date', 'Spot', 'Hedge_Price', 'Rate', 'Asset_Val', 'dt_days', 'Cum_Interest']
        indicators = [col for col in df.columns if col not in exclude_columns]
        return df[indicators]

    sheet.range(output_cell).options(index=False).value = get_df_to_write(result_df)
    

    # print("Backtest Finished. Results and Chart updated.")