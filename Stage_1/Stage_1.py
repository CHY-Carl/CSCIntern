import sys 
from pathlib import Path

# 1. 获取当前脚本文件的绝对路径
# 结果类似: /Users/cuihaoyuan/CSCIntern/Stage_1/Stage_1.py
current_file = Path(__file__).resolve()

# 2. 向上回溯两级，找到 'CSCINTERN' 文件夹 (即 PricingLib 所在的父级目录)
# .parent -> Stage_1 文件夹
# .parent.parent -> CSCIntern 文件夹
project_root = current_file.parent.parent

# 3. 将项目根目录加入 Python 搜索路径 (插入到第0位，确保优先级最高)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- 测试打印 (可选，调试用) ---
# print(f"Project Root added: {project_root}")

from PricingLib.Applications.SingleOption import *
from PricingLib.Applications.SeriesOption import *
from PricingLib.Applications.SharkFinOptions import *
from PricingLib.Applications.DeltaHedgeVanilla import *
from PricingLib.Applications.DeltaHedgeSharkFin import *

import xlwings as xw

def main():
    wb = xw.Book.caller()
    active_sheet = wb.sheets.active

    # #! python debug 
    # active_sheet = wb.sheets['VanillaCall_DeltaHedge_FDM']
    # #! python debug


    import time
    start_time = time.time()
    active_sheet.range('H1').value = "Running..."

    if active_sheet.name == 'Single_Option':
        run_single_option(active_sheet)
    elif active_sheet.name == 'Series_Option':
        run_series_option(active_sheet)
    elif active_sheet.name == 'UpAndOutCall_Option':
        run_up_and_out_call_option(active_sheet)
    elif active_sheet.name == 'DoubleSharkFin_Option':
        run_double_sharkfin_option(active_sheet)
    elif (active_sheet.name == 'VanillaCall_DeltaHedge_FDM') or (active_sheet.name == 'VanillaCall_DeltaHedge_MC'):
        run_hedge_backtest_vanilla(active_sheet)
    elif (active_sheet.name == 'SharkFinCall_DeltaHedge_FDM') or (active_sheet.name == 'SharkFinCall_DeltaHedge_MC'):
        run_hedge_backtest_sharkfin(active_sheet)
    else:
        active_sheet.range('A1').value = "Error: 请在 Single_Option 或 Series_Option 页面运行"

    end_time = time.time()

    active_sheet.range('H1').value = f"Done in {end_time - start_time:.4f}s"

if __name__ == "__main__":
    xw.Book("Stage_1.xlsm").set_mock_caller()
    main()