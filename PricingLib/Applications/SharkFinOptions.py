import xlwings as xw
import random
from ..Base.BaseLayer import MarketEnvironment
from ..Instruments.SharkFinOption import UpAndOutCall  # 假设你把类放在这里
from ..PricingEngines.BS_Engine import AnalyticBSEngine
from ..PricingEngines.MC_Engine import MonteCarloEngine
from ..PricingEngines.FDM_Engine import FDMEngine
from ..Base.Utils import RandomContext
from ..Processes.GBM import GeometricBrownianMotion

def run_up_and_out_call_option(sheet):
    """
    鲨鱼鳍期权 (Up-and-Out Call) 定价入口。
    Excel 布局假设：
    B1: S, B2: K, B3: H (Barrier), B4: T, B5: r, B6: sigma, B7: Rebate
    E2: MC Sims, E5: FDM Space, E6: FDM Time
    """
    # 1. 读取参数
    # 注意：这里多读了 H 和 Rebate
    # 假设 Excel B列顺序: S, K, H, T, r, sigma, Rebate
    params = sheet.range('B1:B7').value
    S, K, H, T, r, sigma, rebate = params
    
    # 确保 M, N 是整数
    M_mc = int(sheet.range('E1').value)
    M_fdm = int(sheet.range('E4').value)
    N_fdm = int(sheet.range('E5').value)

    # 2. 构建积木
    market = MarketEnvironment(S, r, sigma, T)
    gbm_process = GeometricBrownianMotion()

    # 构建鲨鱼鳍产品
    shark_fin = UpAndOutCall(K=K, H=H, T=T, rebate=rebate)

    # 3. 构建引擎
    bs_engine = AnalyticBSEngine()
    mc_engine = MonteCarloEngine(process=gbm_process, n_sims=M_mc, n_steps=N_fdm, rng_type='sobol') 
    fdm_engine = FDMEngine(process=gbm_process, M_space=M_fdm, N_time=N_fdm)

    # 4. 计算 Price & Greeks
    def get_data(engine, option):
        with RandomContext(seed=random.randint(0, 1000000)):
            price = engine.calculate(option, market)['price']
            greeks = [
                engine.get_delta(option, market),
                engine.get_gamma(option, market),
                engine.get_vega(option, market),
                engine.get_theta(option, market),
                engine.get_rho(option, market),
                engine.get_vanna(option, market),
                engine.get_volga(option, market)
            ]
        return price, greeks

    # 5. 执行计算
    bs_p, bs_g = get_data(bs_engine, shark_fin)
    mc_p, mc_g = get_data(mc_engine, shark_fin)
    fdm_p, fdm_g = get_data(fdm_engine, shark_fin)

    # 6. 写入 Excel 
    sheet.range('B10').value = [float(bs_p), float(mc_p), float(fdm_p)]

    # Greeks
    greeks_rows = []
    # 顺序: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga
    for i in range(7):
        row = [bs_g[i], mc_g[i], fdm_g[i]]
        greeks_rows.append(row)

    greeks_data_safe = [[float(x) for x in row] for row in greeks_rows]
    sheet.range('B11').value = greeks_data_safe