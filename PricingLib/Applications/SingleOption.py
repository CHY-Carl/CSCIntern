import sys
sys.path.append('../../')

from PricingLib.Instruments.EuropeanOption import EuropeanOption
from PricingLib.PricingEngines.BS_Engine import AnalyticBSEngine
from PricingLib.PricingEngines.MC_Engine import MonteCarloEngine
from PricingLib.PricingEngines.FDM_Engine import FDMEngine
from PricingLib.Base.BaseLayer import MarketEnvironment
from PricingLib.Base.Utils import RandomContext
from PricingLib.Processes.GBM import GeometricBrownianMotion

import random

def run_single_option(sheet):
    # --- 1. 读取参数 ---
    S, K, T, r, sigma = sheet.range('B1:B5').value
    M_mc = int(sheet.range('E1').value)
    M_fdm, N_fdm = map(int, sheet.range('E4:E5').value)

    # --- 2. 构建对象 (积木) ---
    market = MarketEnvironment(S, r, sigma, T)
    # 分别创建 Call 和 Put 产品
    opt_call = EuropeanOption(K, T, 'call')
    opt_put  = EuropeanOption(K, T, 'put')

    gbm_process = GeometricBrownianMotion()

    # --- 3. 构建引擎 ---
    bs_engine = AnalyticBSEngine()
    mc_engine = MonteCarloEngine(process=gbm_process, n_sims=M_mc, rng_type='sobol') 
    fdm_engine = FDMEngine(process=gbm_process, M_space=M_fdm, N_time=N_fdm)

    # --- 4. 执行计算 (一键获取 Price + Greeks) ---
    
    # BS
    bs_res_c = bs_engine.calculate(opt_call, market)
    bs_res_p = bs_engine.calculate(opt_put, market)
    # 补充 Greeks (基类方法调用)
    # 注意：AnalyticBSEngine 的 calculate 只返回了 price
    # 我们需要手动调 get_delta 等，或者修改 calculate 一次性返回所有
    # 这里演示显式调用，清晰明了

    def get_all_greeks(eng, opt, mkt):
        with RandomContext(seed=random.randint(0, 1000000)):
            return [
                eng.get_delta(opt, mkt),
                eng.get_gamma(opt, mkt),
                eng.get_vega(opt, mkt),
                eng.get_theta(opt, mkt),
                eng.get_rho(opt, mkt),
                eng.get_vanna(opt, mkt),
                eng.get_volga(opt, mkt)
            ]

    bs_greeks_c = get_all_greeks(bs_engine, opt_call, market)
    bs_greeks_p = get_all_greeks(bs_engine, opt_put, market)

    # MC
    mc_res_c = mc_engine.calculate(opt_call, market)
    mc_res_p = mc_engine.calculate(opt_put, market)
    mc_greeks_c = get_all_greeks(mc_engine, opt_call, market)
    mc_greeks_p = get_all_greeks(mc_engine, opt_put, market)

    # FDM
    fdm_res_c = fdm_engine.calculate(opt_call, market)
    fdm_res_p = fdm_engine.calculate(opt_put, market)
    fdm_greeks_c = get_all_greeks(fdm_engine, opt_call, market)
    fdm_greeks_p = get_all_greeks(fdm_engine, opt_put, market)

    # --- 5. 写入 Excel ---
    # Price
    sheet.range('B7').value = [[float(bs_res_c['price'])], [float(bs_res_p['price'])]]
    sheet.range('C7').value = [[float(mc_res_c['price'])], [float(mc_res_p['price'])]]
    sheet.range('D7').value = [[float(fdm_res_c['price'])], [float(fdm_res_p['price'])]]

    # Greeks 数据组装 (Call/Put 交替或按行)
    # 你的 Excel 布局是: 
    # Row 10: Call Delta
    # Row 11: Put Delta
    # Row 12: Gamma
    # ...
    
    # 提取数据
    greeks_data = []
    # Delta
    greeks_data.append([bs_greeks_c[0], mc_greeks_c[0], fdm_greeks_c[0]]) # Call Delta
    greeks_data.append([bs_greeks_p[0], mc_greeks_p[0], fdm_greeks_p[0]]) # Put Delta
    # Gamma (取 Call 的即可，理论上相同)
    greeks_data.append([bs_greeks_c[1], mc_greeks_c[1], fdm_greeks_c[1]])
    # Vega
    greeks_data.append([bs_greeks_c[2], mc_greeks_c[2], fdm_greeks_c[2]])
    # Theta
    greeks_data.append([bs_greeks_c[3], mc_greeks_c[3], fdm_greeks_c[3]]) # Call Theta
    greeks_data.append([bs_greeks_p[3], mc_greeks_p[3], fdm_greeks_p[3]]) # Put Theta
    # Rho
    greeks_data.append([bs_greeks_c[4], mc_greeks_c[4], fdm_greeks_c[4]]) # Call Rho
    greeks_data.append([bs_greeks_p[4], mc_greeks_p[4], fdm_greeks_p[4]]) # Put Rho
    # Vanna
    greeks_data.append([bs_greeks_c[5], mc_greeks_c[5], fdm_greeks_c[5]])
    # Volga
    greeks_data.append([bs_greeks_c[6], mc_greeks_c[6], fdm_greeks_c[6]])

    # 类型转换并写入
    greeks_data_safe = [[float(x) for x in row] for row in greeks_data]
    sheet.range('B9').value = greeks_data_safe