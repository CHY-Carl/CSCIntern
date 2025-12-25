import sys
sys.path.append('../../')

from PricingLib.Instruments.EuropeanOption import EuropeanOption
from PricingLib.PricingEngines.BS_Engine import AnalyticBSEngine
from PricingLib.PricingEngines.MC_Engine import MonteCarloEngine
from PricingLib.PricingEngines.FDM_Engine import FDMEngine
from PricingLib.Base.BaseLayer import MarketEnvironment
from PricingLib.Processes.GBM import GeometricBrownianMotion

import xlwings as xw
import numpy as np
import random

def run_series_option(sheet):
    # --- 1. 读取参数 ---
    S, T, r, sigma = sheet.range('B1:B4').value
    wb = xw.Book.caller()
    single_sheet = wb.sheets['Single_Option']
    M_mc = int(single_sheet.range('E1').value)
    M_fdm, N_fdm = map(int, single_sheet.range('E4:E5').value)

    K_list = sheet.range('A7').expand('down').value
    if isinstance(K_list, (int, float)): K_list = [K_list]
    K_arr = np.array(K_list) # (N,)

    # --- 2. 构建对象 ---
    market = MarketEnvironment(S, r, sigma, T)
    opt_call = EuropeanOption(K_arr, T, 'call')
    opt_put  = EuropeanOption(K_arr, T, 'put')
    gbm_process = GeometricBrownianMotion()

    bs_engine = AnalyticBSEngine()
    mc_engine = MonteCarloEngine(process=gbm_process, n_sims=M_mc, rng_type='sobol')
    fdm_engine = FDMEngine(process=gbm_process, M_space=M_fdm, N_time=N_fdm)

    # --- 3. 批量计算 (核心逻辑) ---
    
    def get_results(eng, opt, market=market, seed=None):
        if seed is None:
            seed = random.randint(0, 1000000)
        price = eng.calculate(opt, market)['price'] # -> (N,)
        delta = eng.get_delta(opt, market, seed=seed)
        gamma = eng.get_gamma(opt, market, seed=seed)
        vega  = eng.get_vega(opt, market, seed=seed)
        theta = eng.get_theta(opt, market, seed=seed)
        rho   = eng.get_rho(opt, market, seed=seed)
        vanna = eng.get_vanna(opt, market, seed=seed)
        volga = eng.get_volga(opt, market, seed=seed)
        
        return price, delta, gamma, vega, theta, rho, vanna, volga

    # 执行计算
    # BS
    bs_c_res = get_results(bs_engine, opt_call)
    bs_p_res = get_results(bs_engine, opt_put)
    
    # MC
    mc_c_res = get_results(mc_engine, opt_call)
    mc_p_res = get_results(mc_engine, opt_put)
    
    # FDM
    fdm_c_res = get_results(fdm_engine, opt_call)
    fdm_p_res = get_results(fdm_engine, opt_put)

    # --- 4. 组装数据 ---
    # Price Columns: BS(C,P), MC(C,P), FDM(C,P)
    price_data = np.column_stack((
        bs_c_res[0], bs_p_res[0],
        mc_c_res[0], mc_p_res[0],
        fdm_c_res[0], fdm_p_res[0]
    ))
    
    # Order per method: Call Delta, Put Delta, Gamma, Vega, Vanna, Volga, Call Theta, Put Theta, Call Rho, Put Rho
    
    def stack_greeks(c_res, p_res):
        return np.column_stack((
            c_res[1], p_res[1], # Delta
            c_res[2],           # Gamma (shared)
            c_res[3],           # Vega (shared)
            c_res[4], p_res[4], # Theta
            c_res[5], p_res[5],  # Rho
            c_res[6],           # Vanna
            c_res[7],           # Volga
        ))

    all_greeks = np.column_stack((
        stack_greeks(bs_c_res, bs_p_res),
        stack_greeks(mc_c_res, mc_p_res),
        stack_greeks(fdm_c_res, fdm_p_res)
    ))

    # --- 5. 写入 ---
    sheet.range('B7').value = price_data.tolist()
    sheet.range('I7').value = all_greeks.tolist()