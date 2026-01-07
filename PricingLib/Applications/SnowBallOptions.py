import sys
sys.path.append('../../')


# --- 导入您项目中的核心类 ---
from PricingLib.Instruments.SnowBallOption import SnowballOption, UnifiedSnowball
from PricingLib.PricingEngines.MC_Engine import MonteCarloEngine
from PricingLib.PricingEngines.FDM_Engine import FDMEngine, EventFDMEngine
from PricingLib.Processes.GBM import GeometricBrownianMotion
from PricingLib.Base.BaseLayer import MarketEnvironment

import numpy as np
import pandas as pd
import xlwings as xw
import random



def calculate_snowball_fdm(S0, r, sigma, T_total, knock_out_pct, knock_in_pct, coupon_rate, rebate_rate, 
                           T_rem=None, start_month=3):
    """
    使用 Unified FDM (双轨并行法) 计算雪球的价格和 Greeks。
    """
    
    if T_rem is None:
        T_rem = T_total
    B_out = S0 * knock_out_pct
    B_in = S0 * knock_in_pct 
    
    obs_months = list(range(start_month, int(T_total * 12) + 1))

    unified_product = UnifiedSnowball(
        initial_price=S0,
        knock_in_barrier=B_in,
        knock_out_barrier=B_out,
        coupon_rate=coupon_rate,
        rebate_rate=rebate_rate,
        T_total=T_total,
        observation_months=obs_months,
        T_rem=T_rem
    )
    
    process = GeometricBrownianMotion()
    engine = EventFDMEngine(process, M_space=4000, N_time=252)
    market = MarketEnvironment(S=S0, r=r, sigma=sigma, T=T_rem)
    
    price_res = engine.calculate(unified_product, market)
    price = float(price_res['price']) if isinstance(price_res, dict) else float(price_res)

    greeks_results = [
        engine.get_delta(unified_product, market),
        engine.get_gamma(unified_product, market),
        engine.get_vega(unified_product, market),
        engine.get_theta(unified_product, market),
        engine.get_rho(unified_product, market),
        engine.get_vanna(unified_product, market),
        engine.get_volga(unified_product, market)
    ]
    
    return price, greeks_results


def run_snowball(sheet: xw.Sheet):
    S0, knock_out_pct, knock_in_pct, T, r, sigma, coupon_rate, rebate_rate = sheet.range('B1:B8').value
    
    n_sims = int(sheet.range('E1').value)
    n_steps_per_year = int(sheet.range('E5').value)
    
    # =================================================================
    # 日历处理：将日期规则转换为模拟索引 
    # =================================================================
    num_steps = int(n_steps_per_year * T)
    dt = T / num_steps
    days_per_month = n_steps_per_year / 12

    knock_in_indices = list(range(1, num_steps + 1))
    
    start_month = 3 
    end_month = int(T * 12)
    # MC 使用的是时间步索引列表
    knock_out_indices = [int(round(m * days_per_month)) for m in range(start_month, end_month + 1) if int(round(m * days_per_month)) <= num_steps]

    # =================================================================
    # MC 
    # =================================================================
    snowball_product_mc = SnowballOption(
        S0=S0,
        knock_out_barrier=S0 * knock_out_pct,
        knock_in_barrier=S0 * knock_in_pct,
        coupon_rate=coupon_rate,
        rebate_rate=rebate_rate,
        T_total=T,
        knock_out_obs_indices=knock_out_indices,
        knock_in_obs_indices=knock_in_indices,
        dt=dt,
    )

    gbm_process = GeometricBrownianMotion()
    mc_engine = MonteCarloEngine(process=gbm_process, n_sims=n_sims, n_steps=num_steps, rng_type='sobol')
    market = MarketEnvironment(S=S0, r=r, sigma=sigma, T=T)

    mc_price_result = mc_engine.calculate(snowball_product_mc, market)
    mc_price = float(mc_price_result['price'])

    def get_all_greeks_mc(eng, opt, mkt, seed=None):
        if seed is None:
            seed = random.randint(0, 1000000)
        return [
            eng.get_delta(opt, mkt, seed=seed),
            eng.get_gamma(opt, mkt, seed=seed),
            eng.get_vega(opt, mkt, seed=seed),
            eng.get_theta(opt, mkt, seed=seed),
            eng.get_rho(opt, mkt, seed=seed),
            eng.get_vanna(opt, mkt, seed=seed),
            eng.get_volga(opt, mkt, seed=seed)
        ]

    mc_greeks = get_all_greeks_mc(mc_engine, snowball_product_mc, market)

    # =================================================================
    #  FDM 计算 
    # =================================================================
    fdm_price, fdm_greeks = calculate_snowball_fdm(
        S0=S0, r=r, sigma=sigma, T_total=T, 
        knock_out_pct=knock_out_pct, knock_in_pct=knock_in_pct, 
        coupon_rate=coupon_rate, rebate_rate=rebate_rate, 
        T_rem=T, start_month=start_month
    )

    # =================================================================
    price_rows = [mc_price, fdm_price]
    greek_rows = []
    
    for i in range(7):
        row = [mc_greeks[i], fdm_greeks[i]]
        greek_rows.append(row)
    
    greeks_data_safe = [[float(x) for x in row] for row in greek_rows]
    
    sheet.range('B11').value = price_rows
    sheet.range('B12').value = greeks_data_safe