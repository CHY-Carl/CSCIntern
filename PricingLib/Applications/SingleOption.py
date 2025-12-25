import sys
sys.path.append('../../')

from PricingLib.Instruments.EuropeanOption import EuropeanOption
from PricingLib.PricingEngines.BS_Engine import AnalyticBSEngine
from PricingLib.PricingEngines.MC_Engine import MonteCarloEngine
from PricingLib.PricingEngines.FDM_Engine import FDMEngine
from PricingLib.Base.BaseLayer import MarketEnvironment
from PricingLib.Processes.GBM import GeometricBrownianMotion

import random

def run_single_option(sheet):
    S, K, T, r, sigma = sheet.range('B1:B5').value
    M_mc = int(sheet.range('E1').value)
    M_fdm, N_fdm = map(int, sheet.range('E4:E5').value)

    market = MarketEnvironment(S, r, sigma, T)
    opt_call = EuropeanOption(K, T, 'call')
    opt_put  = EuropeanOption(K, T, 'put')
    gbm_process = GeometricBrownianMotion()

    bs_engine = AnalyticBSEngine()
    mc_engine = MonteCarloEngine(process=gbm_process, n_sims=M_mc, n_steps=N_fdm, rng_type='sobol', path_gen_method='exact') 
    fdm_engine = FDMEngine(process=gbm_process, M_space=M_fdm, N_time=N_fdm)

    bs_res_c = bs_engine.calculate(opt_call, market)
    bs_res_p = bs_engine.calculate(opt_put, market)
    def get_all_greeks(eng, opt, mkt, seed=None):
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

    bs_greeks_c = get_all_greeks(bs_engine, opt_call, market)
    bs_greeks_p = get_all_greeks(bs_engine, opt_put, market)

    mc_res_c = mc_engine.calculate(opt_call, market)
    mc_res_p = mc_engine.calculate(opt_put, market)
    mc_greeks_c = get_all_greeks(mc_engine, opt_call, market)
    mc_greeks_p = get_all_greeks(mc_engine, opt_put, market)

    fdm_res_c = fdm_engine.calculate(opt_call, market)
    fdm_res_p = fdm_engine.calculate(opt_put, market)
    fdm_greeks_c = get_all_greeks(fdm_engine, opt_call, market)
    fdm_greeks_p = get_all_greeks(fdm_engine, opt_put, market)

    sheet.range('B7').value = [[float(bs_res_c['price'])], [float(bs_res_p['price'])]]
    sheet.range('C7').value = [[float(mc_res_c['price'])], [float(mc_res_p['price'])]]
    sheet.range('D7').value = [[float(fdm_res_c['price'])], [float(fdm_res_p['price'])]]

    greeks_data = []
    # Delta
    greeks_data.append([bs_greeks_c[0], mc_greeks_c[0], fdm_greeks_c[0]]) # Call Delta
    greeks_data.append([bs_greeks_p[0], mc_greeks_p[0], fdm_greeks_p[0]]) # Put Delta
    # Gamma 
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

    greeks_data_safe = [[float(x) for x in row] for row in greeks_data]
    sheet.range('B9').value = greeks_data_safe