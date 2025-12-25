import xlwings as xw
import random
from ..Base.BaseLayer import MarketEnvironment
from ..Instruments.SharkFinOption import UpAndOutCall, DoubleSharkFin  # 假设你把类放在这里
from ..PricingEngines.BS_Engine import AnalyticBSEngine
from ..PricingEngines.MC_Engine import MonteCarloEngine
from ..PricingEngines.FDM_Engine import FDMEngine
from ..Processes.GBM import GeometricBrownianMotion

def run_up_and_out_call_option(sheet):
    """
    鲨鱼鳍期权 (Up-and-Out Call) 定价入口。
    Excel 布局假设：
    B1: S, B2: K, B3: H (Barrier), B4: T, B5: r, B6: sigma, B7: Rebate
    E2: MC Sims, E5: FDM Space, E6: FDM Time
    """
    params = sheet.range('B1:B7').value
    S, K, H, T, r, sigma, rebate = params
    
    M_mc = int(sheet.range('E1').value)
    M_fdm = int(sheet.range('E4').value)
    N_fdm = int(sheet.range('E5').value)

    market = MarketEnvironment(S, r, sigma, T)
    gbm_process = GeometricBrownianMotion()
    shark_fin = UpAndOutCall(K=K, H=H, T=T, rebate=rebate)


    bs_engine = AnalyticBSEngine()
    mc_engine = MonteCarloEngine(process=gbm_process, n_sims=M_mc, n_steps=N_fdm, rng_type='sobol') 
    fdm_engine = FDMEngine(process=gbm_process, M_space=M_fdm, N_time=N_fdm)

    def get_data(engine, option, market=market, seed=None):
        if seed is None:
            seed = random.randint(0, 1000000)

        price = engine.calculate(option, market)['price']
        greeks = [
            engine.get_delta(option, market, seed=seed),
            engine.get_gamma(option, market, seed=seed),
            engine.get_vega(option, market, seed=seed),
            engine.get_theta(option, market, seed=seed),
            engine.get_rho(option, market, seed=seed),
            engine.get_vanna(option, market, seed=seed),
            engine.get_volga(option, market, seed=seed)
        ]
        return price, greeks

    bs_p, bs_g = get_data(bs_engine, shark_fin)
    mc_p, mc_g = get_data(mc_engine, shark_fin)
    fdm_p, fdm_g = get_data(fdm_engine, shark_fin)

    sheet.range('B10').value = [float(bs_p), float(mc_p), float(fdm_p)]

    # Greeks
    greeks_rows = []
    # 顺序: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga
    for i in range(7):
        row = [bs_g[i], mc_g[i], fdm_g[i]]
        greeks_rows.append(row)

    greeks_data_safe = [[float(x) for x in row] for row in greeks_rows]
    sheet.range('B11').value = greeks_data_safe


def run_double_sharkfin_option(sheet):
    """
    双鲨鱼鳍期权 (Double Shark Fin) 定价入口。
    Excel 布局假设：
    B1: S, B2: K_L, B3: K_U, B4: B_L, B5: B_U, B6:R_L, B7:R_U, B8:T, B9: r, B10: sigma
    E1: MC Sims, E4: FDM Space, E5: FDM Time
    """
    params = sheet.range('B1:B10').value
    S, K_L, K_U, H_L, H_U, R_L, R_U, T, r, sigma = params

    M_mc = int(sheet.range('E1').value)
    M_fdm = int(sheet.range('E4').value)
    N_fdm = int(sheet.range('E5').value)

    market = MarketEnvironment(S, r, sigma, T)
    gbm_process = GeometricBrownianMotion()
    shark_fin = DoubleSharkFin(K_L=K_L, K_U=K_U, H_L=H_L, H_U=H_U, T=T, R_L=R_L, R_U=R_U)

    # 3. 构建引擎
    mc_engine = MonteCarloEngine(process=gbm_process, n_sims=M_mc, n_steps=N_fdm, rng_type='sobol')
    fdm_engine = FDMEngine(process=gbm_process, M_space=M_fdm, N_time=N_fdm)
    
    def get_data(engine, prod, market=market, seed=None):
        if seed is None:
            seed = random.randint(0, 1000000)
 
            price = engine.calculate(prod, market)['price']
            greeks = [
                engine.get_delta(prod, market, seed=seed),
                engine.get_gamma(prod, market, seed=seed),
                engine.get_vega(prod, market, seed=seed),
                engine.get_theta(prod, market, seed=seed),
                engine.get_rho(prod, market, seed=seed),
                engine.get_vanna(prod, market, seed=seed),
                engine.get_volga(prod, market, seed=seed)
            ]
        return price, greeks
    
    mc_p, mc_g = get_data(mc_engine, shark_fin)
    fdm_p, fdm_g = get_data(fdm_engine, shark_fin)

    sheet.range('B13').value = [float(mc_p), float(fdm_p)]
    greeks_rows = []
    for i in range(7):
        row = [mc_g[i], fdm_g[i]]
        greeks_rows.append(row)
    
    greeks_data_safe = [[float(x) for x in row] for row in greeks_rows]
    sheet.range('B14').value = greeks_data_safe