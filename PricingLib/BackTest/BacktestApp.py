import pandas as pd
from ..Base.BaseLayer import MarketEnvironment
from ..Instruments.EuropeanOption import EuropeanOption
from ..Instruments.SharkFinOption import UpAndOutCall, DoubleSharkFin
from ..PricingEngines.BS_Engine import AnalyticBSEngine
from ..PricingEngines.MC_Engine import MonteCarloEngine
from ..PricingEngines.FDM_Engine import FDMEngine
from .Account import Account
from .Strategy import DeltaHedgeStrategy
from .Simulator import BacktestSimulator
from .Analytics import BacktestAnalyzer
from ..Processes.GBM import GeometricBrownianMotion

def run_backtest_logic(market_data_df, config, strategy_type='delta_hedge', option_type='VanillaCall'):
    """
    纯业务逻辑入口。
    
    Args:
        market_data_df: 从 Excel 读取并清洗好的 DataFrame。
        config: 从 Excel 读取的配置字典。
        
    Returns:
        result_df: 包含回测结果的 DataFrame。
    """
    print("Initializing Backtest Engine...")

    # 1. 组装对象
    hedge_inst = config.get('hedge_instrument', 'Future')
    initial_cash = config.get('initial_cash', 10_000_000)
    account = Account(initial_cash=initial_cash, mode=hedge_inst)
    
    # 默认使用 BS 引擎
    engine_type = config.get('engine_type', 'BS')
    if engine_type == 'BS':
        engine = AnalyticBSEngine()

    elif engine_type == 'MC':
        n_steps = config.get('mc_n_steps', 50)
        gbm_process = GeometricBrownianMotion()
        engine = MonteCarloEngine(process=gbm_process, n_sims=20000, n_steps=n_steps, rng_type='sobol')

    elif engine_type == 'FDM':
        n_space = config.get('fdm_n_space', 100)
        n_time = config.get('fdm_n_time', 100)
        gbm_process = GeometricBrownianMotion()
        engine = FDMEngine(process=gbm_process, M_space=n_space, N_time=n_time)

    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")
    

    if strategy_type == 'delta_hedge':
        strategy = DeltaHedgeStrategy(
            engine=engine,
            num_options=config['num_options'],
            hedge_instrument=hedge_inst,
            future_multiplier=config.get('future_multiplier', 1),
            threshold=config.get('threshold', 0.0)
        )
    else:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")
    
    if option_type == 'VanillaCall':
        # 创建产品
        opt_product = EuropeanOption(
            K=config['K'], 
            T=1.0, # 初始 T，实际 T 由 Simulator 根据日期动态计算
            option_type='call'
        )
    elif option_type == 'SharkFinCall':
        opt_product = UpAndOutCall(
            K=config['K'],
            H=config['H'],
            rebate=config['rebate'],
            T=1.0,
        )

    #TODO: 支持更多产品
    else:
        raise ValueError(f"Unsupported option type: {option_type}")

    # 2. 启动模拟器
    sim = BacktestSimulator(
        account=account,
        strategy=strategy,
        market_data_df=market_data_df,
        config=config,
        product=opt_product
    )
    
    raw_df = sim.run()
    
    analyzer = BacktestAnalyzer(raw_df, config['initial_cash'])
    clean_df = analyzer.get_clean_df()

    return clean_df