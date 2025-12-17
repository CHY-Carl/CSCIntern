import sys
sys.path.append('../../')
from PricingLib.Base.BaseLayer import MarketEnvironment, PricingEngine, Instrument
from PricingLib.Base.Utils import MathUtils
from PricingLib.Base.Utils import SharkFinFormulas
from PricingLib.Instruments.EuropeanOption import EuropeanOption
from PricingLib.Instruments.SharkFinOption import UpAndOutCall

import numpy as np
from typing import Dict, Any, Union




class AnalyticBSEngine(PricingEngine):
    """
    使用 Black-Scholes 解析公式定价。
    覆盖了基类的数值 Greeks 方法，提供精确解。
    """
    
    # --- 内部辅助函数：计算 d1, d2 ---
    def _calc_d_params(self, S, K, T, r, sigma):
        # 向量化计算，且处理除零等边界情况
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
        
        # 边界修正：当 T->0 或 Sigma->0 时
        # 如果 S > K (ITM), d1, d2 -> +inf
        # 如果 S < K (OTM), d1, d2 -> -inf
        d1 = np.where(np.isfinite(d1), d1, np.inf * np.sign(np.log(S/K)))
        d2 = np.where(np.isfinite(d2), d2, np.inf * np.sign(np.log(S/K)))
        # 如果 S==K 且 T->0，d -> 0
        d1 = np.nan_to_num(d1); d2 = np.nan_to_num(d2)
        
        return d1, d2

    # --- 1. 核心定价实现 ---
    def calculate(self, option: Instrument, market: MarketEnvironment) -> Dict[str, Any]:
        if not isinstance(option, (EuropeanOption, UpAndOutCall)):
            raise ValueError("BS Engine only supports European Options or Up-and-Out Calls.")
            
        S, K, T, r, sigma = market.S, option.K, market.T, market.r, market.sigma
        if isinstance(option, EuropeanOption):
            d1, d2 = self._calc_d_params(S, K, T, r, sigma)
            N = MathUtils.norm_cdf
            if option.option_type == 'call':
                price = S * N(d1) - K * np.exp(-r * T) * N(d2)
            else:
                price = K * np.exp(-r * T) * N(-d2) - S * N(-d1)
                
            # 边界处理 (T=0 or Sigma=0)
            # 虽然公式本身在极限下可能成立，但直接用 payoff 更稳健
            intrinsic_val = option.payoff(S)
            price = np.where((T <= 0) | (sigma <= 0), intrinsic_val, price)
            
            # 这里的字典只返回 calculate 职责范围内的东西，通常是价格
            # Greeks 通过 get_... 接口获取
            return {'price': price}
        elif isinstance(option, UpAndOutCall):
            H = option.H
            K = option.K
            rebate = option.rebate

            dt = 1/252
            price = SharkFinFormulas.up_and_out_call_discrete(S, K, H, T, r, sigma, dt=dt, rebate=rebate)
            return {'price': price}
        
        else:
            raise ValueError("Unsupported option type for AnalyticBSEngine.")

    # --- 2. 覆盖基类接口，提供解析解 (Analytical Greeks) ---

    def get_delta(self, option, market):
        if isinstance(option, EuropeanOption):
            S, K, T, r, sigma = market.S, option.K, option.T, market.r, market.sigma
            if T <= 0 or sigma <= 0:
                return super().get_delta(option, market) # 回退到基类数值解处理边界

            d1, _ = self._calc_d_params(S, K, T, r, sigma)
            if option.option_type == 'call':
                return MathUtils.norm_cdf(d1)
            else:
                return MathUtils.norm_cdf(d1) - 1
        
        return super().get_delta(option, market) # 其他产品回退基类

    def get_gamma(self, option, market):
        if isinstance(option, EuropeanOption):
            S, K, T, r, sigma = market.S, option.K, option.T, market.r, market.sigma
            if T <= 0 or sigma <= 0: return 0.0
            
            d1, _ = self._calc_d_params(S, K, T, r, sigma)
            return MathUtils.norm_pdf(d1) / (S * sigma * np.sqrt(T))
        
        return super().get_gamma(option, market) # 其他产品回退基类

    def get_vega(self, option, market):
        if isinstance(option, EuropeanOption):
            # 单位: per 1% vol
            S, K, T, r, sigma = market.S, option.K, option.T, market.r, market.sigma
            if T <= 0 or sigma <= 0: return 0.0
            
            d1, _ = self._calc_d_params(S, K, T, r, sigma)
            return S * MathUtils.norm_pdf(d1) * np.sqrt(T) / 100

        return super().get_vega(option, market) # 其他产品回退基类

    def get_theta(self, option, market):
        if isinstance(option, EuropeanOption):
            # 单位: per 1 day
            S, K, T, r, sigma = market.S, option.K, option.T, market.r, market.sigma
            if T <= 0 or sigma <= 0: return 0.0
            
            d1, d2 = self._calc_d_params(S, K, T, r, sigma)
            N = MathUtils.norm_cdf
            n = MathUtils.norm_pdf
            
            term1 = -S * n(d1) * sigma / (2 * np.sqrt(T))
            if option.option_type == 'call':
                term2 = -r * K * np.exp(-r*T) * N(d2)
            else:
                term2 = r * K * np.exp(-r*T) * N(-d2)
                
            return (term1 + term2) / 365
        
        return super().get_theta(option, market) # 其他产品回退基类

    def get_rho(self, option, market):
        if isinstance(option, EuropeanOption):
            # 单位: per 1% rate
            S, K, T, r, sigma = market.S, option.K, option.T, market.r, market.sigma
            if T <= 0 or sigma <= 0: return 0.0
            
            _, d2 = self._calc_d_params(S, K, T, r, sigma)
            if option.option_type == 'call':
                rho = K * T * np.exp(-r*T) * MathUtils.norm_cdf(d2)
            else:
                rho = -K * T * np.exp(-r*T) * MathUtils.norm_cdf(-d2)
                
            return rho / 100
        
        return super().get_rho(option, market) # 其他产品回退基类

    def get_vanna(self, option, market):
        if isinstance(option, EuropeanOption):
            # 解析公式: -n(d1) * d2 / sigma
            # 单位: Delta change per 1% vol
            S, K, T, r, sigma = market.S, option.K, option.T, market.r, market.sigma
            if T <= 0 or sigma <= 0: return 0.0
            
            d1, d2 = self._calc_d_params(S, K, T, r, sigma)
            return -MathUtils.norm_pdf(d1) * d2 / sigma / 100
        
        return super().get_vanna(option, market) # 其他产品回退基类

    def get_volga(self, option, market):
        if isinstance(option, EuropeanOption):
            # 解析公式: Vega * d1 * d2 / sigma
            # 单位: Vega change per 1% vol
            S, K, T, r, sigma = market.S, option.K, option.T, market.r, market.sigma
            if T <= 0 or sigma <= 0: return 0.0
            
            d1, d2 = self._calc_d_params(S, K, T, r, sigma)
            raw_vega = S * np.sqrt(T) * MathUtils.norm_pdf(d1)
            return raw_vega * d1 * d2 / sigma / 10000
        
        return super().get_volga(option, market) # 其他产品回退基类