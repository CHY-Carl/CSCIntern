import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_banded
import xlwings as xw

def bs_call(S, K, T, r, sigma):
    S, K, T, r, sigma = map(np.asarray, [S, K, T, r, sigma])

    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

    d1 = np.where(np.isfinite(d1), d1, np.inf * np.sign(d1))
    d2 = np.where(np.isfinite(d2), d2, np.inf * np.sign(d2))

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return np.where((T <= 0) | (sigma <= 0), np.maximum(0, S - K), price)

def bs_put(S, K, T, r, sigma):
    S, K, T, r, sigma = map(np.asarray, [S, K, T, r, sigma])

    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

    d1 = np.where(np.isfinite(d1), d1, np.inf * np.sign(d1))
    d2 = np.where(np.isfinite(d2), d2, np.inf * np.sign(d2))

    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return np.where((T <= 0) | (sigma <= 0), np.maximum(0, K - S), price)

def mc_pricer(S, K, T, r, sigma, option_type, M=100000):
    if (T <= 0) or (sigma <= 0):
        return np.maximum(0., S - K) if option_type == 'call' else np.maximum(0., K - S)
    
    Z = np.random.standard_normal(int(M/2))
    Z = np.concatenate((Z, -Z))

    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)

    return np.mean(payoff) * np.exp(-r * T)

def mc_series_pricer(S, K_array, T, r, sigma, M=100000):
    if (T <= 0) or (sigma <= 0):
        call_prices = np.maximum(0, S - K_array)
        put_prices = np.maximum(0, K_array - S)
        return call_prices, put_prices
        
    Z = np.random.standard_normal(int(M/2))
    Z = np.concatenate((Z, -Z))
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    
    # Broadcasting: (M,) vs (num_K,) -> (M, num_K)
    call_payoffs = np.maximum(ST[:, np.newaxis] - K_array, 0)
    put_payoffs = np.maximum(K_array - ST[:, np.newaxis], 0)
    
    call_prices = np.mean(call_payoffs, axis=0) * np.exp(-r * T)
    put_prices = np.mean(put_payoffs, axis=0) * np.exp(-r * T)

    return call_prices, put_prices

def fdm_pricer(S, K, T, r, sigma, option_type, M_space=100, N_time=100, return_full_grid=False):
    if (T <= 0) or (sigma <= 0):
        price = np.maximum(0., S - K) if option_type == 'call' else np.maximum(0., K - S)
        if not return_full_grid:
            return price
        else: # 如果需要，即使在边界情况下也返回兼容的输出
            S_vec_dummy = np.linspace(0, 2*S if S>0 else 100, M_space + 1)
            V_dummy = np.maximum(0, S_vec_dummy - K) if option_type=='call' else np.maximum(0, K-S_vec_dummy)
            return {'price': price, 'S_vec': S_vec_dummy, 'V_vec': V_dummy}

    k_std = 5.0 
    S_stat_max = S * np.exp(k_std * sigma * np.sqrt(T))
    S_max = max(S_stat_max, K) * 1.1
    
    S_vec = np.linspace(0, S_max, M_space + 1)
    V_matrix = fdm_series_pricer(S_vec, np.array([K]), T, r, sigma, option_type, N_time)
    price = np.interp(S, S_vec, V_matrix[:, 0])
    
    if not return_full_grid:
        return price
    else:
        # 如果调用者需要，返回一个包含所有信息的字典
        return {
            'price': price,
            'S_vec': S_vec,
            'V_vec': V_matrix[:, 0]
        }


def fdm_series_pricer(S_vec, K_array, T, r, sigma, option_type, N_time=100):
    M_space = len(S_vec) - 1
    dt = T / N_time
    
    j = np.arange(1, M_space)
    a = 0.25 * dt * (sigma**2 * j**2 - r * j)
    b = -0.5 * dt * (sigma**2 * j**2 + r)
    c = 0.25 * dt * (sigma**2 * j**2 + r * j)
    
    M1_banded = np.zeros((3, M_space - 1))
    M1_banded[0, 1:] = -c[:-1]
    M1_banded[1, :] = 1 - b
    M1_banded[2, :-1] = -a[1:]
    
    M2 = np.diag(1 + b) + np.diag(a[1:], k=-1) + np.diag(c[:-1], k=1)
    
    V = np.maximum(S_vec[:, np.newaxis] - K_array, 0) if option_type == 'call' else np.maximum(K_array - S_vec[:, np.newaxis], 0)
        
    for i in range(N_time):
        rhs = M2 @ V[1:-1, :]
        
        # --- 边界条件修正 ---
        time_to_expiry = T - (i + 1) * dt
        
        # 1. 上边界 (S=S_max)
        if option_type == 'call':
            boundary_cond_upper = S_vec[-1] - K_array * np.exp(-r * time_to_expiry)
            rhs[-1, :] += c[-1] * boundary_cond_upper
        
        # 2. 下边界 (S=0)
        if option_type == 'put':
            boundary_cond_lower = K_array * np.exp(-r * time_to_expiry)
            # a[0] 是 a 向量的第一个元素，对应 j=1
            # 它会影响 rhs 的第一行（j=1）
            rhs[0, :] += a[0] * boundary_cond_lower

        V[1:-1, :] = solve_banded((1, 1), M1_banded, rhs, check_finite=False)
    return V

def bs_delta(S, K, T, r, sigma, option_type):
    S, K, T, r, sigma = map(np.asarray, [S, K, T, r, sigma])
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type.lower() == 'call':
        return np.where((T <= 0) | (sigma <= 0), np.where(S > K, 1.0, 0.0), norm.cdf(d1))
    else:
        return np.where((T <= 0) | (sigma <= 0), np.where(S < K, -1.0, 0.0), norm.cdf(d1) - 1)

def bs_gamma(S, K, T, r, sigma):
    S, K, T, r, sigma = map(np.asarray, [S, K, T, r, sigma])
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    pdf_d1 = norm.pdf(d1)
    return np.where((T <= 0) | (sigma <= 0), 0.0, pdf_d1 / (S * sigma * np.sqrt(T)))

def bs_vega(S, K, T, r, sigma):
    S, K, T, r, sigma = map(np.asarray, [S, K, T, r, sigma])
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    # 返回的是波动率每变动 1% 对应的价格变化
    return np.where((T <= 0) | (sigma <= 0), 0.0, S * norm.pdf(d1) * np.sqrt(T) / 100)

def bs_theta(S, K, T, r, sigma, option_type):
    S, K, T, r, sigma = map(np.asarray, [S, K, T, r, sigma])
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type.lower() == 'call':
        term2 = - r * K * np.exp(-r * T) * norm.cdf(d2)
        # 返回的是每过一天（1/365年）对应的价格变化
        return np.where((T <= 0) | (sigma <= 0), 0.0, (term1 + term2) / 365)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        return np.where((T <= 0) | (sigma <= 0), 0.0, (term1 + term2) / 365)

# --- 数值法 Greeks (通用微扰法) ---
def numerical_delta(pricer_func, S, K, T, r, sigma, option_type, **kwargs):
    dS = S * 0.01  # 股价变动 1%
    price_up = pricer_func(S + dS, K, T, r, sigma, option_type, **kwargs)
    price_down = pricer_func(S - dS, K, T, r, sigma, option_type, **kwargs)
    return (price_up - price_down) / (2 * dS)

def numerical_gamma(pricer_func, S, K, T, r, sigma, option_type, **kwargs):
    dS = S * 0.01
    price_up = pricer_func(S + dS, K, T, r, sigma, option_type, **kwargs)
    price_mid = pricer_func(S, K, T, r, sigma, option_type, **kwargs)
    price_down = pricer_func(S - dS, K, T, r, sigma, option_type, **kwargs)
    return (price_up - 2 * price_mid + price_down) / (dS ** 2)

def numerical_vega(pricer_func, S, K, T, r, sigma, option_type, **kwargs):
    dSigma = 0.01 # 波动率变动 1%
    price_up = pricer_func(S, K, T, r, sigma + dSigma, option_type, **kwargs)
    price_mid = pricer_func(S, K, T, r, sigma, option_type, **kwargs)
    return (price_up - price_mid) # dSigma is 1%, so this is per 1%

def numerical_theta(pricer_func, S, K, T, r, sigma, option_type, **kwargs):
    if T <= 1/365: return 0.0
    dT = 1/365 # 时间过去一天
    price_later = pricer_func(S, K, T - dT, r, sigma, option_type, **kwargs)
    price_mid = pricer_func(S, K, T, r, sigma, option_type, **kwargs)
    return (price_later - price_mid)


def run_single_option(sheet):
    # 读取参数
    S, K, T, r, sigma = sheet.range('B1:B5').value
    M_mc = int(sheet.range('E1').value)
    M_fdm_space, N_fdm_time = map(int,sheet.range('E4:E5').value)

    mc_kwargs = {'M': M_mc}
    fdm_kwargs = {'M_space': M_fdm_space, 'N_time': N_fdm_time, 'return_full_grid': True}

    # 计算
    bs_c = bs_call(S, K, T, r, sigma)
    bs_p = bs_put(S, K, T, r, sigma)

    mc_c = mc_pricer(S, K, T, r, sigma, 'call', **mc_kwargs)
    mc_p = mc_pricer(S, K, T, r, sigma, 'put', **mc_kwargs)

    fdm_results_call = fdm_pricer(S, K, T, r, sigma, 'call', **fdm_kwargs)
    fdm_results_put = fdm_pricer(S, K, T, r, sigma, 'put', **fdm_kwargs)

    fdm_c = fdm_results_call['price']
    fdm_p = fdm_results_put['price']

    # BS Greeks
    bs_d_c = bs_delta(S, K, T, r, sigma, 'call')
    bs_d_p = bs_delta(S, K, T, r, sigma, 'put')

    bs_g = bs_gamma(S, K, T, r, sigma)
    bs_v = bs_vega(S, K, T, r, sigma)

    bs_t_c = bs_theta(S, K, T, r, sigma, 'call')
    bs_t_p = bs_theta(S, K, T, r, sigma, 'put')

    # MC Greeks
    mc_d_c = numerical_delta(mc_pricer, S, K, T, r, sigma, 'call', **mc_kwargs)
    mc_d_p = numerical_delta(mc_pricer, S, K, T, r, sigma, 'put', **mc_kwargs)
    mc_g = numerical_gamma(mc_pricer, S, K, T, r, sigma, 'call', **mc_kwargs)
    mc_v = numerical_vega(mc_pricer, S, K, T, r, sigma, 'call', **mc_kwargs)
    mc_t_c = numerical_theta(mc_pricer, S, K, T, r, sigma, 'call', **mc_kwargs)
    mc_t_p = numerical_theta(mc_pricer, S, K, T, r, sigma, 'put', **mc_kwargs)

    # FDM Greeks
    S_vec = fdm_results_call['S_vec']
    dS_grid = S_vec[1] - S_vec[0]

    fdm_d_grid = np.gradient(fdm_results_call['V_vec'], dS_grid, edge_order=2)
    fdm_d_c = np.interp(S, S_vec, fdm_d_grid)
    fdm_d_p = fdm_d_c - 1
    
    fdm_g_grid = np.gradient(fdm_d_grid, dS_grid, edge_order=2)
    fdm_g = np.interp(S, S_vec, fdm_g_grid)

    fdm_kwargs_bumping = {'M_space': M_fdm_space, 'N_time': N_fdm_time}
    fdm_v = numerical_vega(fdm_pricer, S, K, T, r, sigma, 'call', **fdm_kwargs_bumping)
    fdm_t_c = numerical_theta(fdm_pricer, S, K, T, r, sigma, 'call', **fdm_kwargs_bumping)
    fdm_t_p = numerical_theta(fdm_pricer, S, K, T, r, sigma, 'put', **fdm_kwargs_bumping)



    # 写入结果
    greeks_data = [
        [bs_d_c, mc_d_c, fdm_d_c],
        [bs_d_p, mc_d_p, fdm_d_p],
        [bs_g, mc_g, fdm_g],
        [bs_v, mc_v, fdm_v],
        [bs_t_c, mc_t_c, fdm_t_c],
        [bs_t_p, mc_t_p, fdm_t_p]
    ]
    greeks_data = [list(map(float, row)) for row in greeks_data]

    sheet.range('B7').value = [[float(bs_c)], [float(bs_p)]]
    sheet.range('C7').value = [[float(mc_c)], [float(mc_p)]]
    sheet.range('D7').value = [[float(fdm_c)], [float(fdm_p)]]
    sheet.range('B9').value = greeks_data

def run_series_option(sheet):
    # 读取参数
    S, T, r, sigma = sheet.range('B1:B4').value
    wb = xw.Book.caller()
    single_sheet = wb.sheets['Single_Option']
    M_mc = int(single_sheet.range('E1').value)
    M_fdm_space, N_fdm_time = map(int, single_sheet.range('E4:E5').value)

    K_list = sheet.range('A7').expand('down').value
    if isinstance(K_list, (int, float)): K_list = [K_list]
    K_array = np.array(K_list)

    mc_kwargs = {'M': M_mc}
    fdm_kwargs = {'N_time': N_fdm_time}

    # --- BS (原生向量化) ---
    bs_calls = bs_call(S, K_array, T, r, sigma)
    bs_puts = bs_put(S, K_array, T, r, sigma)

    # --- MC (调用新的向量化函数) ---
    mc_calls, mc_puts = mc_series_pricer(S, K_array, T, r, sigma, **mc_kwargs)

    # --- FDM (调用新的向量化函数) ---
    k_std = 5.0 
    S_stat_max = S * np.exp(k_std * sigma * np.sqrt(T))

    max_K = np.max(K_array) if K_array.size > 0 else S
    S_max = max(S_stat_max, max_K) * 1.1
    S_vec = np.linspace(0, S_max, M_fdm_space + 1)
    
    V_calls_matrix = fdm_series_pricer(S_vec, K_array, T, r, sigma, 'call', N_fdm_time)
    V_puts_matrix = fdm_series_pricer(S_vec, K_array, T, r, sigma, 'put', N_fdm_time)
    
    fdm_calls = np.apply_along_axis(lambda v: np.interp(S, S_vec, v), 0, V_calls_matrix)
    fdm_puts = np.apply_along_axis(lambda v: np.interp(S, S_vec, v), 0, V_puts_matrix)

    # BS Greeks
    bs_greeks_matrix = np.column_stack((
        bs_delta(S, K_array, T, r, sigma, 'call'), bs_delta(S, K_array, T, r, sigma, 'put'),
        bs_gamma(S, K_array, T, r, sigma), bs_vega(S, K_array, T, r, sigma),
        bs_theta(S, K_array, T, r, sigma, 'call'), bs_theta(S, K_array, T, r, sigma, 'put')
    ))

    # MC Greeks
    dS = S * 0.01; dSigma = 0.01; dT = 1/365
    
    p_up_S_c, _ = mc_series_pricer(S + dS, K_array, T, r, sigma, **mc_kwargs)
    p_down_S_c, _ = mc_series_pricer(S - dS, K_array, T, r, sigma, **mc_kwargs)
    
    mc_d_c = (p_up_S_c - p_down_S_c) / (2 * dS)
    mc_d_p = mc_d_c - 1
    mc_g = (p_up_S_c - 2 * mc_calls + p_down_S_c) / (dS ** 2)

    p_up_sigma_c, _ = mc_series_pricer(S, K_array, T, r, sigma + dSigma, **mc_kwargs)
    mc_v = (p_up_sigma_c - mc_calls)

    p_later_T_c, p_later_T_p = mc_series_pricer(S, K_array, T - dT, r, sigma, **mc_kwargs)
    mc_t_c = p_later_T_c - mc_calls
    mc_t_p = p_later_T_p - mc_puts
    mc_greeks_matrix = np.column_stack((mc_d_c, mc_d_p, mc_g, mc_v, mc_t_c, mc_t_p))

    # FDM Greeks
    dS_grid = S_vec[1] - S_vec[0]
    fdm_d_grid = np.gradient(V_calls_matrix, dS_grid, axis=0, edge_order=2)
    fdm_g_grid = np.gradient(fdm_d_grid, dS_grid, axis=0, edge_order=2)
    fdm_d_c = np.apply_along_axis(lambda v: np.interp(S, S_vec, v), 0, fdm_d_grid)
    fdm_d_p = fdm_d_c - 1
    fdm_g = np.apply_along_axis(lambda v: np.interp(S, S_vec, v), 0, fdm_g_grid)
    
    p_up_sigma_fdm_v_c = fdm_series_pricer(S_vec, K_array, T, r, sigma + dSigma, 'call', **fdm_kwargs)
    p_up_sigma_fdm_c = np.apply_along_axis(lambda v: np.interp(S, S_vec, v), 0, p_up_sigma_fdm_v_c)
    fdm_v = p_up_sigma_fdm_c - fdm_calls
    
    p_later_T_fdm_v_c = fdm_series_pricer(S_vec, K_array, T - dT, r, sigma, 'call', **fdm_kwargs)
    p_later_T_fdm_c = np.apply_along_axis(lambda v: np.interp(S, S_vec, v), 0, p_later_T_fdm_v_c)
    p_later_T_fdm_v_p = fdm_series_pricer(S_vec, K_array, T - dT, r, sigma, 'put', **fdm_kwargs)
    p_later_T_fdm_p = np.apply_along_axis(lambda v: np.interp(S, S_vec, v), 0, p_later_T_fdm_v_p)
    fdm_t_c = p_later_T_fdm_c - fdm_calls
    fdm_t_p = p_later_T_fdm_p - fdm_puts
    fdm_greeks_matrix = np.column_stack((fdm_d_c, fdm_d_p, fdm_g, fdm_v, fdm_t_c, fdm_t_p))

    # --- 组合并写入 ---
    result_data = np.column_stack((
        bs_calls, bs_puts,
        mc_calls, mc_puts,
        fdm_calls, fdm_puts
    ))

    all_greeks_data = np.column_stack((
        bs_greeks_matrix,
        mc_greeks_matrix,
        fdm_greeks_matrix
    ))

    sheet.range('B7').value = result_data.tolist()
    sheet.range('I7').value = all_greeks_data.tolist()


def main():
    wb = xw.Book.caller()
    active_sheet = wb.sheets.active
    
    import time
    start_time = time.time()
    
    if active_sheet.name == 'Single_Option':
        run_single_option(active_sheet)
    elif active_sheet.name == 'Series_Option':
        run_series_option(active_sheet)
    else:
        active_sheet.range('A1').value = "Error: 请在 Single_Option 或 Series_Option 页面运行"

    end_time = time.time()

    active_sheet.range('H1').value = f"Done in {end_time - start_time:.4f}s"