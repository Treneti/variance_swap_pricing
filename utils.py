import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

# entities
@dataclass
class VarianceSwapData:
    sigma_strike: float
    vega_notional: float
    current_vol_env: float 
    risk_free_rate: float
    valuation_date: pd.Timestamp
    
    theta: float  # long-run variance
    kappa: float = 2.0  # how fast the variance reverts to the long-run level 1 means too slow 20 means too fast 
    vol_of_var: float = 0.3 # Vol of variance 
    rho: float = -0.7 # correlation between the volatility and the stock price 
    
    variance_notional: float = 0.0
    trading_days: List[pd.Timestamp] = None
    total_days: int = 0
    past_days: List[pd.Timestamp] = None
    future_days: List[pd.Timestamp] = None
    num_past: int = 0
    num_future: int = 0
    b_days_in_year: int = 252
    days_in_year: int = 365
    dt: float = 1.0 / 252.0

def create_variance_swap(sigma_strike=0.20, 
                         vega_notional=1.0, 
                         current_vol_env=0.15, 
                         risk_free_rate=0.05, 
                         valuation_date='2022-11-09',
                         # Heston Params
                         kappa=2.0, 
                         theta=None,
                         vol_of_var=0.3, 
                         rho=-0.7) -> VarianceSwapData:

    val_date = pd.to_datetime(valuation_date)
    variance_notional = vega_notional / (2 * sigma_strike)

    if theta is None:
        theta_val = current_vol_env ** 2
    else:
        theta_val = theta

    start_date, end_date = '2022-11-01', '2022-11-30'
    b_days = pd.bdate_range(start=start_date, end=end_date)
    holidays = [pd.to_datetime('2022-11-24')] # Thanksgiving 2022
    trading_days = [d for d in b_days if d not in holidays]
    total_days = len(trading_days)
    
    past_days = [d for d in trading_days if d <= val_date]
    future_days = [d for d in trading_days if d > val_date]
    
    return VarianceSwapData(
        sigma_strike=sigma_strike,
        vega_notional=vega_notional,
        current_vol_env=current_vol_env,
        risk_free_rate=risk_free_rate,
        valuation_date=val_date,
        
        # Heston Params
        kappa=kappa,
        theta=theta_val,
        vol_of_var=vol_of_var,
        rho=rho,
        
        variance_notional=variance_notional,
        trading_days=trading_days,
        total_days=total_days,
        past_days=past_days,
        future_days=future_days,
        num_past=len(past_days),
        num_future=len(future_days)
    )

# helper functions
def get_analytical_past_variance(data: VarianceSwapData):
    daily_expected_sq_return = (data.current_vol_env ** 2) / data.b_days_in_year
    past_sum_sq_return = data.num_past * daily_expected_sq_return
    return past_sum_sq_return 

def calculate_final_price(data: VarianceSwapData, sum_sq_past, sum_sq_future, r_param=None):
    r = r_param if r_param is not None else data.risk_free_rate
    total_realized_var = (sum_sq_past + sum_sq_future) * (data.b_days_in_year / data.total_days)
    payoff = data.variance_notional * (total_realized_var - data.sigma_strike**2)
    
    days_to_maturity = (data.trading_days[-1] - data.valuation_date).days
    t_maturity = max(days_to_maturity / data.days_in_year, 0.0)
    return np.exp(-r * t_maturity) * payoff

def get_tomorrow_swap_data(data: VarianceSwapData):
    return create_variance_swap(
        data.sigma_strike, 
        data.vega_notional,
        data.current_vol_env, 
        data.risk_free_rate, 
        data.valuation_date + pd.Timedelta(days=1)
    )

def get_drift(data: VarianceSwapData, r_param=None, vol_param=None):
    # expected trend = rf rate minus ito correction because expected growth of the Log Price is always lower than the risk-free rate all that scaled 
    r = r_param if r_param is not None else data.risk_free_rate
    vol = vol_param if vol_param is not None else data.current_vol_env
    return (r - 0.5 * vol**2) * data.dt

# monte carlo under gbm
def get_mc_gbm_sum_sq_return(data: VarianceSwapData, Z_matrix, vol_param=None, r_param=None):
    vol_param = vol_param if vol_param is not None else data.current_vol_env
    r_param = r_param if r_param is not None else data.risk_free_rate
    drift = get_drift(data, r_param, vol_param) # trend of the returns
    diffusion = vol_param * np.sqrt(data.dt) # magnitute of the randomness 
    future_returns = drift + diffusion * Z_matrix
    future_sum_sq = np.sum(future_returns**2, axis=1)
    return np.mean(future_sum_sq)

def run_gbm_mc_greeks(data: VarianceSwapData, n_simulations=50000):
    # Generate random numbers once
    Z_common = np.random.normal(0, 1, (n_simulations, data.num_future))

    sum_sq_past = get_analytical_past_variance(data)
    sum_sq_future_base = get_mc_gbm_sum_sq_return(data, Z_common)
    current_val = calculate_final_price(data, sum_sq_past, sum_sq_future_base)

    bump = 0.001
    future_up = get_mc_gbm_sum_sq_return(data, Z_common, vol_param=data.current_vol_env + bump)
    future_down = get_mc_gbm_sum_sq_return(data, Z_common, vol_param=data.current_vol_env - bump)
    p_up = calculate_final_price(data, sum_sq_past, future_up)
    p_down = calculate_final_price(data, sum_sq_past, future_down)
    vega = (p_up - p_down) / (2 * bump)
    
    future_r_up = get_mc_gbm_sum_sq_return(data, Z_common, r_param=data.risk_free_rate + 0.0001)
    future_r_down = get_mc_gbm_sum_sq_return(data, Z_common, r_param=data.risk_free_rate - 0.0001)
    p_r_up = calculate_final_price(data, sum_sq_past, future_r_up, r_param=data.risk_free_rate + 0.0001)
    p_r_down = calculate_final_price(data, sum_sq_past, future_r_down, r_param=data.risk_free_rate - 0.0001)
    rho = (p_r_up - p_r_down) / (2 * 0.0001)

    data_tomorrow = get_tomorrow_swap_data(data)
    Z_tomorrow = Z_common[:, 1:] 
    sum_sq_future_tom = get_mc_gbm_sum_sq_return(data_tomorrow, Z_tomorrow)
    sum_sq_past_tom = get_analytical_past_variance(data_tomorrow)
    price_tomorrow = calculate_final_price(data_tomorrow, sum_sq_past_tom, sum_sq_future_tom)
    theta = price_tomorrow - current_val
    
    return {
        "price": current_val,
        "delta": 0,
        "gamma": 0,
        "rho": rho,   
        "vega": vega, 
        "theta": theta 
    }

# heston 
# analytical heston
def run_heston_analytical_results(data: VarianceSwapData):
    def get_heston_expected_future_var(d, init_var=None):
        init_var = init_var if init_var is not None else (d.current_vol_env**2)
        T_future = d.num_future / d.b_days_in_year # annualized time to maturity
        if T_future == 0: return 0.0
        lt_anchor = d.theta * T_future # if we wait a one year, we expect the variance to be theta
        diff_var = init_var - d.theta # diffrence in the initial variance and suppoasbly where the variance will be in the future 
        decay_factor = 1 - np.exp(-d.kappa * T_future)  # how fast the variance goes to the mean 
        shock = (diff_var * decay_factor) / d.kappa  # the shock is the difference between the initial variance and the mean, scaled by the decay factor
        return lt_anchor + shock # expected total variance

    past = get_analytical_past_variance(data)
    future = get_heston_expected_future_var(data)
    val = calculate_final_price(data, past, future)

    days_to_mat = (data.trading_days[-1] - data.valuation_date).days
    rho = -(days_to_mat/365.0) * val
    
    bump = 0.001
    v0_up = (data.current_vol_env + bump)**2
    v0_down = (data.current_vol_env - bump)**2
    
    future_up = get_heston_expected_future_var(data, init_var=v0_up)
    future_down = get_heston_expected_future_var(data, init_var=v0_down)
    p_up = calculate_final_price(data, past, future_up)
    p_down = calculate_final_price(data, past, future_down)
    vega = (p_up - p_down) / (2 * bump)

    data_tom = get_tomorrow_swap_data(data)
    past_tom = get_analytical_past_variance(data_tom)
    future_tom = get_heston_expected_future_var(data_tom)
    price_tom = calculate_final_price(data_tom, past_tom, future_tom)
    theta = price_tom - val

    return {
        "price": val,
        "delta": 0,
        "gamma": 0,
        "rho": rho,
        "vega": vega,
        "theta": theta
    }

# monte carlo under heston
def get_mc_heston_sum_sq_return(data: VarianceSwapData, Z_S, Z_v, v0_param=None, r_param=None):
    v0 = v0_param if v0_param is not None else (data.current_vol_env**2)
    r = r_param if r_param is not None else data.risk_free_rate

    n_sims = Z_S.shape[0]
    n_steps = data.num_future
    v = np.full(n_sims, v0)
    total_sq_returns = np.zeros(n_sims)
    sqrt_dt = np.sqrt(data.dt)
    
    corr_compl = np.sqrt(1 - data.rho**2) # correlation complement comes from Cholesky Decomposition, it shows how much the Stock depends on itself while rho shows how much it depends on the variance
    # TODO: move this explanation 
    # Cholesky Decomposition - with lieanr algebra look for matrix that when multiplied by transpose of itself gives the correlation matrix that is [[1, rho], [rho, 1]]
    # the matrix is [[1, rho], [0, corr_compl]]
    for i in range(n_steps):
        z_v_step = Z_v[:, i]
        z_s_step = Z_S[:, i]
        
        v_trunc = np.maximum(v, 0) # making sure v is not negative because of the square root
        
        # R = ln(S_{t+1}/S_t) approx (r - 0.5*v)*dt + sqrt(v)*dW_S
        drift = get_drift(data, r_param=r, vol_param=np.sqrt(v_trunc))
        
        curr_scaled_vol = np.sqrt(v_trunc) * sqrt_dt # current volatility scared by time step
        direction_shared_component = data.rho * z_v_step # how much stock returns shock are influenced by the variance shock
        stock_noise = corr_compl * z_s_step # correlation complement * new random noise 
        direction = direction_shared_component + stock_noise 
        diffusion = curr_scaled_vol * direction
        
        log_ret = drift + diffusion
        total_sq_returns += (log_ret**2)
        
        #calcualte variance for the next day 
        drift_variance = data.kappa * (data.theta - v_trunc) * data.dt # strength of mean reversion mutiplied by the difference of the long term mean and the current variance
        cir_feature = np.sqrt(v_trunc) #  the "Cox-Ingersoll-Ross" feature - noise decreases when variance is low
        scaled_noise = sqrt_dt * z_v_step # scaled noise
        diffusion_variance = data.vol_of_var * cir_feature * scaled_noise # add random shock to the variance
        delta_variance = drift_variance + diffusion_variance
        v_next = v + delta_variance # adding the original variance to the shock
        v = v_next

    return np.mean(total_sq_returns)

def run_heston_mc_greeks(data: VarianceSwapData, n_simulations=20000):
    np.random.seed(42)
    Z_S_common = np.random.normal(0, 1, (n_simulations, data.num_future))
    Z_v_common = np.random.normal(0, 1, (n_simulations, data.num_future))

    sum_sq_past = get_analytical_past_variance(data)

    sum_sq_future_base = get_mc_heston_sum_sq_return(data, Z_S_common, Z_v_common)
    current_val = calculate_final_price(data, sum_sq_past, sum_sq_future_base)

    bump = 0.001
    v0_up = (data.current_vol_env + bump)**2
    v0_down = (data.current_vol_env - bump)**2
    future_up = get_mc_heston_sum_sq_return(data, Z_S_common, Z_v_common, v0_param=v0_up)
    future_down = get_mc_heston_sum_sq_return(data, Z_S_common, Z_v_common, v0_param=v0_down)
    p_up = calculate_final_price(data, sum_sq_past, future_up)
    p_down = calculate_final_price(data, sum_sq_past, future_down)
    vega = (p_up - p_down) / (2 * bump)
    
    future_r_up = get_mc_heston_sum_sq_return(data, Z_S_common, Z_v_common, r_param=data.risk_free_rate + 0.0001)
    future_r_down = get_mc_heston_sum_sq_return(data, Z_S_common, Z_v_common, r_param=data.risk_free_rate - 0.0001)
    p_r_up = calculate_final_price(data, sum_sq_past, future_r_up, r_param=data.risk_free_rate + 0.0001)
    p_r_down = calculate_final_price(data, sum_sq_past, future_r_down, r_param=data.risk_free_rate - 0.0001)
    rho_val = (p_r_up - p_r_down) / (2 * 0.0001)

    data_tomorrow = get_tomorrow_swap_data(data)
    Z_S_tomorrow = Z_S_common[:, 1:] 
    Z_v_tomorrow = Z_v_common[:, 1:]
    
    sum_sq_future_tom = get_mc_heston_sum_sq_return(data_tomorrow, Z_S_tomorrow, Z_v_tomorrow)
    sum_sq_past_tom = get_analytical_past_variance(data_tomorrow)
    price_tomorrow = calculate_final_price(data_tomorrow, sum_sq_past_tom, sum_sq_future_tom)
    theta = price_tomorrow - current_val
    return {
        "price": current_val,
        "delta": 0,
        "gamma": 0,
        "rho": rho_val,   
        "vega": vega, 
        "theta": theta 
    }

# analytical 
def run_analytical_results(data: VarianceSwapData):
    def get_analytical_future(d, v=None):
        vol = v if v else d.current_vol_env
        return d.num_future * (vol**2 / d.b_days_in_year)

    past = get_analytical_past_variance(data)
    future = get_analytical_future(data)
    val = calculate_final_price(data, past, future)
    
    days_to_mat = (data.trading_days[-1] - data.valuation_date).days
    rho = -(days_to_mat/365.0) * val

    df = np.exp(-data.risk_free_rate * (days_to_mat/365.0))
    weight = data.num_future / data.total_days
    vega = df * data.variance_notional * weight * (2 * data.current_vol_env)

    data_tom = get_tomorrow_swap_data(data)
    past_tom = get_analytical_past_variance(data_tom)
    future_tom = get_analytical_future(data_tom)
    price_tom = calculate_final_price(data_tom, past_tom, future_tom)
    theta = price_tom - val
    
    return {
        "price": val, 
        "delta": 0, 
        "gamma": 0, 
        "rho": rho, 
        "vega": vega, 
        "theta": theta
    }