import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

# entities
@dataclass
class VarianceSwapData:
    # Inputs
    sigma_strike: float
    vega_notional: float
    current_vol_env: float
    risk_free_rate: float
    valuation_date: pd.Timestamp
    
    # Derived Attributes
    variance_notional: float
    trading_days: List[pd.Timestamp]
    total_days: int
    past_days: List[pd.Timestamp]
    future_days: List[pd.Timestamp]
    num_past: int
    num_future: int
    b_days_in_year: int = 252
    days_in_year: int = 365
    dt: float = 1.0 / 252.0

def create_variance_swap(sigma_strike=0.20, 
                         vega_notional=1.0, 
                         current_vol_env=0.15, 
                         risk_free_rate=0.05, 
                         valuation_date='2022-11-09') -> VarianceSwapData:

    val_date = pd.to_datetime(valuation_date)
    variance_notional = vega_notional / (2 * sigma_strike)

    start_date, end_date = '2022-11-01', '2022-11-30'
    b_days = pd.bdate_range(start=start_date, end=end_date)
    holidays = [pd.to_datetime('2022-11-24')]
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

# monte carlo under gbm
def get_mc_gbm_sum_sq_return_crn(data: VarianceSwapData, Z_matrix, vol_param=None, r_param=None):
    vol_param = vol_param if vol_param is not None else data.current_vol_env
    r_param = r_param if r_param is not None else data.risk_free_rate
    drift = (r_param - 0.5 * vol_param**2) * data.dt
    diffusion = vol_param * np.sqrt(data.dt)
    future_returns = drift + diffusion * Z_matrix
    future_sum_sq = np.sum(future_returns**2, axis=1)
    return np.mean(future_sum_sq)

def run_gbm_mc_greeks(data: VarianceSwapData, n_simulations=50000):
    # Generate random numbers once
    Z_common = np.random.normal(0, 1, (n_simulations, data.num_future))

    sum_sq_past = get_analytical_past_variance(data)
    sum_sq_future_base = get_mc_gbm_sum_sq_return_crn(data, Z_common)
    current_val = calculate_final_price(data, sum_sq_past, sum_sq_future_base)

    bump = 0.001
    future_up = get_mc_gbm_sum_sq_return_crn(data, Z_common, vol_param=data.current_vol_env + bump)
    future_down = get_mc_gbm_sum_sq_return_crn(data, Z_common, vol_param=data.current_vol_env - bump)
    p_up = calculate_final_price(data, sum_sq_past, future_up)
    p_down = calculate_final_price(data, sum_sq_past, future_down)
    vega = (p_up - p_down) / (2 * bump)
    
    future_r_up = get_mc_gbm_sum_sq_return_crn(data, Z_common, r_param=data.risk_free_rate + 0.0001)
    future_r_down = get_mc_gbm_sum_sq_return_crn(data, Z_common, r_param=data.risk_free_rate - 0.0001)
    p_r_up = calculate_final_price(data, sum_sq_past, future_r_up, r_param=data.risk_free_rate + 0.0001)
    p_r_down = calculate_final_price(data, sum_sq_past, future_r_down, r_param=data.risk_free_rate - 0.0001)
    rho = (p_r_up - p_r_down) / (2 * 0.0001)

    data_tomorrow = get_tomorrow_swap_data(data)
    Z_tomorrow = Z_common[:, 1:] 
    sum_sq_future_tom = get_mc_gbm_sum_sq_return_crn(data_tomorrow, Z_tomorrow)
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