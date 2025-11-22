import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

# ==========================================
# DATA STRUCTURE
# ==========================================

@dataclass
class VarianceSwapData:
    # Inputs
    sigma_strike: float
    vega_notional: float
    current_vol_env: float
    risk_free_rate: float
    valuation_date: pd.Timestamp
    
    variance_notional: float
    trading_days: List[pd.Timestamp]
    total_days: int
    past_days: List[pd.Timestamp]
    future_days: List[pd.Timestamp]
    num_past: int
    num_future: int
    b_days_in_year: int = 252
    days_in_year: int = 365
    dt: float = 1 / 252

def create_variance_swap(sigma_strike=0.20, 
                         vega_notional=1.0, 
                         current_vol_env=0.15, 
                         risk_free_rate=0.05, 
                         valuation_date='2022-11-09') -> VarianceSwapData:

    val_date = pd.to_datetime(valuation_date)
    
    # Notional conversion
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
        num_future=len(future_days),
        dt=dt
    )

# ==========================================
# CORE FUNCTIONS
# ==========================================

def get_analytical_sum_sq_return_for_days(data: VarianceSwapData, days: int, vol_param=None, r_param=None):
    vol = vol_param if vol_param is not None else data.current_vol_env
    r = r_param if r_param is not None else data.risk_free_rate
    
    daily_expected_sq_return = (vol ** 2) / data.b_days_in_year
    past_sum_sq_return = days * daily_expected_sq_return
    return past_sum_sq_return 

def get_swap_value(data: VarianceSwapData, sum_sq_future:float, vol_param=None, r_param=None):
    vol_param = vol_param if vol_param is not None else data.current_vol_env
    r_param = r_param if r_param is not None else data.risk_free_rate

    sum_sq_past = get_analytical_sum_sq_return_for_days(data, data.num_past, vol_param, r_param)

    total_realized_var = (sum_sq_past + sum_sq_future) * (data.b_days_in_year / data.total_days)
    payoff = data.variance_notional * (total_realized_var - data.sigma_strike**2)

    days_to_maturity = (data.trading_days[-1] - data.valuation_date).days
    t_maturity = max(days_to_maturity / data.days_in_year, 0.0)
    present_value = np.exp(-r_param * t_maturity) * payoff
    return present_value

def get_tomorrow_swap_data(data: VarianceSwapData):
    return create_variance_swap(
        data.sigma_strike, 
        data.vega_notional,
        data.current_vol_env, 
        data.risk_free_rate, 
        data.valuation_date + pd.Timedelta(days=1)
    )

# ==========================================
# OPTION 1: ANALYTICAL FUNCTIONS
# ==========================================

def run_analytical_results(data: VarianceSwapData):
    sum_sq_future = get_analytical_sum_sq_return_for_days(data, data.num_future)
    current_val = get_swap_value(data, sum_sq_future)
    
    days_to_maturity = (data.trading_days[-1] - data.valuation_date).days
    t_maturity = days_to_maturity / data.days_in_year
    discount_factor = np.exp(-data.risk_free_rate * t_maturity)

    # rho - sensitivity to the risk free rate - first derivative of the price with respect to the risk free rate
    rho = -t_maturity * current_val

    # vega - sensitivity to volatility - first derivative of the price of the swap with respect to volatility 
    sensitivity_weight = data.num_future / data.total_days
    vega = discount_factor * data.variance_notional * sensitivity_weight * (2 * data.current_vol_env)
    
    # theta - sensitivity to time - first derivative of the price of the swap with respect to time
    data_tomorrow = get_tomorrow_swap_data(data)
    sum_sq_future_tomorrow = get_analytical_sum_sq_return_for_days(data_tomorrow, data_tomorrow.num_future)
    price_tomorrow = get_swap_value(data_tomorrow, sum_sq_future_tomorrow)
    theta = price_tomorrow - current_val

    return {
        "price": current_val,
        "delta": 0, # delta and gamma are 0 because the swap price is not affected by changes in the underlying price
        "gamma": 0,
        "rho": rho,
        "vega": vega,
        "theta": theta
    }

# ==========================================
# OPTION 2: MONTE CARLO FUNCTIONS (GBM)
# ==========================================

def get_mc_gbm_sum_sq_return_for_days(data: VarianceSwapData, n_simulations=10_000, vol_param=None, r_param=None):
    vol_param = vol_param if vol_param is not None else data.current_vol_env
    r_param = r_param if r_param is not None else data.risk_free_rate

    drift = (r_param - 0.5 * vol_param**2) * data.dt
    diffusion = vol_param * np.sqrt(data.dt)
    Z = np.random.normal(0, 1, (n_simulations, data.num_future))
    future_returns = drift + diffusion * Z
    future_sum_sq = np.sum(future_returns**2, axis=1)
    return np.mean(future_sum_sq)

def run_gbm_mc_greeks(data: VarianceSwapData,n_simulations=10_000):
    sum_sq_future = get_mc_gbm_sum_sq_return_for_days(data, n_simulations)
    current_val = get_swap_value(data, sum_sq_future)

    bump = 0.001
    p_up = get_swap_value(data, sum_sq_future, vol_param=data.current_vol_env + bump)
    p_down = get_swap_value(data, sum_sq_future, vol_param=data.current_vol_env - bump)
    vega = (p_up - p_down) / (2 * bump)
    
    p_r_up = get_swap_value(data, sum_sq_future, r_param=data.risk_free_rate + 0.0001)
    p_r_down = get_swap_value(data, sum_sq_future, r_param=data.risk_free_rate - 0.0001)
    rho = (p_r_up - p_r_down) / (2 * 0.0001)

    data_tomorrow = get_tomorrow_swap_data(data)
    sum_sq_future_tomorrow = get_mc_gbm_sum_sq_return_for_days(data_tomorrow, n_simulations)
    price_tomorrow = get_swap_value(data_tomorrow, sum_sq_future_tomorrow)
    theta = price_tomorrow - current_val
    
    return {
        "price": current_val,
        "delta": 0,
        "gamma": 0,
        "vega": vega,
        "rho": rho,
        "theta": theta
    }   


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Initialize Data Object
    swap_data = create_variance_swap(
        sigma_strike=0.20,
        vega_notional=1.0,
        current_vol_env=0.15,
        risk_free_rate=0.05,
        valuation_date='2022-11-09'
    )
    

    sum_sq_future_analytical = get_analytical_sum_sq_return_for_days(swap_data, swap_data.num_future)
    sum_sq_future_mc_gbm = get_mc_gbm_sum_sq_return_for_days(swap_data, swap_data.num_future)
    
    # 2. Run Reports by passing the data object
    run_analytical_results(swap_data)
    run_gbm_mc_results(swap_data)
