import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import logging
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from joblib import Parallel, delayed
import itertools

# Configure logging for Azure
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def fetch_data(tickers, start, end):
    logger.info(f"Fetching data for {len(tickers)} tickers from yfinance...")
    data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
    return data

def engineer_features(raw_stock, raw_market, raw_vix, stock_ticker):
    try:
        if isinstance(raw_stock.columns, pd.MultiIndex):
            df = raw_stock[stock_ticker].copy()
        else:
            df = raw_stock.copy()
            
        df = df[['Close', 'Volume']].dropna()
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['log_vol_change'] = np.log((df['Volume'] + 1) / (df['Volume'].shift(1) + 1))
        df['roll_mean_20'] = df['log_ret'].rolling(window=20).mean()
        df['roll_vol_20'] = df['log_ret'].rolling(window=20).std()
        
        market_close = raw_market if isinstance(raw_market, pd.Series) else raw_market['Close'].squeeze()
        vix_close    = raw_vix    if isinstance(raw_vix,    pd.Series) else raw_vix['Close'].squeeze()
        market_ret     = np.log(market_close / market_close.shift(1)).rename('market_ret')
        vix_pct_change = vix_close.pct_change().rename('vix_pct_change')
        
        feature_df = df.join(market_ret, how='left').join(vix_pct_change, how='left')
        return feature_df.dropna()
    except Exception as e:
        logger.error(f"Error processing {stock_ticker}: {e}")
        return pd.DataFrame()

def fit_arimax(train_y, train_exog):
    p_values, d_values, q_values = range(0, 3), [0], range(0, 3)
    best_aic, best_model = float("inf"), None
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(endog=train_y, exog=train_exog, order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic, best_model = results.aic, results
        except:
            continue
    return best_model

def fit_garch(residuals):
    scaled_resid = residuals * 100
    try:
        model = arch_model(scaled_resid, vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off')
        forecasts = res.forecast(horizon=1)
        return np.sqrt(forecasts.variance.iloc[-1, 0]) / 100.0
    except:
        return residuals.std()

def walk_forward_validation(df, ticker, train_window=504, step=1):
    if len(df) < train_window + step:
        return None
    
    y = df['log_ret']
    exog_cols = ['market_ret', 'vix_pct_change', 'log_vol_change', 'roll_mean_20', 'roll_vol_20']
    exog = df[exog_cols]
    
    train_y = y.iloc[-train_window-step : -step]
    train_exog = exog.iloc[-train_window-step : -step]
    test_exog = exog.iloc[[-step]]
    
    arimax_model = fit_arimax(train_y, train_exog)
    if arimax_model is None: return None
        
    forecast_vol = fit_garch(arimax_model.resid)
    forecast_ret = arimax_model.forecast(steps=1, exog=test_exog).iloc[0]
    
    return {
        'ticker': ticker,
        'expected_return': float(forecast_ret),
        'expected_volatility': float(forecast_vol)
    }

def run_analysis_pipeline(tickers_subset=None):
    """
    Main entry point for the Azure Function.
    Returns a list of dictionaries for the optimizer.
    """
    tickers = tickers_subset or ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA']
    
    end_date = dt.datetime.today().strftime('%Y-%m-%d')
    start_date = (dt.datetime.today() - dt.timedelta(days=5*365)).strftime('%Y-%m-%d')

    raw_stock = fetch_data(tickers, start_date, end_date)
    
    raw_market = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    raw_vix    = yf.download('^VIX',  start=start_date, end=end_date, progress=False)
    raw_rf     = yf.download('^IRX',  start=start_date, end=end_date, progress=False)

    # Flatten MultiIndex from yfinance 0.2.x+
    if isinstance(raw_market.columns, pd.MultiIndex):
        raw_market = raw_market.xs('^GSPC', axis=1, level=1)
    if isinstance(raw_vix.columns, pd.MultiIndex):
        raw_vix = raw_vix.xs('^VIX', axis=1, level=1)
    if isinstance(raw_rf.columns, pd.MultiIndex):
        raw_rf = raw_rf.xs('^IRX', axis=1, level=1)

    processed_data = {t: engineer_features(raw_stock, raw_market, raw_vix, t) for t in tickers}
    processed_data = {t: df for t, df in processed_data.items() if not df.empty}

    # Use n_jobs=2 for Azure Consumption Plans to avoid OOM
    results = Parallel(n_jobs=2)(
        delayed(walk_forward_validation)(df, ticker) for ticker, df in processed_data.items()
    )
    
    valid_results = [r for r in results if r is not None]
    
    # Calculate Risk Adjusted Score
    daily_rf = (raw_rf['Close'].iloc[-1] / 100.0) / 252.0 if not raw_rf.empty else 0.0001
    for r in valid_results:
        r['risk_adjusted_score'] = (r['expected_return'] - daily_rf) / r['expected_volatility']

    return sorted(valid_results, key=lambda x: x['risk_adjusted_score'], reverse=True)
