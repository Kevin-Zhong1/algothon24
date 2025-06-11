import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore")

#* WIP pipeline

# === Step 1: Load data ===
prices = pd.read_csv("./data/formatted_prices.csv") 
prices["dates"] = pd.to_datetime(prices["dates"])
prices.set_index("dates", inplace=True)
train_prices = prices.iloc[:, :50]  # first 50 stocks

# === Step 2: Compute log returns ===
log_returns = np.log(train_prices / train_prices.shift(1)).dropna()

# === Step 3: Define helper functions ===
def get_acf_info(series, nlags=20):
    acf_vals, confs = acf(series, nlags=nlags, alpha=0.05)
    sig_lags = [
        i for i, (val, (low, high)) in enumerate(zip(acf_vals, confs))
        if i > 0 and (low > 0 or high < 0)
    ]
    return acf_vals, confs, sig_lags

def fit_ar_model(X, max_lag=10):
    selection = ar_select_order(X, maxlag=max_lag, ic='aic', old_names=False)
    
    # Handle the case where no model could be selected
    if selection.ar_lags is None or len(selection.ar_lags) == 0:
        return None

    best_lag = selection.ar_lags[-1]
    model = AutoReg(X, lags=best_lag, old_names=False).fit()
    return {
        'lag': best_lag,
        'model': model,
        'residuals': model.resid,
        'forecast': model.predict(start=len(X), end=len(X))
    }

# === Step 4: Run pipeline for all 50 stocks ===
summary = []

for stock in log_returns.columns:
    series = log_returns[stock]
    
    # Step 4a: Get ACF info
    _, _, significant_lags = get_acf_info(series)
    
    if not significant_lags:
        print(f"{stock}: No significant autocorrelation")
        continue

    # Step 4b: Fit AR model
    result = fit_ar_model(series)
    if result is None:
        print(f"{stock}: Failed to fit AR model")
        continue
    
    # Step 4c: Residual whiteness check
    lb_test = acorr_ljungbox(result["residuals"], lags=[10], return_df=True)
    lb_pval = lb_test["lb_pvalue"].iloc[0]
    residual_white = lb_pval > 0.05
    
    # Step 4d: Store results
    summary.append({
        "stock": stock,
        "best_lag": result["lag"],
        "params": result["model"].params.values,
        "forecast": result["forecast"].values[0],
        "resid_white_noise": residual_white
    })

# === Step 5: Convert summary to DataFrame ===
summary_df = pd.DataFrame(summary)
print("\n=== AR Model Summary ===")
print(summary_df.head())
