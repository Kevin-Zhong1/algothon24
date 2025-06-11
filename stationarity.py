import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Load and prepare price data
prices = pd.read_csv("./data/formatted_prices.csv")
prices["dates"] = pd.to_datetime(prices["dates"])
prices.set_index("dates", inplace=True)

# Just use the first 50 instruments (STOCK1 to STOCK50)
train_prices = prices.iloc[:, :50]


def check_for_stationarity(series, name="", cutoff=0.05):
    series = series.dropna()
    if series.std() == 0:
        print(f"{name}: constant series → NON-STATIONARY")
        return False
    p_value = adfuller(series)[1]
    print(f"{name}: p-value = {p_value} → ", end="")
    if p_value < cutoff:
        print("Likely STATIONARY")
        return True
    else:
        print("Likely NON-STATIONARY")
        return False


print("\n=== ADF Test on Raw Prices ===")
stationary_raw = [
    check_for_stationarity(train_prices[col], name=col) for col in train_prices.columns
]

print("\n=== ADF Test on Additive Returns (diff) ===")
additive_returns = train_prices.diff()
# print(additive_returns)
stationary_additive = [
    check_for_stationarity(additive_returns[col], name=col)
    for col in additive_returns.columns
]

print("\n=== ADF Test on Multiplicative Returns (pct_change) ===")
multiplicative_returns = train_prices.pct_change()
# print(multiplicative_returns)
stationary_multiplicative = [
    check_for_stationarity(multiplicative_returns[col], name=col)
    for col in multiplicative_returns.columns
]

print("\n=== Summary ===")
print(f"Raw Prices Stationary: {sum(stationary_raw)}/50")
print(f"Additive Returns Stationary: {sum(stationary_additive)}/50")
print(f"Multiplicative Returns Stationary: {sum(stationary_multiplicative)}/50")

# train_prices["STOCK1"].plot(title="Raw Price")
# plt.show()

# additive_returns["STOCK1"].dropna().plot(title="Additive Return")
# plt.show()

result = adfuller(additive_returns["STOCK1"].dropna())
print("ADF statistic:", result[0])
print("p-value:", result[1])
print("Used lags:", result[2])
print("Number of observations:", result[3])
print("Critical values:", result[4])