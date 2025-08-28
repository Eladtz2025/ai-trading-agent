import os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

TICKER = os.environ.get("BACKTEST_TICKER", "AAPL")
START = os.environ.get("BACKTEST_START", "2020-01-01")
END   = os.environ.get("BACKTEST_END",   None)  # till today
FAST  = int(os.environ.get("SMA_FAST", "20"))
SLOW  = int(os.environ.get("SMA_SLOW", "50"))

print(f"Running SMA backtest on {TICKER} ({START} → {END or 'today'}) [fast={FAST}, slow={SLOW}]")

data = yf.download(TICKER, start=START, end=END, auto_adjust=True)
if data.empty:
    raise SystemExit("No data downloaded. Check ticker or internet.")

close = data['Close'].dropna()
sma_fast = close.rolling(FAST).mean()
sma_slow = close.rolling(SLOW).mean()

# Signals: long when fast > slow; exit when fast < slow
signal = (sma_fast > sma_slow).astype(int)
signal = signal.shift(1).fillna(0)  # act next day

# Strategy equity (simple, no leverage, assume fully invested when signal=1)
ret = close.pct_change().fillna(0.0)
strat_ret = ret * signal
equity = (1 + strat_ret).cumprod()

# Buy & Hold for comparison
bh_equity = (1 + ret).cumprod()

def stats(e, r):
    # e = equity curve (Series), r = strategy daily returns (Series)
    total = float(e.iloc[-1] - 1)
    cagr = float(e.iloc[-1]**(252/len(e)) - 1)
    dd = float((e / e.cummax() - 1).min())

    r = r.fillna(0.0)
    r_mean = float(r.mean())
    r_std = float(r.std())  # pandas -> float

    Vol = r_std * np.sqrt(252)
    if r_std == 0 or np.isnan(r_std):
        sharpe = float("nan")
    else:
        sharpe = (r_mean * 252) / Vol

    return total, cagr, dd, Vol, sharpe


s_total, s_cagr, s_dd, s_vol, s_sharpe = stats(equity, strat_ret)
b_total, b_cagr, b_dd, b_vol, b_sharpe = stats(bh_equity, ret)

print("\n=== Results (No fees/slippage) ===")
print(f"Strategy Total Return: {s_total:.2%}, CAGR: {s_cagr:.2%}, MaxDD: {s_dd:.2%}, Vol: {s_vol:.2%}, Sharpe~: {s_sharpe:.2f}")
print(f"Buy&Hold Total Return: {b_total:.2%}, CAGR: {b_cagr:.2%}, MaxDD: {b_dd:.2%}")

# Plot
plt.figure(figsize=(10,6))
plt.plot(equity, label="SMA Strategy")
plt.plot(bh_equity, label="Buy & Hold", alpha=0.8)
plt.title(f"SMA {FAST}/{SLOW} – {TICKER}")
plt.legend()
png_path = OUT / f"sma_{TICKER}_{FAST}_{SLOW}.png"
plt.savefig(png_path, dpi=140, bbox_inches="tight")
print(f"Saved chart to: {png_path}")
