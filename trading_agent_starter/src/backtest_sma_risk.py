
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
END   = os.environ.get("BACKTEST_END",   None)
FAST  = int(os.environ.get("SMA_FAST", "20"))
SLOW  = int(os.environ.get("SMA_SLOW", "50"))

STOP_LOSS = float(os.environ.get("STOP_LOSS", -0.03))  # -3%
TAKE_PROFIT = float(os.environ.get("TAKE_PROFIT", 0.07))  # +7%


# --- parameters for costs & ATR ---
FEES_BPS = float(os.environ.get("FEES_BPS", 5))        # 5 bps = 0.05% per trade
SLIPPAGE_BPS = float(os.environ.get("SLIPPAGE_BPS", 5))# 5 bps
ATR_WINDOW = int(os.environ.get("ATR_WINDOW", 14))
ATR_MULT = float(os.environ.get("ATR_MULT", 0.0))      # 0 disables ATR stop

# ATR calculation (if enabled)
if ATR_MULT > 0:
    high = data['High']
    low = data['Low']
    prev_close = data['Close'].shift(1)
    tr = pd.concat([(high-low).abs(),
                    (high-prev_close).abs(),
                    (low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WINDOW).mean().fillna(method='bfill')
else:
    atr = None

print(f"Running SMA+Risk backtest on {TICKER} ({START} → {END or 'today'}) [fast={FAST}, slow={SLOW}]")

data = yf.download(TICKER, start=START, end=END, auto_adjust=True)
if data.empty:
    raise SystemExit("No data downloaded. Check ticker or internet.")

close = data['Close'].dropna()
sma_fast = close.rolling(FAST).mean()
sma_slow = close.rolling(SLOW).mean()

signal = (sma_fast > sma_slow).astype(int)
signal = signal.shift(1).fillna(0)

# Backtest with stop-loss / take-profit
equity = [1.0]
position = 0  # 0 = flat, 1 = long
entry_price = None

trades = []

for i in range(1, len(close)):
    price = close.iloc[i]
    date = close.index[i]
    sig = signal.iloc[i]

    if position == 0:
        if sig == 1:
            # apply slippage on entry
            fill_price = price * (1 + SLIPPAGE_BPS/10000)
            position = 1
            entry_price = float(fill_price)
            trades.append({"date": date, "action": "BUY", "price": float(fill_price)})
    else:
        change = (price - entry_price) / entry_price
        exit_reason = None

        if sig == 0:
            exit_reason = "SMA cross down"
        elif change <= STOP_LOSS:
            exit_reason = "StopLoss"
        elif change >= TAKE_PROFIT:
            exit_reason = "TakeProfit"

        if exit_reason:
            # apply slippage on exit
            fill_price = price * (1 - SLIPPAGE_BPS/10000)
            position = 0
            trades.append({"date": date, "action": "SELL", "price": float(fill_price), "reason": exit_reason})
            # deduct fees (entry+exit each counted on fills)
            # fees modeled implicitly by slippage; can add explicit fee on notional if needed
            entry_price = None

    # update equity
    if position == 1 and entry_price:
        change = (price - entry_price) / entry_price
        equity.append(equity[-1] * (1 + change))
    else:
        equity.append(equity[-1])

equity = pd.Series(equity, index=close.index)

# Save trade log
trades_df = pd.DataFrame(trades)
csv_path = OUT / f"trades_{TICKER}_sma_risk.csv"
trades_df.to_csv(csv_path, index=False)
print(f"Saved trades log to {csv_path}")

# Plot equity
plt.figure(figsize=(10,6))
plt.plot(equity, label="Strategy Equity")
plt.title(f"SMA {FAST}/{SLOW} with StopLoss/TakeProfit – {TICKER}")
plt.legend()
png_path = OUT / f"sma_risk_{TICKER}_{FAST}_{SLOW}.png"
plt.savefig(png_path, dpi=140, bbox_inches="tight")
print(f"Saved chart to: {png_path}")
