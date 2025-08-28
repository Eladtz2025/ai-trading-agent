
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from utils_logger import log_trade, log_signal, log_error

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

TICKER = os.environ.get("BACKTEST_TICKER", "AAPL")
START = os.environ.get("BACKTEST_START", "2020-01-01")
END   = os.environ.get("BACKTEST_END",   None)
FAST  = int(os.environ.get("SMA_FAST", "20"))
SLOW  = int(os.environ.get("SMA_SLOW", "50"))

STOP_LOSS = float(os.environ.get("STOP_LOSS", -0.03))      # -3%
TAKE_PROFIT = float(os.environ.get("TAKE_PROFIT", 0.07))   # +7%
FEES_BPS = float(os.environ.get("FEES_BPS", 5))            # 5 bps
SLIPPAGE_BPS = float(os.environ.get("SLIPPAGE_BPS", 5))    # 5 bps
ATR_WINDOW = int(os.environ.get("ATR_WINDOW", 14))
ATR_MULT = float(os.environ.get("ATR_MULT", 0.0))          # 0 = disabled

print(f"Running SMA+Risk v2 on {TICKER} ({START} → {END or 'today'}) [fast={FAST}, slow={SLOW}]")

try:
    data = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
except Exception as e:
    log_error("yfinance.download", e, {"ticker": TICKER})
    raise

if data.empty:
    raise SystemExit("No data downloaded. Check ticker or internet.")

close = data['Close'].dropna()
sma_fast = close.rolling(FAST).mean()
sma_slow = close.rolling(SLOW).mean()

# Sanity: shift by 1 bar to avoid look-ahead
signal = (sma_fast > sma_slow).astype(float).fillna(0.0).shift(1).fillna(0.0)
ret = close.pct_change().fillna(0.0)

# ATR (optional)
atr = None
if ATR_MULT > 0 and {'High','Low','Close'}.issubset(set(data.columns)):
    high, low, prev_close = data['High'], data['Low'], data['Close'].shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WINDOW).mean().bfill()

equity = [1.0]
position = 0
entry_price = None

for i in range(1, len(close)):
    price = float(close.iloc[i].item())
    ts = close.index[i]
    # --- safe scalar signal ---
raw_sig = signal.iloc[i]
# אם חוזר אובייקט שאינו סקלר, נמיר לסקלר
try:
    if hasattr(raw_sig, "item"):
        raw_sig = raw_sig.item()
except Exception:
    pass
try:
    sig = int(float(raw_sig))
except Exception:
    sig = 0



    # log signal
    log_signal(ts, TICKER, sig, {"fast": float(sma_fast.iloc[i]) if pd.notna(sma_fast.iloc[i]) else None,
                                  "slow": float(sma_slow.iloc[i]) if pd.notna(sma_slow.iloc[i]) else None})

    if position == 0 and sig == 1:
        fill_price = price * (1 + SLIPPAGE_BPS/10000)
        entry_price = float(fill_price)
        position = 1
        log_trade(ts, TICKER, "BUY", entry_price, reason="SMA cross up",
                  meta={"slip_bps": SLIPPAGE_BPS, "fees_bps": FEES_BPS})
    elif position == 1:
        change_from_entry = (price - entry_price) / entry_price
        exit_reason = None

        # ATR protective stop
        if atr is not None and pd.notna(atr.iloc[i]) and ATR_MULT > 0:
            if price <= entry_price - ATR_MULT * float(atr.iloc[i]):
                exit_reason = f"ATR stop ({ATR_MULT}x)"

        if sig == 0:
            exit_reason = "SMA cross down"
        elif change_from_entry <= STOP_LOSS:
            exit_reason = "StopLoss"
        elif change_from_entry >= TAKE_PROFIT:
            exit_reason = "TakeProfit"

        if exit_reason:
            fill_price = price * (1 - SLIPPAGE_BPS/10000)
            position = 0
            log_trade(ts, TICKER, "SELL", float(fill_price), reason=exit_reason,
                      meta={"slip_bps": SLIPPAGE_BPS, "fees_bps": FEES_BPS})
            entry_price = None

    day_ret = float(ret.iloc[i]) if position == 1 else 0.0
    equity.append(equity[-1] * (1.0 + day_ret))
# --- align equity length to price index (safety) ---
if len(equity) < len(close):
    equity = equity + [equity[-1]] * (len(close) - len(equity))
elif len(equity) > len(close):
    equity = equity[:len(close)]

equity = pd.Series(equity, index=close.index)

# Chart
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(equity, label="Strategy Equity")
plt.title(f"SMA {FAST}/{SLOW} + Risk v2 — {TICKER}")
plt.legend()
png_path = OUT / f"sma_risk_v2_{TICKER}_{FAST}_{SLOW}.png"
plt.savefig(png_path, dpi=140, bbox_inches="tight")
print(f"Saved chart to: {png_path}")
