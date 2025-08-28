# src/utils_screener.py
from __future__ import annotations

import time
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf

from paper_execute_sma import generate_recommendation

# yfinance download tuning for cloud environments
YF_KW = dict(period="9mo", interval="1d", auto_adjust=False, progress=False, threads=False)

# Symbol fixes (e.g., renamed tickers that commonly fail)
SYMBOL_FIXES: Dict[str, str] = {
    "FISV": "FI",
}

def fix_symbol(sym: str) -> str:
    s = sym.strip().upper()
    return SYMBOL_FIXES.get(s, s)

def safe_download(sym: str) -> Optional[pd.DataFrame]:
    """
    Best-effort Yahoo download with graceful failure (returns None if not usable).
    """
    try:
        df = yf.download(sym, **YF_KW)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return df
    except Exception:
        return None

def get_universe(name: str = "Nasdaq 100 (wide)") -> List[str]:
    """
    Returns a 'wide' Nasdaq 100 list (can include a few extras).
    Adjust/extend as you like; invalids will be skipped gracefully.
    """
    base = [
        "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","TSLA","AVGO","COST","NFLX","ADBE",
        "CMCSA","PEP","TMUS","AMD","INTC","QCOM","TXN","AMAT","PDD","MU","CSCO","ISRG","LIN",
        "SBUX","REGN","MRVL","ISRG","ADI","BKNG","LRCX","VRTX","INTU","PYPL","TEAM","WDAY",
        "ORLY","CDNS","ADP","PANW","SNPS","MNST","KDP","CRWD","MELI","ZS","DDOG","FTNT",
        "GILD","ABNB","CTAS","KHC","HON","MDLZ","VRTX","ADSK","CSX","AEP","AON","MDT",
        # a few more liquid names
        "ISRG","MRVL","AMAT","NXPI","ON","MAR","EA","SNPS"
    ]
    # de-dup
    return sorted(set(base))

def _score_row(confidence: int, atr_pct: Optional[float]) -> int:
    """
    Produce a simple sortable 'score' similar in spirit to what you had.
    Higher confidence + lower ATR% => higher score.
    """
    base = int(confidence or 0)
    if atr_pct is None:
        return 100 + base
    bonus = max(0, int((0.05 - min(0.05, float(atr_pct))) * 1000))  # small reward for low ATR%
    return 100 + base + bonus

def screen_top_picks(
    *,
    universe_name: str = "Nasdaq 100 (wide)",
    top_n: int = 10,
    per_order_budget: float = 100.0,
    fast: int = 20,
    slow: int = 50,
    use_rsi: bool = True,
    min_confidence: int = 0,
    min_avg_vol_20: int = 1_000_000,
    max_price_allowed: float = 400.0,
) -> pd.DataFrame:
    """
    Scans a ticker universe and returns a table of candidates with
    columns expected by the dashboard:
      ticker, score, action, confidence, price, amount, stop, take,
      atr_pct, rsi, sma_fast, sma_slow, avg_vol_20, horizon,
      days_to_stop, days_to_take, reason
    """
    tickers = get_universe(universe_name)
    rows = []

    for sym in tickers:
        t = fix_symbol(sym)
        # quick preflight to avoid heavy calls on dead symbols
        df = safe_download(t)
        if df is None:
            continue

        # call the same engine used by the single-ticker flow
        rec = generate_recommendation(
            ticker=t, fast=fast, slow=slow,
            notional=per_order_budget, use_rsi=use_rsi
        )

        # quality gates enforced inside generate_recommendation already
        # we apply UI-level filters here
        if rec.get("confidence", 0) < int(min_confidence or 0):
            continue
        # rec contains avg_vol_20, price, etc. (computed inside)
        avg_vol_20 = rec.get("avg_vol_20")
        if avg_vol_20 is not None and float(avg_vol_20) < float(min_avg_vol_20):
            continue
        price = rec.get("price")
        if price is not None and float(price) > float(max_price_allowed) and float(rec.get("amount", 0)) < float(price):
            # too expensive vs budget and no fractional – skip
            continue

        # build output row
        atr_pct = rec.get("atr_pct")
        confidence = int(rec.get("confidence") or 0)
        rows.append({
            "ticker": t,
            "score": _score_row(confidence, atr_pct),
            "action": rec.get("action"),
            "confidence": confidence,
            "price": rec.get("price"),
            "amount": rec.get("amount"),
            "stop": rec.get("stop"),
            "take": rec.get("take"),
            "atr_pct": atr_pct,
            "rsi": rec.get("rsi"),
            "sma_fast": rec.get("sma_fast"),
            "sma_slow": rec.get("sma_slow"),
            "avg_vol_20": avg_vol_20,
            "horizon": rec.get("horizon"),
            "days_to_stop": rec.get("days_to_stop"),
            "days_to_take": rec.get("days_to_take"),
            "reason": rec.get("reason"),
        })

        # be gentle with Yahoo while running in the cloud
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(columns=[
            "ticker","score","action","confidence","price","amount",
            "stop","take","atr_pct","rsi","sma_fast","sma_slow","avg_vol_20",
            "horizon","days_to_stop","days_to_take","reason"
        ])

    df_out = pd.DataFrame(rows)
    # BUY first, then by score desc
    action_order = pd.Categorical(df_out["action"], categories=["BUY","SELL","WAIT"], ordered=True)
    df_out = df_out.assign(_a=action_order).sort_values(by=["_a","score"], ascending=[True, False]).drop(columns=["_a"])
    return df_out.head(int(top_n or 10)).reset_index(drop=True)

# --- Back-compat adapters (keep for dashboard compatibility) -----------------

import pandas as pd  # אם כבר קיים בתחילת הקובץ, אפשר להשאיר

def screen_tickers(
    universe: str = "Nasdaq 100 (wide)",
    # גם השם הישן וגם החדש:
    picks: int | None = None,
    top_n: int | None = None,
    # גם השם הישן וגם החדש:
    order_budget: float | None = None,
    per_order_budget: float | None = None,
    fast: int = 20,
    slow: int = 50,
    use_rsi: bool = True,
    min_confidence: int = 0,
    min_avg_vol_20: int = 1_000_000,
    max_price_allowed: float = 400.0,
) -> pd.DataFrame:
    """
    Legacy entry-point used by the Streamlit dashboard.
    Accepts both old (picks, order_budget) and new (top_n, per_order_budget) names,
    and delegates to screen_top_picks().
    """
    # ה־N וה־budget מחושבים מהפרמטר בשם שקיים
    n = top_n if top_n is not None else (picks if picks is not None else 10)
    budget = (
        per_order_budget
        if per_order_budget is not None
        else (order_budget if order_budget is not None else 100.0)
    )

    return screen_top_picks(
        universe_name=universe,
        top_n=int(n),
        per_order_budget=float(budget),
        fast=int(fast),
        slow=int(slow),
        use_rsi=bool(use_rsi),
        min_confidence=int(min_confidence),
        min_avg_vol_20=int(min_avg_vol_20),
        max_price_allowed=float(max_price_allowed),
    )


def find_top_picks(**kwargs) -> pd.DataFrame:
    """Alias לשם ישן נוסף אם מישהו עוד קורא אליו."""
    return screen_top_picks(**kwargs)


