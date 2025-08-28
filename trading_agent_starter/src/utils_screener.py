# src/utils_screener.py
from __future__ import annotations

import concurrent.futures as cf
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

from paper_execute_sma import generate_recommendation

def _avg_volume_20(ticker: str) -> float:
    try:
        df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
        if df.empty or "Volume" not in df.columns:
            return 0.0
        return float(df["Volume"].tail(20).mean())
    except Exception:
        return 0.0

def _score_row(rec: Dict[str,Any], avg_vol_20: float) -> float:
    action = rec.get("action")
    price = rec.get("price") or 0.0
    sma_f = rec.get("sma_fast") or 0.0
    sma_s = rec.get("sma_slow") or 0.0
    rsi   = rec.get("rsi")
    atrp  = rec.get("atr_pct")

    score = 0.0
    if action == "BUY":
        score += float(rec.get("confidence", 0))
    elif action == "SELL":
        score -= float(rec.get("confidence", 0))

    if price > 0:
        dist = abs(sma_f - sma_s) / price
        score += min(dist * 1000.0, 18.0)

    if sma_f > sma_s:
        score += 8.0

    if rsi is not None and not np.isnan(rsi):
        if 55 <= rsi <= 70:
            score += 6.0
        elif rsi > 80 or rsi < 40:
            score -= 6.0

    if atrp is not None:
        if atrp >= 0.06:
            score -= 18.0
        elif atrp >= 0.04:
            score -= 8.0

    if avg_vol_20 < 1_000_000:
        score -= 20.0

    return float(score)

def _one(ticker: str, fast: int, slow: int, per_order_budget: float, use_rsi: bool) -> Dict[str,Any]:
    try:
        rec = generate_recommendation(ticker, fast, slow, per_order_budget, use_rsi=use_rsi)
        av20 = rec.get("avg_vol_20")
        if av20 is None:
            av20 = _avg_volume_20(ticker)
        rec["avg_vol_20"] = float(av20)
        rec["score"] = _score_row(rec, av20)
        return rec
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "score": -1e9, "action":"WAIT", "confidence":0}

def screen_tickers(
    tickers: List[str],
    fast: int,
    slow: int,
    per_order_budget: float,
    use_rsi: bool = True,
    top_n: int = 10,
    min_confidence: int = 0,
) -> pd.DataFrame:
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t and t.strip()]))[:300]
    results: List[Dict[str,Any]] = []

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_one, t, fast, slow, per_order_budget, use_rsi) for t in tickers]
        for f in cf.as_completed(futs):
            results.append(f.result())

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Keep only known actions
    df = df[df["action"].isin(["BUY","WAIT","SELL"])].copy()

    # NEW: confidence threshold
    if "confidence" in df.columns:
        df = df[df["confidence"] >= int(min_confidence)].copy()

    for col in ["atr_pct","confidence","avg_vol_20"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values(
        by=["score","confidence","avg_vol_20"],
        ascending=[False, False, False],
        na_position="last"
    )

    keep_cols = [
        "ticker","score","action","confidence","price","amount","stop","take",
        "atr_pct","rsi","sma_fast","sma_slow","avg_vol_20",
        "horizon","days_to_stop","days_to_take","reason"
    ]
    show_cols = [c for c in keep_cols if c in df.columns]
    return df[show_cols].head(int(top_n)).reset_index(drop=True)
