# src/paper_execute_sma.py
# Plain-English SMA/ATR/RSI engine + Paper execution (alpaca-py) + DuckDB logs
# Confidence v2.1, Horizon, ~Days to Stop/TP, risk sizing, quality gates,
# live-price TP/SL validation with legal tick rounding, asset status check,
# optional fractional orders, and trade follow-up view.

from __future__ import annotations

import os, re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from decimal import Decimal, ROUND_HALF_UP, ROUND_UP

import numpy as np
import pandas as pd
import yfinance as yf
import duckdb

# ---------- Alpaca SDK (alpaca-py) ----------
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.trading.requests import (
    MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetOrdersRequest
)
try:
    # Notional (fractional) order support – if available on account.
    from alpaca.trading.requests import NotionalOrderRequest
    HAS_NOTIONAL = True
except Exception:
    HAS_NOTIONAL = False

# Alpaca data for live base price (validation)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

# ---------- ENV / KEYS ----------
@dataclass
class AlpacaKeys:
    key: str
    secret: str
    base_url: str = "https://paper-api.alpaca.markets"

def _parse_env_file(fp: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not fp.exists():
        return out
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out

def load_env_alpaca() -> AlpacaKeys:
    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_SECRET_KEY")
    base = os.getenv("ALPACA_PAPER_BASE_URL") or "https://paper-api.alpaca.markets"
    if key and sec:
        return AlpacaKeys(key=key, secret=sec, base_url=base)

    vals = _parse_env_file(Path("config/.env"))
    key = vals.get("ALPACA_API_KEY") or key
    sec = vals.get("ALPACA_SECRET_KEY") or sec
    base = vals.get("ALPACA_PAPER_BASE_URL") or base
    if key and sec:
        return AlpacaKeys(key=key, secret=sec, base_url=base)

    raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in config/.env or environment.")

# ---------- RISK & QUALITY ----------
RISK_PCT_PER_TRADE = 0.005      # 0.5% של ההון לכל טרייד
MIN_AVG_VOL_20     = 1_000_000  # סף נזילות
MAX_PRICE_ALLOWED  = 400.0      # להגביל שמות יקרים כשאין fractional

def configure_risk(*, risk_pct: Optional[float] = None,
                   min_avg_vol_20: Optional[int] = None,
                   max_price_allowed: Optional[float] = None) -> None:
    global RISK_PCT_PER_TRADE, MIN_AVG_VOL_20, MAX_PRICE_ALLOWED
    if risk_pct is not None:
        RISK_PCT_PER_TRADE = max(0.0001, float(risk_pct))
    if min_avg_vol_20 is not None:
        MIN_AVG_VOL_20 = int(max(0, min_avg_vol_20))
    if max_price_allowed is not None:
        MAX_PRICE_ALLOWED = float(max(0.0, max_price_allowed))

# ---------- DUCKDB ----------
DB_PATH = Path("logs/trade_log.duckdb")

def _ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    # signals – כולל stop/take כבר בסכימה
    con.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            ts TIMESTAMP,
            ticker TEXT,
            fast INTEGER,
            slow INTEGER,
            sma_fast DOUBLE,
            sma_slow DOUBLE,
            atr DOUBLE,
            rsi DOUBLE,
            atr_pct DOUBLE,
            decision TEXT,
            confidence INTEGER,
            reason TEXT,
            stop DOUBLE,
            take DOUBLE
        );
    """)
    # trades
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            ts TIMESTAMP,
            ticker TEXT,
            side TEXT,
            qty DOUBLE,
            notional DOUBLE,
            price DOUBLE,
            status TEXT,
            order_id TEXT
        );
    """)
    return con

def _log_signal(rows: List[Dict[str, Any]]):
    try:
        con = _ensure_db()
        for r in rows:
            con.execute(
                """INSERT INTO signals
                   (ts,ticker,fast,slow,sma_fast,sma_slow,atr,rsi,atr_pct,decision,confidence,reason,stop,take)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    datetime.now(timezone.utc),
                    r.get("ticker"),
                    int(r.get("fast") or 0),
                    int(r.get("slow") or 0),
                    float(r.get("sma_fast")) if r.get("sma_fast") is not None else None,
                    float(r.get("sma_slow")) if r.get("sma_slow") is not None else None,
                    float(r.get("atr")) if r.get("atr") is not None else None,
                    float(r.get("rsi")) if r.get("rsi") is not None else None,
                    float(r.get("atr_pct")) if r.get("atr_pct") is not None else None,
                    r.get("decision"),
                    int(r.get("confidence") or 0),
                    (r.get("reason") or "")[:2000],
                    float(r.get("stop")) if r.get("stop") is not None else None,
                    float(r.get("take")) if r.get("take") is not None else None,
                ],
            )
        con.close()
    except Exception:
        pass  # best-effort

def _log_trade(ticker: str, side: str, qty: float, notional: float, price: float, status: str, order_id: str):
    try:
        con = _ensure_db()
        con.execute(
            """INSERT INTO trades (ts,ticker,side,qty,notional,price,status,order_id)
               VALUES (?,?,?,?,?,?,?,?)""",
            [datetime.now(timezone.utc), ticker, side, float(qty), float(notional), float(price), status, order_id],
        )
        con.close()
    except Exception:
        pass

# ---------- INDICATORS ----------
def _days_since_last_cross(sma_fast_s: pd.Series, sma_slow_s: pd.Series) -> int:
    diff = (sma_fast_s - sma_slow_s).dropna()
    if diff.empty:
        return 9999
    sign = np.sign(diff)
    cross_idx = np.where(np.diff(sign) != 0)[0]
    if len(cross_idx) == 0:
        return 9999
    last_cross = cross_idx[-1]
    return int((len(sign) - 1) - last_cross)

def _horizon_from_atr_pct(atr_pct):
    if atr_pct is None:
        return "short-term (days–weeks)"
    if atr_pct < 0.02:
        return "short-term (1–4 weeks)"
    elif atr_pct < 0.04:
        return "short-term (5–15 trading days)"
    else:
        return "very short-term (1–7 trading days)"

def _confidence_v2(price, sma_f, sma_s, atr_pct, rsi, avg_vol_20, sma200, days_since_cross):
    parts, score = [], 50.0
    # Trend distance (עם קנס קל על ה־overheat)
    if price and sma_f and sma_s:
        dist = abs(sma_f - sma_s) / price
        add = min(dist / 0.01 * 18.0, 18.0)
        score += add; parts.append(f"+{add:.0f} trend-distance")
        if dist > 0.06:
            score -= 8; parts.append("-8 over-extended")
    # Fresh cross
    if days_since_cross is not None:
        if days_since_cross <= 3:   score += 12; parts.append("+12 fresh cross (≤3d)")
        elif days_since_cross <= 10: score += 6;  parts.append("+6 recent cross (≤10d)")
    # Regime vs SMA200
    if sma200:
        if price > sma200: score += 6; parts.append("+6 above SMA200")
        else:              score -= 6; parts.append("-6 below SMA200")
    # RSI
    if rsi is not None and not np.isnan(rsi):
        if 55 <= rsi <= 65: score += 12; parts.append("+12 RSI sweet-spot")
        elif rsi > 80 or rsi < 40: score -= 10; parts.append("-10 RSI extreme")
    # Volatility (ATR%)
    if atr_pct is not None and not np.isnan(atr_pct):
        if atr_pct <= 0.02:      score += 10; parts.append("+10 low ATR%")
        elif atr_pct <= 0.04:    score += 5;  parts.append("+5 moderate ATR%")
        elif atr_pct >= 0.06:    score -= 18; parts.append("-18 high ATR%")
        else:                    score -= 8;  parts.append("-8 elevated ATR%")
    # Liquidity
    if avg_vol_20 is not None:
        if avg_vol_20 >= 5_000_000: score += 6;  parts.append("+6 high liquidity")
        elif avg_vol_20 < 1_000_000: score -= 10; parts.append("-10 low liquidity")
    return int(max(0, min(100, round(score)))), parts

# ---------- CLIENT/EQUITY ----------
def _get_client() -> TradingClient:
    keys = load_env_alpaca()
    return TradingClient(keys.key, keys.secret, paper=True)

def _get_equity_fallback() -> float:
    try:
        client = _get_client()
        acct = client.get_account()
        eq = float(getattr(acct, "equity", 10000))
        return eq if eq > 0 else 10000.0
    except Exception:
        return 10000.0

# ---------- PRICE / TICK ----------
def _live_price(t: str) -> Optional[float]:
    try:
        ti = yf.Ticker(t)
        fi = getattr(ti, "fast_info", {}) or {}
        p = fi.get("last_price") or fi.get("last_trade_price") or fi.get("regular_market_price")
        if p:
            return float(p)
        h = ti.history(period="1d", interval="1m")
        if not h.empty and "Close" in h.columns:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None

def _alpaca_live_base_price(symbol: str) -> Optional[float]:
    """Latest trade price from Alpaca Data API – הכי אחיד מול הביצוע."""
    try:
        keys = load_env_alpaca()
        data_client = StockHistoricalDataClient(keys.key, keys.secret)
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        lt = data_client.get_stock_latest_trade(req)
        if isinstance(lt, dict):
            obj = lt.get(symbol)
            if obj and getattr(obj, "price", None) is not None:
                return float(obj.price)
        if getattr(lt, "price", None) is not None:
            return float(lt.price)
    except Exception:
        pass
    return None

def _tick_size(base_price: float) -> Decimal:
    return Decimal('0.01') if base_price >= 1 else Decimal('0.0001')

def _round_to_tick(value: float, base_price: float) -> float:
    tick = _tick_size(base_price)
    return float(Decimal(str(value)).quantize(tick, rounding=ROUND_HALF_UP))

# ---------- SIZING ----------
def size_notional_by_atr(price: float, atr: Optional[float], ui_budget: float) -> float:
    equity = _get_equity_fallback()
    risk_per_trade = equity * RISK_PCT_PER_TRADE
    if atr is None or atr <= 0 or price <= 0:
        return float(max(50.0, ui_budget))
    risk_per_share = 1.5 * atr
    if risk_per_share <= 0:
        return float(max(50.0, ui_budget))
    shares_by_risk = max(1, int(risk_per_trade / risk_per_share))
    notional_by_risk = shares_by_risk * price
    return float(max(50.0, min(ui_budget, notional_by_risk)))

# ---------- CORE ENGINE ----------
def generate_recommendation(ticker: str, fast: int, slow: int, notional: float, use_rsi: bool = True) -> dict:
    t = ticker.strip().upper()
    df = yf.download(t, period="9mo", interval="1d", auto_adjust=False, progress=False)
    if df.empty or "Close" not in df.columns:
        return {"action":"WAIT","amount":notional,"reason":"No data.","stop":None,"take":None,"confidence":0,
                "ticker":t,"price":None,"fast":fast,"slow":slow,"sma_fast":None,"sma_slow":None,"atr":None,"atr_pct":None,"rsi":None,
                "horizon":None,"days_to_stop":None,"days_to_take":None,"avg_vol_20":None}

    df = df.dropna().copy()
    close = df["Close"]

    df["SMA_FAST"] = close.rolling(int(fast), min_periods=int(fast)).mean()
    df["SMA_SLOW"] = close.rolling(int(slow), min_periods=int(slow)).mean()

    high, low = df["High"], df["Low"]
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14, min_periods=14).mean()

    rsi_val = None
    if use_rsi:
        n = 14
        delta = close.diff()
        gain = (delta.clip(lower=0)).rolling(n, min_periods=n).mean()
        loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        rsi_val = float(df["RSI"].iloc[-1]) if not np.isnan(df["RSI"].iloc[-1]) else None

    price = float(close.iloc[-1])
    sma_f = float(df["SMA_FAST"].iloc[-1])
    sma_s = float(df["SMA_SLOW"].iloc[-1])
    atr = float(df["ATR"].iloc[-1]) if not np.isnan(df["ATR"].iloc[-1]) else None
    atr_pct = float(atr / price) if (atr is not None and price) else None
    avg_vol_20 = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else None

    # Decision
    if sma_f > sma_s:
        action, base_reason = "BUY", "It likely goes UP."
    elif sma_f < sma_s:
        action, base_reason = "SELL", "Trend is weak; likely DOWN."
    else:
        action, base_reason = "WAIT", "Flat trend; I would WAIT."

    # Quality gates
    if avg_vol_20 is not None and avg_vol_20 < MIN_AVG_VOL_20:
        return {"action":"WAIT","amount":float(notional),
                "reason":"Low liquidity; I would WAIT.",
                "stop":None,"take":None,"confidence":0,
                "ticker":t,"price":price,"fast":int(fast),"slow":int(slow),
                "sma_fast":sma_f,"sma_slow":sma_s,"atr":atr,"atr_pct":atr_pct,"rsi":rsi_val,
                "horizon":None,"days_to_stop":None,"days_to_take":None,"avg_vol_20":avg_vol_20}

    if price is not None and price > MAX_PRICE_ALLOWED and notional < price:
        return {"action":"WAIT","amount":float(notional),
                "reason":"Price too high for current budget; pick a cheaper stock or enable fractional.",
                "stop":None,"take":None,"confidence":0,
                "ticker":t,"price":price,"fast":int(fast),"slow":int(slow),
                "sma_fast":sma_f,"sma_slow":sma_s,"atr":atr,"atr_pct":atr_pct,"rsi":rsi_val,
                "horizon":None,"days_to_stop":None,"days_to_take":None,"avg_vol_20":avg_vol_20}

    # Risk sizing
    amount = size_notional_by_atr(price, atr, notional)
    amount = float(round(amount, 2))

    # Stops/Take
    if atr is not None:
        stop = round(price - 1.5 * atr, 2)
        take = round(price + 2.0 * atr, 2)
    else:
        stop = round(price * 0.97, 2)
        take = round(price * 1.05, 2)

    # Horizon & ~days
    horizon = _horizon_from_atr_pct(atr_pct)
    d_stop = d_take = None
    if atr:
        if action == "BUY":
            d_stop = max(1, round((price - stop) / atr, 1))
            d_take = max(1, round((take - price) / atr, 1))
        elif action == "SELL":
            d_stop = max(1, round((take - price) / atr, 1))
            d_take = max(1, round((price - stop) / atr, 1))

    # Confidence
    sma200 = float(close.rolling(200, min_periods=200).mean().iloc[-1]) if len(df) >= 200 else None
    days_since_cross = _days_since_last_cross(df["SMA_FAST"], df["SMA_SLOW"])
    confidence, factors = _confidence_v2(price, sma_f, sma_s, atr_pct, rsi_val, avg_vol_20, sma200, days_since_cross)

    # English text
    if action == "BUY":
        reason = f"{base_reason} I would BUY ${int(amount)} and set a Stop at {stop} and a Take-Profit at {take}."
    elif action == "SELL":
        reason = f"{base_reason} I would SELL ${int(amount)} (paper short) with Stop {take} above and Take-Profit {stop} below."
    else:
        reason = base_reason
    reason += f" Time horizon: {horizon}. Rough reach → Stop ~{d_stop}d, Take ~{d_take}d."
    reason += " Confidence factors: " + ", ".join(factors)
    if atr and atr > 0:
        r = 1.5 * atr
        days_ts = max(10, int(round((price / r) * 2)))
        reason += f" Exit plan: if price reaches +1R, move Stop to entry; if not at target by ~{days_ts} sessions, close manually."

    # Log signal (now includes stop/take)
    try:
        _log_signal([{
            "ticker": t, "fast": fast, "slow": slow,
            "sma_fast": sma_f, "sma_slow": sma_s,
            "atr": atr, "rsi": rsi_val, "atr_pct": atr_pct,
            "decision": action, "confidence": confidence, "reason": reason,
            "stop": stop, "take": take
        }])
    except Exception:
        pass

    return {
        "action": action, "amount": amount, "reason": reason,
        "stop": stop, "take": take, "confidence": confidence,
        "ticker": t, "price": price, "fast": int(fast), "slow": int(slow),
        "sma_fast": sma_f, "sma_slow": sma_s, "atr": atr, "atr_pct": atr_pct, "rsi": rsi_val,
        "horizon": horizon, "days_to_stop": d_stop, "days_to_take": d_take, "avg_vol_20": avg_vol_20,
    }

# ---------- ORDERING ----------
def _dynamic_pad_dec(base: float) -> Decimal:
    """Dynamic cushion so TP/SL satisfy Alpaca base_price rules even if price moves."""
    if base >= 1:
        min_pad = Decimal('0.05')
        pct_pad = Decimal('0.001') * Decimal(str(base))   # 0.1%
    else:
        min_pad = Decimal('0.002')
        pct_pad = Decimal('0.001') * Decimal(str(base))
    pad = pct_pad if pct_pad > min_pad else min_pad
    tick = _tick_size(base)
    steps = (pad / tick).to_integral_value(rounding=ROUND_UP)
    return steps * tick

def execute_order(
    ticker: str,
    side: str,
    notional: float,
    entry_price: float,
    stop: float,
    take: float,
    dry_run: bool = True,
) -> Tuple[str, str, float]:
    side = (side or "BUY").upper()
    if side not in ("BUY", "SELL"):
        side = "BUY"

    base = _alpaca_live_base_price(ticker) or _live_price(ticker) or float(entry_price)
    base_dec = Decimal(str(base))
    tick_dec = _tick_size(base)
    one_min_dec = Decimal('0.01') if base >= 1 else Decimal('0.0001')
    pad_dec = _dynamic_pad_dec(base)

    def _build_tp_sl(_base_dec: Decimal, _tp: float, _sl: float, _side: str) -> Tuple[float,float]:
        tp_d = Decimal(str(_tp)); sl_d = Decimal(str(_sl))
        if _side == "BUY":
            tp_d = max(tp_d, _base_dec + pad_dec)
            sl_d = min(sl_d, _base_dec - pad_dec)
        else:
            tp_d = min(tp_d, _base_dec - pad_dec)
            sl_d = max(sl_d, _base_dec + pad_dec)
        tp_d = tp_d.quantize(tick_dec, rounding=ROUND_HALF_UP)
        sl_d = sl_d.quantize(tick_dec, rounding=ROUND_HALF_UP)
        if _side == "BUY":
            if tp_d < _base_dec + one_min_dec:
                tp_d = (_base_dec + one_min_dec + tick_dec).quantize(tick_dec, rounding=ROUND_HALF_UP)
            if sl_d > _base_dec - one_min_dec:
                sl_d = (_base_dec - one_min_dec - tick_dec).quantize(tick_dec, rounding=ROUND_HALF_UP)
        else:
            if tp_d > _base_dec - one_min_dec:
                tp_d = (_base_dec - one_min_dec - tick_dec).quantize(tick_dec, rounding=ROUND_HALF_UP)
            if sl_d < _base_dec + one_min_dec:
                sl_d = (_base_dec + one_min_dec + tick_dec).quantize(tick_dec, rounding=ROUND_HALF_UP)
        return float(tp_d), float(sl_d)

    tp, sl = _build_tp_sl(base_dec, float(take), float(stop), side)

    est_qty = max(1, int(notional / max(0.01, float(base))))
    if dry_run:
        _log_trade(ticker, side, est_qty, notional, float(base), "simulated", "DRY_RUN")
        return "DRY_RUN", "simulated", float(est_qty)

    client = _get_client()

    fractionable = True
    try:
        asset = client.get_asset(ticker)
        if getattr(asset, "status", "active") != "active" or not getattr(asset, "tradable", True):
            msg = f"asset {ticker} is not active/tradable on Alpaca"
            _log_trade(ticker, side, 0.0, notional, float(base), f"error: {msg}", "ERROR")
            return "ERROR", f"error: {msg}", 0.0
        fractionable = bool(getattr(asset, "fractionable", True))
    except Exception:
        pass

    def _submit(tp_val: float, sl_val: float) -> Tuple[str,str,float]:
        if HAS_NOTIONAL and fractionable and notional < float(base):
            req = NotionalOrderRequest(
                symbol=ticker,
                notional=float(notional),
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_val),
                stop_loss=StopLossRequest(stop_price=sl_val),
            )
            order = client.submit_order(order_data=req)
            order_qty = float(notional / float(base))
        else:
            if est_qty < 1:
                msg = f"Budget ${notional:.0f} too small vs live price ${float(base):.2f} for qty-based order."
                _log_trade(ticker, side, 0.0, notional, float(base), f"error: {msg}", "ERROR")
                return "ERROR", f"error: {msg}", 0.0
            req = MarketOrderRequest(
                symbol=ticker,
                qty=int(est_qty),
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_val),
                stop_loss=StopLossRequest(stop_price=sl_val),
            )
            order = client.submit_order(order_data=req)
            order_qty = float(est_qty)
        order_id = str(getattr(order, "id", None) or getattr(order, "client_order_id", "unknown"))
        status = str(getattr(order, "status", "accepted"))
        return order_id, status, order_qty

    try:
        order_id, status, order_qty = _submit(tp, sl)
        _log_trade(ticker, side, order_qty, notional, float(base), status, order_id)
        return order_id, status, order_qty
    except Exception as e:
        msg = str(e)
        m = re.search(r'"base_price":"([\d\.]+)"', msg)
        if m and ("must be" in msg or "minimum pricing criteria" in msg):
            try:
                new_base_dec = Decimal(m.group(1))
                tp2, sl2 = _build_tp_sl(new_base_dec, float(take), float(stop), side)
                order_id, status, order_qty = _submit(tp2, sl2)
                _log_trade(ticker, side, order_qty, notional, float(new_base_dec), status, order_id)
                return order_id, status, order_qty
            except Exception as e2:
                msg2 = str(e2)
                _log_trade(ticker, side, 0.0, notional, float(base), f"error: {msg2[:120]}", "ERROR")
                return "ERROR", f"error: {msg2}", 0.0
        _log_trade(ticker, side, 0.0, notional, float(base), f"error: {msg[:120]}", "ERROR")
        return "ERROR", f"error: {msg}", 0.0

# ---------- QUERIES ----------
def get_open_positions() -> pd.DataFrame:
    try:
        client = _get_client()
        pos = client.get_all_positions()
        if not pos:
            return pd.DataFrame()
        rows = [p.model_dump() if hasattr(p, "model_dump") else p.__dict__ for p in pos]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def get_open_orders() -> pd.DataFrame:
    try:
        client = _get_client()
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = client.get_orders(filter=req)
        if not orders:
            return pd.DataFrame()
        rows = []
        for o in orders:
            d = o.model_dump() if hasattr(o, "model_dump") else o.__dict__
            rows.append({
                "symbol": d.get("symbol"),
                "side": d.get("side"),
                "type": d.get("type"),
                "qty": d.get("qty"),
                "limit_price": d.get("limit_price"),
                "stop_price": d.get("stop_price"),
                "status": d.get("status"),
                "submitted_at": d.get("submitted_at"),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def close_all_positions_paper():
    client = _get_client()
    client.close_all_positions(cancel_orders=True)

# ---------- FOLLOW-UP ----------
def get_trade_followup(days_back: int = 5) -> pd.DataFrame:
    """
    Recent trades + same-day (closest prior) recommendation from `signals`,
    with live price, P&L%, and distances to stop/take when available.
    """
    try:
        con = _ensure_db()
        df_tr = con.execute("SELECT * FROM trades ORDER BY ts DESC").fetchdf()
        con.close()
    except Exception:
        return pd.DataFrame()

    if df_tr.empty:
        return pd.DataFrame()

    now_utc = pd.Timestamp.utcnow()
    df_tr["ts"] = pd.to_datetime(df_tr["ts"], utc=True, errors="coerce")
    df_tr = df_tr[df_tr["ts"] >= (now_utc - pd.Timedelta(days=days_back))].copy()
    if df_tr.empty:
        return pd.DataFrame()

    rows = []
    for _, r in df_tr.iterrows():
        tkr = str(r["ticker"]).upper()
        ts_trade = r["ts"]
        try:
            con = duckdb.connect(str(DB_PATH))
            sig = con.execute(
                """
                SELECT * FROM signals
                WHERE ticker = ? AND ts <= ?
                ORDER BY ts DESC
                LIMIT 1
                """,
                [tkr, ts_trade],
            ).fetchdf()
            con.close()
        except Exception:
            sig = pd.DataFrame()

        live = None
        try:
            ti = yf.Ticker(tkr)
            info = getattr(ti, "fast_info", {}) or {}
            live = info.get("last_price") or info.get("regular_market_price")
            live = float(live) if live else None
        except Exception:
            pass

        entry = float(r.get("price") or 0.0)
        pnl_pct = ((live - entry) / entry * 100.0) if (live and entry) else None

        rec_dec = conf = rec_stop = rec_take = reason = None
        if not isinstance(sig, pd.DataFrame) or sig.empty is False:
            if isinstance(sig, pd.DataFrame) and not sig.empty:
                s = sig.iloc[0]
                rec_dec  = s.get("decision")
                conf     = s.get("confidence")
                rec_stop = s.get("stop")
                rec_take = s.get("take")
                reason   = s.get("reason")

        to_stop_pct = ((live - rec_stop) / entry * 100.0) if (live and rec_stop and entry) else None
        to_take_pct = ((rec_take - live) / entry * 100.0) if (live and rec_take and entry) else None

        rows.append({
            "ticker": tkr,
            "trade_ts": ts_trade,
            "side": r.get("side"),
            "qty": r.get("qty"),
            "entry_price": entry,
            "status": r.get("status"),
            "order_id": r.get("order_id"),
            "rec_action": rec_dec,
            "confidence": conf,
            "rec_stop": rec_stop,
            "rec_take": rec_take,
            "current_price": live,
            "pnl_pct": pnl_pct,
            "to_stop_%": to_stop_pct,
            "to_take_%": to_take_pct,
            "reason": reason
        })

    out = pd.DataFrame(rows)
    return out.sort_values(by="trade_ts", ascending=False).reset_index(drop=True)

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=50)
    ap.add_argument("--notional", type=float, default=100.0)
    ap.add_argument("--dry", type=int, default=1)
    args = ap.parse_args()

    rec = generate_recommendation(args.ticker, args.fast, args.slow, args.notional, use_rsi=True)
    print("Recommendation:", rec["action"], "| amount:", rec["amount"], "| stop:", rec["stop"], "| take:", rec["take"], "| conf:", rec["confidence"])
    print(rec["reason"])

    if rec["action"] in ("BUY","SELL"):
        oid, status, qty = execute_order(
            ticker=rec["ticker"], side=rec["action"], notional=rec["amount"],
            entry_price=rec["price"], stop=rec["stop"], take=rec["take"], dry_run=bool(args.dry)
        )
        print(f"Order result: id={oid}, status={status}, qty={qty}")
    else:
        print("Action is WAIT – no order placed.")
