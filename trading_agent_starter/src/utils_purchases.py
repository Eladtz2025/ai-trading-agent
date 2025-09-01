# src/utils_purchases.py
from __future__ import annotations
import os
from datetime import timedelta, date
from typing import List, Optional, Dict, Any

import duckdb
import pandas as pd

# business-day calendar (US holidays); safe fallback if not available
try:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    _HAS_US_CAL = True
except Exception:
    from pandas.tseries.offsets import BDay as CustomBusinessDay  # Mon–Fri only
    _HAS_US_CAL = False

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
DEFAULT_DB = os.getenv("TRADE_LOG_DUCKDB", "logs/trade_log.duckdb")
DEFAULT_MAX_HOLD_DAYS = int(os.getenv("DEFAULT_MAX_HOLD_DAYS", "10"))

# --------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------
def _connect(db_path: str = DEFAULT_DB):
    """Open (and ensure dir exists) DuckDB file."""
    dirpath = os.path.dirname(db_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    return duckdb.connect(db_path, read_only=False)

def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def _pick(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def _normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Return minimal normalized BUY trades with qty/ts/date/horizon."""
    if df is None or df.empty:
        return pd.DataFrame()

    w = df.copy()

    sym = _pick(w, ["symbol", "ticker", "asset_symbol", "asset"])
    w["symbol"] = (w[sym] if sym else "").astype(str).str.upper()

    side = _pick(w, ["side", "action", "order_side"])
    w["side"] = (w[side] if side else "").astype(str).str.upper()

    qty = _pick(w, ["qty", "quantity", "filled_qty", "filled_quantity"])
    w["qty"] = pd.to_numeric(w[qty], errors="coerce") if qty else None

    hz = _pick(w, ["horizon_days", "eta_days", "eta_to_tp_days"])
    w["horizon_days"] = w[hz] if hz else None

    tscol = _pick(w, ["filled_at", "executed_at", "ts", "timestamp", "created_at", "time", "date"])
    ts = _safe_to_datetime(w[tscol]) if tscol else pd.Series(pd.NaT, index=w.index, dtype="datetime64[ns, UTC]")
    w["purchase_ts"] = ts
    w["purchase_date"] = (
        w["purchase_ts"].dt.tz_convert(None).dt.date
        if getattr(w["purchase_ts"].dt, "tz", None) is not None
        else w["purchase_ts"].dt.date
    )

    # keep only BUY/LONG or positive qty rows
    mask_buy = w["side"].str.contains("BUY", na=False) | w["side"].str.contains("LONG", na=False)
    mask_pos = (pd.to_numeric(w["qty"], errors="coerce") > 0)
    w = w[mask_buy | mask_pos].copy()

    keep = ["symbol", "purchase_ts", "purchase_date", "horizon_days", "qty"]
    for c in keep:
        if c not in w.columns:
            w[c] = None
    return w[keep].dropna(subset=["symbol"])

# --------------------------------------------------------------------
# Public helpers used by the Streamlit app
# --------------------------------------------------------------------
def load_latest_buy_info(db_path: str = DEFAULT_DB) -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping like:
      {
        "AAPL": {
          "buy_date": <quantity-weighted buy date>,
          "buy_date_first": <first BUY date>,
          "buy_date_last": <last BUY date>,
          "horizon_days": int|None
        },
        ...
      }
    """
    try:
        con = _connect(db_path)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        if "trades" not in tables:
            return {}
        df = con.execute("SELECT * FROM trades").df()
        if df is None or df.empty:
            return {}

        norm = _normalize_trades(df)
        if norm.empty:
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        for sym, g in norm.groupby("symbol"):
            g2 = g.sort_values("purchase_ts").copy()
            # first and last buy dates
            buy_first = g2["purchase_date"].min()
            buy_last = g2["purchase_date"].max()
            # quantity-weighted average timestamp
            g2["w"] = pd.to_numeric(g2["qty"], errors="coerce").fillna(0.0)
            if g2["w"].sum() > 0 and g2["purchase_ts"].notna().any():
                # convert to epoch ns, weight, back to date
                ts_ns = g2["purchase_ts"].astype("int64")
                wavg_ns = (ts_ns * g2["w"]).sum() / g2["w"].sum()
                buy_weighted = pd.to_datetime(wavg_ns).tz_localize("UTC").tz_convert(None).date()
            else:
                buy_weighted = buy_last  # fallback

            # Horizon: use last row's horizon if any
            hz = None
            if "horizon_days" in g2.columns:
                try:
                    hz = int(g2["horizon_days"].dropna().iloc[-1])
                except Exception:
                    hz = None

            out[str(sym).upper()] = {
                "buy_date": buy_weighted,            # main date we display/use
                "buy_date_first": buy_first,
                "buy_date_last": buy_last,
                "horizon_days": hz,
            }
        return out
    except Exception:
        return {}

def _business_day_add(start: date, days: int) -> date:
    """Add trading days (Mon–Fri; US holidays excluded if available)."""
    try:
        if _HAS_US_CAL:
            cbd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        else:
            cbd = CustomBusinessDay()  # Mon–Fri
        return (pd.Timestamp(start) + cbd * int(days)).date()
    except Exception:
        # simple Mon–Fri fallback
        d = pd.Timestamp(start).date()
        left = int(days)
        while left > 0:
            d += timedelta(days=1)
            if d.weekday() < 5:
                left -= 1
        return d

def compute_desired_sell_date(buy_date, horizon_days=None, fallback_days: int = DEFAULT_MAX_HOLD_DAYS):
    """Calendar-days version (kept for backward-compat)."""
    if buy_date is None or pd.isna(buy_date):
        return None
    days = int(horizon_days) if (horizon_days is not None and not pd.isna(horizon_days)) else int(fallback_days)
    return pd.to_datetime(buy_date).date() + timedelta(days=days)

def compute_desired_sell_date_trading(buy_date, horizon_days=None, fallback_days: int = DEFAULT_MAX_HOLD_DAYS):
    """Trading-days version (requested): Mon–Fri, US holidays excluded (when available)."""
    if buy_date is None or pd.isna(buy_date):
        return None
    days = int(horizon_days) if (horizon_days is not None and not pd.isna(horizon_days)) else int(fallback_days)
    return _business_day_add(pd.to_datetime(buy_date).date(), days)

def load_latest_targets(db_path: str = DEFAULT_DB) -> Dict[str, float]:
    """
    Returns mapping of latest target (take-profit) per symbol from 'trades'.
    Preference: 'rec_take' > 'take' > 'take_price' > 'tp' > 'takeprofit'.
    """
    try:
        con = _connect(db_path)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        if "trades" not in tables:
            return {}

        df = con.execute("SELECT * FROM trades").df()
        if df is None or df.empty:
            return {}

        sym_col = _pick(df, ["symbol", "ticker", "asset_symbol", "asset"])
        take_col = _pick(df, ["rec_take", "take", "take_price", "tp", "takeprofit"])
        ts_col = _pick(df, ["filled_at", "executed_at", "ts", "timestamp", "created_at", "time", "date"])
        if not sym_col or not take_col:
            return {}

        d = df.copy()
        d["symbol"] = d[sym_col].astype(str).str.upper()
        d["take"] = pd.to_numeric(d[take_col], errors="coerce")
        d["ts"] = _safe_to_datetime(d[ts_col]) if ts_col else pd.to_datetime(pd.Series([], dtype="datetime64[ns, UTC]"))
        d = d.dropna(subset=["take"])
        if d.empty:
            return {}

        d = d.sort_values("ts").groupby("symbol", as_index=False).tail(1)
        return {row["symbol"]: float(row["take"]) for _, row in d.iterrows() if pd.notnull(row["take"])}
    except Exception:
        return {}

def load_latest_stops(db_path: str = DEFAULT_DB) -> Dict[str, float]:
    """
    Returns mapping of latest stop price per symbol from 'trades'.
    Preference: 'rec_stop' > 'stop' > 'stop_price' > 'sl' > 'stoploss'.
    """
    try:
        con = _connect(db_path)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        if "trades" not in tables:
            return {}

        df = con.execute("SELECT * FROM trades").df()
        if df is None or df.empty:
            return {}

        sym_col = _pick(df, ["symbol", "ticker", "asset_symbol", "asset"])
        stop_col = _pick(df, ["rec_stop", "stop", "stop_price", "sl", "stoploss"])
        ts_col = _pick(df, ["filled_at", "executed_at", "ts", "timestamp", "created_at", "time", "date"])
        if not sym_col or not stop_col:
            return {}

        d = df.copy()
        d["symbol"] = d[sym_col].astype(str).str.upper()
        d["stop"] = pd.to_numeric(d[stop_col], errors="coerce")
        d["ts"] = _safe_to_datetime(d[ts_col]) if ts_col else pd.to_datetime(pd.Series([], dtype="datetime64[ns, UTC]"))
        d = d.dropna(subset=["stop"])
        if d.empty:
            return {}

        d = d.sort_values("ts").groupby("symbol", as_index=False).tail(1)
        return {row["symbol"]: float(row["stop"]) for _, row in d.iterrows() if pd.notnull(row["stop"])}
    except Exception:
        return {}

# --------------------------------------------------------------------
# Backward-compat: avoid ImportError if some old code still imports it
# --------------------------------------------------------------------
def render_purchases_by_day_panel(*args, **kwargs):
    """Deprecated placeholder. Kept to avoid import errors if an old import remains."""
    return None
