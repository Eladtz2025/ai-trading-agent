# src/utils_purchases.py
from __future__ import annotations
import os
from datetime import timedelta
from typing import List, Optional, Dict, Any

import duckdb
import pandas as pd

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
    """Return minimal normalized view of trades we need."""
    if df is None or df.empty:
        return pd.DataFrame()

    w = df.copy()

    sym = _pick(w, ["symbol", "ticker", "asset_symbol", "asset"])
    w["symbol"] = (w[sym] if sym else "").astype(str).str.upper()

    side = _pick(w, ["side", "action", "order_side"])
    w["side"] = (w[side] if side else "").astype(str).str.upper()

    qty = _pick(w, ["qty", "quantity", "filled_qty", "filled_quantity"])
    notional = _pick(w, ["notional", "order_notional", "amount", "value"])
    avgp = _pick(w, ["avg_price", "avg_entry_price", "price", "fill_price", "executed_price"])

    w["qty"] = pd.to_numeric(w[qty], errors="coerce") if qty else None
    w["notional"] = pd.to_numeric(w[notional], errors="coerce") if notional else None
    w["avg_price"] = pd.to_numeric(w[avgp], errors="coerce") if avgp else None

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

    # keep buys / positive amounts
    mask_buy = w["side"].str.contains("BUY", na=False) | w["side"].str.contains("LONG", na=False)
    mask_pos = (pd.to_numeric(w["qty"], errors="coerce") > 0) | (pd.to_numeric(w["notional"], errors="coerce") > 0)
    w = w[mask_buy | mask_pos].copy()

    keep = ["symbol", "purchase_ts", "purchase_date", "horizon_days"]
    for c in keep:
        if c not in w.columns:
            w[c] = None
    return w[keep]

# --------------------------------------------------------------------
# Public helpers used by the Streamlit app
# --------------------------------------------------------------------
def load_latest_buy_info(db_path: str = DEFAULT_DB) -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping like:
      {"AAPL": {"buy_date": date, "horizon_days": int|None}, ...}
    Based on the last BUY (or positive qty/notional) per symbol from 'trades'.
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

        grp = (
            norm.groupby("symbol", as_index=False)
                .agg({"purchase_ts": "max", "purchase_date": "max", "horizon_days": "last"})
        )

        out: Dict[str, Dict[str, Any]] = {}
        for _, r in grp.iterrows():
            out[str(r["symbol"]).upper()] = {
                "buy_date": r["purchase_date"],
                "horizon_days": (int(r["horizon_days"]) if pd.notnull(r["horizon_days"]) else None),
            }
        return out
    except Exception:
        return {}

def compute_desired_sell_date(buy_date, horizon_days=None, fallback_days: int = DEFAULT_MAX_HOLD_DAYS):
    """If horizon_days is provided → buy_date + horizon_days; else → +fallback_days (default 10)."""
    if buy_date is None or pd.isna(buy_date):
        return None
    days = int(horizon_days) if (horizon_days is not None and not pd.isna(horizon_days)) else int(fallback_days)
    return buy_date + timedelta(days=days)

def load_latest_targets(db_path: str = DEFAULT_DB) -> Dict[str, float]:
    """
    Returns mapping of latest target (take-profit) per symbol from 'trades'.
    Preference order for columns: 'rec_take', 'take', 'take_price', 'tp', 'takeprofit'.
    Chooses the most recent record per symbol using timestamp columns if available.
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

        # latest per symbol
        d = d.sort_values("ts").groupby("symbol", as_index=False).tail(1)
        return {row["symbol"]: float(row["take"]) for _, row in d.iterrows() if pd.notnull(row["take"])}
    except Exception:
        return {}

# --------------------------------------------------------------------
# Backward-compat: avoid ImportError if some old code still imports it
# --------------------------------------------------------------------
def render_purchases_by_day_panel(*args, **kwargs):
    """Deprecated placeholder. Kept to avoid import errors if an old import remains."""
    return None
