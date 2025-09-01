# src/utils_purchases.py
# Render "Purchases by Day" tables from logs/trade_log.duckdb (table: trades),
# and add a per-row "Exit by (date)" based on a configurable Max Hold (days).

from __future__ import annotations
import os
import calendar
from datetime import date, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import duckdb
import streamlit as st

DEFAULT_DB = os.getenv("TRADE_LOG_DUCKDB", "logs/trade_log.duckdb")

def _connect(db_path: str = DEFAULT_DB):
    if not os.path.exists(os.path.dirname(db_path)) and os.path.dirname(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return duckdb.connect(db_path, read_only=False)

def _safe_to_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", utc=True)
    except Exception:
        return pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))

def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to: symbol, side, qty, notional, avg_price, stop, take,
    confidence, reason, purchase_ts (UTC), purchase_date (YYYY-MM-DD)."""
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    sym_col = _pick_first_existing(work, ["symbol", "ticker", "asset_symbol", "asset"])
    work["symbol"] = (work[sym_col] if sym_col else "").astype(str).str.upper()

    side_col = _pick_first_existing(work, ["side", "action", "order_side"])
    work["side"] = (work[side_col] if side_col else "").astype(str).str.upper()

    qty_col = _pick_first_existing(work, ["qty", "quantity", "filled_qty", "filled_quantity"])
    notional_col = _pick_first_existing(work, ["notional", "order_notional", "amount", "value"])
    avg_price_col = _pick_first_existing(work, ["avg_price", "avg_entry_price", "price", "fill_price", "executed_price"])

    work["qty"] = pd.to_numeric(work[qty_col], errors="coerce") if qty_col else None
    work["notional"] = pd.to_numeric(work[notional_col], errors="coerce") if notional_col else None
    work["avg_price"] = pd.to_numeric(work[avg_price_col], errors="coerce") if avg_price_col else None

    for field, candidates in {
        "stop": ["stop", "stop_price", "sl", "stoploss"],
        "take": ["take", "take_price", "tp", "takeprofit"],
        "confidence": ["confidence"],
        "reason": ["reason", "note", "comment", "text"],
        "horizon_days": ["horizon_days", "eta_days", "eta_to_tp_days"],
    }.items():
        col = _pick_first_existing(work, candidates)
        work[field] = work[col] if col else None

    time_candidates = ["filled_at", "executed_at", "ts", "timestamp", "created_at", "time", "date"]
    ts_col = _pick_first_existing(work, time_candidates)
    if ts_col:
        ts = _safe_to_datetime(work[ts_col])
    else:
        ts = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")

    work["purchase_ts"] = ts
    work["purchase_date"] = work["purchase_ts"].dt.tz_convert(None).dt.date if work["purchase_ts"].dt.tz is not None else work["purchase_ts"].dt.date

    mask_buy = work["side"].str.contains("BUY", na=False) | work["side"].str.contains("LONG", na=False)
    mask_pos = (pd.to_numeric(work["qty"], errors="coerce") > 0) | (pd.to_numeric(work["notional"], errors="coerce") > 0)
    work = work[mask_buy | mask_pos].copy()

    cols = ["purchase_ts", "purchase_date", "symbol", "side", "qty", "notional", "avg_price",
            "stop", "take", "confidence", "reason", "horizon_days"]
    for c in cols:
        if c not in work.columns:
            work[c] = None
    return work[cols].sort_values(["purchase_ts", "symbol"])

def _month_bounds(year: int, month: int):
    start = date(year, month, 1)
    end = date(year, month, calendar.monthrange(year, month)[1])
    return start, end

def render_purchases_by_day_panel(db_path: str = DEFAULT_DB) -> None:
    st.subheader("Purchases by Day")

    today = date.today()
    c1, c2, c3 = st.columns([1,1,1])
    year = c1.number_input("Year", min_value=2015, max_value=2100, value=today.year, step=1)
    month = c2.selectbox("Month", options=list(range(1,13)), index=today.month-1, format_func=lambda m: f"{m:02d}")
    max_hold_days = c3.number_input(
        "Max Hold (days) for 'Exit by'",
        min_value=1, max_value=120, value=10, step=1,
        help="Used to compute 'Exit by (date)' as: purchase_date + Max Hold"
    )

    month_start, month_end = _month_bounds(int(year), int(month))
    st.caption(f"Showing purchases from {month_start} to {month_end}")

    try:
        con = _connect(db_path)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        if "trades" not in tables:
            st.info("No 'trades' table found in DuckDB yet.")
            return
        df = con.execute("SELECT * FROM trades").df()
    except Exception as e:
        st.error(f"DuckDB error: {e}")
        return

    if df.empty:
        st.write("No trade records yet.")
        return

    norm = _normalize_trades(df)
    if norm.empty:
        st.write("No BUY trades found.")
        return

    mm = (norm["purchase_date"] >= month_start) & (norm["purchase_date"] <= month_end)
    view = norm[mm].copy()
    if view.empty:
        st.write("No purchases in the selected month.")
        return

    view["Exit by (date)"] = view["purchase_date"].apply(lambda d: d + timedelta(days=int(max_hold_days)))
    if "horizon_days" in view.columns and pd.api.types.is_numeric_dtype(view["horizon_days"]):
        view["ETA exit (model)"] = view.apply(
            lambda r: (r["purchase_date"] + timedelta(days=int(r["horizon_days"]))) if pd.notnull(r["horizon_days"]) else None,
            axis=1
        )

    display_cols = ["symbol", "qty", "notional", "avg_price", "stop", "take",
                    "confidence", "reason", "purchase_ts", "Exit by (date)"]
    if "ETA exit (model)" in view.columns:
        display_cols.append("ETA exit (model)")

    for d in sorted(view["purchase_date"].unique()):
        sub = view[view["purchase_date"] == d].copy()
        st.markdown(f"### {d} — {len(sub)} purchase(s)")
        st.dataframe(sub[display_cols], use_container_width=True)

# ---- Helpers for Open Positions table (append at end of file) ----
import pandas as pd
from datetime import timedelta

def load_latest_buy_info(db_path: str = DEFAULT_DB):
    """
    Returns: dict like {"AAPL": {"buy_date": date, "horizon_days": int|None}, ...}
    Based on the 'trades' table (BUY/LONG or positive qty/notional).
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

        # latest purchase per symbol
        grp = (
            norm
            .groupby("symbol", as_index=False)
            .agg({"purchase_ts": "max", "purchase_date": "max", "horizon_days": "last"})
        )

        out = {}
        for _, r in grp.iterrows():
            sym = str(r["symbol"]).upper()
            out[sym] = {
                "buy_date": r["purchase_date"],
                "horizon_days": (int(r["horizon_days"]) if pd.notnull(r["horizon_days"]) else None),
            }
        return out
    except Exception:
        return {}

def compute_desired_sell_date(buy_date, horizon_days=None, f_
