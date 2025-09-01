# streamlit_dashboard_pro.py
import os, sys
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from paper_execute_sma import (
    generate_recommendation, execute_order, get_open_positions,
    close_all_positions_paper, get_open_orders, configure_risk
)
from utils_screener import screen_tickers
from utils_universe import UNIVERSES
from paper_execute_sma import get_trade_followup
from src.utils_purchases import load_latest_buy_info, compute_desired_sell_date

st.set_page_config(page_title="AI Trading Agent – Paper", layout="wide")
st.title("AI Trading Agent – Paper Trading")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker (US stock)", value="AAPL").upper().strip()
    fast = st.number_input("SMA Fast", min_value=3, max_value=200, value=20, step=1)
    slow = st.number_input("SMA Slow", min_value=5, max_value=400, value=50, step=1)
    notional = st.number_input("Order Budget (USD)", min_value=50.0, max_value=100000.0, value=100.0, step=10.0, format="%.2f")

    st.divider()
    st.subheader("Risk & Quality settings")
    risk_pct = st.slider("Risk per trade (% of equity)", min_value=0.1, max_value=2.0, value=0.5, step=0.1) / 100.0
    min_vol = st.number_input("Min Avg Volume (20d)", min_value=0, max_value=50_000_000, value=1_000_000, step=100_000)
    max_price_allowed = st.number_input("Max Price Allowed", min_value=1.0, max_value=2000.0, value=400.0, step=1.0, format="%.2f")

    dry_run = st.toggle("DRY_RUN (Simulation only)", value=True, help="If ON, no real paper order will be sent.")
    st.caption("All outputs are plain English. Bracket orders use stop/take aligned to live price with tick rounding.")

# Apply risk knobs globally
configure_risk(risk_pct=risk_pct, min_avg_vol_20=min_vol, max_price_allowed=max_price_allowed)

if dry_run:
    st.warning("Simulation only – no real paper order.", icon="⚠️")

# ---------- Chart helper ----------
@st.cache_data(ttl=300)
def load_chart_data(sym: str, f: int, s: int):
    df = yf.download(sym, period="6mo", interval="1d", auto_adjust=True, progress=False)
    if df.empty: return pd.DataFrame()
    df = df.dropna().copy()
    df["SMA_FAST"] = df["Close"].rolling(int(f), min_periods=int(f)).mean()
    df["SMA_SLOW"] = df["Close"].rolling(int(s), min_periods=int(s)).mean()
    return df

def draw_chart(sym: str, f: int, s: int, stop, take):
    df = load_chart_data(sym, f, s)
    if df.empty:
        st.info("No chart data.")
        return
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["SMA_FAST"], label=f"SMA {f}")
    ax.plot(df.index, df["SMA_SLOW"], label=f"SMA {s}")
    if stop: ax.axhline(stop, linestyle="--", linewidth=1, label=f"Stop {stop:.2f}")
    if take: ax.axhline(take, linestyle="--", linewidth=1, label=f"Take {take:.2f}")
    ax.legend(loc="best"); ax.set_title(f"{sym} — 6M Daily")
    st.pyplot(fig)

# ---------- Recommendation panel ----------
colL, colR = st.columns([1.25, 1])

with colL:
    st.subheader("Plain English Recommendation")

    if st.button("Get Recommendation", type="primary"):
        st.session_state["rec"] = generate_recommendation(ticker, int(fast), int(slow), float(notional), use_rsi=True)

    rec = st.session_state.get("rec")
    if rec:
        st.markdown(
            f"**Action:** {rec['action']}  |  **Amount:** ${rec['amount']:.0f}  |  "
            f"**Stop:** {rec['stop']}  |  **Take:** {rec['take']}  |  "
            f"**Confidence:** {rec['confidence']}  |  **Horizon:** {rec.get('horizon','—')}  |  "
            f"**~Days → Stop/TP:** {rec.get('days_to_stop','—')}/{rec.get('days_to_take','—')}"
        )
        st.info(rec["reason"])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Price", f"{rec['price']:.2f}" if rec.get("price") else "—")
        m2.metric(f"SMA {rec['fast']}", f"{rec['sma_fast']:.2f}" if rec.get("sma_fast") else "—")
        m3.metric(f"SMA {rec['slow']}", f"{rec['sma_slow']:.2f}" if rec.get("sma_slow") else "—")
        m4.metric("ATR (14)", f"{rec['atr']:.2f}" if rec.get("atr") is not None else "—")
        m5.metric("RSI (14)", f"{rec['rsi']:.0f}" if rec.get("rsi") is not None else "—")

        st.markdown("**Price Chart**")
        draw_chart(rec["ticker"], rec["fast"], rec["slow"],
                   rec["stop"] if rec["action"]!="WAIT" else None,
                   rec["take"] if rec["action"]!="WAIT" else None)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Place a market **bracket** order (TP & SL).")
            if st.button("Place order now (Paper)", disabled=(rec["action"] == "WAIT")):
                if dry_run:
                    st.warning("DRY_RUN is ON → simulated only. Toggle OFF to send a real paper order.", icon="⚠️")
                order_id, status, qty = execute_order(
                    ticker=rec["ticker"], side=rec["action"],
                    notional=rec["amount"], entry_price=rec["price"],
                    stop=rec["stop"], take=rec["take"], dry_run=dry_run
                )
                st.success(f"Order result: id={order_id}, status={status}, qty={qty}")

        with c2:
            st.caption("Close all open paper positions on Alpaca.")
            if st.button("Close ALL paper positions"):
                try:
                    close_all_positions_paper()
                    st.success("Requested to close all positions.")
                except Exception as e:
                    st.error(f"Close-all failed: {e}")

st.subheader("Open Positions (Paper)")
try:
    pos_df = get_open_positions()  # הפונקציה הקיימת אצלך
    if pos_df.empty:
        st.info("No open positions.")
    else:
        # העמודות הבסיסיות שלך
        cols = ["symbol","side","qty","avg_entry_price","current_price","market_value","unrealized_pl","unrealized_plpc"]
        cols = [c for c in cols if c in pos_df.columns]
        pos_df_show = pos_df[cols].copy()

        # המרות מספריות
        for c in ["qty","avg_entry_price","current_price","market_value","unrealized_pl","unrealized_plpc"]:
            if c in pos_df_show.columns:
                pos_df_show[c] = pd.to_numeric(pos_df_show[c], errors="coerce")
        if "unrealized_plpc" in pos_df_show.columns:
            pos_df_show["unrealized_plpc"] = (pos_df_show["unrealized_plpc"] * 100).round(2)

        # === NEW: Change % (current vs entry)
        if "avg_entry_price" in pos_df_show.columns and "current_price" in pos_df_show.columns:
            base = pos_df_show["avg_entry_price"].replace(0, pd.NA)
            pos_df_show["change_pct"] = ((pos_df_show["current_price"] / base) - 1.0) * 100
            pos_df_show["change_pct"] = pos_df_show["change_pct"].round(2)

        # === NEW: Buy date + Desired sell date (מ-DuckDB)
        buy_map = load_latest_buy_info()

        def _buy_date(sym):
            info = buy_map.get(str(sym).upper())
            return info["buy_date"] if info else None

        def _desired(sym):
            info = buy_map.get(str(sym).upper())
            if info and info.get("buy_date"):
                return compute_desired_sell_date(info["buy_date"], info.get("horizon_days"))
            return None

        pos_df_show["buy_date"] = pos_df_show["symbol"].map(_buy_date)
        pos_df_show["desired_sell_date"] = pos_df_show["symbol"].map(_desired)

        # סידור העמודות: Change % אחרי current_price; Buy/Desired אחרי avg_entry_price
        display_cols = cols.copy()
        if "change_pct" in pos_df_show.columns:
            idx_cp = display_cols.index("current_price")+1 if "current_price" in display_cols else len(display_cols)
            if "change_pct" not in display_cols:
                display_cols.insert(idx_cp, "change_pct")
        idx_avg = display_cols.index("avg_entry_price")+1 if "avg_entry_price" in display_cols else len(display_cols)
        for newc in ["buy_date","desired_sell_date"]:
            if newc not in display_cols:
                display_cols.insert(idx_avg, newc); idx_avg += 1

        df_ui = pos_df_show[display_cols].rename(columns={
            "change_pct": "Change %",
            "buy_date": "Buy date",
            "desired_sell_date": "Desired sell date"
        })
        st.dataframe(df_ui, use_container_width=True)
except Exception as e:
    st.error(f"Failed to fetch positions: {e}")

if st.button("Refresh positions"):
    st.rerun()

st.divider()
st.subheader("Purchased – Follow-up (last 5 days)")

colFU1, colFU2 = st.columns([1, 0.15])
with colFU1:
    try:
        fu_df = get_trade_followup(days_back=5)
        if fu_df.empty:
            st.info("No recent trades to follow-up.")
        else:
            show_cols = [
                "trade_ts","ticker","side","qty","entry_price","current_price",
                "pnl_pct","rec_action","confidence","rec_stop","rec_take","to_stop_%","to_take_%"
            ]
            show = [c for c in show_cols if c in fu_df.columns]
            df_disp = fu_df[show].copy()
            # formatting
            for c in ["entry_price","current_price","rec_stop","rec_take"]:
                if c in df_disp.columns:
                    df_disp[c] = df_disp[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            for c in ["pnl_pct","to_stop_%","to_take_%"]:
                if c in df_disp.columns:
                    df_disp[c] = df_disp[c].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")
            st.dataframe(df_disp, use_container_width=True, height=300)
    except Exception as e:
        st.error(f"Follow-up failed: {e}")

with colFU2:
    if st.button("Refresh follow-up"):
        st.rerun()

# ---------- Screener ----------
st.divider()
st.header("Top Picks Screener (model-based)")

colA, colB = st.columns([1.4, 1])
with colA:
    uni_name = st.selectbox("Universe", list(UNIVERSES.keys()), index=1, help="Start with Nasdaq 100 or Dow 30 for speed.")
    top_n = st.number_input("How many picks?", min_value=5, max_value=25, value=10, step=1)
    per_order_budget = st.number_input("Per-order Budget (USD)", min_value=50.0, max_value=10000.0, value=100.0, step=10.0)
    use_rsi = st.toggle("Use RSI in scoring", value=True)
    min_conf = st.slider("Min Confidence", min_value=0, max_value=100, value=50, step=1)

    if st.button("Find Top N Now", type="primary"):
        with st.spinner("Screening universe..."):
            tickers = UNIVERSES[uni_name]
            df_top = screen_tickers(
    tickers, int(fast), int(slow), float(per_order_budget),
    use_rsi=use_rsi, top_n=int(top_n), min_confidence=int(min_conf)
)

            st.session_state["screener_df"] = df_top

    df_top = st.session_state.get("screener_df")
    if isinstance(df_top, pd.DataFrame) and not df_top.empty:
        st.success(f"Found {len(df_top)} candidates from **{uni_name}**.")
        st.dataframe(df_top, use_container_width=True)

with colB:
    st.subheader("Place orders for selected (Paper)")
    df_top = st.session_state.get("screener_df")
    if isinstance(df_top, pd.DataFrame) and not df_top.empty:
        picks = st.multiselect("Select tickers to order", df_top["ticker"].tolist(), default=df_top["ticker"].tolist()[:3])
        if st.button("Place selected orders (Paper)"):
            if dry_run:
                st.warning("DRY_RUN is ON → simulated only. Toggle OFF to send real paper orders.", icon="⚠️")
            results = []
            for t in picks:
                row = df_top[df_top["ticker"]==t].iloc[0].to_dict()
                if row["action"] != "BUY":
                    results.append(f"{t}: skipped (action={row['action']})")
                    continue
                order_id, status, qty = execute_order(
                    ticker=row["ticker"], side="BUY",
                    notional=float(row["amount"]), entry_price=float(row["price"]),
                    stop=float(row["stop"]), take=float(row["take"]), dry_run=dry_run
                )
                results.append(f"{t}: id={order_id}, status={status}, qty={qty}")
            st.success(" | ".join(results))

        sel = st.selectbox("Explain pick:", df_top["ticker"].tolist())
        if sel:
            row = df_top[df_top["ticker"]==sel].iloc[0]
            st.markdown(f"**Horizon:** {row.get('horizon','—')}  |  **~Days → Stop/TP:** {row.get('days_to_stop','—')}/{row.get('days_to_take','—')}")
            st.info(row.get("reason","No reason."))
    else:
        st.info("Run the screener first.")

st.divider()
render_purchases_by_day_panel()  # Tables by purchase date + "Exit by (date)"

st.divider()
st.caption("This tool ranks candidates, not guarantees. Paper only – for learning. Logs: logs/trade_log.duckdb")







