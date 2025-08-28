# streamlit_dashboard.py
import os, sys, subprocess
from pathlib import Path
import duckdb, pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent  # שמור קובץ זה בשורש הפרויקט
load_dotenv(PROJECT_ROOT / "config" / ".env")

st.set_page_config(page_title="AI Trading Agent Dashboard", layout="wide")
st.title("AI Trading Agent — Dashboard")
st.caption("צפייה ב-signals/trades מ-DuckDB + הרצת paper_execute_sma.py בכפתור.")

db_path = PROJECT_ROOT / "logs" / "trade_log.duckdb"
src_script = PROJECT_ROOT / "src" / "paper_execute_sma.py"

if not db_path.exists():
    st.warning(f"DuckDB לא נמצא: {db_path}. הרץ backtest או paper_execute_sma.py לפחות פעם אחת.")
else:
    con = duckdb.connect(str(db_path))
    tabs = st.tabs(["Signals", "Trades", "Run (Paper)"])

    with tabs[0]:
        n = st.slider("Rows", 10, 300, 50, 10)
        st.dataframe(con.execute(f"SELECT * FROM signals ORDER BY ts DESC LIMIT {n}").fetchdf(), use_container_width=True)

    with tabs[1]:
        n2 = st.slider("Rows ", 10, 300, 50, 10, key="rows2")
        st.dataframe(con.execute(f"SELECT * FROM trades ORDER BY ts DESC LIMIT {n2}").fetchdf(), use_container_width=True)

    with tabs[2]:
        st.subheader("Trigger paper_execute_sma.py")
        c1, c2, c3 = st.columns(3)
        with c1:
            ticker = st.text_input("Ticker", value=os.getenv("BACKTEST_TICKER","AAPL"))
        with c2:
            notional = st.number_input("Order Notional ($)", min_value=10.0, max_value=10000.0, value=100.0, step=10.0)
        with c3:
            dry = st.toggle("Dry run (no order)", value=True)
        fast = st.number_input("SMA Fast", 2, 200, int(os.getenv("SMA_FAST","20")))
        slow = st.number_input("SMA Slow", 2, 500, int(os.getenv("SMA_SLOW","50")))

        if st.button("Run now"):
            if not src_script.exists():
                st.error(f"לא נמצא סקריפט: {src_script}")
            else:
                env = os.environ.copy()
                env["BACKTEST_TICKER"] = ticker
                env["ORDER_NOTIONAL"]  = str(notional)
                env["DRY_RUN"]         = "1" if dry else "0"
                env["SMA_FAST"]        = str(fast)
                env["SMA_SLOW"]        = str(slow)
                try:
                    out = subprocess.run(
                        [sys.executable, str(src_script)],
                        cwd=str(PROJECT_ROOT), env=env,
                        capture_output=True, text=True, timeout=120
                    )
                    st.code(out.stdout or "(no stdout)")
                    if out.stderr:
                        st.error(out.stderr)
                except Exception as e:
                    st.exception(e)

    con.close()
