# sys_check.py
# Professional checklist validator for your AI Trading Agent setup.
# Run: python sys_check.py

import os, sys
from pathlib import Path

REPORT = []

def section(title):
    REPORT.append("\n" + "="*len(title) + f"\n{title}\n" + "="*len(title))

def add(line):
    REPORT.append(line)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CFG = ROOT / "config"
LOGS = ROOT / "logs"
OUT = ROOT / "outputs"

section("Paths")
add(f"Project root: {ROOT}")
add(f"src exists: {SRC.exists()}")
add(f"config exists: {CFG.exists()}")
add(f"logs exists: {LOGS.exists()}")
add(f"outputs exists: {OUT.exists()}")

section("Python & packages")
add(f"Python: {sys.version.split()[0]}")

# package name -> import name
pkg_imports = {
    "numpy": "numpy",
    "pandas": "pandas",
    "yfinance": "yfinance",
    "duckdb": "duckdb",
    "streamlit": "streamlit",
    "python-dotenv": "dotenv",   # ← הייבוא הנכון
    "alpaca-py": "alpaca",       # ← הייבוא הנכון
    "matplotlib": "matplotlib",
}
for pkg, imod in pkg_imports.items():
    try:
        __import__(imod)
        add(f"[OK] {pkg}")
    except Exception as e:
        add(f"[MISS] {pkg} -> {e}")



section("Core files present")
files = [
    SRC/"backtest_sma_risk_v2.py",
    SRC/"inspect_logs.py",
    SRC/"paper_execute_sma.py",
    ROOT/"streamlit_dashboard_pro.py",
    CFG/".env",
]
for f in files:
    add(f"[{'OK' if f.exists() else 'MISS'}] {f.relative_to(ROOT)}")

section("DuckDB tables")
try:
    import duckdb
    db = LOGS/"trade_log.duckdb"
    if not db.exists():
        add("[MISS] logs/trade_log.duckdb (run a backtest or paper_execute once)")
    else:
        con = duckdb.connect(str(db))
        tables = [r[0] for r in con.execute("PRAGMA show_tables").fetchall()]
        add(f"Tables: {tables}")
        for t in ("signals","trades"):
            if t in tables:
                cnt = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                add(f"[OK] {t} count={cnt}")
            else:
                add(f"[MISS] table {t}")
        con.close()
except Exception as e:
    add(f"[ERR] DuckDB check failed: {e}")

section("Alpaca keys (.env)")
try:
    from dotenv import load_dotenv
    load_dotenv(CFG/".env")
    key = os.getenv("ALPACA_API_KEY","")
    sec = os.getenv("ALPACA_API_SECRET","")
    add(f"ALPACA_API_KEY length: {len(key)} (last4: {key[-4:] if key else ''})")
    add(f"ALPACA_API_SECRET length: {len(sec)} (last4: {sec[-4:] if sec else ''})")
except Exception as e:
    add(f"[ERR] dotenv: {e}")

section("Alpaca connection (optional)")
add("Tip: run `python src/check_alpaca_py.py` for full check.")

print("\n".join(REPORT))
