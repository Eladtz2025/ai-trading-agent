
from pathlib import Path
import datetime as dt
import duckdb
import json

BASE = Path(__file__).resolve().parents[1]
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(exist_ok=True)

DB_PATH = LOG_DIR / "trade_log.duckdb"
ERR_PATH = LOG_DIR / "errors.log"

def _conn():
    return duckdb.connect(str(DB_PATH))

def log_error(where:str, err:Exception, extra:dict=None):
    msg = f"[{dt.datetime.utcnow().isoformat()}Z] {where}: {repr(err)}"
    if extra: msg += " | extra=" + json.dumps(extra, ensure_ascii=False, default=str)
    ERR_PATH.open("a", encoding="utf-8").write(msg + "\n")

def log_trade(ts, ticker:str, action:str, price:float, reason:str="", meta:dict=None):
    con = _conn()
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            ts TIMESTAMP,
            ticker VARCHAR,
            action VARCHAR,
            price DOUBLE,
            reason VARCHAR,
            meta JSON
        )
    """)
    con.execute(
        "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?)",
        [ts, ticker, action, float(price), reason, json.dumps(meta or {}, ensure_ascii=False, default=str)]
    )
    con.close()

def log_signal(ts, ticker:str, signal:float, meta:dict=None):
    con = _conn()
    con.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            ts TIMESTAMP,
            ticker VARCHAR,
            signal DOUBLE,
            meta JSON
        )
    """)
    con.execute(
        "INSERT INTO signals VALUES (?, ?, ?, ?)",
        [ts, ticker, float(signal), json.dumps(meta or {}, ensure_ascii=False, default=str)]
    )
    con.close()
