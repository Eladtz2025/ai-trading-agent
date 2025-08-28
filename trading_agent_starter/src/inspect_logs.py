import duckdb, pathlib
con = duckdb.connect(str(pathlib.Path("logs/trade_log.duckdb")))
for tbl in ("signals","trades"):
    try:
        print(f"\n=== {tbl} (last 10) ===")
        print(con.execute(f"SELECT * FROM {tbl} ORDER BY ts DESC LIMIT 10").fetchdf())
    except Exception as e:
        print(f"{tbl}: not found ({e})")
con.close()
