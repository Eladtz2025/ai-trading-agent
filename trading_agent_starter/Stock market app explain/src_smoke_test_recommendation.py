# src/smoke_test_recommendation.py
# Quick smoke test for the recommendation and dry-run execution.
# Usage (PowerShell from project root with venv active):
#   python src\smoke_test_recommendation.py

from pathlib import Path
import sys

# Ensure src is on path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from paper_execute_sma import generate_recommendation, execute_order

def main():
    ticker = "AAPL"
    fast, slow = 20, 50
    notional = 100.0

    rec = generate_recommendation(ticker, fast, slow, notional, use_rsi=True)
    required = {"action","amount","reason","stop","take","confidence","price","fast","slow"}
    missing = required - set(rec.keys())
    assert not missing, f"Missing keys in recommendation: {missing}"

    print("Recommendation OK:")
    for k in ["action","amount","stop","take","confidence","price"]:
        print(f"  {k}: {rec.get(k)}")

    if rec["action"] in ("BUY","SELL"):
        order_id, status, qty = execute_order(
            ticker=rec["ticker"],
            side=rec["action"],
            notional=float(rec["amount"]),
            entry_price=float(rec["price"]),
            stop=float(rec["stop"]),
            take=float(rec["take"]),
            dry_run=True,  # DRY RUN on purpose
        )
        assert order_id == "DRY_RUN" and status == "simulated", "Dry-run execute_order failed"
        print(f"Dry-run order OK: qty={qty}")
    else:
        print("Action is WAIT – skipping order test.")

if __name__ == "__main__":
    main()
