import os
from pathlib import Path
from dotenv import load_dotenv

# load keys from config/.env
env_file = Path(__file__).resolve().parents[1] / "config" / ".env"
if env_file.exists():
    load_dotenv(env_file)

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

if not API_KEY or not API_SECRET:
    raise SystemExit("Missing Alpaca keys. Edit config/.env with ALPACA_API_KEY/SECRET.")

# alpaca‑py (new SDK)
from alpaca.trading.client import TradingClient

# use paper trading endpoint
trading = TradingClient(API_KEY, API_SECRET, paper=True)

# simple checks
account = trading.get_account()
print("Account status:", account.status)
print("Buying power:", account.buying_power)

# optional: list open positions (safe call)
positions = trading.get_all_positions()
print("Open positions:", len(positions))

print("Alpaca trading client OK.")
