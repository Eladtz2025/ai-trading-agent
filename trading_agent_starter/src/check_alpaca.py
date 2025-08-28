import os
from utils import load_env

load_env()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not API_SECRET:
    raise SystemExit("Missing Alpaca keys. Create config/.env from .env.template and add keys.")

from alpaca_trade_api.rest import REST
api = REST(API_KEY, API_SECRET, base_url=BASE_URL)

account = api.get_account()
print("Status:", account.status)
print("Buying Power:", account.buying_power)

try:
    clock = api.get_clock()
    print("Market is open?", clock.is_open)
except Exception as e:
    print("Clock endpoint not available or permission issue:", e)
