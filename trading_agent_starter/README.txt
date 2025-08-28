# AI Trading Agent – Starter Project

This starter lets you verify your setup and run a **simple moving-average backtest** (no API keys needed).
Then you can connect to paper trading (Alpaca).

## 1) Activate your virtual environment
Windows:
  venv\Scripts\activate
Mac/Linux:
  source venv/bin/activate

## 2) Install dependencies (already done in your case)
  pip install numpy pandas matplotlib vectorbt backtrader requests alpaca-trade-api tiingo yfinance openai python-dotenv

## 3) First run (no keys needed)
  python src/backtest_sma.py

You should see summary stats and a PNG chart in the `outputs/` folder.

## 4) Configure APIs (optional for later)
Create a file at `config/.env` using `config/.env.template` as reference and add your keys.
- Alpaca (paper): https://app.alpaca.markets/
- Tiingo: https://api.tiingo.com/
- Polygon: https://polygon.io/

## 5) Test Alpaca Paper connection (optional)
  python src/check_alpaca.py

If successful, you'll see your paper account status and buying power.

## Files
- src/backtest_sma.py  – Simple SMA cross strategy with yfinance.
- src/check_alpaca.py  – Tests connection to Alpaca paper.
- src/utils.py         – Helpers (env loading).
- config/.env.template – Template for your API keys.
- outputs/             – Charts and logs.
