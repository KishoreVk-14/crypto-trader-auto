# Crypto Paper Trader

A crypto strategy backtesting and forward-testing tool. **Paper trading only**: this repository does not configure real-money order execution.

## Local setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python paper_trader.py --help
```

Commands:

- `python paper_trader.py backtest` — fetch market history and run a historical backtest.
- `python paper_trader.py run` — execute one paper-trading scan.
- `python paper_trader.py loop` — continuously scan after candle closes.
- `python paper_trader.py report` — rebuild the HTML dashboard from the local journal.

## GitHub Actions

The **Paper trader validation** workflow runs on pushes to `main`, pull requests, and manual dispatch. Automatic runs perform dependency installation, Python compilation, and a CLI smoke check.

To run a historical backtest, open **Actions → Paper trader validation → Run workflow** and enable **Run the network-dependent historical backtest**. The generated logs, JSON results, and HTML report are uploaded as a workflow artifact for 14 days. Backtests require access to the exchange's public market-data API and can fail if the provider is unavailable or rate-limits requests.

## Safety and limitations

- CI does not deploy to AWS, place real orders, or automatically edit/merge strategy code.
- Do not commit exchange keys, Telegram tokens, `.env` files, databases, or private logs.
- Backtest output is diagnostic, not evidence of future profitability. We should audit the accounting and exit assumptions and add regression tests before using performance metrics to tune strategies.
- AWS deployment automation should be a separate phase with explicit approval, paper-trading-only configuration, health checks, and rollback.

## Current validation scope

The initial workflow checks syntax and CLI startup. It does **not** yet provide comprehensive unit tests or prove the correctness of backtest accounting.
