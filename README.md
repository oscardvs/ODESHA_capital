# ODESHA_capital

A personal experiment from April 2025 in machine-learning-assisted options trading, written in Python. It contains the parts such a system needs: market and options data from Interactive Brokers and Yahoo Finance, feature engineering (Greeks, volatility metrics, technical indicators, event proximity, optional sentiment), two ML models (a price direction classifier and an implied volatility predictor), three option strategy classes, a backtesting engine with stress tests, an order execution layer for IBKR, and a Streamlit dashboard.

## Status

Every module contains a full implementation rather than a stub, and each has a small `__main__` demo that runs on random sample data. The whole codebase was committed in one go in April 2025 and has not changed since. There are no tests. Things to know before running it:

- `dashboard/app.py` imports `DataStore` from `data/store.py`, but that module defines `DataStorage`, so the dashboard does not start as committed. Renaming one of the two fixes the import.
- `execution/ibkr_connection.py` uses the official `ibapi` package, which is not in `requirements.txt`. When it is missing, the code falls back to an `IBKRSimulator` that returns random market data and simulates order fills. `data/fetch_ibkr.py` uses `ib_insync` instead.
- The modules load `../config/settings.yaml` relative to the working directory, so run them from inside their own folder.
- `Dockerfile`, `docker-compose.yml`, `deploy.sh`, `USER_GUIDE.md` and `API_DOCS.md` refer to a `quant_ml_trader/` directory and a `settings.yaml.example` file. Neither exists: the modules sit at the repository root and the config file is `config/settings.yaml`.
- The code also imports PyYAML, seaborn and tqdm, which are not in either requirements file.

## Layout

```
data/         fetching from IBKR (ib_insync) and Yahoo Finance, preprocessing, SQLite or Parquet storage
features/     Greeks (py_vollib, with a Black-Scholes fallback), volatility, technicals, event proximity, sentiment
models/       direction classifier (XGBoost, CatBoost or scikit-learn), IV predictor (LightGBM), model registry
strategies/   base class, directional, volatility and multi-leg strategies
backtests/    backtest engine and stress tester
execution/    IBKR connection (with simulator fallback) and order executor
dashboard/    Streamlit app with Market Overview, Strategy Recommendations, Portfolio, Backtesting and Settings tabs
config/       settings.yaml
```

## Install

```bash
git clone https://github.com/oscardvs/ODESHA_capital.git
cd ODESHA_capital
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists every library the code can use, including the optional ones (PyTorch, transformers, QuantLib, TA-Lib, MLflow). `requirements_core.txt` is a shorter list with pandas, NumPy, scikit-learn and the plotting libraries. The dashboard needs Streamlit and Plotly, which are only in the full file.

## Usage

Edit `config/settings.yaml` (IBKR host and port, data sources, strategy filters, backtest dates, paper trading flag). Then, after fixing the import noted above:

```bash
cd dashboard
streamlit run app.py
```

The backtester can also be driven from Python, as in the `__main__` block of `backtests/backtest_engine.py`:

```python
from backtests.backtest_engine import BacktestEngine
from strategies.directional_strategy import DirectionalOptionsStrategy

backtest = BacktestEngine()
backtest.set_strategy(DirectionalOptionsStrategy())
backtest.load_market_data("../data/market_data.csv")
backtest.load_options_data("../data/options_data.csv")
backtest.generate_features()
backtest.generate_signals()
backtest.run_backtest()
backtest.plot_equity_curve("equity_curve.png")
report = backtest.generate_report("backtest_report.md")
```

`backtests/stress_tester.py` wraps a `BacktestEngine` and reruns it under price, volatility, correlation, liquidity, slippage and commission shocks.

## License

This repository does not include a license file.

## Disclaimer

This software is for educational and informational purposes only. It is not financial advice. Trading options involves significant risk and may not be suitable for all investors. You should carefully consider your investment objectives, level of experience, and risk appetite before using this system for live trading.
