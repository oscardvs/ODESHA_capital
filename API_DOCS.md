# API Documentation: Quant ML Options Trading System

This document provides technical documentation for the APIs and components of the Quant ML Options Trading System.

## Table of Contents

1. [Data Layer](#data-layer)
2. [Feature Engineering](#feature-engineering)
3. [Machine Learning Models](#machine-learning-models)
4. [Strategy Engine](#strategy-engine)
5. [Backtesting Framework](#backtesting-framework)
6. [Execution Layer](#execution-layer)
7. [Dashboard Components](#dashboard-components)

## Data Layer

### DataStore

The `DataStore` class provides a unified interface for accessing market and options data from various sources.

```python
from data.store import DataStore

# Initialize data store
data_store = DataStore()

# Get market data
market_data = data_store.get_market_data(symbol='AAPL', timeframe='1d')

# Get options chain
options_chain = data_store.get_options_chain(symbol='AAPL')

# Get options data for specific expiration
options_data = data_store.get_options_for_expiration(symbol='AAPL', expiration='20230721')
```

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `get_market_data` | `symbol` (str), `timeframe` (str), `start_date` (str, optional), `end_date` (str, optional) | `pd.DataFrame` | Retrieves historical market data for a symbol |
| `get_options_chain` | `symbol` (str) | `Dict` | Retrieves options chain for a symbol |
| `get_options_for_expiration` | `symbol` (str), `expiration` (str) | `pd.DataFrame` | Retrieves options data for a specific expiration |
| `save_market_data` | `symbol` (str), `data` (pd.DataFrame) | `bool` | Saves market data to storage |
| `save_options_data` | `symbol` (str), `data` (Dict) | `bool` | Saves options data to storage |

### IBKR Data Fetcher

The `IBKRDataFetcher` class fetches data from Interactive Brokers.

```python
from data.fetch_ibkr import IBKRDataFetcher

# Initialize fetcher
ibkr_fetcher = IBKRDataFetcher()

# Connect to IBKR
ibkr_fetcher.connect()

# Fetch historical data
historical_data = ibkr_fetcher.fetch_historical_data(symbol='AAPL', timeframe='1d')

# Fetch options chain
options_chain = ibkr_fetcher.fetch_options_chain(symbol='AAPL')
```

### YFinance Data Fetcher

The `YFinanceDataFetcher` class fetches data from Yahoo Finance.

```python
from data.fetch_yfinance import YFinanceDataFetcher

# Initialize fetcher
yf_fetcher = YFinanceDataFetcher()

# Fetch historical data
historical_data = yf_fetcher.fetch_historical_data(symbol='AAPL', timeframe='1d')

# Fetch options chain
options_chain = yf_fetcher.fetch_options_chain(symbol='AAPL')
```

## Feature Engineering

### GreeksCalculator

The `GreeksCalculator` class calculates option Greeks using various methods.

```python
from features.greeks import GreeksCalculator

# Initialize calculator
calculator = GreeksCalculator()

# Calculate implied volatility
iv = calculator.calculate_implied_volatility(
    option_price=5.0,
    underlying_price=100.0,
    strike=100.0,
    time_to_expiry=0.25,
    risk_free_rate=0.03,
    option_type='call'
)

# Calculate delta
delta = calculator.calculate_delta(
    underlying_price=100.0,
    strike=100.0,
    time_to_expiry=0.25,
    risk_free_rate=0.03,
    implied_volatility=0.3,
    option_type='call'
)
```

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `calculate_implied_volatility` | `option_price` (float), `underlying_price` (float), `strike` (float), `time_to_expiry` (float), `risk_free_rate` (float), `option_type` (str) | `float` | Calculates implied volatility |
| `calculate_delta` | `underlying_price` (float), `strike` (float), `time_to_expiry` (float), `risk_free_rate` (float), `implied_volatility` (float), `option_type` (str) | `float` | Calculates delta |
| `calculate_gamma` | `underlying_price` (float), `strike` (float), `time_to_expiry` (float), `risk_free_rate` (float), `implied_volatility` (float), `option_type` (str) | `float` | Calculates gamma |
| `calculate_theta` | `underlying_price` (float), `strike` (float), `time_to_expiry` (float), `risk_free_rate` (float), `implied_volatility` (float), `option_type` (str) | `float` | Calculates theta |
| `calculate_vega` | `underlying_price` (float), `strike` (float), `time_to_expiry` (float), `risk_free_rate` (float), `implied_volatility` (float), `option_type` (str) | `float` | Calculates vega |
| `calculate_all_greeks` | `underlying_price` (float), `strike` (float), `time_to_expiry` (float), `risk_free_rate` (float), `implied_volatility` (float), `option_type` (str) | `Dict[str, float]` | Calculates all Greeks |

### VolatilityMetrics

The `VolatilityMetrics` class calculates various volatility-related metrics.

```python
from features.volatility_metrics import VolatilityMetrics

# Initialize calculator
vol_metrics = VolatilityMetrics()

# Calculate IV Rank
iv_rank = vol_metrics.calculate_iv_rank(
    current_iv=0.3,
    historical_iv=historical_iv_series
)

# Calculate IV Percentile
iv_percentile = vol_metrics.calculate_iv_percentile(
    current_iv=0.3,
    historical_iv=historical_iv_series
)
```

### TechnicalIndicators

The `TechnicalIndicators` class calculates various technical indicators.

```python
from features.technicals import TechnicalIndicators

# Initialize calculator
tech_indicators = TechnicalIndicators()

# Calculate RSI
rsi = tech_indicators.calculate_rsi(
    prices=price_series,
    period=14
)

# Calculate MACD
macd, signal, histogram = tech_indicators.calculate_macd(
    prices=price_series,
    fast_period=12,
    slow_period=26,
    signal_period=9
)
```

## Machine Learning Models

### DirectionClassifier

The `DirectionClassifier` class predicts price movement direction.

```python
from models.direction_classifier import DirectionClassifier

# Initialize model
model = DirectionClassifier()

# Train model
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
metrics = model.evaluate(X_test, y_test)
```

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `train` | `X` (pd.DataFrame), `y` (pd.Series) | `None` | Trains the model |
| `predict` | `X` (pd.DataFrame) | `np.ndarray` | Makes predictions |
| `predict_proba` | `X` (pd.DataFrame) | `np.ndarray` | Returns probability estimates |
| `evaluate` | `X` (pd.DataFrame), `y` (pd.Series) | `Dict[str, float]` | Evaluates model performance |
| `save_model` | `path` (str) | `bool` | Saves model to disk |
| `load_model` | `path` (str) | `bool` | Loads model from disk |

### IVPredictor

The `IVPredictor` class predicts implied volatility changes.

```python
from models.iv_predictor import IVPredictor

# Initialize model
model = IVPredictor()

# Train model
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
metrics = model.evaluate(X_test, y_test)
```

### ModelManager

The `ModelManager` class manages model lifecycle and versioning.

```python
from models.model_manager import ModelManager

# Initialize manager
manager = ModelManager()

# Register model
manager.register_model('direction_classifier', model)

# Get model
model = manager.get_model('direction_classifier')

# List models
models = manager.list_models()
```

## Strategy Engine

### StrategyBase

The `StrategyBase` class is the base class for all strategies.

```python
from strategies.strategy_base import StrategyBase

class MyStrategy(StrategyBase):
    def __init__(self):
        super().__init__()
        self.name = "My Custom Strategy"
        self.parameters = {
            'param1': 10,
            'param2': 0.5
        }
    
    def generate_signals(self, data):
        # Implementation
        return signals
```

#### Methods to Implement

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `generate_signals` | `data` (Dict) | `List[Dict]` | Generates trading signals |
| `get_parameters` | None | `Dict` | Returns strategy parameters |
| `set_parameters` | `parameters` (Dict) | `None` | Sets strategy parameters |

### DirectionalStrategy

The `DirectionalOptionsStrategy` class implements directional trading strategies.

```python
from strategies.directional_strategy import DirectionalOptionsStrategy

# Initialize strategy
strategy = DirectionalOptionsStrategy()

# Set parameters
strategy.set_parameters({
    'confidence_threshold': 0.7,
    'use_stop_loss': True
})

# Generate signals
signals = strategy.generate_signals(data)
```

### VolatilityStrategy

The `VolatilityStrategy` class implements volatility-based trading strategies.

```python
from strategies.volatility_strategy import VolatilityStrategy

# Initialize strategy
strategy = VolatilityStrategy()

# Generate signals
signals = strategy.generate_signals(data)
```

### MultiLegStrategy

The `MultiLegStrategy` class implements multi-leg option strategies.

```python
from strategies.multi_leg_strategy import MultiLegStrategy

# Initialize strategy
strategy = MultiLegStrategy()

# Generate signals
signals = strategy.generate_signals(data)
```

## Backtesting Framework

### BacktestEngine

The `BacktestEngine` class provides functionality for backtesting strategies.

```python
from backtests.backtest_engine import BacktestEngine

# Initialize engine
backtest = BacktestEngine()

# Set strategy
backtest.set_strategy(strategy)

# Load data
backtest.load_market_data("data/market_data.csv")
backtest.load_options_data("data/options_data.csv")

# Run backtest
backtest.run_backtest()

# Get results
results = backtest.get_results()
```

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `set_strategy` | `strategy` (StrategyBase) | `None` | Sets the strategy to backtest |
| `set_initial_capital` | `capital` (float) | `None` | Sets initial capital |
| `load_market_data` | `file_path` (str) or `symbol` (str), `data` (pd.DataFrame) | `None` | Loads market data |
| `load_options_data` | `file_path` (str) or `symbol` (str), `data` (Dict) | `None` | Loads options data |
| `generate_features` | None | `None` | Generates features for backtesting |
| `generate_signals` | None | `None` | Generates signals using the strategy |
| `run_backtest` | None | `None` | Runs the backtest |
| `get_results` | None | `Dict` | Returns backtest results |
| `plot_equity_curve` | None | `plt.Figure` | Plots equity curve |
| `plot_drawdowns` | None | `plt.Figure` | Plots drawdowns |
| `generate_report` | `file_path` (str, optional) | `str` | Generates backtest report |

### StressTester

The `StressTester` class provides functionality for stress testing strategies.

```python
from backtests.stress_tester import StressTester

# Initialize tester with backtest engine
stress_tester = StressTester(backtest)

# Apply price shock
stress_tester.apply_price_shock(-0.20)  # -20% price shock

# Run stress test
stress_tester.run_stress_test("Market Crash")

# Run standard stress tests
stress_tester.run_standard_stress_tests()

# Generate report
report = stress_tester.generate_stress_test_report()
```

## Execution Layer

### IBKRConnection

The `IBKRConnection` class provides connectivity to Interactive Brokers.

```python
from execution.ibkr_connection import create_ibkr_connection

# Create connection
ibkr = create_ibkr_connection()

# Connect to IBKR
ibkr.connect()

# Get market data
market_data = ibkr.get_market_data('AAPL')

# Place order
contract = ibkr.create_stock_contract('AAPL')
order = ibkr.create_market_order('BUY', 100)
order_id = ibkr.place_order(contract, order)
```

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `connect` | None | `bool` | Connects to IBKR |
| `disconnect` | None | `None` | Disconnects from IBKR |
| `check_connection` | None | `bool` | Checks if connected |
| `get_account_summary` | None | `Dict` | Gets account summary |
| `get_positions` | None | `Dict` | Gets current positions |
| `get_market_data` | `symbol` (str), `sec_type` (str, optional), `exchange` (str, optional), `currency` (str, optional) | `Dict` | Gets market data |
| `get_options_chain` | `symbol` (str), `exchange` (str, optional), `currency` (str, optional) | `Dict` | Gets options chain |
| `place_order` | `contract` (Contract), `order` (Order) | `int` | Places an order |
| `cancel_order` | `order_id` (int) | `bool` | Cancels an order |
| `get_order_status` | `order_id` (int) | `Dict` | Gets order status |
| `get_executions` | None | `Dict` | Gets executions |

### OrderExecutor

The `OrderExecutor` class executes trading signals.

```python
from execution.order_executor import OrderExecutor

# Initialize executor
executor = OrderExecutor()

# Connect to broker
executor.connect()

# Execute signals
result = executor.execute_signals(signals)

# Get positions
positions = executor.get_positions()

# Cancel all orders
executor.cancel_all_orders()
```

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `connect` | None | `bool` | Connects to broker |
| `disconnect` | None | `None` | Disconnects from broker |
| `check_connection` | None | `bool` | Checks if connected |
| `update_account_info` | None | `None` | Updates account information |
| `execute_signals` | `signals` (List[Dict]) | `Dict` | Executes trading signals |
| `cancel_all_orders` | None | `Dict` | Cancels all open orders |
| `get_order_status` | `order_id` (int) | `Dict` | Gets order status |
| `get_all_orders` | None | `Dict` | Gets all orders |
| `get_positions` | None | `Dict` | Gets current positions |
| `get_account_summary` | None | `Dict` | Gets account summary |
| `calculate_daily_pnl` | None | `float` | Calculates daily P&L |
| `reset_daily_pnl` | None | `None` | Resets daily P&L |

## Dashboard Components

### Dashboard

The `Dashboard` class is the main dashboard application.

```python
from dashboard.app import Dashboard

# Initialize dashboard
dashboard = Dashboard()

# Render dashboard
dashboard.render_dashboard()
```

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `initialize_components` | None | `None` | Initializes system components |
| `connect_broker` | None | `None` | Connects to broker |
| `disconnect_broker` | None | `None` | Disconnects from broker |
| `refresh_data` | None | `None` | Refreshes market data |
| `run_backtest` | `strategy_name` (str), `start_date` (str), `end_date` (str), `initial_capital` (float, optional) | `Dict` | Runs backtest |
| `execute_signals` | `signals` (List[Dict], optional) | `Dict` | Executes trading signals |
| `render_dashboard` | None | `None` | Renders the dashboard |

### UI Components

The dashboard includes several reusable UI components:

```python
from dashboard.components import MarketOverviewComponents, StrategyComponents, PortfolioComponents, BacktestComponents

# Render market summary
MarketOverviewComponents.render_market_summary(symbols, market_data)

# Render price chart
MarketOverviewComponents.render_price_chart(symbol, historical_data, signals, show_signals)

# Render strategy parameters
StrategyComponents.render_strategy_parameters(strategy)

# Render performance metrics
StrategyComponents.render_performance_metrics(metrics)

# Render account summary
PortfolioComponents.render_account_summary(account_summary)

# Render backtest form
BacktestComponents.render_backtest_form(strategies, default_strategy)
```
