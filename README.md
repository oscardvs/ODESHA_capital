# Quant ML Options Trading System

A comprehensive machine learning-based options trading system for quantitative analysis, strategy development, backtesting, and automated execution.

## Overview

The Quant ML Options Trading System is a complete end-to-end solution for options trading that leverages machine learning to generate trading signals, evaluate strategies, and execute trades. The system includes data ingestion from multiple sources, feature engineering, ML model training and prediction, strategy development, backtesting, and live trading execution through Interactive Brokers.

## Features

- **Data Layer**: Ingest market data from IBKR, Yahoo Finance, and alternative sources
- **Feature Engineering**: Calculate Greeks, volatility metrics, technical indicators, and sentiment analysis
- **Machine Learning**: Train and deploy models for price direction and implied volatility prediction
- **Strategy Engine**: Implement directional, volatility-based, and multi-leg option strategies
- **Backtesting Framework**: Test strategies against historical data with realistic simulations
- **Execution Layer**: Connect to Interactive Brokers for automated trade execution
- **Dashboard UI**: Monitor markets, analyze strategies, and manage trades through an intuitive interface

## System Architecture

The system is built with a modular architecture that allows for flexibility and extensibility:

```
quant_ml_trader/
├── data/               # Data ingestion and storage
├── features/           # Feature engineering components
├── models/             # Machine learning models
├── strategies/         # Trading strategies
├── backtests/          # Backtesting framework
├── execution/          # Order execution and broker connection
├── dashboard/          # User interface
└── config/             # Configuration files
```

## Technology Stack

- **Python 3.10+**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computations
- **Scikit-learn, XGBoost, LightGBM**: Machine learning frameworks
- **QuantLib**: Quantitative finance library for options pricing and Greeks
- **Streamlit**: Interactive dashboard interface
- **Docker**: Containerization for deployment
- **PostgreSQL** (optional): Database for data persistence

## Installation

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- Interactive Brokers account (for live trading)

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quant-ml-trader.git
   cd quant-ml-trader
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r quant_ml_trader/requirements.txt
   ```

4. Configure settings:
   ```bash
   cp quant_ml_trader/config/settings.yaml.example quant_ml_trader/config/settings.yaml
   # Edit settings.yaml with your configuration
   ```

### Docker Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quant-ml-trader.git
   cd quant-ml-trader
   ```

2. Run the deployment script:
   ```bash
   ./deploy.sh
   ```

   For more options:
   ```bash
   ./deploy.sh --help
   ```

## Usage

### Starting the Dashboard

#### Local Environment

```bash
cd quant-ml-trader
source venv/bin/activate  # On Windows: venv\Scripts\activate
streamlit run quant_ml_trader/dashboard/app.py
```

#### Docker Environment

The dashboard will be automatically started when using the deployment script. Access it at:

```
http://localhost:8501
```

### Dashboard Sections

1. **Market Overview**: View real-time market data, options chains, and price charts
2. **Strategy Recommendations**: Configure strategies and generate trading signals
3. **Portfolio**: Monitor positions, orders, and trade history
4. **Backtesting**: Test strategies against historical data and analyze performance
5. **Settings**: Configure system parameters and risk management settings

## Development Guide

### Adding a New Strategy

1. Create a new strategy class in the `strategies` directory, inheriting from `StrategyBase`
2. Implement the required methods: `generate_signals()`, `get_parameters()`, etc.
3. Register the strategy in the dashboard by adding it to the `strategies` dictionary

### Adding a New Data Source

1. Create a new data fetcher in the `data` directory
2. Implement methods to fetch and process the data
3. Update the `DataStore` class to integrate the new data source

### Adding a New ML Model

1. Create a new model class in the `models` directory
2. Implement training, prediction, and evaluation methods
3. Register the model with the `ModelManager` class

## Backtesting

The system includes a comprehensive backtesting framework that allows you to:

- Test strategies against historical data
- Simulate realistic market conditions
- Apply various stress tests (market crashes, volatility spikes, etc.)
- Generate detailed performance reports

Example:

```python
from backtests.backtest_engine import BacktestEngine
from strategies.directional_strategy import DirectionalOptionsStrategy

# Create backtest engine
backtest = BacktestEngine()

# Set strategy
strategy = DirectionalOptionsStrategy()
backtest.set_strategy(strategy)

# Load data
backtest.load_market_data("data/market_data.csv")
backtest.load_options_data("data/options_data.csv")

# Run backtest
backtest.run_backtest()

# Get results
results = backtest.get_results()
```

## Live Trading

To enable live trading:

1. Configure your IBKR account details in `config/settings.yaml`
2. Connect to IBKR TWS or IB Gateway
3. Use the Dashboard's Portfolio section to monitor and execute trades

## Configuration

The system is configured through YAML files in the `config` directory:

- `settings.yaml`: Main configuration file
- Environment-specific configurations can be created as needed

Example configuration:

```yaml
data:
  sources:
    - ibkr
    - yfinance
  cache_dir: ./data/cache

models:
  direction_classifier:
    type: xgboost
    hyperparameters:
      max_depth: 6
      learning_rate: 0.1
  iv_predictor:
    type: lightgbm
    hyperparameters:
      num_leaves: 31
      learning_rate: 0.05

execution:
  max_position_size: 100
  max_loss_per_trade: 1000.0
  max_daily_loss: 5000.0
  use_stop_loss: true
  stop_loss_pct: 0.05

dashboard:
  default_symbols:
    - SPY
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
  refresh_interval: 60
  theme: dark
```

## Deployment

The system can be deployed in various environments using Docker:

- **Local**: Development and testing
- **Development**: Staging environment
- **Production**: Live trading environment

Use the deployment script with appropriate options:

```bash
./deploy.sh --env prod --use-db
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and informational purposes only. It is not financial advice. Trading options involves significant risk and may not be suitable for all investors. You should carefully consider your investment objectives, level of experience, and risk appetite before using this system for live trading.
