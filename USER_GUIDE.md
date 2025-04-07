# User Guide: Quant ML Options Trading System

This user guide provides detailed instructions on how to use the Quant ML Options Trading System for options trading, strategy development, backtesting, and automated execution.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Market Analysis](#market-analysis)
4. [Strategy Development](#strategy-development)
5. [Backtesting](#backtesting)
6. [Portfolio Management](#portfolio-management)
7. [Live Trading](#live-trading)
8. [System Configuration](#system-configuration)
9. [Troubleshooting](#troubleshooting)

## Getting Started

### System Requirements

- **Hardware**: 
  - Minimum: 4GB RAM, dual-core processor
  - Recommended: 8GB+ RAM, quad-core processor
- **Operating System**: 
  - Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)
- **Software**:
  - Docker and Docker Compose (for containerized deployment)
  - Python 3.10+ (for local installation)
- **Network**:
  - Stable internet connection
  - Access to Interactive Brokers (for live trading)

### Installation

#### Docker Installation (Recommended)

1. Ensure Docker and Docker Compose are installed on your system
2. Clone or download the Quant ML Options Trading System
3. Open a terminal and navigate to the system directory
4. Run the deployment script:
   ```bash
   ./deploy.sh
   ```
5. Access the dashboard at `http://localhost:8501`

#### Local Installation

1. Ensure Python 3.10+ is installed on your system
2. Clone or download the Quant ML Options Trading System
3. Open a terminal and navigate to the system directory
4. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   pip install -r quant_ml_trader/requirements.txt
   ```
6. Start the dashboard:
   ```bash
   streamlit run quant_ml_trader/dashboard/app.py
   ```
7. Access the dashboard at `http://localhost:8501`

### Initial Configuration

Before using the system, you need to configure a few settings:

1. Copy the example configuration file:
   ```bash
   cp quant_ml_trader/config/settings.yaml.example quant_ml_trader/config/settings.yaml
   ```

2. Edit the configuration file with your preferred settings:
   - Data sources
   - Model parameters
   - Risk management settings
   - Dashboard preferences

3. If using Interactive Brokers for live trading, configure your IBKR credentials in the settings file

## Dashboard Overview

The dashboard is divided into five main sections:

1. **Market Overview**: Real-time market data and visualization
2. **Strategy Recommendations**: Strategy configuration and signal generation
3. **Portfolio**: Position management and order execution
4. **Backtesting**: Strategy testing and performance analysis
5. **Settings**: System configuration and preferences

### Navigation

- Use the tabs at the top of the dashboard to switch between sections
- The sidebar provides additional controls and filters for each section
- Most charts and tables are interactive - hover, click, or drag to explore data

## Market Analysis

### Market Overview

The Market Overview section provides real-time market data and visualization tools:

1. **Market Summary**: Quick view of key market indicators and selected symbols
2. **Price Charts**: Interactive price charts with technical indicators
3. **Options Chains**: Detailed options data including prices, Greeks, and implied volatility
4. **Volatility Analysis**: IV rank, IV percentile, and volatility term structure

### Using the Market Overview

1. Enter symbols in the sidebar to track specific assets
2. Select timeframes to view different chart periods
3. Use the options chain selector to view options for different expirations
4. Toggle display options to show/hide Greeks, IV, and other metrics

### Technical Analysis

The system provides various technical indicators:

1. Select a symbol and timeframe
2. Choose indicators from the dropdown menu (RSI, MACD, Bollinger Bands, etc.)
3. Adjust indicator parameters as needed
4. Indicators will be displayed on the price chart

## Strategy Development

### Available Strategies

The system includes several pre-built strategies:

1. **Directional Strategy**: Based on price movement predictions
2. **Volatility Strategy**: Based on implied volatility predictions
3. **Multi-Leg Strategy**: Complex option spreads for various market conditions

### Customizing Strategies

To customize a strategy:

1. Navigate to the Strategy Recommendations tab
2. Select a strategy from the dropdown menu
3. Adjust strategy parameters in the form
4. Click "Update Parameters" to save changes

### Creating Signals

To generate trading signals:

1. Configure your strategy parameters
2. Click "Generate Signals" button
3. Review the signals in the table below
4. If connected to a broker, you can execute signals directly

### Signal Analysis

Each signal includes:

- Symbol and direction (BUY/SELL)
- Entry and exit criteria
- Risk/reward metrics
- Confidence score (for ML-based signals)

## Backtesting

### Running a Backtest

To backtest a strategy:

1. Navigate to the Backtesting tab
2. Select a strategy from the dropdown menu
3. Set the backtest period (start and end dates)
4. Set initial capital amount
5. Click "Run Backtest" to start the simulation

### Analyzing Results

Backtest results include:

1. **Performance Metrics**:
   - Total return and annualized return
   - Sharpe ratio and Sortino ratio
   - Maximum drawdown
   - Win rate and profit factor

2. **Equity Curve**: Visual representation of portfolio value over time

3. **Drawdown Chart**: Visualization of drawdowns during the backtest period

4. **Monthly Returns**: Heatmap showing returns by month and year

5. **Trade Statistics**: Detailed information about individual trades

### Stress Testing

To stress test a strategy:

1. Run a normal backtest first
2. Click "Run Stress Tests" to simulate extreme market conditions
3. Review how the strategy performs under different scenarios:
   - Market crash (-20%)
   - Volatility spike (+100%)
   - Liquidity crisis (widened spreads)
   - Correlation shock

## Portfolio Management

### Account Overview

The Portfolio tab provides an overview of your trading account:

1. **Account Summary**: Net liquidation value, cash balance, buying power
2. **Positions**: Current holdings with unrealized P&L
3. **Open Orders**: Pending orders and their status
4. **Trade History**: Record of executed trades

### Managing Positions

To manage your positions:

1. View current positions in the Positions table
2. Click "Close Position" next to a specific position to exit
3. Click "Close All Positions" to exit all positions

### Order Management

To manage orders:

1. View open orders in the Orders table
2. Click "Cancel Order" next to a specific order to cancel it
3. Click "Cancel All Orders" to cancel all pending orders

### Executing Trades

To execute trades manually:

1. Navigate to the Strategy Recommendations tab
2. Generate signals or create a custom order
3. Review the order details
4. Click "Execute" to send the order to your broker

## Live Trading

### Connecting to Interactive Brokers

To enable live trading:

1. Ensure IBKR TWS or IB Gateway is running
2. Configure your IBKR connection settings in the Settings tab
3. Click "Connect to Broker" in the sidebar

### Automated Trading

To enable automated trading:

1. Navigate to the Settings tab
2. Enable "Automated Trading" option
3. Configure trading parameters:
   - Trading hours
   - Maximum positions
   - Risk limits
4. The system will automatically execute signals based on your strategies

### Risk Management

The system includes several risk management features:

1. **Position Sizing**: Automatically sizes positions based on account value and risk tolerance
2. **Stop Losses**: Optional automatic stop losses for all positions
3. **Maximum Loss Limits**: Daily and per-trade loss limits
4. **Exposure Limits**: Maximum exposure to any single asset or sector

## System Configuration

### General Settings

Configure general system settings in the Settings tab:

1. **Dashboard**: Theme, refresh interval, default symbols
2. **Data**: Data sources, update frequency, storage options
3. **Models**: Model parameters, retraining frequency
4. **Execution**: Order types, slippage assumptions, commission rates

### Risk Management Settings

Configure risk management settings:

1. **Position Sizing**: Maximum position size as percentage of account
2. **Stop Loss**: Enable/disable automatic stop losses and set percentage
3. **Take Profit**: Enable/disable automatic take profit orders and set percentage
4. **Maximum Loss**: Set maximum daily loss and per-trade loss limits

### Saving and Loading Configurations

To save your configuration:

1. Navigate to the Settings tab
2. Make your desired changes
3. Click "Save Settings" to store your configuration

To load a saved configuration:

1. Navigate to the Settings tab
2. Click "Load Configuration"
3. Select the configuration file to load

## Troubleshooting

### Common Issues

#### Dashboard Not Loading

1. Check if the Docker container is running:
   ```bash
   docker ps
   ```
2. Check the logs for errors:
   ```bash
   docker logs quant-ml-trader
   ```
3. Ensure port 8501 is not being used by another application

#### Connection to IBKR Failed

1. Ensure TWS or IB Gateway is running
2. Check that your API settings in TWS/Gateway allow connections
3. Verify your connection settings in the system configuration
4. Check the logs for specific error messages

#### Data Not Updating

1. Check your internet connection
2. Verify data source credentials in your configuration
3. Try manually refreshing the data using the "Refresh Data" button
4. Check the logs for API rate limit errors

### Getting Help

If you encounter issues not covered in this guide:

1. Check the logs for error messages
2. Consult the README.md file for additional information
3. Search for similar issues in the project repository
4. Contact support with detailed information about your issue

## Appendix

### Keyboard Shortcuts

- `R`: Refresh data
- `S`: Generate signals
- `B`: Run backtest
- `C`: Connect/disconnect broker
- `Esc`: Cancel current operation

### Glossary

- **IV**: Implied Volatility
- **IV Rank**: Current IV relative to 52-week range
- **IV Percentile**: Percentage of days with lower IV in past year
- **Greeks**: Delta, Gamma, Theta, Vega, Rho
- **Sharpe Ratio**: Risk-adjusted return measure
- **Drawdown**: Decline from peak to trough in portfolio value
