"""
Dashboard Application for Quant ML Options Trading System

This module provides a Streamlit-based dashboard for monitoring and controlling
the trading system.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.store import DataStore
from features.greeks import GreeksCalculator
from features.volatility_metrics import VolatilityMetrics
from models.model_manager import ModelManager
from strategies.strategy_base import StrategyBase
from strategies.directional_strategy import DirectionalOptionsStrategy
from strategies.volatility_strategy import VolatilityStrategy
from strategies.multi_leg_strategy import MultiLegStrategy
from backtests.backtest_engine import BacktestEngine
from execution.order_executor import OrderExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Dashboard:
    """
    Main dashboard application for the Quant ML Options Trading System.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the dashboard.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        self.dashboard_config = self.config.get('dashboard', {})
        
        # Initialize components
        self.data_store = None
        self.greeks_calculator = None
        self.volatility_metrics = None
        self.model_manager = None
        self.strategies = {}
        self.backtest_engine = None
        self.order_executor = None
        
        # Dashboard state
        self.selected_symbols = self.dashboard_config.get('default_symbols', ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN'])
        self.selected_strategy = None
        self.selected_timeframe = '1d'
        self.selected_expiration = None
        self.show_greeks = True
        self.show_iv = True
        self.show_signals = True
        self.auto_refresh = False
        self.refresh_interval = 60  # seconds
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.connected = False
            st.session_state.market_data = {}
            st.session_state.options_data = {}
            st.session_state.signals = []
            st.session_state.positions = {}
            st.session_state.orders = {}
            st.session_state.account_summary = {}
            st.session_state.backtest_results = {}
            st.session_state.last_refresh = datetime.now()
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return default configuration
            return {
                'dashboard': {
                    'default_symbols': ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN'],
                    'refresh_interval': 60,
                    'theme': 'dark'
                }
            }
    
    def initialize_components(self):
        """
        Initialize system components.
        """
        try:
            # Initialize data store
            self.data_store = DataStore()
            
            # Initialize feature calculators
            self.greeks_calculator = GreeksCalculator()
            self.volatility_metrics = VolatilityMetrics()
            
            # Initialize model manager
            self.model_manager = ModelManager()
            
            # Initialize strategies
            self.strategies = {
                'directional': DirectionalOptionsStrategy(),
                'volatility': VolatilityStrategy(),
                'multi_leg': MultiLegStrategy()
            }
            self.selected_strategy = 'directional'
            
            # Initialize backtest engine
            self.backtest_engine = BacktestEngine()
            
            # Initialize order executor
            self.order_executor = OrderExecutor()
            
            # Mark as initialized
            st.session_state.initialized = True
            
            logger.info("Dashboard components initialized")
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            st.error(f"Error initializing components: {e}")
    
    def connect_broker(self):
        """
        Connect to broker.
        """
        if not st.session_state.initialized:
            st.warning("System not initialized")
            return
        
        try:
            # Connect to broker
            if self.order_executor.connect():
                st.session_state.connected = True
                
                # Get account summary
                account_info = self.order_executor.get_account_summary()
                if account_info['success']:
                    st.session_state.account_summary = account_info['account_summary']
                
                # Get positions
                positions = self.order_executor.get_positions()
                if positions['success']:
                    st.session_state.positions = positions['positions']
                
                st.success("Connected to broker")
                logger.info("Connected to broker")
            else:
                st.error("Failed to connect to broker")
                logger.error("Failed to connect to broker")
        except Exception as e:
            st.error(f"Error connecting to broker: {e}")
            logger.error(f"Error connecting to broker: {e}")
    
    def disconnect_broker(self):
        """
        Disconnect from broker.
        """
        if not st.session_state.initialized or not st.session_state.connected:
            st.warning("Not connected to broker")
            return
        
        try:
            # Disconnect from broker
            self.order_executor.disconnect()
            st.session_state.connected = False
            st.success("Disconnected from broker")
            logger.info("Disconnected from broker")
        except Exception as e:
            st.error(f"Error disconnecting from broker: {e}")
            logger.error(f"Error disconnecting from broker: {e}")
    
    def refresh_data(self):
        """
        Refresh market data.
        """
        if not st.session_state.initialized:
            st.warning("System not initialized")
            return
        
        try:
            # Update last refresh time
            st.session_state.last_refresh = datetime.now()
            
            # Get market data for selected symbols
            for symbol in self.selected_symbols:
                # Get market data
                if st.session_state.connected:
                    # Get from broker
                    market_data = self.order_executor.ibkr.get_market_data(symbol)
                    if market_data:
                        st.session_state.market_data[symbol] = market_data
                else:
                    # Get from data store
                    market_data = self.data_store.get_market_data(symbol, self.selected_timeframe)
                    if not market_data.empty:
                        # Convert to dict for last bar
                        last_bar = market_data.iloc[-1].to_dict()
                        st.session_state.market_data[symbol] = {
                            'last': last_bar['close'],
                            'open': last_bar['open'],
                            'high': last_bar['high'],
                            'low': last_bar['low'],
                            'volume': last_bar['volume'],
                            'date': last_bar['date']
                        }
                
                # Get options chain
                if st.session_state.connected:
                    # Get from broker
                    options_chain = self.order_executor.ibkr.get_options_chain(symbol)
                    if options_chain:
                        st.session_state.options_data[symbol] = options_chain
                else:
                    # Get from data store
                    options_chain = self.data_store.get_options_chain(symbol)
                    if options_chain:
                        st.session_state.options_data[symbol] = options_chain
            
            # Generate signals if strategy selected
            if self.selected_strategy and self.selected_strategy in self.strategies:
                strategy = self.strategies[self.selected_strategy]
                
                # Prepare data for strategy
                data = {
                    'market_data': st.session_state.market_data,
                    'options_data': st.session_state.options_data
                }
                
                # Generate signals
                signals = strategy.generate_signals(data)
                st.session_state.signals = signals
            
            # Update account info if connected
            if st.session_state.connected:
                # Get account summary
                account_info = self.order_executor.get_account_summary()
                if account_info['success']:
                    st.session_state.account_summary = account_info['account_summary']
                
                # Get positions
                positions = self.order_executor.get_positions()
                if positions['success']:
                    st.session_state.positions = positions['positions']
                
                # Get orders
                orders = self.order_executor.get_all_orders()
                if orders['success']:
                    st.session_state.orders = orders['orders']
            
            logger.info("Data refreshed")
        except Exception as e:
            st.error(f"Error refreshing data: {e}")
            logger.error(f"Error refreshing data: {e}")
    
    def run_backtest(self, strategy_name: str, start_date: str, end_date: str, initial_capital: float = 100000.0):
        """
        Run backtest for a strategy.
        
        Args:
            strategy_name: Name of the strategy to backtest
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Initial capital for backtest
        """
        if not st.session_state.initialized:
            st.warning("System not initialized")
            return
        
        try:
            # Check if strategy exists
            if strategy_name not in self.strategies:
                st.error(f"Strategy {strategy_name} not found")
                return
            
            # Get strategy
            strategy = self.strategies[strategy_name]
            
            # Set strategy for backtest engine
            self.backtest_engine.set_strategy(strategy)
            
            # Set initial capital
            self.backtest_engine.set_initial_capital(initial_capital)
            
            # Load market data
            for symbol in self.selected_symbols:
                market_data = self.data_store.get_market_data(
                    symbol, '1d', start_date=start_date, end_date=end_date
                )
                if not market_data.empty:
                    self.backtest_engine.add_market_data(symbol, market_data)
                
                # Load options data
                options_data = self.data_store.get_options_data(
                    symbol, start_date=start_date, end_date=end_date
                )
                if options_data:
                    self.backtest_engine.add_options_data(symbol, options_data)
            
            # Run backtest
            self.backtest_engine.run_backtest()
            
            # Get results
            results = self.backtest_engine.get_results()
            
            # Store results
            st.session_state.backtest_results[strategy_name] = results
            
            st.success(f"Backtest completed for {strategy_name}")
            logger.info(f"Backtest completed for {strategy_name}")
            
            return results
        except Exception as e:
            st.error(f"Error running backtest: {e}")
            logger.error(f"Error running backtest: {e}")
            return None
    
    def execute_signals(self, signals: List[Dict] = None):
        """
        Execute trading signals.
        
        Args:
            signals: List of signals to execute (if None, use current signals)
        """
        if not st.session_state.initialized or not st.session_state.connected:
            st.warning("Not connected to broker")
            return
        
        try:
            # Use current signals if none provided
            if signals is None:
                signals = st.session_state.signals
            
            if not signals:
                st.warning("No signals to execute")
                return
            
            # Execute signals
            result = self.order_executor.execute_signals(signals)
            
            if result['success']:
                st.success(f"Executed {len(result['results'])} signals")
                logger.info(f"Executed {len(result['results'])} signals")
                
                # Refresh data
                self.refresh_data()
                
                return result
            else:
                st.error(f"Error executing signals: {result.get('error', 'Unknown error')}")
                logger.error(f"Error executing signals: {result.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            st.error(f"Error executing signals: {e}")
            logger.error(f"Error executing signals: {e}")
            return None
    
    def cancel_all_orders(self):
        """
        Cancel all open orders.
        """
        if not st.session_state.initialized or not st.session_state.connected:
            st.warning("Not connected to broker")
            return
        
        try:
            # Cancel all orders
            result = self.order_executor.cancel_all_orders()
            
            if result['success']:
                st.success(f"Cancelled {len(result['results'])} orders")
                logger.info(f"Cancelled {len(result['results'])} orders")
                
                # Refresh data
                self.refresh_data()
                
                return result
            else:
                st.error(f"Error cancelling orders: {result.get('error', 'Unknown error')}")
                logger.error(f"Error cancelling orders: {result.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            st.error(f"Error cancelling orders: {e}")
            logger.error(f"Error cancelling orders: {e}")
            return None
    
    def render_dashboard(self):
        """
        Render the dashboard.
        """
        # Set page config
        st.set_page_config(
            page_title="Quant ML Options Trading System",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        st.title("Quant ML Options Trading System")
        
        # Initialize if not already
        if not st.session_state.initialized:
            self.initialize_components()
        
        # Auto-refresh
        if self.auto_refresh and (datetime.now() - st.session_state.last_refresh).total_seconds() > self.refresh_interval:
            self.refresh_data()
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Market Overview", "Strategy Recommendations", "Portfolio", "Backtesting", "Settings"
        ])
        
        # Market Overview tab
        with tab1:
            self._render_market_overview()
        
        # Strategy Recommendations tab
        with tab2:
            self._render_strategy_recommendations()
        
        # Portfolio tab
        with tab3:
            self._render_portfolio()
        
        # Backtesting tab
        with tab4:
            self._render_backtesting()
        
        # Settings tab
        with tab5:
            self._render_settings()
    
    def _render_sidebar(self):
        """
        Render the sidebar.
        """
        st.sidebar.title("Controls")
        
        # Connection status
        if st.session_state.connected:
            st.sidebar.success("Connected to Broker")
            if st.sidebar.button("Disconnect"):
                self.disconnect_broker()
        else:
            st.sidebar.warning("Not Connected to Broker")
            if st.sidebar.button("Connect"):
                self.connect_broker()
        
        # Symbol selection
        st.sidebar.subheader("Symbols")
        symbols_input = st.sidebar.text_input(
            "Enter symbols (comma-separated)",
            ",".join(self.selected_symbols)
        )
        self.selected_symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
        
        # Strategy selection
        st.sidebar.subheader("Strategy")
        self.selected_strategy = st.sidebar.selectbox(
            "Select strategy",
            list(self.strategies.keys()) if hasattr(self, 'strategies') and self.strategies else ['directional', 'volatility', 'multi_leg']
        )
        
        # Timeframe selection
        st.sidebar.subheader("Timeframe")
        self.selected_timeframe = st.sidebar.selectbox(
            "Select timeframe",
            ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        )
        
        # Display options
        st.sidebar.subheader("Display Options")
        self.show_greeks = st.sidebar.checkbox("Show Greeks", value=self.show_greeks)
        self.show_iv = st.sidebar.checkbox("Show IV", value=self.show_iv)
        self.show_signals = st.sidebar.checkbox("Show Signals", value=self.show_signals)
        
        # Auto-refresh
        st.sidebar.subheader("Auto-Refresh")
        self.auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=self.auto_refresh)
        if self.auto_refresh:
            self.refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=10,
                max_value=300,
                value=self.refresh_interval,
                step=10
            )
        
        # Manual refresh
        if st.sidebar.button("Refresh Data"):
            self.refresh_data()
        
        # Last refresh time
        st.sidebar.text(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    def _render_market_overview(self):
        """
        Render the market overview tab.
        """
        st.header("Market Overview")
        
        # Market summary
        st.subheader("Market Summary")
        
        # Create columns for market data
        cols = st.columns(len(self.selected_symbols))
        
        for i, symbol in enumerate(self.selected_symbols):
            with cols[i]:
                if symbol in st.session_state.market_data:
                    market_data = st.session_state.market_data[symbol]
                    
                    # Display price and change
                    last_price = market_data.get('last', 0.0)
                    open_price = market_data.get('open', last_price)
                    change = last_price - open_price
                    change_pct = (change / open_price) * 100 if open_price > 0 else 0.0
                    
                    # Color based on change
                    color = "green" if change >= 0 else "red"
                    
                    st.metric(
                        label=symbol,
                        value=f"${last_price:.2f}",
                        delta=f"{change:.2f} ({change_pct:.2f}%)",
                        delta_color="normal" if change >= 0 else "inverse"
                    )
                    
                    # Display additional info
                    st.text(f"High: ${market_data.get('high', 0.0):.2f}")
                    st.text(f"Low: ${market_data.get('low', 0.0):.2f}")
                    st.text(f"Volume: {market_data.get('volume', 0):,}")
                    
                    # Display IV if available and enabled
                    if self.show_iv and 'implied_volatility' in market_data:
                        iv = market_data['implied_volatility']
                        st.text(f"IV: {iv:.2%}")
                else:
                    st.metric(label=symbol, value="N/A")
                    st.text("No data available")
        
        # Options chains
        st.subheader("Options Chains")
        
        # Symbol selection for options
        selected_symbol = st.selectbox("Select Symbol for Options Chain", self.selected_symbols)
        
        if selected_symbol in st.session_state.options_data:
            options_data = st.session_state.options_data[selected_symbol]
            
            # Expiration selection
            if 'expirations' in options_data and options_data['expirations']:
                expirations = options_data['expirations']
                self.selected_expiration = st.selectbox(
                    "Select Expiration",
                    expirations,
                    index=0 if self.selected_expiration not in expirations else expirations.index(self.selected_expiration)
                )
                
                # Get options for selected expiration
                if st.session_state.connected:
                    # Get from broker
                    options = self._get_options_for_expiration(selected_symbol, self.selected_expiration)
                else:
                    # Get from data store
                    options = self.data_store.get_options_for_expiration(selected_symbol, self.selected_expiration)
                
                if options:
                    # Display options table
                    self._display_options_table(options)
                else:
                    st.warning(f"No options data available for {selected_symbol} on {self.selected_expiration}")
            else:
                st.warning(f"No expirations available for {selected_symbol}")
        else:
            st.warning(f"No options data available for {selected_symbol}")
        
        # Market charts
        st.subheader("Market Charts")
        
        # Symbol selection for chart
        chart_symbol = st.selectbox("Select Symbol for Chart", self.selected_symbols)
        
        # Get historical data
        historical_data = self.data_store.get_market_data(chart_symbol, self.selected_timeframe)
        
        if not historical_data.empty:
            # Create price chart
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=historical_data['date'],
                open=historical_data['open'],
                high=historical_data['high'],
                low=historical_data['low'],
                close=historical_data['close'],
                name=chart_symbol
            ))
            
            # Add volume as bar chart
            fig.add_trace(go.Bar(
                x=historical_data['date'],
                y=historical_data['volume'],
                name='Volume',
                marker_color='rgba(0, 0, 255, 0.3)',
                opacity=0.3,
                yaxis='y2'
            ))
            
            # Add signals if available and enabled
            if self.show_signals and st.session_state.signals:
                # Filter signals for selected symbol
                symbol_signals = [s for s in st.session_state.signals if s['symbol'] == chart_symbol]
                
                for signal in symbol_signals:
                    # Get signal date
                    signal_date = signal.get('date', datetime.now())
                    
                    # Determine color based on direction
                    color = 'green' if signal['direction'] == 'BUY' else 'red'
                    
                    # Add signal marker
                    fig.add_trace(go.Scatter(
                        x=[signal_date],
                        y=[historical_data.loc[historical_data['date'] == signal_date, 'high'].iloc[0] if not historical_data.loc[historical_data['date'] == signal_date].empty else 0],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down' if signal['direction'] == 'SELL' else 'triangle-up',
                            size=15,
                            color=color
                        ),
                        name=f"{signal['direction']} Signal"
                    ))
            
            # Update layout
            fig.update_layout(
                title=f"{chart_symbol} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right",
                    showgrid=False
                )
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No historical data available for {chart_symbol}")
    
    def _render_strategy_recommendations(self):
        """
        Render the strategy recommendations tab.
        """
        st.header("Strategy Recommendations")
        
        # Strategy selection
        st.subheader("Strategy Settings")
        
        # Display strategy parameters
        if self.selected_strategy and self.selected_strategy in self.strategies:
            strategy = self.strategies[self.selected_strategy]
            
            # Get strategy parameters
            params = strategy.get_parameters()
            
            # Create form for parameters
            with st.form(key=f"strategy_params_{self.selected_strategy}"):
                new_params = {}
                
                for param_name, param_value in params.items():
                    if isinstance(param_value, bool):
                        new_params[param_name] = st.checkbox(param_name, value=param_value)
                    elif isinstance(param_value, int):
                        new_params[param_name] = st.number_input(param_name, value=param_value, step=1)
                    elif isinstance(param_value, float):
                        new_params[param_name] = st.number_input(param_name, value=param_value, step=0.01)
                    elif isinstance(param_value, str):
                        new_params[param_name] = st.text_input(param_name, value=param_value)
                    elif isinstance(param_value, list):
                        if all(isinstance(x, str) for x in param_value):
                            new_params[param_name] = st.multiselect(param_name, options=param_value, default=param_value)
                        else:
                            new_params[param_name] = st.text_input(param_name, value=str(param_value))
                
                # Submit button
                if st.form_submit_button("Update Parameters"):
                    strategy.set_parameters(new_params)
                    st.success(f"Parameters updated for {self.selected_strategy} strategy")
        
        # Generate signals
        st.subheader("Generate Signals")
        
        if st.button("Generate Signals"):
            if self.selected_strategy and self.selected_strategy in self.strategies:
                strategy = self.strategies[self.selected_strategy]
                
                # Prepare data for strategy
                data = {
                    'market_data': st.session_state.market_data,
                    'options_data': st.session_state.options_data
                }
                
                # Generate signals
                signals = strategy.generate_signals(data)
                st.session_state.signals = signals
                
                st.success(f"Generated {len(signals)} signals")
            else:
                st.warning("No strategy selected")
        
        # Display signals
        st.subheader("Current Signals")
        
        if st.session_state.signals:
            # Create DataFrame from signals
            signals_df = pd.DataFrame(st.session_state.signals)
            
            # Display signals table
            st.dataframe(signals_df)
            
            # Execute signals button
            if st.session_state.connected and st.button("Execute Signals"):
                self.execute_signals()
        else:
            st.info("No signals available. Generate signals first.")
        
        # Strategy performance
        st.subheader("Strategy Performance")
        
        if self.selected_strategy in st.session_state.backtest_results:
            results = st.session_state.backtest_results[self.selected_strategy]
            
            # Display performance metrics
            metrics = results.get('metrics', {})
            
            if metrics:
                # Create columns for metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{metrics.get('total_return', 0.0):.2%}")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0.0):.2%}")
                
                with col4:
                    st.metric("Win Rate", f"{metrics.get('win_rate', 0.0):.2%}")
                
                # Display equity curve
                if 'equity_curve' in results:
                    equity_curve = results['equity_curve']
                    
                    fig = px.line(
                        equity_curve,
                        x='date',
                        y='equity',
                        title=f"{self.selected_strategy.capitalize()} Strategy Equity Curve"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance metrics available. Run backtest first.")
        else:
            st.info("No backtest results available for selected strategy. Run backtest first.")
    
    def _render_portfolio(self):
        """
        Render the portfolio tab.
        """
        st.header("Portfolio")
        
        # Account summary
        st.subheader("Account Summary")
        
        if st.session_state.account_summary:
            # Create columns for account metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Net Liquidation Value",
                    f"${st.session_state.account_summary.get('NetLiquidation', 0.0):,.2f}"
                )
            
            with col2:
                st.metric(
                    "Cash Balance",
                    f"${st.session_state.account_summary.get('TotalCashValue', 0.0):,.2f}"
                )
            
            with col3:
                st.metric(
                    "Available Funds",
                    f"${st.session_state.account_summary.get('AvailableFunds', 0.0):,.2f}"
                )
            
            with col4:
                st.metric(
                    "Buying Power",
                    f"${st.session_state.account_summary.get('BuyingPower', 0.0):,.2f}"
                )
        else:
            st.info("No account data available. Connect to broker first.")
        
        # Positions
        st.subheader("Current Positions")
        
        if st.session_state.positions:
            # Create DataFrame from positions
            positions_data = []
            
            for symbol, position in st.session_state.positions.items():
                # Get market data for current price
                current_price = 0.0
                if symbol in st.session_state.market_data:
                    current_price = st.session_state.market_data[symbol].get('last', 0.0)
                
                # Calculate P&L
                position_size = position.get('position', 0.0)
                avg_cost = position.get('avg_cost', 0.0)
                market_value = position_size * current_price
                cost_basis = position_size * avg_cost
                unrealized_pnl = market_value - cost_basis
                unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis != 0 else 0.0
                
                positions_data.append({
                    'Symbol': symbol,
                    'Position': position_size,
                    'Avg Cost': avg_cost,
                    'Current Price': current_price,
                    'Market Value': market_value,
                    'Unrealized P&L': unrealized_pnl,
                    'Unrealized P&L %': unrealized_pnl_pct
                })
            
            positions_df = pd.DataFrame(positions_data)
            
            # Format DataFrame
            positions_df['Avg Cost'] = positions_df['Avg Cost'].map('${:,.2f}'.format)
            positions_df['Current Price'] = positions_df['Current Price'].map('${:,.2f}'.format)
            positions_df['Market Value'] = positions_df['Market Value'].map('${:,.2f}'.format)
            positions_df['Unrealized P&L'] = positions_df['Unrealized P&L'].map('${:,.2f}'.format)
            positions_df['Unrealized P&L %'] = positions_df['Unrealized P&L %'].map('{:,.2f}%'.format)
            
            # Display positions table
            st.dataframe(positions_df)
            
            # Close positions button
            if st.session_state.connected and st.button("Close All Positions"):
                # Create close signals
                close_signals = []
                
                for symbol, position in st.session_state.positions.items():
                    position_size = position.get('position', 0.0)
                    
                    if position_size > 0:
                        # Long position, sell to close
                        close_signals.append({
                            'symbol': symbol,
                            'direction': 'SELL',
                            'quantity': abs(position_size)
                        })
                    elif position_size < 0:
                        # Short position, buy to close
                        close_signals.append({
                            'symbol': symbol,
                            'direction': 'BUY',
                            'quantity': abs(position_size)
                        })
                
                if close_signals:
                    self.execute_signals(close_signals)
                else:
                    st.info("No positions to close")
        else:
            st.info("No positions available")
        
        # Orders
        st.subheader("Open Orders")
        
        if st.session_state.orders:
            # Filter open orders
            open_orders = {
                order_id: order_info for order_id, order_info in st.session_state.orders.items()
                if order_info['status'] in ['Submitted', 'PreSubmitted', 'PendingSubmit']
            }
            
            if open_orders:
                # Create DataFrame from orders
                orders_data = []
                
                for order_id, order_info in open_orders.items():
                    orders_data.append({
                        'Order ID': order_id,
                        'Symbol': order_info.get('symbol', 'N/A'),
                        'Action': order_info.get('action', 'N/A'),
                        'Quantity': order_info.get('quantity', 0),
                        'Order Type': order_info.get('order_type', 'N/A'),
                        'Price': order_info.get('market_price', order_info.get('limit_price', 0.0)),
                        'Status': order_info.get('status', 'N/A'),
                        'Time': order_info.get('timestamp', datetime.now()).strftime('%H:%M:%S')
                    })
                
                orders_df = pd.DataFrame(orders_data)
                
                # Format DataFrame
                orders_df['Price'] = orders_df['Price'].map('${:,.2f}'.format)
                
                # Display orders table
                st.dataframe(orders_df)
                
                # Cancel orders button
                if st.session_state.connected and st.button("Cancel All Orders"):
                    self.cancel_all_orders()
            else:
                st.info("No open orders")
        else:
            st.info("No orders available")
        
        # Trade history
        st.subheader("Trade History")
        
        if st.session_state.connected:
            # Get executions
            executions = self.order_executor.ibkr.get_executions()
            
            if executions:
                # Create DataFrame from executions
                executions_data = []
                
                for exec_id, exec_info in executions.items():
                    executions_data.append({
                        'Execution ID': exec_id,
                        'Order ID': exec_info.get('order_id', 'N/A'),
                        'Symbol': exec_info.get('symbol', 'N/A'),
                        'Side': exec_info.get('side', 'N/A'),
                        'Quantity': exec_info.get('shares', 0),
                        'Price': exec_info.get('price', 0.0),
                        'Time': exec_info.get('time', 'N/A'),
                        'Commission': exec_info.get('commission', 0.0)
                    })
                
                executions_df = pd.DataFrame(executions_data)
                
                # Format DataFrame
                executions_df['Price'] = executions_df['Price'].map('${:,.2f}'.format)
                executions_df['Commission'] = executions_df['Commission'].map('${:,.2f}'.format)
                
                # Display executions table
                st.dataframe(executions_df)
            else:
                st.info("No executions available")
        else:
            st.info("Connect to broker to view trade history")
    
    def _render_backtesting(self):
        """
        Render the backtesting tab.
        """
        st.header("Backtesting")
        
        # Backtest settings
        st.subheader("Backtest Settings")
        
        # Create form for backtest settings
        with st.form(key="backtest_settings"):
            # Strategy selection
            backtest_strategy = st.selectbox(
                "Select Strategy",
                list(self.strategies.keys()) if hasattr(self, 'strategies') and self.strategies else ['directional', 'volatility', 'multi_leg'],
                index=list(self.strategies.keys()).index(self.selected_strategy) if hasattr(self, 'strategies') and self.strategies and self.selected_strategy in self.strategies else 0
            )
            
            # Date range
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365)
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now()
                )
            
            # Initial capital
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=1000.0,
                max_value=10000000.0,
                value=100000.0,
                step=10000.0
            )
            
            # Submit button
            if st.form_submit_button("Run Backtest"):
                # Convert dates to string format
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                # Run backtest
                results = self.run_backtest(backtest_strategy, start_date_str, end_date_str, initial_capital)
                
                if results:
                    st.success(f"Backtest completed for {backtest_strategy} strategy")
        
        # Backtest results
        st.subheader("Backtest Results")
        
        # Strategy selection for results
        result_strategy = st.selectbox(
            "Select Strategy for Results",
            list(st.session_state.backtest_results.keys()) if st.session_state.backtest_results else ['No results available'],
            disabled=not st.session_state.backtest_results
        )
        
        if result_strategy in st.session_state.backtest_results:
            results = st.session_state.backtest_results[result_strategy]
            
            # Display performance metrics
            metrics = results.get('metrics', {})
            
            if metrics:
                # Create columns for metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{metrics.get('total_return', 0.0):.2%}")
                    st.metric("Annualized Return", f"{metrics.get('annualized_return', 0.0):.2%}")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}")
                    st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0.0):.2f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0.0):.2%}")
                    st.metric("Volatility", f"{metrics.get('annualized_volatility', 0.0):.2%}")
                
                with col4:
                    st.metric("Win Rate", f"{metrics.get('win_rate', 0.0):.2%}")
                    st.metric("Profit Factor", f"{metrics.get('profit_factor', 0.0):.2f}")
                
                # Display equity curve
                if 'equity_curve' in results:
                    equity_curve = results['equity_curve']
                    
                    fig = px.line(
                        equity_curve,
                        x='date',
                        y='equity',
                        title=f"{result_strategy.capitalize()} Strategy Equity Curve"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display drawdown chart
                if 'drawdowns' in results:
                    drawdowns = results['drawdowns']
                    
                    fig = px.area(
                        drawdowns,
                        x='date',
                        y='drawdown',
                        title=f"{result_strategy.capitalize()} Strategy Drawdowns"
                    )
                    
                    fig.update_traces(line_color='red')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display monthly returns
                if 'monthly_returns' in results:
                    monthly_returns = results['monthly_returns']
                    
                    # Create heatmap
                    fig = px.imshow(
                        monthly_returns,
                        labels=dict(x="Month", y="Year", color="Return"),
                        x=monthly_returns.columns,
                        y=monthly_returns.index,
                        color_continuous_scale='RdYlGn',
                        title=f"{result_strategy.capitalize()} Strategy Monthly Returns"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display trade statistics
                if 'trades' in results:
                    trades = results['trades']
                    
                    st.subheader("Trade Statistics")
                    
                    # Create DataFrame from trades
                    trades_df = pd.DataFrame(trades)
                    
                    # Display trade metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Trades", len(trades))
                    
                    with col2:
                        st.metric("Avg Trade", f"${trades_df['pnl'].mean():.2f}")
                    
                    with col3:
                        st.metric("Best Trade", f"${trades_df['pnl'].max():.2f}")
                    
                    with col4:
                        st.metric("Worst Trade", f"${trades_df['pnl'].min():.2f}")
                    
                    # Display trades table
                    st.dataframe(trades_df)
            else:
                st.info("No performance metrics available")
        else:
            st.info("No backtest results available. Run a backtest first.")
    
    def _render_settings(self):
        """
        Render the settings tab.
        """
        st.header("Settings")
        
        # System settings
        st.subheader("System Settings")
        
        # Create form for system settings
        with st.form(key="system_settings"):
            # Dashboard settings
            st.write("Dashboard Settings")
            
            # Theme
            theme = st.selectbox(
                "Theme",
                ["light", "dark"],
                index=1 if self.dashboard_config.get('theme') == 'dark' else 0
            )
            
            # Refresh interval
            refresh_interval = st.slider(
                "Default Refresh Interval (seconds)",
                min_value=10,
                max_value=300,
                value=self.dashboard_config.get('refresh_interval', 60),
                step=10
            )
            
            # Risk management settings
            st.write("Risk Management Settings")
            
            # Max position size
            max_position_size = st.number_input(
                "Max Position Size",
                min_value=1,
                max_value=1000,
                value=self.execution_config.get('max_position_size', 100) if hasattr(self, 'execution_config') else 100,
                step=1
            )
            
            # Max loss per trade
            max_loss_per_trade = st.number_input(
                "Max Loss Per Trade ($)",
                min_value=100.0,
                max_value=100000.0,
                value=self.execution_config.get('max_loss_per_trade', 1000.0) if hasattr(self, 'execution_config') else 1000.0,
                step=100.0
            )
            
            # Max daily loss
            max_daily_loss = st.number_input(
                "Max Daily Loss ($)",
                min_value=100.0,
                max_value=100000.0,
                value=self.execution_config.get('max_daily_loss', 5000.0) if hasattr(self, 'execution_config') else 5000.0,
                step=100.0
            )
            
            # Use stop loss
            use_stop_loss = st.checkbox(
                "Use Stop Loss",
                value=self.execution_config.get('use_stop_loss', True) if hasattr(self, 'execution_config') else True
            )
            
            # Stop loss percentage
            stop_loss_pct = st.slider(
                "Stop Loss Percentage",
                min_value=1.0,
                max_value=20.0,
                value=self.execution_config.get('stop_loss_pct', 5.0) * 100 if hasattr(self, 'execution_config') else 5.0,
                step=0.5
            ) / 100.0
            
            # Use take profit
            use_take_profit = st.checkbox(
                "Use Take Profit",
                value=self.execution_config.get('use_take_profit', False) if hasattr(self, 'execution_config') else False
            )
            
            # Take profit percentage
            take_profit_pct = st.slider(
                "Take Profit Percentage",
                min_value=1.0,
                max_value=50.0,
                value=self.execution_config.get('take_profit_pct', 10.0) * 100 if hasattr(self, 'execution_config') else 10.0,
                step=0.5
            ) / 100.0
            
            # Submit button
            if st.form_submit_button("Save Settings"):
                # Update dashboard config
                self.dashboard_config['theme'] = theme
                self.dashboard_config['refresh_interval'] = refresh_interval
                
                # Update execution config if available
                if hasattr(self, 'execution_config'):
                    self.execution_config['max_position_size'] = max_position_size
                    self.execution_config['max_loss_per_trade'] = max_loss_per_trade
                    self.execution_config['max_daily_loss'] = max_daily_loss
                    self.execution_config['use_stop_loss'] = use_stop_loss
                    self.execution_config['stop_loss_pct'] = stop_loss_pct
                    self.execution_config['use_take_profit'] = use_take_profit
                    self.execution_config['take_profit_pct'] = take_profit_pct
                
                # Update order executor if available
                if hasattr(self, 'order_executor'):
                    self.order_executor.max_position_size = max_position_size
                    self.order_executor.max_loss_per_trade = max_loss_per_trade
                    self.order_executor.max_daily_loss = max_daily_loss
                    self.order_executor.use_stop_loss = use_stop_loss
                    self.order_executor.stop_loss_pct = stop_loss_pct
                    self.order_executor.use_take_profit = use_take_profit
                    self.order_executor.take_profit_pct = take_profit_pct
                
                # Save config to file
                self._save_config()
                
                st.success("Settings saved")
        
        # Data management
        st.subheader("Data Management")
        
        # Create columns for data actions
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Cache"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    if key != 'initialized':
                        del st.session_state[key]
                
                st.session_state.market_data = {}
                st.session_state.options_data = {}
                st.session_state.signals = []
                st.session_state.positions = {}
                st.session_state.orders = {}
                st.session_state.account_summary = {}
                st.session_state.backtest_results = {}
                st.session_state.last_refresh = datetime.now()
                
                st.success("Cache cleared")
        
        with col2:
            if st.button("Reset System"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                
                # Reinitialize
                st.session_state.initialized = False
                st.session_state.connected = False
                st.session_state.market_data = {}
                st.session_state.options_data = {}
                st.session_state.signals = []
                st.session_state.positions = {}
                st.session_state.orders = {}
                st.session_state.account_summary = {}
                st.session_state.backtest_results = {}
                st.session_state.last_refresh = datetime.now()
                
                # Reset components
                self.data_store = None
                self.greeks_calculator = None
                self.volatility_metrics = None
                self.model_manager = None
                self.strategies = {}
                self.backtest_engine = None
                self.order_executor = None
                
                st.success("System reset")
                st.experimental_rerun()
    
    def _get_options_for_expiration(self, symbol: str, expiration: str) -> pd.DataFrame:
        """
        Get options data for a specific expiration.
        
        Args:
            symbol: Symbol
            expiration: Expiration date
            
        Returns:
            DataFrame containing options data
        """
        if not st.session_state.connected:
            return pd.DataFrame()
        
        try:
            # Get options chain
            if symbol not in st.session_state.options_data:
                return pd.DataFrame()
            
            options_chain = st.session_state.options_data[symbol]
            
            if 'strikes' not in options_chain:
                return pd.DataFrame()
            
            strikes = options_chain['strikes']
            
            # Get options data for each strike
            options_data = []
            
            for strike in strikes:
                # Get call option
                call_data = self.order_executor.ibkr.get_option_market_data(symbol, expiration, strike, 'C')
                
                if call_data:
                    call_data['symbol'] = symbol
                    call_data['expiration'] = expiration
                    call_data['strike'] = strike
                    call_data['option_type'] = 'CALL'
                    options_data.append(call_data)
                
                # Get put option
                put_data = self.order_executor.ibkr.get_option_market_data(symbol, expiration, strike, 'P')
                
                if put_data:
                    put_data['symbol'] = symbol
                    put_data['expiration'] = expiration
                    put_data['strike'] = strike
                    put_data['option_type'] = 'PUT'
                    options_data.append(put_data)
            
            # Create DataFrame
            if options_data:
                df = pd.DataFrame(options_data)
                
                # Calculate days to expiration
                expiration_date = datetime.strptime(expiration, '%Y%m%d')
                days_to_expiration = (expiration_date - datetime.now()).days
                df['days_to_expiration'] = days_to_expiration
                
                # Calculate mid price
                if 'bid' in df.columns and 'ask' in df.columns:
                    df['mid'] = (df['bid'] + df['ask']) / 2
                
                # Get underlying price
                underlying_price = 0.0
                if symbol in st.session_state.market_data:
                    underlying_price = st.session_state.market_data[symbol].get('last', 0.0)
                
                df['underlying_price'] = underlying_price
                
                # Calculate moneyness
                df['moneyness'] = (underlying_price / df['strike']) - 1
                df.loc[df['option_type'] == 'PUT', 'moneyness'] = -df.loc[df['option_type'] == 'PUT', 'moneyness']
                
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting options for expiration: {e}")
            return pd.DataFrame()
    
    def _display_options_table(self, options_df: pd.DataFrame):
        """
        Display options table.
        
        Args:
            options_df: DataFrame containing options data
        """
        if options_df.empty:
            st.warning("No options data available")
            return
        
        # Split into calls and puts
        calls = options_df[options_df['option_type'] == 'CALL'].copy()
        puts = options_df[options_df['option_type'] == 'PUT'].copy()
        
        # Sort by strike
        calls = calls.sort_values('strike')
        puts = puts.sort_values('strike')
        
        # Select columns to display
        display_columns = ['strike', 'bid', 'ask', 'mid', 'implied_volatility']
        
        if self.show_greeks:
            greek_columns = ['delta', 'gamma', 'theta', 'vega']
            for col in greek_columns:
                if col in options_df.columns:
                    display_columns.append(col)
        
        # Create merged table
        merged = pd.DataFrame()
        
        for col in display_columns:
            if col in calls.columns:
                merged[f'call_{col}'] = calls[col].values
            
            if col == 'strike':
                merged['strike'] = calls['strike'].values
            
            if col in puts.columns:
                merged[f'put_{col}'] = puts[col].values
        
        # Format columns
        for col in merged.columns:
            if 'price' in col or 'bid' in col or 'ask' in col or 'mid' in col:
                merged[col] = merged[col].map('${:,.2f}'.format)
            elif 'implied_volatility' in col:
                merged[col] = merged[col].map('{:,.2%}'.format)
            elif 'delta' in col or 'gamma' in col or 'theta' in col or 'vega' in col:
                merged[col] = merged[col].map('{:,.4f}'.format)
        
        # Display table
        st.dataframe(merged)
    
    def _save_config(self):
        """
        Save configuration to file.
        """
        try:
            # Create config dict
            config = {
                'dashboard': self.dashboard_config
            }
            
            if hasattr(self, 'execution_config'):
                config['execution'] = self.execution_config
            
            # Save to file
            with open('../config/settings.yaml', 'w') as file:
                yaml.dump(config, file)
            
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            st.error(f"Error saving configuration: {e}")


# Run dashboard
if __name__ == "__main__":
    # This will run if the script is executed directly
    dashboard = Dashboard()
    dashboard.render_dashboard()
