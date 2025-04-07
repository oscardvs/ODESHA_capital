"""
Backtesting Framework Base Module

This module provides the core functionality for backtesting options trading strategies
against historical data and performing stress testing.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Base class for backtesting options trading strategies.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the backtest engine with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        self.backtest_config = self.config.get('backtest', {})
        
        # Backtest parameters
        self.initial_capital = self.backtest_config.get('initial_capital', 100000)
        self.commission_per_contract = self.backtest_config.get('commission_per_contract', 0.65)
        self.slippage_pct = self.backtest_config.get('slippage_pct', 0.01)
        self.start_date = self.backtest_config.get('start_date', '2020-01-01')
        self.end_date = self.backtest_config.get('end_date', '2023-12-31')
        self.rebalance_frequency = self.backtest_config.get('rebalance_frequency', 'daily')
        self.max_positions = self.backtest_config.get('max_positions', 10)
        self.max_position_size_pct = self.backtest_config.get('max_position_size_pct', 0.05)
        self.risk_free_rate = self.backtest_config.get('risk_free_rate', 0.02)
        
        # Backtest state
        self.portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'history': [],
            'trades': []
        }
        
        # Performance metrics
        self.metrics = {}
        
        # Data storage
        self.market_data = {}
        self.options_data = {}
        self.features_data = {}
        self.signals_data = {}
        
        # Strategy instance
        self.strategy = None
    
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
                'backtest': {
                    'initial_capital': 100000,
                    'commission_per_contract': 0.65,
                    'slippage_pct': 0.01,
                    'start_date': '2020-01-01',
                    'end_date': '2023-12-31',
                    'rebalance_frequency': 'daily',
                    'max_positions': 10,
                    'max_position_size_pct': 0.05,
                    'risk_free_rate': 0.02
                }
            }
    
    def set_strategy(self, strategy):
        """
        Set the strategy to backtest.
        
        Args:
            strategy: Strategy instance
        """
        self.strategy = strategy
    
    def load_market_data(self, data_path: str, symbols: List[str] = None):
        """
        Load historical market data for backtesting.
        
        Args:
            data_path: Path to market data
            symbols: List of symbols to load (if None, load all available)
        """
        try:
            if os.path.isdir(data_path):
                # Load data from directory of CSV files
                files = os.listdir(data_path)
                for file in files:
                    if file.endswith('.csv'):
                        symbol = file.split('.')[0]
                        if symbols is None or symbol in symbols:
                            file_path = os.path.join(data_path, file)
                            df = pd.read_csv(file_path)
                            
                            # Convert date column to datetime
                            if 'date' in df.columns:
                                df['date'] = pd.to_datetime(df['date'])
                            
                            # Filter by date range
                            if 'date' in df.columns:
                                df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
                            
                            self.market_data[symbol] = df
                            logger.info(f"Loaded market data for {symbol}: {len(df)} rows")
            elif os.path.isfile(data_path) and data_path.endswith('.csv'):
                # Load data from single CSV file with multiple symbols
                df = pd.read_csv(data_path)
                
                # Convert date column to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Filter by date range
                if 'date' in df.columns:
                    df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
                
                # Group by symbol
                if 'symbol' in df.columns:
                    for symbol, group in df.groupby('symbol'):
                        if symbols is None or symbol in symbols:
                            self.market_data[symbol] = group.reset_index(drop=True)
                            logger.info(f"Loaded market data for {symbol}: {len(group)} rows")
            elif os.path.isfile(data_path) and data_path.endswith('.parquet'):
                # Load data from parquet file
                df = pd.read_parquet(data_path)
                
                # Convert date column to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Filter by date range
                if 'date' in df.columns:
                    df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
                
                # Group by symbol
                if 'symbol' in df.columns:
                    for symbol, group in df.groupby('symbol'):
                        if symbols is None or symbol in symbols:
                            self.market_data[symbol] = group.reset_index(drop=True)
                            logger.info(f"Loaded market data for {symbol}: {len(group)} rows")
            else:
                logger.error(f"Unsupported data path: {data_path}")
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
    
    def load_options_data(self, data_path: str, symbols: List[str] = None):
        """
        Load historical options data for backtesting.
        
        Args:
            data_path: Path to options data
            symbols: List of symbols to load (if None, load all available)
        """
        try:
            if os.path.isdir(data_path):
                # Load data from directory of CSV files
                files = os.listdir(data_path)
                for file in files:
                    if file.endswith('.csv'):
                        symbol = file.split('.')[0]
                        if symbols is None or symbol in symbols:
                            file_path = os.path.join(data_path, file)
                            df = pd.read_csv(file_path)
                            
                            # Convert date column to datetime
                            if 'date' in df.columns:
                                df['date'] = pd.to_datetime(df['date'])
                            
                            # Filter by date range
                            if 'date' in df.columns:
                                df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
                            
                            self.options_data[symbol] = df
                            logger.info(f"Loaded options data for {symbol}: {len(df)} rows")
            elif os.path.isfile(data_path) and data_path.endswith('.csv'):
                # Load data from single CSV file with multiple symbols
                df = pd.read_csv(data_path)
                
                # Convert date column to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Filter by date range
                if 'date' in df.columns:
                    df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
                
                # Group by symbol
                if 'symbol' in df.columns:
                    for symbol, group in df.groupby('symbol'):
                        if symbols is None or symbol in symbols:
                            self.options_data[symbol] = group.reset_index(drop=True)
                            logger.info(f"Loaded options data for {symbol}: {len(group)} rows")
            elif os.path.isfile(data_path) and data_path.endswith('.parquet'):
                # Load data from parquet file
                df = pd.read_parquet(data_path)
                
                # Convert date column to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Filter by date range
                if 'date' in df.columns:
                    df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
                
                # Group by symbol and date
                if 'symbol' in df.columns and 'date' in df.columns:
                    self.options_data = {}
                    for symbol, symbol_group in df.groupby('symbol'):
                        if symbols is None or symbol in symbols:
                            # Group by date to create a dictionary of options chains by date
                            date_groups = {}
                            for date, date_group in symbol_group.groupby('date'):
                                date_groups[date] = date_group.reset_index(drop=True)
                            
                            self.options_data[symbol] = date_groups
                            logger.info(f"Loaded options data for {symbol}: {len(date_groups)} dates")
            else:
                logger.error(f"Unsupported data path: {data_path}")
        except Exception as e:
            logger.error(f"Error loading options data: {e}")
    
    def generate_features(self):
        """
        Generate features for backtesting.
        """
        try:
            if not self.market_data:
                logger.error("No market data loaded")
                return
            
            # Process each symbol
            for symbol, market_df in self.market_data.items():
                logger.info(f"Generating features for {symbol}")
                
                # Create a copy of market data
                features_df = market_df.copy()
                
                # Add basic technical indicators
                features_df = self._add_technical_indicators(features_df)
                
                # Add volatility metrics
                features_df = self._add_volatility_metrics(features_df)
                
                # Add options-specific features if options data is available
                if symbol in self.options_data:
                    features_df = self._add_options_features(features_df, symbol)
                
                # Store features
                self.features_data[symbol] = features_df
                
                logger.info(f"Generated features for {symbol}: {len(features_df)} rows")
        except Exception as e:
            logger.error(f"Error generating features: {e}")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Make a copy to avoid modifying the original
            result = df.copy()
            
            # Check if we have the required columns
            if 'close' not in result.columns:
                logger.warning("Missing 'close' column for technical indicators")
                return result
            
            # Add SMA
            result['sma_20'] = result['close'].rolling(window=20).mean()
            result['sma_50'] = result['close'].rolling(window=50).mean()
            result['sma_200'] = result['close'].rolling(window=200).mean()
            
            # Add EMA
            result['ema_12'] = result['close'].ewm(span=12, adjust=False).mean()
            result['ema_26'] = result['close'].ewm(span=26, adjust=False).mean()
            
            # Add MACD
            result['macd'] = result['ema_12'] - result['ema_26']
            result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
            result['macd_hist'] = result['macd'] - result['macd_signal']
            
            # Add RSI
            delta = result['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result['rsi'] = 100 - (100 / (1 + rs))
            
            # Add Bollinger Bands
            result['bb_middle'] = result['close'].rolling(window=20).mean()
            result['bb_std'] = result['close'].rolling(window=20).std()
            result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
            result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']
            
            # Add ATR
            if all(col in result.columns for col in ['high', 'low', 'close']):
                high_low = result['high'] - result['low']
                high_close = (result['high'] - result['close'].shift()).abs()
                low_close = (result['low'] - result['close'].shift()).abs()
                
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                
                result['atr'] = true_range.rolling(window=14).mean()
            
            return result
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _add_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility metrics to the dataframe.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with added volatility metrics
        """
        try:
            # Make a copy to avoid modifying the original
            result = df.copy()
            
            # Check if we have the required columns
            if 'close' not in result.columns:
                logger.warning("Missing 'close' column for volatility metrics")
                return result
            
            # Calculate returns
            result['returns'] = result['close'].pct_change()
            
            # Calculate historical volatility (20-day)
            result['hv_20'] = result['returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Calculate historical volatility (60-day)
            result['hv_60'] = result['returns'].rolling(window=60).std() * np.sqrt(252)
            
            # Calculate historical volatility (120-day)
            result['hv_120'] = result['returns'].rolling(window=120).std() * np.sqrt(252)
            
            # Calculate HV rank (percentile of current HV in 1-year lookback)
            result['hv_rank'] = result['hv_20'].rolling(window=252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
            
            return result
        except Exception as e:
            logger.error(f"Error adding volatility metrics: {e}")
            return df
    
    def _add_options_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add options-specific features to the dataframe.
        
        Args:
            df: DataFrame with market data
            symbol: Symbol to process
            
        Returns:
            DataFrame with added options features
        """
        try:
            # Make a copy to avoid modifying the original
            result = df.copy()
            
            # Check if we have options data
            if symbol not in self.options_data:
                logger.warning(f"No options data for {symbol}")
                return result
            
            options_data = self.options_data[symbol]
            
            # Check if options data is a dictionary of dates
            if isinstance(options_data, dict):
                # Initialize columns for options features
                result['iv_atm'] = np.nan
                result['iv_skew'] = np.nan
                result['put_call_ratio'] = np.nan
                result['iv_term_structure'] = np.nan
                result['iv_rank'] = np.nan
                
                # Process each date
                for date, options_df in options_data.items():
                    if date in result['date'].values:
                        # Find the row index for this date
                        idx = result[result['date'] == date].index[0]
                        
                        # Calculate ATM IV
                        if 'implied_volatility' in options_df.columns and 'strike' in options_df.columns:
                            # Get the underlying price
                            if 'underlying_price' in options_df.columns:
                                underlying_price = options_df['underlying_price'].iloc[0]
                            elif 'close' in result.columns:
                                underlying_price = result.loc[idx, 'close']
                            else:
                                underlying_price = None
                            
                            if underlying_price is not None:
                                # Find ATM options
                                options_df['strike_distance'] = abs(options_df['strike'] - underlying_price)
                                atm_options = options_df.nsmallest(5, 'strike_distance')
                                
                                if not atm_options.empty:
                                    # Calculate ATM IV
                                    result.loc[idx, 'iv_atm'] = atm_options['implied_volatility'].mean()
                                    
                                    # Calculate IV skew
                                    if 'right' in options_df.columns:
                                        calls = options_df[options_df['right'].isin(['C', 'c', 'CALL', 'call'])]
                                        puts = options_df[options_df['right'].isin(['P', 'p', 'PUT', 'put'])]
                                        
                                        if not calls.empty and not puts.empty:
                                            # Calculate put/call ratio
                                            result.loc[idx, 'put_call_ratio'] = len(puts) / len(calls)
                                            
                                            # Calculate IV skew (25-delta put IV / 25-delta call IV)
                                            if 'delta' in options_df.columns:
                                                calls['delta_25_distance'] = abs(calls['delta'] - 0.25)
                                                puts['delta_25_distance'] = abs(puts['delta'] + 0.25)
                                                
                                                call_25 = calls.nsmallest(3, 'delta_25_distance')
                                                put_25 = puts.nsmallest(3, 'delta_25_distance')
                                                
                                                if not call_25.empty and not put_25.empty:
                                                    call_25_iv = call_25['implied_volatility'].mean()
                                                    put_25_iv = put_25['implied_volatility'].mean()
                                                    
                                                    if call_25_iv > 0:
                                                        result.loc[idx, 'iv_skew'] = put_25_iv / call_25_iv
                        
                        # Calculate IV term structure
                        if 'days_to_expiration' in options_df.columns and 'implied_volatility' in options_df.columns:
                            # Group by expiration
                            expirations = options_df.groupby('days_to_expiration')['implied_volatility'].mean()
                            
                            if len(expirations) >= 2:
                                # Sort by days to expiration
                                expirations = expirations.sort_index()
                                
                                # Calculate term structure (front month / back month)
                                front_month = expirations.iloc[0]
                                back_month = expirations.iloc[-1]
                                
                                if back_month > 0:
                                    result.loc[idx, 'iv_term_structure'] = front_month / back_month
                
                # Calculate IV rank
                if 'iv_atm' in result.columns:
                    result['iv_rank'] = result['iv_atm'].rolling(window=252).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                    )
            
            return result
        except Exception as e:
            logger.error(f"Error adding options features: {e}")
            return df
    
    def generate_signals(self):
        """
        Generate trading signals for backtesting.
        """
        try:
            if not self.features_data:
                logger.error("No features data generated")
                return
            
            if self.strategy is None:
                logger.error("No strategy set")
                return
            
            # Process each symbol
            for symbol, features_df in self.features_data.items():
                logger.info(f"Generating signals for {symbol}")
                
                # Generate signals using the strategy
                signals_df = self.strategy.generate_signals(features_df)
                
                # Store signals
                self.signals_data[symbol] = signals_df
                
                logger.info(f"Generated signals for {symbol}: {len(signals_df)} rows")
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
    
    def run_backtest(self):
        """
        Run the backtest simulation.
        """
        try:
            if not self.signals_data:
                logger.error("No signals data generated")
                return
            
            if self.strategy is None:
                logger.error("No strategy set")
                return
            
            # Get unique dates across all symbols
            all_dates = set()
            for symbol, signals_df in self.signals_data.items():
                if 'date' in signals_df.columns:
                    all_dates.update(signals_df['date'].unique())
            
            # Sort dates
            all_dates = sorted(all_dates)
            
            # Initialize portfolio
            self.portfolio = {
                'cash': self.initial_capital,
                'positions': {},
                'history': [],
                'trades': []
            }
            
            # Run simulation for each date
            logger.info(f"Running backtest from {all_dates[0]} to {all_dates[-1]}")
            for date in tqdm(all_dates, desc="Backtesting"):
                # Process each symbol for this date
                self._process_date(date)
                
                # Record portfolio value
                self._record_portfolio_value(date)
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            logger.info("Backtest completed")
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
    
    def _process_date(self, date):
        """
        Process a single date in the backtest.
        
        Args:
            date: Date to process
        """
        try:
            # Update positions with current prices
            self._update_positions(date)
            
            # Check for expired options
            self._check_expired_options(date)
            
            # Process exit signals
            self._process_exit_signals(date)
            
            # Process entry signals
            self._process_entry_signals(date)
        except Exception as e:
            logger.error(f"Error processing date {date}: {e}")
    
    def _update_positions(self, date):
        """
        Update positions with current prices.
        
        Args:
            date: Current date
        """
        try:
            # Process each position
            for symbol, position in list(self.portfolio['positions'].items()):
                # Check if we have data for this symbol
                if position['type'] == 'option':
                    # For options, we need to find the current price in options data
                    underlying = position.get('underlying', symbol.split('_')[0])
                    
                    if underlying in self.options_data:
                        options_data = self.options_data[underlying]
                        
                        # Check if options data is a dictionary of dates
                        if isinstance(options_data, dict) and date in options_data:
                            options_df = options_data[date]
                            
                            # Find the option
                            option_row = None
                            if 'option_symbol' in position and 'option_symbol' in options_df.columns:
                                option_matches = options_df[options_df['option_symbol'] == position['option_symbol']]
                                if not option_matches.empty:
                                    option_row = option_matches.iloc[0]
                            
                            # If we couldn't find by symbol, try to match by strike, expiration, and right
                            if option_row is None and all(k in position for k in ['strike', 'expiration', 'right']):
                                option_matches = options_df[
                                    (options_df['strike'] == position['strike']) &
                                    (options_df['expiration'] == position['expiration']) &
                                    (options_df['right'] == position['right'])
                                ]
                                if not option_matches.empty:
                                    option_row = option_matches.iloc[0]
                            
                            # Update position with current price
                            if option_row is not None and 'option_price' in option_row:
                                old_value = position['current_price'] * position['quantity']
                                position['current_price'] = option_row['option_price']
                                new_value = position['current_price'] * position['quantity']
                                
                                # Record P&L
                                position['unrealized_pnl'] = new_value - position['cost_basis']
                                position['daily_pnl'] = new_value - old_value
                            else:
                                # If we can't find the option, estimate the price using a simple model
                                # This is a very simplified approach and should be replaced with a proper pricing model
                                days_to_expiration = (position['expiration'] - date).days
                                if days_to_expiration > 0:
                                    # Simple time decay estimate
                                    theta_decay = position.get('theta', 0) * position['quantity']
                                    old_value = position['current_price'] * position['quantity']
                                    position['current_price'] *= (1 - 0.01)  # Simple 1% daily decay
                                    new_value = position['current_price'] * position['quantity']
                                    
                                    # Record P&L
                                    position['unrealized_pnl'] = new_value - position['cost_basis']
                                    position['daily_pnl'] = new_value - old_value
                                else:
                                    # Option expired
                                    self._close_position(symbol, date, "Expired")
                else:
                    # For stocks, we can use market data
                    if symbol in self.market_data:
                        market_df = self.market_data[symbol]
                        
                        # Find the row for this date
                        if 'date' in market_df.columns:
                            date_rows = market_df[market_df['date'] == date]
                            
                            if not date_rows.empty and 'close' in date_rows.columns:
                                old_value = position['current_price'] * position['quantity']
                                position['current_price'] = date_rows['close'].iloc[0]
                                new_value = position['current_price'] * position['quantity']
                                
                                # Record P&L
                                position['unrealized_pnl'] = new_value - position['cost_basis']
                                position['daily_pnl'] = new_value - old_value
        except Exception as e:
            logger.error(f"Error updating positions for date {date}: {e}")
    
    def _check_expired_options(self, date):
        """
        Check for expired options and close positions.
        
        Args:
            date: Current date
        """
        try:
            # Process each position
            for symbol, position in list(self.portfolio['positions'].items()):
                # Check if this is an option position
                if position['type'] == 'option' and 'expiration' in position:
                    # Convert expiration to datetime if it's a string
                    if isinstance(position['expiration'], str):
                        position['expiration'] = pd.to_datetime(position['expiration'])
                    
                    # Check if the option has expired
                    if position['expiration'] <= date:
                        # Calculate intrinsic value at expiration
                        intrinsic_value = 0
                        
                        # Get underlying price
                        underlying = position.get('underlying', symbol.split('_')[0])
                        underlying_price = None
                        
                        if underlying in self.market_data:
                            market_df = self.market_data[underlying]
                            date_rows = market_df[market_df['date'] == date]
                            
                            if not date_rows.empty and 'close' in date_rows.columns:
                                underlying_price = date_rows['close'].iloc[0]
                        
                        if underlying_price is not None and 'strike' in position and 'right' in position:
                            # Calculate intrinsic value
                            if position['right'] in ['C', 'c', 'CALL', 'call']:
                                intrinsic_value = max(0, underlying_price - position['strike'])
                            elif position['right'] in ['P', 'p', 'PUT', 'put']:
                                intrinsic_value = max(0, position['strike'] - underlying_price)
                        
                        # Close the position at intrinsic value
                        old_value = position['current_price'] * position['quantity']
                        position['current_price'] = intrinsic_value
                        new_value = position['current_price'] * position['quantity']
                        
                        # Record P&L
                        position['unrealized_pnl'] = new_value - position['cost_basis']
                        position['daily_pnl'] = new_value - old_value
                        
                        # Close the position
                        self._close_position(symbol, date, "Expired")
        except Exception as e:
            logger.error(f"Error checking expired options for date {date}: {e}")
    
    def _process_exit_signals(self, date):
        """
        Process exit signals for existing positions.
        
        Args:
            date: Current date
        """
        try:
            # Process each position
            for symbol, position in list(self.portfolio['positions'].items()):
                # Check stop loss
                if position['unrealized_pnl'] < -position.get('stop_loss', float('inf')):
                    self._close_position(symbol, date, "Stop Loss")
                    continue
                
                # Check take profit
                if position['unrealized_pnl'] > position.get('take_profit', float('inf')):
                    self._close_position(symbol, date, "Take Profit")
                    continue
                
                # Check for exit signals from strategy
                underlying = position.get('underlying', symbol.split('_')[0])
                
                if underlying in self.signals_data:
                    signals_df = self.signals_data[underlying]
                    
                    # Find the row for this date
                    if 'date' in signals_df.columns:
                        date_rows = signals_df[signals_df['date'] == date]
                        
                        if not date_rows.empty:
                            # Check for exit signal
                            signal = date_rows['signal'].iloc[0]
                            
                            # Exit if signal is opposite to position direction
                            if (position['direction'] > 0 and signal < 0) or (position['direction'] < 0 and signal > 0):
                                self._close_position(symbol, date, "Signal Reversal")
        except Exception as e:
            logger.error(f"Error processing exit signals for date {date}: {e}")
    
    def _process_entry_signals(self, date):
        """
        Process entry signals for new positions.
        
        Args:
            date: Current date
        """
        try:
            # Check if we have capacity for new positions
            if len(self.portfolio['positions']) >= self.max_positions:
                return
            
            # Process each symbol
            for symbol, signals_df in self.signals_data.items():
                # Find the row for this date
                if 'date' in signals_df.columns:
                    date_rows = signals_df[signals_df['date'] == date]
                    
                    if not date_rows.empty:
                        # Check for entry signal
                        signal_row = date_rows.iloc[0]
                        signal = signal_row['signal']
                        
                        # Skip if no signal
                        if signal == 0:
                            continue
                        
                        # Check if we already have a position for this symbol
                        if any(pos.get('underlying', '') == symbol for pos in self.portfolio['positions'].values()):
                            continue
                        
                        # Generate trade plan
                        if self.strategy is not None and symbol in self.options_data:
                            options_data = self.options_data[symbol]
                            
                            # Check if options data is a dictionary of dates
                            if isinstance(options_data, dict) and date in options_data:
                                options_df = options_data[date]
                                
                                # Generate trade plan
                                account_value = self._calculate_portfolio_value(date)
                                trade_plan = self.strategy.generate_trade_plan(
                                    date_rows,
                                    {symbol: options_df},
                                    account_value
                                )
                                
                                # Execute trade plan
                                if not trade_plan.empty:
                                    for _, trade in trade_plan.iterrows():
                                        # Check if we have capacity for new positions
                                        if len(self.portfolio['positions']) >= self.max_positions:
                                            break
                                        
                                        # Execute the trade
                                        self._execute_trade(trade, date)
        except Exception as e:
            logger.error(f"Error processing entry signals for date {date}: {e}")
    
    def _execute_trade(self, trade, date):
        """
        Execute a trade based on the trade plan.
        
        Args:
            trade: Trade plan row
            date: Current date
        """
        try:
            # Check if this is a multi-leg trade
            if 'legs' in trade:
                # Execute each leg
                for leg in trade['legs']:
                    self._execute_single_leg(leg, trade, date)
            else:
                # Execute as a single-leg trade
                self._execute_single_leg(trade, trade, date)
        except Exception as e:
            logger.error(f"Error executing trade for date {date}: {e}")
    
    def _execute_single_leg(self, leg, trade, date):
        """
        Execute a single leg of a trade.
        
        Args:
            leg: Leg details
            trade: Overall trade plan
            date: Current date
        """
        try:
            # Extract trade details
            symbol = trade.get('symbol', '')
            underlying = trade.get('underlying', symbol)
            direction = trade.get('signal', 0)
            
            # Extract leg details
            option_symbol = leg.get('option_symbol', f"{underlying}_{leg.get('strike', 0)}_{leg.get('expiration', '')}_{leg.get('right', '')}")
            strike = leg.get('strike', 0)
            expiration = leg.get('expiration', '')
            right = leg.get('right', '')
            action = leg.get('action', 'BUY')
            option_price = leg.get('option_price', 0)
            quantity = leg.get('quantity', 1) * trade.get('position_size', 1)
            
            # Calculate cost
            cost = option_price * quantity * 100  # Each option contract is for 100 shares
            
            # Add commission
            commission = self.commission_per_contract * quantity
            total_cost = cost + commission
            
            # Check if we have enough cash
            if action == 'BUY' and total_cost > self.portfolio['cash']:
                logger.warning(f"Not enough cash to execute trade: {total_cost} > {self.portfolio['cash']}")
                return
            
            # Execute the trade
            if action == 'BUY':
                # Deduct cash
                self.portfolio['cash'] -= total_cost
                
                # Add position
                self.portfolio['positions'][option_symbol] = {
                    'type': 'option',
                    'underlying': underlying,
                    'option_symbol': option_symbol,
                    'strike': strike,
                    'expiration': pd.to_datetime(expiration) if isinstance(expiration, str) else expiration,
                    'right': right,
                    'entry_price': option_price,
                    'current_price': option_price,
                    'quantity': quantity,
                    'direction': direction,
                    'cost_basis': total_cost,
                    'entry_date': date,
                    'unrealized_pnl': 0,
                    'daily_pnl': 0,
                    'stop_loss': trade.get('stop_loss', 0) * quantity * 100,
                    'take_profit': trade.get('take_profit', float('inf')) * quantity * 100,
                    'delta': leg.get('delta', 0),
                    'gamma': leg.get('gamma', 0),
                    'theta': leg.get('theta', 0),
                    'vega': leg.get('vega', 0)
                }
            elif action == 'SELL':
                # Add cash (credit)
                self.portfolio['cash'] += cost - commission
                
                # Add position (short)
                self.portfolio['positions'][option_symbol] = {
                    'type': 'option',
                    'underlying': underlying,
                    'option_symbol': option_symbol,
                    'strike': strike,
                    'expiration': pd.to_datetime(expiration) if isinstance(expiration, str) else expiration,
                    'right': right,
                    'entry_price': option_price,
                    'current_price': option_price,
                    'quantity': -quantity,  # Negative for short
                    'direction': -direction,  # Opposite for short
                    'cost_basis': -(cost - commission),  # Negative for short
                    'entry_date': date,
                    'unrealized_pnl': 0,
                    'daily_pnl': 0,
                    'stop_loss': trade.get('stop_loss', 0) * quantity * 100,
                    'take_profit': trade.get('take_profit', float('inf')) * quantity * 100,
                    'delta': -leg.get('delta', 0),  # Opposite for short
                    'gamma': -leg.get('gamma', 0),  # Opposite for short
                    'theta': -leg.get('theta', 0),  # Opposite for short
                    'vega': -leg.get('vega', 0)  # Opposite for short
                }
            
            # Record the trade
            self.portfolio['trades'].append({
                'date': date,
                'symbol': option_symbol,
                'underlying': underlying,
                'action': action,
                'quantity': quantity if action == 'BUY' else -quantity,
                'price': option_price,
                'cost': total_cost if action == 'BUY' else -(cost - commission),
                'commission': commission,
                'type': 'ENTRY'
            })
        except Exception as e:
            logger.error(f"Error executing single leg for date {date}: {e}")
    
    def _close_position(self, symbol, date, reason):
        """
        Close a position.
        
        Args:
            symbol: Symbol of the position to close
            date: Current date
            reason: Reason for closing the position
        """
        try:
            # Check if we have this position
            if symbol not in self.portfolio['positions']:
                logger.warning(f"Position {symbol} not found")
                return
            
            # Get position details
            position = self.portfolio['positions'][symbol]
            
            # Calculate exit value
            exit_price = position['current_price']
            quantity = abs(position['quantity'])
            exit_value = exit_price * quantity * 100  # Each option contract is for 100 shares
            
            # Add commission
            commission = self.commission_per_contract * quantity
            
            # Update cash
            if position['quantity'] > 0:  # Long position
                self.portfolio['cash'] += exit_value - commission
            else:  # Short position
                self.portfolio['cash'] -= exit_value + commission
            
            # Calculate realized P&L
            realized_pnl = position['unrealized_pnl']
            
            # Record the trade
            self.portfolio['trades'].append({
                'date': date,
                'symbol': symbol,
                'underlying': position.get('underlying', symbol.split('_')[0]),
                'action': 'SELL' if position['quantity'] > 0 else 'BUY',
                'quantity': quantity,
                'price': exit_price,
                'cost': exit_value,
                'commission': commission,
                'pnl': realized_pnl,
                'type': 'EXIT',
                'reason': reason
            })
            
            # Remove the position
            del self.portfolio['positions'][symbol]
        except Exception as e:
            logger.error(f"Error closing position {symbol} for date {date}: {e}")
    
    def _record_portfolio_value(self, date):
        """
        Record the portfolio value for a date.
        
        Args:
            date: Current date
        """
        try:
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(date)
            
            # Record history
            self.portfolio['history'].append({
                'date': date,
                'cash': self.portfolio['cash'],
                'positions_value': portfolio_value - self.portfolio['cash'],
                'total_value': portfolio_value
            })
        except Exception as e:
            logger.error(f"Error recording portfolio value for date {date}: {e}")
    
    def _calculate_portfolio_value(self, date):
        """
        Calculate the total portfolio value.
        
        Args:
            date: Current date
            
        Returns:
            Total portfolio value
        """
        try:
            # Start with cash
            total_value = self.portfolio['cash']
            
            # Add value of positions
            for symbol, position in self.portfolio['positions'].items():
                position_value = position['current_price'] * abs(position['quantity']) * 100
                total_value += position_value
            
            return total_value
        except Exception as e:
            logger.error(f"Error calculating portfolio value for date {date}: {e}")
            return self.portfolio['cash']
    
    def _calculate_performance_metrics(self):
        """
        Calculate performance metrics for the backtest.
        """
        try:
            # Check if we have history
            if not self.portfolio['history']:
                logger.warning("No history to calculate performance metrics")
                return
            
            # Convert history to DataFrame
            history_df = pd.DataFrame(self.portfolio['history'])
            
            # Calculate daily returns
            history_df['daily_return'] = history_df['total_value'].pct_change()
            
            # Calculate cumulative returns
            history_df['cumulative_return'] = (1 + history_df['daily_return']).cumprod() - 1
            
            # Calculate annualized return
            days = (history_df['date'].max() - history_df['date'].min()).days
            years = days / 365
            annualized_return = (1 + history_df['cumulative_return'].iloc[-1]) ** (1 / years) - 1
            
            # Calculate volatility
            daily_volatility = history_df['daily_return'].std()
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            # Calculate Sharpe ratio
            sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
            
            # Calculate drawdown
            history_df['peak'] = history_df['total_value'].cummax()
            history_df['drawdown'] = (history_df['total_value'] - history_df['peak']) / history_df['peak']
            max_drawdown = history_df['drawdown'].min()
            
            # Calculate win rate
            trades_df = pd.DataFrame(self.portfolio['trades'])
            if not trades_df.empty and 'pnl' in trades_df.columns:
                exit_trades = trades_df[trades_df['type'] == 'EXIT']
                win_rate = len(exit_trades[exit_trades['pnl'] > 0]) / len(exit_trades) if len(exit_trades) > 0 else 0
            else:
                win_rate = 0
            
            # Store metrics
            self.metrics = {
                'initial_capital': self.initial_capital,
                'final_value': history_df['total_value'].iloc[-1],
                'total_return': history_df['total_value'].iloc[-1] / self.initial_capital - 1,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_trades': len(trades_df[trades_df['type'] == 'EXIT']),
                'history': history_df
            }
            
            # Log metrics
            logger.info(f"Performance Metrics:")
            logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
            logger.info(f"Final Value: ${history_df['total_value'].iloc[-1]:.2f}")
            logger.info(f"Total Return: {self.metrics['total_return']:.2%}")
            logger.info(f"Annualized Return: {annualized_return:.2%}")
            logger.info(f"Annualized Volatility: {annualized_volatility:.2%}")
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {max_drawdown:.2%}")
            logger.info(f"Win Rate: {win_rate:.2%}")
            logger.info(f"Number of Trades: {self.metrics['num_trades']}")
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
    
    def plot_equity_curve(self, save_path=None):
        """
        Plot the equity curve.
        
        Args:
            save_path: Path to save the plot (if None, display the plot)
        """
        try:
            # Check if we have metrics
            if not self.metrics or 'history' not in self.metrics:
                logger.warning("No metrics to plot equity curve")
                return
            
            # Get history
            history_df = self.metrics['history']
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot equity curve
            plt.subplot(2, 1, 1)
            plt.plot(history_df['date'], history_df['total_value'], label='Portfolio Value')
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Value ($)')
            plt.grid(True)
            plt.legend()
            
            # Plot drawdown
            plt.subplot(2, 1, 2)
            plt.fill_between(history_df['date'], history_df['drawdown'], 0, color='red', alpha=0.3)
            plt.plot(history_df['date'], history_df['drawdown'], color='red', label='Drawdown')
            plt.title('Drawdown')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Equity curve saved to {save_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
    
    def plot_returns_distribution(self, save_path=None):
        """
        Plot the distribution of returns.
        
        Args:
            save_path: Path to save the plot (if None, display the plot)
        """
        try:
            # Check if we have metrics
            if not self.metrics or 'history' not in self.metrics:
                logger.warning("No metrics to plot returns distribution")
                return
            
            # Get history
            history_df = self.metrics['history']
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot returns distribution
            plt.subplot(2, 1, 1)
            sns.histplot(history_df['daily_return'].dropna(), kde=True)
            plt.title('Daily Returns Distribution')
            plt.xlabel('Daily Return')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # Plot QQ plot
            plt.subplot(2, 1, 2)
            import scipy.stats as stats
            stats.probplot(history_df['daily_return'].dropna(), plot=plt)
            plt.title('QQ Plot of Daily Returns')
            plt.grid(True)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Returns distribution saved to {save_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Error plotting returns distribution: {e}")
    
    def generate_report(self, save_path=None):
        """
        Generate a backtest report.
        
        Args:
            save_path: Path to save the report (if None, return the report as a string)
            
        Returns:
            Report as a string if save_path is None
        """
        try:
            # Check if we have metrics
            if not self.metrics:
                logger.warning("No metrics to generate report")
                return "No metrics available"
            
            # Create report
            report = []
            report.append("# Backtest Report")
            report.append("")
            report.append("## Performance Metrics")
            report.append("")
            report.append(f"Initial Capital: ${self.metrics['initial_capital']:.2f}")
            report.append(f"Final Value: ${self.metrics['final_value']:.2f}")
            report.append(f"Total Return: {self.metrics['total_return']:.2%}")
            report.append(f"Annualized Return: {self.metrics['annualized_return']:.2%}")
            report.append(f"Annualized Volatility: {self.metrics['annualized_volatility']:.2%}")
            report.append(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
            report.append(f"Max Drawdown: {self.metrics['max_drawdown']:.2%}")
            report.append(f"Win Rate: {self.metrics['win_rate']:.2%}")
            report.append(f"Number of Trades: {self.metrics['num_trades']}")
            report.append("")
            
            # Add trade statistics
            if self.portfolio['trades']:
                trades_df = pd.DataFrame(self.portfolio['trades'])
                exit_trades = trades_df[trades_df['type'] == 'EXIT']
                
                if not exit_trades.empty and 'pnl' in exit_trades.columns:
                    report.append("## Trade Statistics")
                    report.append("")
                    report.append(f"Total Trades: {len(exit_trades)}")
                    report.append(f"Winning Trades: {len(exit_trades[exit_trades['pnl'] > 0])}")
                    report.append(f"Losing Trades: {len(exit_trades[exit_trades['pnl'] <= 0])}")
                    report.append(f"Win Rate: {len(exit_trades[exit_trades['pnl'] > 0]) / len(exit_trades):.2%}")
                    report.append(f"Average Profit: ${exit_trades[exit_trades['pnl'] > 0]['pnl'].mean():.2f}")
                    report.append(f"Average Loss: ${exit_trades[exit_trades['pnl'] <= 0]['pnl'].mean():.2f}")
                    report.append(f"Profit Factor: {abs(exit_trades[exit_trades['pnl'] > 0]['pnl'].sum() / exit_trades[exit_trades['pnl'] <= 0]['pnl'].sum()) if exit_trades[exit_trades['pnl'] <= 0]['pnl'].sum() != 0 else float('inf'):.2f}")
                    report.append(f"Average Holding Period: {(exit_trades['date'] - pd.to_datetime(trades_df[trades_df['type'] == 'ENTRY']['date'])).mean().days:.1f} days")
                    report.append("")
            
            # Join report
            report_str = "\n".join(report)
            
            # Save or return
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_str)
                logger.info(f"Report saved to {save_path}")
                return None
            else:
                return report_str
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    from strategies.directional_strategy import DirectionalOptionsStrategy
    
    # Create backtest engine
    backtest = BacktestEngine()
    
    # Set strategy
    strategy = DirectionalOptionsStrategy()
    backtest.set_strategy(strategy)
    
    # Load data
    backtest.load_market_data("../data/market_data.csv")
    backtest.load_options_data("../data/options_data.csv")
    
    # Generate features
    backtest.generate_features()
    
    # Generate signals
    backtest.generate_signals()
    
    # Run backtest
    backtest.run_backtest()
    
    # Plot results
    backtest.plot_equity_curve("equity_curve.png")
    backtest.plot_returns_distribution("returns_distribution.png")
    
    # Generate report
    report = backtest.generate_report("backtest_report.md")
