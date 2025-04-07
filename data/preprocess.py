"""
Data Preprocessing Module

This module handles data preprocessing and integration of different data sources.
It provides functionality to clean, merge, and prepare data for feature engineering.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
try:
    from data.fetch_ibkr import IBKRDataFetcher
    from data.fetch_yfinance import YFinanceDataFetcher
    from data.store import DataStorage
except ImportError:
    logging.warning("Unable to import local modules. Make sure they are in the correct location.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Class to handle data preprocessing and integration.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the data preprocessor with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize data sources and storage
        self.ibkr_fetcher = None
        self.yfinance_fetcher = None
        self.storage = DataStorage(config_path)
        
        # Initialize data sources based on configuration
        self._initialize_data_sources()
    
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
                'data_sources': {
                    'ibkr': {
                        'enabled': True
                    },
                    'yfinance': {
                        'enabled': True
                    }
                }
            }
    
    def _initialize_data_sources(self):
        """Initialize data sources based on configuration."""
        # Initialize IBKR fetcher if enabled
        if self.config['data_sources']['ibkr']['enabled']:
            try:
                self.ibkr_fetcher = IBKRDataFetcher(config_path="../config/settings.yaml")
            except Exception as e:
                logger.error(f"Error initializing IBKR fetcher: {e}")
        
        # Initialize YFinance fetcher if enabled
        if self.config['data_sources']['yfinance']['enabled']:
            try:
                self.yfinance_fetcher = YFinanceDataFetcher(config_path="../config/settings.yaml")
            except Exception as e:
                logger.error(f"Error initializing YFinance fetcher: {e}")
    
    def fetch_and_store_options_data(self, symbol: str, min_dte: int = 1, max_dte: int = 120) -> pd.DataFrame:
        """
        Fetch options data from available sources and store it.
        
        Args:
            symbol: Stock symbol
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            
        Returns:
            DataFrame containing options data
        """
        options_df = pd.DataFrame()
        
        # Try IBKR first if available
        if self.ibkr_fetcher and self.ibkr_fetcher.connected:
            try:
                logger.info(f"Fetching options data for {symbol} from IBKR")
                options_df = self.ibkr_fetcher.get_option_chain(symbol, min_dte=min_dte, max_dte=max_dte)
            except Exception as e:
                logger.error(f"Error fetching options data from IBKR: {e}")
        
        # If IBKR failed or returned empty data, try YFinance
        if options_df.empty and self.yfinance_fetcher:
            try:
                logger.info(f"Fetching options data for {symbol} from YFinance")
                options_df = self.yfinance_fetcher.get_option_chain(symbol, min_dte=min_dte, max_dte=max_dte)
            except Exception as e:
                logger.error(f"Error fetching options data from YFinance: {e}")
        
        # Store the data if we got any
        if not options_df.empty:
            logger.info(f"Storing options data for {symbol}")
            self.storage.store_options_chain(options_df)
        else:
            logger.warning(f"No options data retrieved for {symbol}")
        
        return options_df
    
    def fetch_and_store_historical_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical price data from available sources and store it.
        
        Args:
            symbol: Stock symbol
            period: Time period (e.g., '1y', '6mo', '3mo')
            interval: Data interval (e.g., '1d', '1h')
            
        Returns:
            DataFrame containing historical price data
        """
        prices_df = pd.DataFrame()
        
        # Try IBKR first if available
        if self.ibkr_fetcher and self.ibkr_fetcher.connected:
            try:
                logger.info(f"Fetching historical data for {symbol} from IBKR")
                # Convert period to IBKR format
                if period == '1y':
                    duration = '1 Y'
                elif period == '6mo':
                    duration = '6 M'
                elif period == '3mo':
                    duration = '3 M'
                else:
                    duration = '1 Y'  # Default
                
                # Convert interval to IBKR format
                if interval == '1d':
                    bar_size = '1 day'
                elif interval == '1h':
                    bar_size = '1 hour'
                else:
                    bar_size = '1 day'  # Default
                
                prices_df = self.ibkr_fetcher.get_historical_data(symbol, duration=duration, bar_size=bar_size)
            except Exception as e:
                logger.error(f"Error fetching historical data from IBKR: {e}")
        
        # If IBKR failed or returned empty data, try YFinance
        if prices_df.empty and self.yfinance_fetcher:
            try:
                logger.info(f"Fetching historical data for {symbol} from YFinance")
                prices_df = self.yfinance_fetcher.get_historical_data(symbol, period=period, interval=interval)
            except Exception as e:
                logger.error(f"Error fetching historical data from YFinance: {e}")
        
        # Store the data if we got any
        if not prices_df.empty:
            logger.info(f"Storing historical data for {symbol}")
            self.storage.store_historical_prices(prices_df)
        else:
            logger.warning(f"No historical data retrieved for {symbol}")
        
        return prices_df
    
    def fetch_and_store_volatility_data(self, symbol: str, lookback_days: List[int] = [10, 20, 60, 120]) -> Dict[int, float]:
        """
        Calculate historical volatility for different lookback periods and store as features.
        
        Args:
            symbol: Stock symbol
            lookback_days: List of lookback periods in days
            
        Returns:
            Dictionary mapping lookback period to volatility value
        """
        volatility_data = {}
        
        # Try IBKR first if available
        if self.ibkr_fetcher and self.ibkr_fetcher.connected:
            try:
                logger.info(f"Calculating volatility for {symbol} using IBKR data")
                volatility_data = self.ibkr_fetcher.get_historical_volatility(symbol, lookback_days)
            except Exception as e:
                logger.error(f"Error calculating volatility using IBKR data: {e}")
        
        # If IBKR failed or returned empty data, try YFinance
        if not volatility_data and self.yfinance_fetcher:
            try:
                logger.info(f"Calculating volatility for {symbol} using YFinance data")
                volatility_data = self.yfinance_fetcher.get_historical_volatility(symbol, lookback_days)
            except Exception as e:
                logger.error(f"Error calculating volatility using YFinance data: {e}")
        
        # Store the data if we got any
        if volatility_data:
            logger.info(f"Storing volatility data for {symbol}")
            
            # Convert to DataFrame for storage
            today = datetime.now().strftime('%Y-%m-%d')
            data = []
            
            for days, value in volatility_data.items():
                if value is not None:
                    data.append({
                        'symbol': symbol,
                        'date': today,
                        'feature_name': f'hv_{days}',
                        'feature_value': value,
                        'feature_group': 'volatility'
                    })
            
            if data:
                feature_df = pd.DataFrame(data)
                self.storage.store_feature_data(feature_df)
            
        else:
            logger.warning(f"No volatility data calculated for {symbol}")
        
        return volatility_data
    
    def clean_options_data(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize options data.
        
        Args:
            options_df: DataFrame containing options data
            
        Returns:
            Cleaned DataFrame
        """
        if options_df.empty:
            return options_df
        
        try:
            # Make a copy to avoid modifying the original
            df = options_df.copy()
            
            # Standardize column names
            column_mapping = {
                'contractSymbol': 'contract_symbol',
                'lastTradeDate': 'last_trade_date',
                'lastPrice': 'last',
                'openInterest': 'open_interest',
                'impliedVolatility': 'implied_volatility',
                'inTheMoney': 'in_the_money'
            }
            
            # Apply mapping for columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Fill missing values
            numeric_cols = ['bid', 'ask', 'last', 'volume', 'open_interest', 'implied_volatility', 
                           'delta', 'gamma', 'vega', 'theta']
            
            for col in numeric_cols:
                if col in df.columns:
                    # Replace 0 with NaN for certain columns
                    if col in ['bid', 'ask', 'last']:
                        df[col] = df[col].replace(0, np.nan)
                    
                    # Fill NaN values
                    if col == 'last':
                        # Fill last price with mid price if available
                        if 'bid' in df.columns and 'ask' in df.columns:
                            df[col] = df[col].fillna((df['bid'] + df['ask']) / 2)
                    elif col in ['volume', 'open_interest']:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(df[col].median())
            
            # Calculate mid price
            if 'bid' in df.columns and 'ask' in df.columns:
                df['mid'] = (df['bid'] + df['ask']) / 2
            
            # Calculate bid-ask spread
            if 'bid' in df.columns and 'ask' in df.columns:
                df['bid_ask_spread'] = df['ask'] - df['bid']
                # Calculate spread as percentage of mid price
                if 'mid' in df.columns:
                    df['bid_ask_spread_pct'] = df['bid_ask_spread'] / df['mid']
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning options data: {e}")
            return options_df
    
    def clean_price_data(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize price data.
        
        Args:
            price_df: DataFrame containing price data
            
        Returns:
            Cleaned DataFrame
        """
        if price_df.empty:
            return price_df
        
        try:
            # Make a copy to avoid modifying the original
            df = price_df.copy()
            
            # Standardize column names
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adjusted_close'
            }
            
            # Apply mapping for columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Ensure date is in datetime format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Fill missing values
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'adjusted_close']
            
            for col in numeric_cols:
                if col in df.columns:
                    # Forward fill price data
                    if col != 'volume':
                        df[col] = df[col].fillna(method='ffill')
                    else:
                        # Fill volume with 0
                        df[col] = df[col].fillna(0)
            
            # Calculate returns
            if 'close' in df.columns:
                df['daily_return'] = df['close'].pct_change()
                df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning price data: {e}")
            return price_df
    
    def merge_options_with_underlying(self, options_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge options data with underlying price data.
        
        Args:
            options_df: DataFrame containing options data
            price_df: DataFrame containing price data
            
        Returns:
            Merged DataFrame
        """
        if options_df.empty or price_df.empty:
            return options_df
        
        try:
            # Make copies to avoid modifying the originals
            options = options_df.copy()
            prices = price_df.copy()
            
            # Ensure date columns are in datetime format
            if 'date' in prices.columns:
                prices['date'] = pd.to_datetime(prices['date'])
            
            if 'timestamp' in options.columns:
                options['timestamp'] = pd.to_datetime(options['timestamp'])
            
            # Get the latest price data
            latest_price = prices.sort_values('date').iloc[-1]
            
            # Add or update underlying price information
            if 'underlying_price' not in options.columns:
                options['underlying_price'] = latest_price['close']
            
            # Add additional underlying information
            options['underlying_daily_return'] = latest_price.get('daily_return', np.nan)
            options['underlying_volume'] = latest_price.get('volume', np.nan)
            
            return options
            
        except Exception as e:
            logger.error(f"Error merging options with underlying: {e}")
            return options_df


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    preprocessor = DataPreprocessor()
    
    # Example: Fetch and store data for AAPL
    symbol = 'AAPL'
    
    # Fetch historical data
    prices = preprocessor.fetch_and_store_historical_data(symbol)
    print(f"Retrieved {len(prices)} historical price records for {symbol}")
    
    # Fetch options data
    options = preprocessor.fetch_and_store_options_data(symbol)
    print(f"Retrieved {len(options)} option contracts for {symbol}")
    
    # Calculate volatility
    volatility = preprocessor.fetch_and_store_volatility_data(symbol)
    print(f"Calculated volatility for {symbol}: {volatility}")
    
    # Clean and merge data
    if not options.empty and not prices.empty:
        clean_options = preprocessor.clean_options_data(options)
        clean_prices = preprocessor.clean_price_data(prices)
        merged_data = preprocessor.merge_options_with_underlying(clean_options, clean_prices)
        print(f"Merged data has {len(merged_data)} rows and {len(merged_data.columns)} columns")
