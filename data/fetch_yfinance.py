"""
YFinance Data Fetcher Module

This module handles data fetching from Yahoo Finance using the yfinance library.
It provides functionality to fetch options chains, historical data, and other market information.
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

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logging.warning("yfinance not installed. Yahoo Finance functionality will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YFinanceDataFetcher:
    """
    Class to handle data fetching from Yahoo Finance.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the Yahoo Finance data fetcher with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        
        if not HAS_YFINANCE:
            logger.error("yfinance package is required for Yahoo Finance data fetching")
    
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
                    'yfinance': {
                        'enabled': True,
                        'cache_dir': '../data/cache/yfinance'
                    }
                }
            }
    
    def get_ticker_info(self, symbol: str) -> Dict:
        """
        Get basic information about a ticker.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing ticker information
        """
        if not HAS_YFINANCE:
            logger.error("yfinance package is required")
            return {}
            
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            logger.error(f"Error fetching ticker info for {symbol}: {e}")
            return {}
    
    def get_historical_data(self, 
                           symbol: str, 
                           period: str = '1y', 
                           interval: str = '1d',
                           include_dividends: bool = True) -> pd.DataFrame:
        """
        Fetch historical price data for a symbol.
        
        Args:
            symbol: Stock symbol
            period: Time period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            include_dividends: Whether to include dividends in the data
            
        Returns:
            DataFrame containing historical price data
        """
        if not HAS_YFINANCE:
            logger.error("yfinance package is required")
            return pd.DataFrame()
            
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Add symbol column
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_option_chain(self, 
                         symbol: str, 
                         min_dte: int = 1, 
                         max_dte: int = 120) -> pd.DataFrame:
        """
        Fetch the option chain for a given symbol.
        
        Args:
            symbol: Stock symbol
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            
        Returns:
            DataFrame containing option chain data
        """
        if not HAS_YFINANCE:
            logger.error("yfinance package is required")
            return pd.DataFrame()
            
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current stock price
            current_price = ticker.info.get('regularMarketPrice', 0)
            if current_price == 0:
                current_price = ticker.history(period='1d').iloc[-1]['Close']
            
            # Get all available expiration dates
            expirations = ticker.options
            
            if not expirations:
                logger.warning(f"No options available for {symbol}")
                return pd.DataFrame()
            
            # Filter expirations based on DTE
            today = datetime.now().date()
            filtered_expirations = []
            
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                dte = (exp_date - today).days
                
                if min_dte <= dte <= max_dte:
                    filtered_expirations.append(exp_str)
            
            if not filtered_expirations:
                logger.warning(f"No options within DTE range {min_dte}-{max_dte} for {symbol}")
                return pd.DataFrame()
            
            # Fetch option chains for each expiration
            all_options = []
            
            for expiration in filtered_expirations:
                try:
                    # Get calls and puts
                    opt = ticker.option_chain(expiration)
                    
                    # Process calls
                    calls_df = opt.calls.copy()
                    calls_df['right'] = 'C'
                    calls_df['expiration'] = expiration
                    
                    # Process puts
                    puts_df = opt.puts.copy()
                    puts_df['right'] = 'P'
                    puts_df['expiration'] = expiration
                    
                    # Combine and add to list
                    combined_df = pd.concat([calls_df, puts_df])
                    combined_df['symbol'] = symbol
                    combined_df['underlying_price'] = current_price
                    
                    # Calculate days to expiration
                    exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                    dte = (exp_date - today).days
                    combined_df['dte'] = dte
                    
                    all_options.append(combined_df)
                    
                except Exception as e:
                    logger.warning(f"Error fetching options for {symbol} expiration {expiration}: {e}")
                    continue
            
            if not all_options:
                logger.warning(f"Failed to retrieve any option data for {symbol}")
                return pd.DataFrame()
            
            # Combine all options into a single DataFrame
            result_df = pd.concat(all_options, ignore_index=True)
            
            # Rename columns to match our standard format
            column_mapping = {
                'contractSymbol': 'contract_symbol',
                'lastTradeDate': 'last_trade_date',
                'strike': 'strike',
                'lastPrice': 'last',
                'bid': 'bid',
                'ask': 'ask',
                'change': 'change',
                'percentChange': 'percent_change',
                'volume': 'volume',
                'openInterest': 'open_interest',
                'impliedVolatility': 'implied_volatility',
                'inTheMoney': 'in_the_money',
                'contractSize': 'contract_size',
                'currency': 'currency'
            }
            
            result_df = result_df.rename(columns=column_mapping)
            
            # Calculate Greeks if not provided by yfinance
            # Note: yfinance doesn't provide Greeks directly, so we'll calculate delta as a simple approximation
            # For a more accurate calculation, we would need to use a proper options pricing model
            
            # Simple delta approximation for educational purposes
            # This is not accurate for real trading - use a proper model like Black-Scholes
            result_df['delta'] = np.nan
            
            # For calls: simple delta approximation
            calls_mask = result_df['right'] == 'C'
            result_df.loc[calls_mask, 'delta'] = np.where(
                result_df.loc[calls_mask, 'in_the_money'],
                0.5 + 0.5 * (result_df.loc[calls_mask, 'underlying_price'] / result_df.loc[calls_mask, 'strike']),
                0.5 - 0.5 * (result_df.loc[calls_mask, 'strike'] / result_df.loc[calls_mask, 'underlying_price'])
            )
            
            # For puts: simple delta approximation
            puts_mask = result_df['right'] == 'P'
            result_df.loc[puts_mask, 'delta'] = np.where(
                result_df.loc[puts_mask, 'in_the_money'],
                -0.5 - 0.5 * (result_df.loc[puts_mask, 'underlying_price'] / result_df.loc[puts_mask, 'strike']),
                -0.5 + 0.5 * (result_df.loc[puts_mask, 'strike'] / result_df.loc[puts_mask, 'underlying_price'])
            )
            
            # Clip delta values to valid range
            result_df['delta'] = result_df['delta'].clip(-1.0, 1.0)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_earnings_dates(self, symbol: str, limit: int = 4) -> pd.DataFrame:
        """
        Get upcoming and past earnings dates.
        
        Args:
            symbol: Stock symbol
            limit: Number of earnings dates to retrieve
            
        Returns:
            DataFrame containing earnings dates
        """
        if not HAS_YFINANCE:
            logger.error("yfinance package is required")
            return pd.DataFrame()
            
        try:
            ticker = yf.Ticker(symbol)
            earnings_df = ticker.earnings_dates
            
            if earnings_df is None or earnings_df.empty:
                logger.warning(f"No earnings dates available for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            earnings_df = earnings_df.reset_index()
            
            # Rename columns to lowercase
            earnings_df.columns = [col.lower() for col in df.columns]
            
            # Add symbol column
            earnings_df['symbol'] = symbol
            
            return earnings_df.head(limit)
            
        except Exception as e:
            logger.error(f"Error fetching earnings dates for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_historical_volatility(self, symbol: str, lookback_days: List[int] = [10, 20, 60, 120]) -> Dict[int, float]:
        """
        Calculate historical volatility for different lookback periods.
        
        Args:
            symbol: Stock symbol
            lookback_days: List of lookback periods in days
            
        Returns:
            Dictionary mapping lookback period to volatility value
        """
        try:
            # Get daily historical data for the maximum lookback period
            max_days = max(lookback_days)
            
            # Add buffer days and convert to appropriate period string
            if max_days <= 60:
                period = '3mo'
            elif max_days <= 120:
                period = '6mo'
            else:
                period = '1y'
            
            df = self.get_historical_data(symbol, period, '1d')
            if df.empty:
                return {}
                
            # Calculate daily returns
            df['return'] = df['close'].pct_change()
            
            # Calculate volatility for each lookback period
            result = {}
            for days in lookback_days:
                if len(df) > days:
                    # Annualized volatility (standard deviation of returns * sqrt(252))
                    vol = df['return'].tail(days).std() * (252 ** 0.5)
                    result[days] = vol
                else:
                    result[days] = None
                    
            return result
            
        except Exception as e:
            logger.error(f"Error calculating historical volatility for {symbol}: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    fetcher = YFinanceDataFetcher()
    
    # Example: Get ticker info for AAPL
    info = fetcher.get_ticker_info('AAPL')
    print(f"Retrieved ticker info for AAPL: {len(info)} fields")
    
    # Example: Get historical data for AAPL
    hist_df = fetcher.get_historical_data('AAPL', '1mo')
    print(f"Retrieved {len(hist_df)} historical bars for AAPL")
    
    # Example: Get option chain for AAPL
    options_df = fetcher.get_option_chain('AAPL')
    print(f"Retrieved {len(options_df)} option contracts for AAPL")
    
    # Example: Get historical volatility for AAPL
    hv = fetcher.get_historical_volatility('AAPL')
    print(f"Historical volatility for AAPL: {hv}")
