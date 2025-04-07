"""
IBKR Data Fetcher Module

This module handles data fetching from Interactive Brokers using the ib_insync library.
It provides functionality to fetch options chains, historical data, and real-time market data.
"""

import os
import sys
import logging
import yaml
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ib_insync import IB, Contract, Option, Stock, BarData, util
    HAS_IB_INSYNC = True
except ImportError:
    HAS_IB_INSYNC = False
    logging.warning("ib_insync not installed. IBKR functionality will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IBKRDataFetcher:
    """
    Class to handle data fetching from Interactive Brokers.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the IBKR data fetcher with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.ib = None
        self.connected = False
        
        if not HAS_IB_INSYNC:
            logger.error("ib_insync package is required for IBKR data fetching")
            return
            
        # Initialize connection if enabled in config
        if self.config['data_sources']['ibkr']['enabled']:
            self._connect()
    
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
                        'enabled': False,
                        'client_id': 1,
                        'host': '127.0.0.1',
                        'port': 7497,
                        'read_only': True
                    }
                }
            }
    
    def _connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS or Gateway.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not HAS_IB_INSYNC:
            return False
            
        try:
            ibkr_config = self.config['data_sources']['ibkr']
            self.ib = IB()
            self.ib.connect(
                host=ibkr_config['host'],
                port=ibkr_config['port'],
                clientId=ibkr_config['client_id'],
                readonly=ibkr_config['read_only']
            )
            self.connected = True
            logger.info(f"Connected to IBKR at {ibkr_config['host']}:{ibkr_config['port']}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def get_stock_contract(self, symbol: str) -> Optional[Contract]:
        """
        Create a stock contract for the given symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Contract object or None if error
        """
        if not self.connected:
            logger.error("Not connected to IBKR")
            return None
            
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            qualified_contracts = self.ib.qualifyContracts(contract)
            if qualified_contracts:
                return qualified_contracts[0]
            else:
                logger.error(f"Could not qualify contract for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Error creating stock contract for {symbol}: {e}")
            return None
    
    def get_option_chain(self, 
                         symbol: str, 
                         right: str = None,  # 'C' for calls, 'P' for puts, None for both
                         min_dte: int = 1, 
                         max_dte: int = 120) -> pd.DataFrame:
        """
        Fetch the option chain for a given symbol.
        
        Args:
            symbol: Stock symbol
            right: Option right ('C' for calls, 'P' for puts, None for both)
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            
        Returns:
            DataFrame containing option chain data
        """
        if not self.connected:
            logger.error("Not connected to IBKR")
            return pd.DataFrame()
            
        try:
            # Get the stock contract
            stock = self.get_stock_contract(symbol)
            if not stock:
                return pd.DataFrame()
                
            # Get the stock price for reference
            self.ib.reqMarketDataType(1)  # 1 = Live, 2 = Frozen, 3 = Delayed, 4 = Delayed frozen
            ticker = self.ib.reqMktData(stock)
            self.ib.sleep(1)  # Give time for market data to arrive
            stock_price = ticker.marketPrice()
            
            # Calculate date range for options
            today = datetime.now()
            min_date = today + timedelta(days=min_dte)
            max_date = today + timedelta(days=max_dte)
            
            # Request option chains
            chains = self.ib.reqSecDefOptParams(
                stock.symbol, 
                '', 
                stock.secType, 
                stock.conId
            )
            
            if not chains:
                logger.error(f"No option chains found for {symbol}")
                return pd.DataFrame()
                
            # Process chains to get contracts
            all_options_data = []
            
            for chain in chains:
                # Filter expirations within our range
                valid_expirations = [exp for exp in chain.expirations 
                                    if min_date <= datetime.strptime(exp, '%Y%m%d') <= max_date]
                
                # For each valid expiration, get options at different strikes
                for expiration in valid_expirations:
                    exp_date = datetime.strptime(expiration, '%Y%m%d')
                    dte = (exp_date - today).days
                    
                    # Determine which strikes to request based on current price
                    strikes = [strike for strike in chain.strikes 
                              if 0.7 * stock_price <= strike <= 1.3 * stock_price]
                    
                    # Request contracts for these strikes
                    rights = ['C', 'P'] if right is None else [right]
                    
                    for strike in strikes:
                        for opt_right in rights:
                            option_contract = Option(
                                symbol, 
                                expiration, 
                                strike, 
                                opt_right, 
                                'SMART'
                            )
                            
                            try:
                                # Qualify the contract
                                qualified_contracts = self.ib.qualifyContracts(option_contract)
                                if not qualified_contracts:
                                    continue
                                    
                                qualified_contract = qualified_contracts[0]
                                
                                # Request market data
                                ticker = self.ib.reqMktData(qualified_contract, '', False, False)
                                self.ib.sleep(0.1)  # Small delay to avoid overwhelming TWS
                                
                                # Get option Greeks if available
                                implied_vol = ticker.modelGreeks.impliedVol if ticker.modelGreeks else None
                                delta = ticker.modelGreeks.delta if ticker.modelGreeks else None
                                gamma = ticker.modelGreeks.gamma if ticker.modelGreeks else None
                                vega = ticker.modelGreeks.vega if ticker.modelGreeks else None
                                theta = ticker.modelGreeks.theta if ticker.modelGreeks else None
                                
                                # Collect data
                                option_data = {
                                    'symbol': symbol,
                                    'expiration': exp_date.strftime('%Y-%m-%d'),
                                    'strike': strike,
                                    'right': opt_right,
                                    'dte': dte,
                                    'bid': ticker.bid,
                                    'ask': ticker.ask,
                                    'last': ticker.last,
                                    'volume': ticker.volume,
                                    'open_interest': ticker.openInterest,
                                    'implied_volatility': implied_vol,
                                    'delta': delta,
                                    'gamma': gamma,
                                    'vega': vega,
                                    'theta': theta,
                                    'underlying_price': stock_price
                                }
                                
                                all_options_data.append(option_data)
                                
                                # Cancel market data subscription to avoid hitting limits
                                self.ib.cancelMktData(qualified_contract)
                                
                            except Exception as e:
                                logger.warning(f"Error fetching data for {symbol} {expiration} {strike} {opt_right}: {e}")
                                continue
            
            # Convert to DataFrame
            if all_options_data:
                df = pd.DataFrame(all_options_data)
                return df
            else:
                logger.warning(f"No option data retrieved for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, 
                           symbol: str, 
                           duration: str = '1 Y', 
                           bar_size: str = '1 day',
                           what_to_show: str = 'TRADES') -> pd.DataFrame:
        """
        Fetch historical price data for a symbol.
        
        Args:
            symbol: Stock symbol
            duration: Time duration (e.g., '1 Y', '6 M', '30 D')
            bar_size: Bar size (e.g., '1 day', '1 hour', '5 mins')
            what_to_show: Type of data to retrieve
            
        Returns:
            DataFrame containing historical price data
        """
        if not self.connected:
            logger.error("Not connected to IBKR")
            return pd.DataFrame()
            
        try:
            # Get the stock contract
            contract = self.get_stock_contract(symbol)
            if not contract:
                return pd.DataFrame()
                
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',  # '' for latest data
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1  # 1 for yyyyMMdd format
            )
            
            if not bars:
                logger.warning(f"No historical data retrieved for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = util.df(bars)
            
            # Add symbol column
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
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
            duration = f'{max_days + 10} D'  # Add buffer days
            
            df = self.get_historical_data(symbol, duration, '1 day', 'TRADES')
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
    fetcher = IBKRDataFetcher()
    
    if fetcher.connected:
        # Example: Get option chain for AAPL
        options_df = fetcher.get_option_chain('AAPL')
        print(f"Retrieved {len(options_df)} option contracts for AAPL")
        
        # Example: Get historical data for AAPL
        hist_df = fetcher.get_historical_data('AAPL', '1 M')
        print(f"Retrieved {len(hist_df)} historical bars for AAPL")
        
        # Example: Get historical volatility for AAPL
        hv = fetcher.get_historical_volatility('AAPL')
        print(f"Historical volatility for AAPL: {hv}")
        
        # Disconnect when done
        fetcher.disconnect()
    else:
        print("Not connected to IBKR. Make sure TWS or IB Gateway is running.")
