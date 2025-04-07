"""
Technical Indicators Module

This module handles the calculation of various technical indicators
including RSI, MACD, ATR, Bollinger Bands, and other market microstructure metrics.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Class to handle the calculation of technical indicators.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the technical indicators calculator with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.enabled_indicators = self.config.get('features', {}).get('technicals', {}).get('indicators', 
                                                                                          ["rsi", "macd", "bollinger", "atr"])
    
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
                'features': {
                    'technicals': {
                        'enabled': True,
                        'indicators': ["rsi", "macd", "bollinger", "atr"]
                    }
                }
            }
    
    def calculate_all_indicators(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all enabled technical indicators.
        
        Args:
            price_df: DataFrame containing price data
            
        Returns:
            DataFrame with added technical indicator columns
        """
        if price_df.empty:
            return price_df
            
        try:
            # Make a copy to avoid modifying the original
            df = price_df.copy()
            
            # Check if we have the required columns
            required_cols = ['close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns for technical indicators: {missing_cols}")
                return df
            
            # Calculate enabled indicators
            if "rsi" in self.enabled_indicators:
                df = self.calculate_rsi(df)
                
            if "macd" in self.enabled_indicators:
                df = self.calculate_macd(df)
                
            if "bollinger" in self.enabled_indicators:
                df = self.calculate_bollinger_bands(df)
                
            if "atr" in self.enabled_indicators:
                df = self.calculate_atr(df)
                
            if "volume_profile" in self.enabled_indicators:
                df = self.calculate_volume_profile(df)
                
            if "momentum" in self.enabled_indicators:
                df = self.calculate_momentum(df)
                
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return price_df
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            df: DataFrame containing price data
            period: RSI period
            
        Returns:
            DataFrame with added RSI column
        """
        try:
            # Check if we have the required columns
            if 'close' not in df.columns:
                logger.error("Missing 'close' column for RSI calculation")
                return df
            
            # Calculate price changes
            delta = df['close'].diff()
            
            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS
            rs = avg_gain / avg_loss
            
            # Calculate RSI
            df['rsi'] = 100 - (100 / (1 + rs))
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return df
    
    def calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            df: DataFrame containing price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            
        Returns:
            DataFrame with added MACD columns
        """
        try:
            # Check if we have the required columns
            if 'close' not in df.columns:
                logger.error("Missing 'close' column for MACD calculation")
                return df
            
            # Calculate EMAs
            ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            df['macd_line'] = ema_fast - ema_slow
            
            # Calculate signal line
            df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
            
            # Calculate histogram
            df['macd_histogram'] = df['macd_line'] - df['macd_signal']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame containing price data
            period: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with added Bollinger Bands columns
        """
        try:
            # Check if we have the required columns
            if 'close' not in df.columns:
                logger.error("Missing 'close' column for Bollinger Bands calculation")
                return df
            
            # Calculate middle band (SMA)
            df['bb_middle'] = df['close'].rolling(window=period).mean()
            
            # Calculate standard deviation
            rolling_std = df['close'].rolling(window=period).std()
            
            # Calculate upper and lower bands
            df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
            df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
            
            # Calculate bandwidth
            df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Calculate %B
            df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: DataFrame containing price data
            period: ATR period
            
        Returns:
            DataFrame with added ATR column
        """
        try:
            # Check if we have the required columns
            required_cols = ['high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns for ATR calculation: {missing_cols}")
                return df
            
            # Calculate true range
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # Calculate ATR
            df['atr'] = df['true_range'].rolling(window=period).mean()
            
            # Drop temporary columns
            df = df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1, errors='ignore')
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return df
    
    def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
        """
        Calculate Volume Profile.
        
        Args:
            df: DataFrame containing price data
            bins: Number of price bins
            
        Returns:
            DataFrame with added volume profile columns
        """
        try:
            # Check if we have the required columns
            required_cols = ['close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns for volume profile calculation: {missing_cols}")
                return df
            
            # Get price range
            price_min = df['close'].min()
            price_max = df['close'].max()
            
            # Create price bins
            bin_edges = np.linspace(price_min, price_max, bins + 1)
            
            # Assign each price to a bin
            df['price_bin'] = pd.cut(df['close'], bins=bin_edges, labels=False)
            
            # Calculate volume per bin
            volume_profile = df.groupby('price_bin')['volume'].sum()
            
            # Find the bin with the highest volume (Point of Control)
            poc_bin = volume_profile.idxmax()
            
            # Calculate Value Area (70% of volume)
            total_volume = volume_profile.sum()
            value_area_volume = total_volume * 0.7
            
            # Sort bins by volume in descending order
            sorted_bins = volume_profile.sort_values(ascending=False)
            
            # Find bins in the Value Area
            cumulative_volume = 0
            value_area_bins = []
            
            for bin_idx, bin_volume in sorted_bins.items():
                cumulative_volume += bin_volume
                value_area_bins.append(bin_idx)
                
                if cumulative_volume >= value_area_volume:
                    break
            
            # Find Value Area High and Low
            value_area_high = bin_edges[max(value_area_bins) + 1]
            value_area_low = bin_edges[min(value_area_bins)]
            
            # Add volume profile metrics to DataFrame
            df['volume_poc'] = bin_edges[poc_bin]
            df['volume_vah'] = value_area_high
            df['volume_val'] = value_area_low
            
            # Drop temporary columns
            df = df.drop(['price_bin'], axis=1, errors='ignore')
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return df
    
    def calculate_momentum(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Calculate price momentum over different periods.
        
        Args:
            df: DataFrame containing price data
            periods: List of periods for momentum calculation
            
        Returns:
            DataFrame with added momentum columns
        """
        try:
            # Check if we have the required columns
            if 'close' not in df.columns:
                logger.error("Missing 'close' column for momentum calculation")
                return df
            
            # Calculate momentum for each period
            for period in periods:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return df
    
    def calculate_market_microstructure(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market microstructure metrics for options.
        
        Args:
            options_df: DataFrame containing options data
            
        Returns:
            DataFrame with added market microstructure columns
        """
        if options_df.empty:
            return options_df
            
        try:
            # Make a copy to avoid modifying the original
            df = options_df.copy()
            
            # Check if we have the required columns
            required_cols = ['bid', 'ask', 'volume', 'open_interest']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns for market microstructure calculation: {missing_cols}")
                return df
            
            # Calculate bid-ask spread
            df['bid_ask_spread'] = df['ask'] - df['bid']
            
            # Calculate bid-ask spread as percentage of mid price
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            df['bid_ask_spread_pct'] = df['bid_ask_spread'] / df['mid_price']
            
            # Calculate volume to open interest ratio
            df['volume_oi_ratio'] = df['volume'] / df['open_interest'].replace(0, np.nan)
            
            # Calculate liquidity score (higher is more liquid)
            # Factors: tight spread, high volume, high open interest
            df['liquidity_score'] = (
                (1 / df['bid_ask_spread_pct'].replace(0, np.inf)) * 
                np.log1p(df['volume']) * 
                np.log1p(df['open_interest'])
            )
            
            # Normalize liquidity score
            max_score = df['liquidity_score'].max()
            if max_score > 0:
                df['liquidity_score_norm'] = df['liquidity_score'] / max_score
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating market microstructure metrics: {e}")
            return options_df


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    indicators = TechnicalIndicators()
    
    # Example: Create sample price data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    price_data = {
        'date': dates,
        'open': np.random.normal(loc=100, scale=1, size=len(dates)).cumsum(),
        'high': np.random.normal(loc=102, scale=1, size=len(dates)).cumsum(),
        'low': np.random.normal(loc=98, scale=1, size=len(dates)).cumsum(),
        'close': np.random.normal(loc=100, scale=1, size=len(dates)).cumsum(),
        'volume': np.random.randint(1000, 10000, size=len(dates))
    }
    
    # Ensure high is the highest and low is the lowest
    for i in range(len(dates)):
        values = [price_data['open'][i], price_data['close'][i]]
        price_data['high'][i] = max(values) + abs(np.random.normal(0, 0.5))
        price_data['low'][i] = min(values) - abs(np.random.normal(0, 0.5))
    
    price_df = pd.DataFrame(price_data)
    
    # Calculate technical indicators
    result_df = indicators.calculate_all_indicators(price_df)
    
    # Print results
    print("Technical Indicators:")
    print(result_df[['date', 'close', 'rsi', 'macd_line', 'bb_upper', 'bb_lower', 'atr']].tail())
