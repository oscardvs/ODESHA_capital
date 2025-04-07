"""
Volatility Metrics Module

This module handles the calculation of various volatility metrics including
IV Rank, IV Percentile, Historical Volatility, and IV-HV spreads.
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


class VolatilityMetrics:
    """
    Class to handle the calculation of volatility metrics.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the volatility metrics calculator with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.lookback_periods = self.config.get('features', {}).get('volatility', {}).get('lookback_periods', [10, 20, 60, 120])
        self.iv_percentile_days = self.config.get('features', {}).get('volatility', {}).get('iv_percentile_days', 252)
    
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
                    'volatility': {
                        'lookback_periods': [10, 20, 60, 120],
                        'iv_percentile_days': 252
                    }
                }
            }
    
    def calculate_historical_volatility(self, price_df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate historical volatility for different lookback periods.
        
        Args:
            price_df: DataFrame containing price data with 'close' column
            periods: List of lookback periods in days
            
        Returns:
            DataFrame with added historical volatility columns
        """
        if price_df.empty:
            return price_df
            
        try:
            # Make a copy to avoid modifying the original
            df = price_df.copy()
            
            # Use default periods if none provided
            if periods is None:
                periods = self.lookback_periods
            
            # Check if we have the required columns
            if 'close' not in df.columns:
                logger.error("Missing 'close' column for historical volatility calculation")
                return df
            
            # Calculate daily returns
            if 'daily_return' not in df.columns:
                df['daily_return'] = df['close'].pct_change()
            
            # Calculate historical volatility for each period
            for period in periods:
                if len(df) > period:
                    # Rolling standard deviation of returns
                    rolling_std = df['daily_return'].rolling(window=period).std()
                    
                    # Annualized volatility (standard deviation * sqrt(252))
                    df[f'hv_{period}'] = rolling_std * np.sqrt(252)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating historical volatility: {e}")
            return price_df
    
    def calculate_iv_rank(self, 
                         current_iv: float, 
                         iv_history: pd.Series) -> float:
        """
        Calculate IV Rank (current IV relative to 52-week high/low).
        
        Args:
            current_iv: Current implied volatility
            iv_history: Series of historical implied volatility values
            
        Returns:
            IV Rank (0-1)
        """
        try:
            if iv_history.empty:
                return np.nan
                
            iv_min = iv_history.min()
            iv_max = iv_history.max()
            
            # Avoid division by zero
            if iv_max == iv_min:
                return 0.5
                
            iv_rank = (current_iv - iv_min) / (iv_max - iv_min)
            
            # Clip to 0-1 range
            iv_rank = np.clip(iv_rank, 0, 1)
            
            return iv_rank
            
        except Exception as e:
            logger.error(f"Error calculating IV Rank: {e}")
            return np.nan
    
    def calculate_iv_percentile(self, 
                              current_iv: float, 
                              iv_history: pd.Series) -> float:
        """
        Calculate IV Percentile (percentage of days IV was below current IV).
        
        Args:
            current_iv: Current implied volatility
            iv_history: Series of historical implied volatility values
            
        Returns:
            IV Percentile (0-1)
        """
        try:
            if iv_history.empty:
                return np.nan
                
            # Count days below current IV
            days_below = (iv_history < current_iv).sum()
            
            # Calculate percentile
            iv_percentile = days_below / len(iv_history)
            
            return iv_percentile
            
        except Exception as e:
            logger.error(f"Error calculating IV Percentile: {e}")
            return np.nan
    
    def calculate_iv_metrics(self, 
                           options_df: pd.DataFrame, 
                           iv_history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate IV Rank and IV Percentile for options.
        
        Args:
            options_df: DataFrame containing options data
            iv_history_df: DataFrame containing historical IV data
            
        Returns:
            DataFrame with added IV metrics columns
        """
        if options_df.empty or iv_history_df.empty:
            return options_df
            
        try:
            # Make a copy to avoid modifying the original
            df = options_df.copy()
            
            # Check if we have the required columns
            if 'implied_volatility' not in df.columns:
                logger.error("Missing 'implied_volatility' column for IV metrics calculation")
                return df
            
            # Get the latest IV for each expiration and strike
            iv_history = iv_history_df['implied_volatility']
            
            # Calculate IV Rank and IV Percentile for each option
            df['iv_rank'] = df['implied_volatility'].apply(
                lambda x: self.calculate_iv_rank(x, iv_history)
            )
            
            df['iv_percentile'] = df['implied_volatility'].apply(
                lambda x: self.calculate_iv_percentile(x, iv_history)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating IV metrics: {e}")
            return options_df
    
    def calculate_iv_skew(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate IV Skew (difference in IV between OTM puts and calls).
        
        Args:
            options_df: DataFrame containing options data
            
        Returns:
            DataFrame with added IV skew metrics
        """
        if options_df.empty:
            return options_df
            
        try:
            # Make a copy to avoid modifying the original
            df = options_df.copy()
            
            # Check if we have the required columns
            required_cols = ['strike', 'right', 'implied_volatility', 'underlying_price']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns for IV skew calculation: {missing_cols}")
                return df
            
            # Calculate moneyness (strike / underlying price)
            df['moneyness'] = df['strike'] / df['underlying_price']
            
            # Separate calls and puts
            calls = df[df['right'].isin(['C', 'c', 'CALL', 'call'])]
            puts = df[df['right'].isin(['P', 'p', 'PUT', 'put'])]
            
            # Group by expiration
            for expiration in df['expiration'].unique():
                exp_calls = calls[calls['expiration'] == expiration]
                exp_puts = puts[puts['expiration'] == expiration]
                
                if exp_calls.empty or exp_puts.empty:
                    continue
                
                # Find ATM options (closest to moneyness = 1)
                atm_call = exp_calls.iloc[(exp_calls['moneyness'] - 1).abs().argsort()[:1]]
                atm_put = exp_puts.iloc[(exp_puts['moneyness'] - 1).abs().argsort()[:1]]
                
                if atm_call.empty or atm_put.empty:
                    continue
                
                # Get ATM IV
                atm_call_iv = atm_call['implied_volatility'].values[0]
                atm_put_iv = atm_put['implied_volatility'].values[0]
                
                # Calculate ATM skew
                atm_skew = atm_put_iv - atm_call_iv
                
                # Find 5% OTM options
                otm_call_moneyness = 1.05
                otm_put_moneyness = 0.95
                
                otm_call = exp_calls.iloc[(exp_calls['moneyness'] - otm_call_moneyness).abs().argsort()[:1]]
                otm_put = exp_puts.iloc[(exp_puts['moneyness'] - otm_put_moneyness).abs().argsort()[:1]]
                
                if otm_call.empty or otm_put.empty:
                    continue
                
                # Get OTM IV
                otm_call_iv = otm_call['implied_volatility'].values[0]
                otm_put_iv = otm_put['implied_volatility'].values[0]
                
                # Calculate OTM skew
                otm_skew = otm_put_iv - otm_call_iv
                
                # Add skew values to all options with this expiration
                mask = df['expiration'] == expiration
                df.loc[mask, 'atm_skew'] = atm_skew
                df.loc[mask, 'otm_skew'] = otm_skew
                df.loc[mask, 'skew_ratio'] = otm_skew / atm_skew if atm_skew != 0 else np.nan
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating IV skew: {e}")
            return options_df
    
    def calculate_iv_hv_spread(self, options_df: pd.DataFrame, hv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate IV-HV spread (difference between implied and historical volatility).
        
        Args:
            options_df: DataFrame containing options data
            hv_df: DataFrame containing historical volatility data
            
        Returns:
            DataFrame with added IV-HV spread columns
        """
        if options_df.empty or hv_df.empty:
            return options_df
            
        try:
            # Make a copy to avoid modifying the original
            df = options_df.copy()
            
            # Check if we have the required columns
            if 'implied_volatility' not in df.columns:
                logger.error("Missing 'implied_volatility' column for IV-HV spread calculation")
                return df
            
            # Get the latest HV values
            latest_hv = hv_df.iloc[-1]
            
            # Calculate IV-HV spread for each lookback period
            for period in self.lookback_periods:
                hv_col = f'hv_{period}'
                
                if hv_col in latest_hv:
                    hv_value = latest_hv[hv_col]
                    
                    # Calculate spread
                    df[f'iv_hv_{period}_spread'] = df['implied_volatility'] - hv_value
                    
                    # Calculate spread as percentage
                    if hv_value != 0:
                        df[f'iv_hv_{period}_spread_pct'] = (df['implied_volatility'] - hv_value) / hv_value
                    else:
                        df[f'iv_hv_{period}_spread_pct'] = np.nan
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating IV-HV spread: {e}")
            return options_df
    
    def calculate_term_structure(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility term structure (IV across different expirations).
        
        Args:
            options_df: DataFrame containing options data
            
        Returns:
            DataFrame with added term structure metrics
        """
        if options_df.empty:
            return options_df
            
        try:
            # Make a copy to avoid modifying the original
            df = options_df.copy()
            
            # Check if we have the required columns
            required_cols = ['expiration', 'dte', 'implied_volatility', 'strike', 'underlying_price']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns for term structure calculation: {missing_cols}")
                return df
            
            # Calculate moneyness
            df['moneyness'] = df['strike'] / df['underlying_price']
            
            # Find ATM options for each expiration
            atm_options = []
            
            for expiration in df['expiration'].unique():
                exp_options = df[df['expiration'] == expiration]
                
                # Find the option closest to ATM
                atm_option = exp_options.iloc[(exp_options['moneyness'] - 1).abs().argsort()[:1]]
                
                if not atm_option.empty:
                    atm_options.append(atm_option)
            
            if not atm_options:
                logger.warning("No ATM options found for term structure calculation")
                return df
            
            # Create term structure DataFrame
            term_structure = pd.concat(atm_options)
            term_structure = term_structure.sort_values('dte')
            
            # Calculate term structure metrics
            if len(term_structure) > 1:
                # Calculate IV ratio between adjacent expirations
                term_structure['next_dte'] = term_structure['dte'].shift(-1)
                term_structure['next_iv'] = term_structure['implied_volatility'].shift(-1)
                
                term_structure['term_structure_ratio'] = term_structure['next_iv'] / term_structure['implied_volatility']
                
                # Calculate annualized IV difference
                term_structure['dte_diff'] = term_structure['next_dte'] - term_structure['dte']
                term_structure['iv_diff'] = term_structure['next_iv'] - term_structure['implied_volatility']
                
                term_structure['term_structure_slope'] = term_structure['iv_diff'] / term_structure['dte_diff']
                
                # Map term structure metrics back to original DataFrame
                for expiration in df['expiration'].unique():
                    ts_data = term_structure[term_structure['expiration'] == expiration]
                    
                    if not ts_data.empty:
                        mask = df['expiration'] == expiration
                        
                        if 'term_structure_ratio' in ts_data.columns:
                            df.loc[mask, 'term_structure_ratio'] = ts_data['term_structure_ratio'].values[0]
                        
                        if 'term_structure_slope' in ts_data.columns:
                            df.loc[mask, 'term_structure_slope'] = ts_data['term_structure_slope'].values[0]
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating term structure: {e}")
            return options_df


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    volatility = VolatilityMetrics()
    
    # Example: Create sample price data
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    
    price_data = {
        'date': dates,
        'close': np.random.normal(loc=100, scale=1, size=len(dates)).cumsum()
    }
    
    price_df = pd.DataFrame(price_data)
    
    # Calculate historical volatility
    hv_df = volatility.calculate_historical_volatility(price_df)
    
    # Print results
    print("Historical Volatility:")
    print(hv_df[['date', 'close', 'hv_10', 'hv_20', 'hv_60', 'hv_120']].tail())
