"""
Directional Options Strategy Module

This module implements a directional options trading strategy based on
ML predictions for price movement and volatility changes.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path to import config and base strategy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.strategy_base import StrategyBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DirectionalOptionsStrategy(StrategyBase):
    """
    Strategy for directional options trading based on ML predictions.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the directional options strategy with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        # Initialize base strategy
        super().__init__(config_path)
        
        # Load strategy-specific configuration
        self.config = self._load_config(config_path)
        self.strategy_config = self.config.get('strategies', {}).get('directional', {})
        
        # Strategy metadata
        self.name = "DirectionalOptionsStrategy"
        self.description = "ML-based directional options trading strategy"
        
        # Strategy parameters
        self.direction_threshold = self.strategy_config.get('direction_threshold', 0.65)
        self.iv_change_threshold = self.strategy_config.get('iv_change_threshold', 0.05)
        self.min_expected_return = self.strategy_config.get('min_expected_return', 0.2)
        self.preferred_delta = self.strategy_config.get('preferred_delta', 0.3)
        self.delta_range = self.strategy_config.get('delta_range', 0.15)
        self.use_iv_filter = self.strategy_config.get('use_iv_filter', True)
        self.iv_rank_threshold = self.strategy_config.get('iv_rank_threshold', 0.3)
        self.use_earnings_filter = self.strategy_config.get('use_earnings_filter', True)
        self.earnings_buffer_days = self.strategy_config.get('earnings_buffer_days', 5)
    
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
                'strategies': {
                    'directional': {
                        'direction_threshold': 0.65,
                        'iv_change_threshold': 0.05,
                        'min_expected_return': 0.2,
                        'preferred_delta': 0.3,
                        'delta_range': 0.15,
                        'use_iv_filter': True,
                        'iv_rank_threshold': 0.3,
                        'use_earnings_filter': True,
                        'earnings_buffer_days': 5
                    }
                }
            }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on ML predictions.
        
        Args:
            data: DataFrame containing market data, features, and predictions
            
        Returns:
            DataFrame with added signal columns
        """
        if data.empty:
            return data
            
        try:
            # Create a copy to avoid modifying the original
            result = data.copy()
            
            # Check if we have the required columns
            required_cols = ['predicted_direction', 'predicted_up_probability', 'predicted_iv_change']
            missing_cols = [col for col in required_cols if col not in result.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns for signal generation: {missing_cols}")
                # Add neutral signal column
                result['signal'] = 0
                return result
            
            # Initialize signal column (0 = neutral, 1 = buy call, -1 = buy put)
            result['signal'] = 0
            
            # Generate signals based on direction prediction
            # Buy call (1) when predicted_up_probability > threshold
            result.loc[
                result['predicted_up_probability'] >= self.direction_threshold,
                'signal'
            ] = 1
            
            # Buy put (-1) when predicted_up_probability < (1 - threshold)
            result.loc[
                result['predicted_up_probability'] <= (1 - self.direction_threshold),
                'signal'
            ] = -1
            
            # Add signal strength based on prediction confidence
            result['signal_strength'] = result['predicted_up_probability'].apply(
                lambda p: abs(p - 0.5) * 2  # Scale to 0-1 range
            )
            
            # Apply IV change filter if enabled
            if self.use_iv_filter and 'iv_rank' in result.columns:
                # For calls, prefer low IV rank (potential for IV expansion)
                result.loc[
                    (result['signal'] == 1) & (result['iv_rank'] > self.iv_rank_threshold),
                    'signal'
                ] = 0
                
                # For puts, prefer high IV rank (potential for IV contraction)
                result.loc[
                    (result['signal'] == -1) & (result['iv_rank'] < (1 - self.iv_rank_threshold)),
                    'signal'
                ] = 0
            
            # Apply earnings filter if enabled
            if self.use_earnings_filter and 'days_to_earnings' in result.columns:
                # Avoid trading close to earnings
                result.loc[
                    result['days_to_earnings'] <= self.earnings_buffer_days,
                    'signal'
                ] = 0
            
            # Calculate expected return based on predictions
            if 'delta' in result.columns and 'vega' in result.columns and 'underlying_price' in result.columns:
                # For calls (signal = 1)
                call_mask = result['signal'] == 1
                if call_mask.any():
                    # Expected price change (as percentage)
                    price_change_pct = 0.01  # Default 1% move
                    
                    # Expected vol change
                    vol_change = result.loc[call_mask, 'predicted_iv_change']
                    
                    # Calculate expected return
                    result.loc[call_mask] = self.calculate_expected_return(
                        result.loc[call_mask],
                        price_change_pct,
                        vol_change
                    )
                
                # For puts (signal = -1)
                put_mask = result['signal'] == -1
                if put_mask.any():
                    # Expected price change (as percentage)
                    price_change_pct = -0.01  # Default -1% move
                    
                    # Expected vol change
                    vol_change = result.loc[put_mask, 'predicted_iv_change']
                    
                    # Calculate expected return
                    result.loc[put_mask] = self.calculate_expected_return(
                        result.loc[put_mask],
                        price_change_pct,
                        vol_change
                    )
                
                # Filter by minimum expected return
                result.loc[
                    result['expected_return'] < self.min_expected_return,
                    'signal'
                ] = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            # Return original data with neutral signal
            data['signal'] = 0
            return data
    
    def select_best_options(self, options_df: pd.DataFrame, signal: int) -> pd.DataFrame:
        """
        Select the best options contracts based on strategy criteria.
        
        Args:
            options_df: DataFrame containing options data
            signal: Signal direction (1 for calls, -1 for puts)
            
        Returns:
            DataFrame with selected options
        """
        if options_df.empty:
            return options_df
            
        try:
            # Filter options by basic criteria
            filtered_options = self.filter_options_by_criteria(options_df)
            
            if filtered_options.empty:
                return filtered_options
            
            # Filter by option type based on signal
            if 'right' in filtered_options.columns:
                if signal == 1:  # Calls
                    filtered_options = filtered_options[
                        filtered_options['right'].isin(['C', 'c', 'CALL', 'call'])
                    ]
                elif signal == -1:  # Puts
                    filtered_options = filtered_options[
                        filtered_options['right'].isin(['P', 'p', 'PUT', 'put'])
                    ]
            
            if filtered_options.empty:
                return filtered_options
            
            # Filter by delta range
            if 'delta' in filtered_options.columns:
                delta_min = self.preferred_delta - self.delta_range
                delta_max = self.preferred_delta + self.delta_range
                
                if signal == 1:  # Calls
                    filtered_options = filtered_options[
                        (filtered_options['delta'] >= delta_min) &
                        (filtered_options['delta'] <= delta_max)
                    ]
                elif signal == -1:  # Puts
                    # For puts, delta is negative
                    filtered_options = filtered_options[
                        (filtered_options['delta'] >= -delta_max) &
                        (filtered_options['delta'] <= -delta_min)
                    ]
            
            if filtered_options.empty:
                return filtered_options
            
            # Rank options by criteria
            if 'expected_return' in filtered_options.columns:
                # Rank by expected return
                filtered_options = filtered_options.sort_values('expected_return', ascending=False)
            elif 'delta' in filtered_options.columns:
                # Rank by closest to preferred delta
                filtered_options['delta_distance'] = abs(
                    filtered_options['delta'] - (self.preferred_delta if signal == 1 else -self.preferred_delta)
                )
                filtered_options = filtered_options.sort_values('delta_distance')
            
            return filtered_options
            
        except Exception as e:
            logger.error(f"Error selecting best options: {e}")
            return options_df
    
    def generate_trade_plan(self, 
                          signals: pd.DataFrame, 
                          options_data: Dict[str, pd.DataFrame], 
                          account_value: float) -> pd.DataFrame:
        """
        Generate a complete trade plan based on signals and available options.
        
        Args:
            signals: DataFrame containing trading signals
            options_data: Dictionary mapping symbols to options DataFrames
            account_value: Current account value
            
        Returns:
            DataFrame with trade plan
        """
        if signals.empty:
            return pd.DataFrame()
            
        try:
            # Filter signals to only include non-zero signals
            active_signals = signals[signals['signal'] != 0]
            
            if active_signals.empty:
                return pd.DataFrame()
            
            # Initialize trade plan
            trade_plans = []
            
            # Process each signal
            for _, row in active_signals.iterrows():
                symbol = row['symbol']
                signal = row['signal']
                
                # Check if we have options data for this symbol
                if symbol not in options_data or options_data[symbol].empty:
                    continue
                
                # Select best options for this signal
                options = self.select_best_options(options_data[symbol], signal)
                
                if options.empty:
                    continue
                
                # Get the best option
                best_option = options.iloc[0]
                
                # Calculate position size
                if 'option_price' in best_option:
                    position_size = self.calculate_position_size(
                        account_value, 
                        best_option['option_price']
                    )
                else:
                    position_size = 1  # Default
                
                # Create trade plan entry
                trade_plan = {
                    'symbol': symbol,
                    'underlying': row.get('underlying', symbol),
                    'signal': signal,
                    'option_symbol': best_option.get('option_symbol', ''),
                    'strike': best_option.get('strike', 0),
                    'expiration': best_option.get('expiration', ''),
                    'right': best_option.get('right', 'C' if signal == 1 else 'P'),
                    'option_price': best_option.get('option_price', 0),
                    'delta': best_option.get('delta', 0),
                    'gamma': best_option.get('gamma', 0),
                    'theta': best_option.get('theta', 0),
                    'vega': best_option.get('vega', 0),
                    'iv': best_option.get('implied_volatility', 0),
                    'position_size': position_size,
                    'expected_return': best_option.get('expected_return', 0),
                    'signal_strength': row.get('signal_strength', 0),
                    'stop_loss': best_option.get('option_price', 0) * (1 - self.stop_loss_pct),
                    'take_profit': best_option.get('option_price', 0) * (1 + self.take_profit_pct)
                }
                
                trade_plans.append(trade_plan)
            
            # Convert to DataFrame
            if trade_plans:
                trade_plan_df = pd.DataFrame(trade_plans)
                
                # Rank by expected return or signal strength
                if 'expected_return' in trade_plan_df.columns:
                    trade_plan_df = trade_plan_df.sort_values('expected_return', ascending=False)
                elif 'signal_strength' in trade_plan_df.columns:
                    trade_plan_df = trade_plan_df.sort_values('signal_strength', ascending=False)
                
                return trade_plan_df
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error generating trade plan: {e}")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    strategy = DirectionalOptionsStrategy()
    
    # Example: Create sample data with predictions
    data = {
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        'underlying_price': [150.0, 280.0, 2800.0, 3200.0],
        'predicted_direction': [1, 0, 1, 0],
        'predicted_up_probability': [0.75, 0.55, 0.80, 0.40],
        'predicted_iv_change': [0.02, -0.01, 0.03, -0.02],
        'iv_rank': [0.2, 0.6, 0.3, 0.7],
        'days_to_earnings': [20, 3, 15, 30]
    }
    
    df = pd.DataFrame(data)
    
    # Generate signals
    signals = strategy.generate_signals(df)
    
    print("Generated Signals:")
    print(signals[['symbol', 'predicted_up_probability', 'signal', 'signal_strength']])
