"""
Strategy Base Module

This module provides the base class for all trading strategies in the system.
It handles common functionality like signal generation, position sizing, and risk management.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrategyBase:
    """
    Base class for all trading strategies.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the strategy with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.strategy_config = self.config.get('strategies', {}).get('base', {})
        self.name = "BaseStrategy"
        self.description = "Base strategy class"
        self.risk_per_trade = self.strategy_config.get('risk_per_trade', 0.01)  # 1% of account
        self.max_position_size = self.strategy_config.get('max_position_size', 0.05)  # 5% of account
        self.max_correlated_positions = self.strategy_config.get('max_correlated_positions', 3)
        self.position_sizing_method = self.strategy_config.get('position_sizing', 'risk_based')
        self.stop_loss_pct = self.strategy_config.get('stop_loss_pct', 0.5)  # 50% of premium
        self.take_profit_pct = self.strategy_config.get('take_profit_pct', 1.0)  # 100% of premium
        self.max_days_to_expiry = self.strategy_config.get('max_days_to_expiry', 45)
        self.min_days_to_expiry = self.strategy_config.get('min_days_to_expiry', 7)
    
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
                    'base': {
                        'risk_per_trade': 0.01,
                        'max_position_size': 0.05,
                        'max_correlated_positions': 3,
                        'position_sizing': 'risk_based',
                        'stop_loss_pct': 0.5,
                        'take_profit_pct': 1.0,
                        'max_days_to_expiry': 45,
                        'min_days_to_expiry': 7
                    }
                }
            }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on strategy rules.
        
        Args:
            data: DataFrame containing market data and features
            
        Returns:
            DataFrame with added signal columns
        """
        # This is a base method to be overridden by specific strategies
        logger.warning("Using base generate_signals method, should be overridden by specific strategy")
        
        if data.empty:
            return data
            
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Add signal column (neutral by default)
        result['signal'] = 0
        
        return result
    
    def calculate_position_size(self, 
                              account_value: float, 
                              option_price: float, 
                              risk_level: float = None) -> int:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account_value: Current account value
            option_price: Price of the option
            risk_level: Optional override for risk_per_trade
            
        Returns:
            Number of contracts to trade
        """
        if risk_level is None:
            risk_level = self.risk_per_trade
            
        try:
            # Calculate dollar risk amount
            risk_amount = account_value * risk_level
            
            # Calculate max position value
            max_position_value = account_value * self.max_position_size
            
            if self.position_sizing_method == 'risk_based':
                # Risk-based position sizing
                # Risk amount is the maximum we're willing to lose on the trade
                # For long options, this is the premium paid
                # For short options, this is more complex and depends on stop loss
                
                # For simplicity, assume long options for now
                position_value = risk_amount
                
            elif self.position_sizing_method == 'kelly':
                # Kelly criterion position sizing
                # This requires win rate and risk/reward ratio
                # For now, use a simplified version
                position_value = risk_amount * 2  # Simplified
                
            else:
                # Default to fixed fractional
                position_value = risk_amount
            
            # Limit position value to max position size
            position_value = min(position_value, max_position_value)
            
            # Calculate number of contracts
            num_contracts = int(position_value / (option_price * 100))
            
            # Ensure at least 1 contract
            num_contracts = max(1, num_contracts)
            
            return num_contracts
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1  # Default to 1 contract on error
    
    def apply_risk_management(self, 
                            signals: pd.DataFrame, 
                            positions: pd.DataFrame, 
                            account_value: float) -> pd.DataFrame:
        """
        Apply risk management rules to signals.
        
        Args:
            signals: DataFrame containing trading signals
            positions: DataFrame containing current positions
            account_value: Current account value
            
        Returns:
            DataFrame with filtered signals
        """
        if signals.empty:
            return signals
            
        try:
            # Create a copy to avoid modifying the original
            filtered_signals = signals.copy()
            
            # Filter out signals that would exceed max correlated positions
            if not positions.empty:
                # Group positions by underlying
                positions_by_underlying = positions.groupby('underlying').size()
                
                # Filter signals
                for underlying, count in positions_by_underlying.items():
                    if count >= self.max_correlated_positions:
                        # Remove signals for this underlying
                        filtered_signals = filtered_signals[
                            filtered_signals['underlying'] != underlying
                        ]
            
            # Calculate position sizes
            if 'option_price' in filtered_signals.columns:
                filtered_signals['position_size'] = filtered_signals['option_price'].apply(
                    lambda price: self.calculate_position_size(account_value, price)
                )
            
            # Add stop loss and take profit levels
            if 'option_price' in filtered_signals.columns:
                filtered_signals['stop_loss'] = filtered_signals['option_price'] * (1 - self.stop_loss_pct)
                filtered_signals['take_profit'] = filtered_signals['option_price'] * (1 + self.take_profit_pct)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error applying risk management: {e}")
            return signals
    
    def filter_options_by_criteria(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter options based on strategy criteria.
        
        Args:
            options_df: DataFrame containing options data
            
        Returns:
            DataFrame with filtered options
        """
        if options_df.empty:
            return options_df
            
        try:
            # Create a copy to avoid modifying the original
            filtered_options = options_df.copy()
            
            # Filter by days to expiry
            if 'dte' in filtered_options.columns:
                filtered_options = filtered_options[
                    (filtered_options['dte'] >= self.min_days_to_expiry) &
                    (filtered_options['dte'] <= self.max_days_to_expiry)
                ]
            
            # Filter by liquidity (if available)
            if 'open_interest' in filtered_options.columns:
                filtered_options = filtered_options[
                    filtered_options['open_interest'] > 0
                ]
            
            # Filter by bid-ask spread (if available)
            if 'bid' in filtered_options.columns and 'ask' in filtered_options.columns:
                filtered_options['bid_ask_spread_pct'] = (
                    (filtered_options['ask'] - filtered_options['bid']) / 
                    ((filtered_options['bid'] + filtered_options['ask']) / 2)
                )
                
                # Filter out options with wide spreads (>10%)
                filtered_options = filtered_options[
                    filtered_options['bid_ask_spread_pct'] <= 0.1
                ]
            
            return filtered_options
            
        except Exception as e:
            logger.error(f"Error filtering options: {e}")
            return options_df
    
    def rank_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Rank signals by quality or expected return.
        
        Args:
            signals: DataFrame containing trading signals
            
        Returns:
            DataFrame with ranked signals
        """
        if signals.empty:
            return signals
            
        try:
            # Create a copy to avoid modifying the original
            ranked_signals = signals.copy()
            
            # Rank by signal strength (if available)
            if 'signal_strength' in ranked_signals.columns:
                ranked_signals = ranked_signals.sort_values('signal_strength', ascending=False)
            
            # Rank by expected return (if available)
            elif 'expected_return' in ranked_signals.columns:
                ranked_signals = ranked_signals.sort_values('expected_return', ascending=False)
            
            # Rank by probability (if available)
            elif 'probability' in ranked_signals.columns:
                ranked_signals = ranked_signals.sort_values('probability', ascending=False)
            
            return ranked_signals
            
        except Exception as e:
            logger.error(f"Error ranking signals: {e}")
            return signals
    
    def calculate_expected_return(self, 
                                option_data: pd.DataFrame, 
                                price_prediction: float, 
                                vol_prediction: float) -> pd.DataFrame:
        """
        Calculate expected return for options based on price and volatility predictions.
        
        Args:
            option_data: DataFrame containing options data
            price_prediction: Predicted price change (percentage)
            vol_prediction: Predicted volatility change (percentage)
            
        Returns:
            DataFrame with added expected return column
        """
        if option_data.empty:
            return option_data
            
        try:
            # Create a copy to avoid modifying the original
            result = option_data.copy()
            
            # Check if we have the required columns
            required_cols = ['delta', 'vega', 'option_price']
            missing_cols = [col for col in required_cols if col not in result.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns for expected return calculation: {missing_cols}")
                return option_data
            
            # Calculate expected price change contribution
            if 'underlying_price' in result.columns:
                price_change_amount = result['underlying_price'] * price_prediction
                delta_contribution = result['delta'] * price_change_amount
            else:
                delta_contribution = 0
            
            # Calculate expected volatility change contribution
            vega_contribution = result['vega'] * vol_prediction * 100  # Vega is per 1% vol change
            
            # Calculate total expected change
            expected_change = delta_contribution + vega_contribution
            
            # Calculate expected return percentage
            result['expected_return'] = expected_change / (result['option_price'] * 100)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating expected return: {e}")
            return option_data
    
    def generate_exit_signals(self, 
                            positions: pd.DataFrame, 
                            current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate exit signals for current positions.
        
        Args:
            positions: DataFrame containing current positions
            current_data: DataFrame containing current market data
            
        Returns:
            DataFrame with exit signals
        """
        if positions.empty:
            return pd.DataFrame()
            
        try:
            # Create a copy to avoid modifying the original
            exit_signals = positions.copy()
            
            # Add exit signal column (0 = hold, 1 = exit)
            exit_signals['exit_signal'] = 0
            
            # Check stop loss and take profit levels
            if 'current_price' in exit_signals.columns:
                # Stop loss exit
                exit_signals.loc[
                    exit_signals['current_price'] <= exit_signals['stop_loss'],
                    'exit_signal'
                ] = 1
                
                # Take profit exit
                exit_signals.loc[
                    exit_signals['current_price'] >= exit_signals['take_profit'],
                    'exit_signal'
                ] = 1
            
            # Check time-based exits
            if 'dte' in exit_signals.columns:
                # Exit positions close to expiry
                exit_signals.loc[
                    exit_signals['dte'] <= 2,  # Exit positions with 2 or fewer days to expiry
                    'exit_signal'
                ] = 1
            
            # Check for signal reversal
            if 'signal' in exit_signals.columns and 'current_signal' in current_data.columns:
                # Merge current signals with positions
                merged = pd.merge(
                    exit_signals,
                    current_data[['symbol', 'current_signal']],
                    on='symbol',
                    how='left'
                )
                
                # Exit on signal reversal
                exit_signals.loc[
                    merged['signal'] != merged['current_signal'],
                    'exit_signal'
                ] = 1
            
            # Filter to only include positions with exit signals
            exit_signals = exit_signals[exit_signals['exit_signal'] == 1]
            
            return exit_signals
            
        except Exception as e:
            logger.error(f"Error generating exit signals: {e}")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    strategy = StrategyBase()
    
    # Example: Calculate position size
    account_value = 100000
    option_price = 2.50
    
    position_size = strategy.calculate_position_size(account_value, option_price)
    print(f"Position size: {position_size} contracts")
    
    # Example: Filter options
    options_data = {
        'symbol': ['AAPL_220121C150', 'AAPL_220218C150', 'AAPL_220318C150'],
        'dte': [5, 30, 60],
        'open_interest': [100, 200, 50],
        'bid': [2.0, 2.5, 3.0],
        'ask': [2.1, 2.7, 3.3]
    }
    
    options_df = pd.DataFrame(options_data)
    filtered_options = strategy.filter_options_by_criteria(options_df)
    
    print("\nFiltered options:")
    print(filtered_options)
