"""
Multi-Leg Strategy Module

This module implements multi-legged option strategies including spreads,
condors, butterflies, and custom combinations.
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


class MultiLegStrategy(StrategyBase):
    """
    Strategy for multi-legged options trading.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the multi-leg strategy with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        # Initialize base strategy
        super().__init__(config_path)
        
        # Load strategy-specific configuration
        self.config = self._load_config(config_path)
        self.strategy_config = self.config.get('strategies', {}).get('multi_leg', {})
        
        # Strategy metadata
        self.name = "MultiLegStrategy"
        self.description = "Multi-legged options trading strategy"
        
        # Strategy parameters
        self.strategy_type = self.strategy_config.get('strategy_type', 'vertical_spread')
        self.spread_width = self.strategy_config.get('spread_width', 5)
        self.min_credit_debit_ratio = self.strategy_config.get('min_credit_debit_ratio', 0.3)
        self.max_risk_reward_ratio = self.strategy_config.get('max_risk_reward_ratio', 2.0)
        self.preferred_delta = self.strategy_config.get('preferred_delta', 0.3)
        self.use_ml_predictions = self.strategy_config.get('use_ml_predictions', True)
        self.min_expected_return = self.strategy_config.get('min_expected_return', 0.15)
        self.min_probability_of_profit = self.strategy_config.get('min_probability_of_profit', 0.6)
    
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
                    'multi_leg': {
                        'strategy_type': 'vertical_spread',
                        'spread_width': 5,
                        'min_credit_debit_ratio': 0.3,
                        'max_risk_reward_ratio': 2.0,
                        'preferred_delta': 0.3,
                        'use_ml_predictions': True,
                        'min_expected_return': 0.15,
                        'min_probability_of_profit': 0.6
                    }
                }
            }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for multi-leg strategies.
        
        Args:
            data: DataFrame containing market data and features
            
        Returns:
            DataFrame with added signal columns
        """
        if data.empty:
            return data
            
        try:
            # Create a copy to avoid modifying the original
            result = data.copy()
            
            # Initialize signal column (0 = neutral, 1 = bullish, -1 = bearish)
            result['signal'] = 0
            
            # Check strategy type
            if self.strategy_type == 'vertical_spread':
                # Generate vertical spread signals
                result = self._generate_vertical_spread_signals(result)
            elif self.strategy_type == 'iron_condor':
                # Generate iron condor signals
                result = self._generate_iron_condor_signals(result)
            elif self.strategy_type == 'butterfly':
                # Generate butterfly signals
                result = self._generate_butterfly_signals(result)
            elif self.strategy_type == 'calendar_spread':
                # Generate calendar spread signals
                result = self._generate_calendar_spread_signals(result)
            else:
                # Default to vertical spread
                result = self._generate_vertical_spread_signals(result)
            
            # Apply ML predictions if enabled
            if self.use_ml_predictions:
                # For directional strategies
                if self.strategy_type in ['vertical_spread', 'calendar_spread']:
                    if 'predicted_up_probability' in result.columns:
                        # Adjust bullish signals based on predicted direction
                        result.loc[
                            (result['signal'] == 1) & (result['predicted_up_probability'] < 0.5),
                            'signal'
                        ] = 0
                        
                        # Adjust bearish signals based on predicted direction
                        result.loc[
                            (result['signal'] == -1) & (result['predicted_up_probability'] > 0.5),
                            'signal'
                        ] = 0
                
                # For volatility-based strategies
                if self.strategy_type in ['iron_condor', 'butterfly']:
                    if 'predicted_iv_change' in result.columns:
                        # For iron condors and butterflies, we want IV to decrease or stay stable
                        result.loc[
                            (result['signal'] != 0) & (result['predicted_iv_change'] > 0.02),
                            'signal'
                        ] = 0
            
            # Calculate signal strength
            result['signal_strength'] = 0.0
            
            # For directional strategies
            if self.strategy_type in ['vertical_spread', 'calendar_spread'] and 'predicted_up_probability' in result.columns:
                # For bullish signals, strength is proportional to predicted_up_probability
                result.loc[result['signal'] == 1, 'signal_strength'] = result.loc[result['signal'] == 1, 'predicted_up_probability']
                
                # For bearish signals, strength is proportional to (1 - predicted_up_probability)
                result.loc[result['signal'] == -1, 'signal_strength'] = 1 - result.loc[result['signal'] == -1, 'predicted_up_probability']
            
            # For volatility-based strategies
            elif self.strategy_type in ['iron_condor', 'butterfly'] and 'iv_rank' in result.columns:
                # For these strategies, we prefer high IV rank
                result.loc[result['signal'] != 0, 'signal_strength'] = result.loc[result['signal'] != 0, 'iv_rank']
            
            # Normalize signal strength to 0-1 range
            if result['signal_strength'].max() > 0:
                result['signal_strength'] = result['signal_strength'] / result['signal_strength'].max()
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating multi-leg signals: {e}")
            # Return original data with neutral signal
            data['signal'] = 0
            return data
    
    def _generate_vertical_spread_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for vertical spread strategy.
        
        Args:
            data: DataFrame containing market data and features
            
        Returns:
            DataFrame with added signal columns
        """
        result = data.copy()
        
        # For vertical spreads, we need directional bias
        if 'predicted_up_probability' in result.columns:
            # Bullish signal (bull call spread or bull put spread)
            result.loc[
                result['predicted_up_probability'] >= 0.65,  # Strong bullish prediction
                'signal'
            ] = 1
            
            # Bearish signal (bear put spread or bear call spread)
            result.loc[
                result['predicted_up_probability'] <= 0.35,  # Strong bearish prediction
                'signal'
            ] = -1
        elif 'rsi' in result.columns:
            # Alternative: Use RSI for directional bias
            # Bullish signal when RSI is oversold
            result.loc[
                result['rsi'] <= 30,
                'signal'
            ] = 1
            
            # Bearish signal when RSI is overbought
            result.loc[
                result['rsi'] >= 70,
                'signal'
            ] = -1
        
        # Consider IV environment for spread type selection
        if 'iv_rank' in result.columns:
            # For high IV environments, prefer credit spreads
            result.loc[
                (result['signal'] == 1) & (result['iv_rank'] >= 0.7),
                'spread_type'
            ] = 'bull_put'
            
            result.loc[
                (result['signal'] == -1) & (result['iv_rank'] >= 0.7),
                'spread_type'
            ] = 'bear_call'
            
            # For low IV environments, prefer debit spreads
            result.loc[
                (result['signal'] == 1) & (result['iv_rank'] < 0.7),
                'spread_type'
            ] = 'bull_call'
            
            result.loc[
                (result['signal'] == -1) & (result['iv_rank'] < 0.7),
                'spread_type'
            ] = 'bear_put'
        else:
            # Default spread types
            result.loc[result['signal'] == 1, 'spread_type'] = 'bull_call'
            result.loc[result['signal'] == -1, 'spread_type'] = 'bear_put'
        
        return result
    
    def _generate_iron_condor_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for iron condor strategy.
        
        Args:
            data: DataFrame containing market data and features
            
        Returns:
            DataFrame with added signal columns
        """
        result = data.copy()
        
        # Iron condors are neutral strategies that benefit from low volatility
        # They work best in high IV environments with expected decrease in IV
        
        # Check for high IV rank
        if 'iv_rank' in result.columns:
            # Generate signal for high IV rank
            result.loc[
                result['iv_rank'] >= 0.7,  # High IV environment
                'signal'
            ] = 2  # Neutral signal (2 for iron condor)
        
        # Check for expected IV decrease
        if 'predicted_iv_change' in result.columns:
            # Remove signal if IV is expected to increase
            result.loc[
                (result['signal'] == 2) & (result['predicted_iv_change'] > 0),
                'signal'
            ] = 0
        
        # Check for low expected price movement
        if 'predicted_up_probability' in result.columns:
            # Remove signal if strong directional move is expected
            result.loc[
                (result['signal'] == 2) & 
                ((result['predicted_up_probability'] >= 0.7) | (result['predicted_up_probability'] <= 0.3)),
                'signal'
            ] = 0
        
        # Set spread type
        result.loc[result['signal'] == 2, 'spread_type'] = 'iron_condor'
        
        return result
    
    def _generate_butterfly_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for butterfly strategy.
        
        Args:
            data: DataFrame containing market data and features
            
        Returns:
            DataFrame with added signal columns
        """
        result = data.copy()
        
        # Butterflies are precise strategies that benefit from low volatility
        # and price movement to a specific target
        
        # Check for moderate to high IV rank
        if 'iv_rank' in result.columns:
            # Generate signal for moderate to high IV rank
            result.loc[
                result['iv_rank'] >= 0.5,  # Moderate to high IV environment
                'signal'
            ] = 3  # Butterfly signal
        
        # Check for expected IV decrease
        if 'predicted_iv_change' in result.columns:
            # Remove signal if IV is expected to increase significantly
            result.loc[
                (result['signal'] == 3) & (result['predicted_iv_change'] > 0.05),
                'signal'
            ] = 0
        
        # Check for expected price stability
        if 'predicted_up_probability' in result.columns:
            # For long call butterfly, we want price to move up slightly
            result.loc[
                (result['signal'] == 3) & (result['predicted_up_probability'] >= 0.55) & (result['predicted_up_probability'] <= 0.65),
                'spread_type'
            ] = 'call_butterfly'
            
            # For long put butterfly, we want price to move down slightly
            result.loc[
                (result['signal'] == 3) & (result['predicted_up_probability'] >= 0.35) & (result['predicted_up_probability'] <= 0.45),
                'spread_type'
            ] = 'put_butterfly'
            
            # For iron butterfly, we want price to stay stable
            result.loc[
                (result['signal'] == 3) & (result['predicted_up_probability'] > 0.45) & (result['predicted_up_probability'] < 0.55),
                'spread_type'
            ] = 'iron_butterfly'
            
            # Remove signal if strong directional move is expected
            result.loc[
                (result['signal'] == 3) & 
                ((result['predicted_up_probability'] > 0.65) | (result['predicted_up_probability'] < 0.35)),
                'signal'
            ] = 0
        else:
            # Default to iron butterfly
            result.loc[result['signal'] == 3, 'spread_type'] = 'iron_butterfly'
        
        return result
    
    def _generate_calendar_spread_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for calendar spread strategy.
        
        Args:
            data: DataFrame containing market data and features
            
        Returns:
            DataFrame with added signal columns
        """
        result = data.copy()
        
        # Calendar spreads benefit from increasing IV and time decay
        # They work well when IV is low with expected increase
        
        # Check for low IV rank
        if 'iv_rank' in result.columns:
            # Generate signal for low IV rank
            result.loc[
                result['iv_rank'] <= 0.3,  # Low IV environment
                'signal'
            ] = 4  # Calendar spread signal
        
        # Check for expected IV increase
        if 'predicted_iv_change' in result.columns:
            # Remove signal if IV is expected to decrease
            result.loc[
                (result['signal'] == 4) & (result['predicted_iv_change'] < 0),
                'signal'
            ] = 0
            
            # Strengthen signal if significant IV increase is expected
            result.loc[
                (result['signal'] == 4) & (result['predicted_iv_change'] >= 0.05),
                'signal_strength'
            ] = result.loc[
                (result['signal'] == 4) & (result['predicted_iv_change'] >= 0.05),
                'predicted_iv_change'
            ] * 10  # Scale to reasonable range
        
        # Check for directional bias
        if 'predicted_up_probability' in result.columns:
            # For call calendar, we want slight bullish bias
            result.loc[
                (result['signal'] == 4) & (result['predicted_up_probability'] >= 0.5),
                'spread_type'
            ] = 'call_calendar'
            
            # For put calendar, we want slight bearish bias
            result.loc[
                (result['signal'] == 4) & (result['predicted_up_probability'] < 0.5),
                'spread_type'
            ] = 'put_calendar'
        else:
            # Default to call calendar
            result.loc[result['signal'] == 4, 'spread_type'] = 'call_calendar'
        
        return result
    
    def select_strategy_options(self, options_df: pd.DataFrame, signal: int, spread_type: str) -> Dict[str, pd.DataFrame]:
        """
        Select the appropriate options for the multi-leg strategy.
        
        Args:
            options_df: DataFrame containing options data
            signal: Signal direction
            spread_type: Type of spread to create
            
        Returns:
            Dictionary with selected options for each leg
        """
        if options_df.empty:
            return {}
            
        try:
            # Filter options by basic criteria
            filtered_options = self.filter_options_by_criteria(options_df)
            
            if filtered_options.empty:
                return {}
            
            # Strategy-specific filtering
            if spread_type == 'bull_call':
                return self._select_bull_call_spread_options(filtered_options)
            elif spread_type == 'bear_put':
                return self._select_bear_put_spread_options(filtered_options)
            elif spread_type == 'bull_put':
                return self._select_bull_put_spread_options(filtered_options)
            elif spread_type == 'bear_call':
                return self._select_bear_call_spread_options(filtered_options)
            elif spread_type == 'iron_condor':
                return self._select_iron_condor_options(filtered_options)
            elif spread_type in ['call_butterfly', 'put_butterfly', 'iron_butterfly']:
                return self._select_butterfly_options(filtered_options, spread_type)
            elif spread_type in ['call_calendar', 'put_calendar']:
                return self._select_calendar_spread_options(filtered_options, spread_type)
            else:
                logger.warning(f"Unknown spread type: {spread_type}")
                return {}
            
        except Exception as e:
            logger.error(f"Error selecting strategy options: {e}")
            return {}
    
    def _select_bull_call_spread_options(self, options_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Select options for a bull call spread.
        
        Args:
            options_df: DataFrame containing filtered options data
            
        Returns:
            Dictionary with selected options for each leg
        """
        try:
            # Filter for calls
            calls = options_df[options_df['right'].isin(['C', 'c', 'CALL', 'call'])]
            
            if calls.empty:
                return {}
            
            # For bull call spread, we need:
            # 1. Buy lower strike call
            # 2. Sell higher strike call (spread_width away)
            
            # Find options with delta close to preferred delta
            if 'delta' in calls.columns:
                calls['delta_distance'] = abs(calls['delta'] - self.preferred_delta)
                calls = calls.sort_values('delta_distance')
                
                # Get the long call (closest to preferred delta)
                long_call = calls.iloc[0:1]
                
                if long_call.empty:
                    return {}
                
                long_strike = long_call['strike'].values[0]
                
                # Find short call with strike spread_width away
                target_short_strike = long_strike + self.spread_width
                
                # Find the closest strike
                calls['strike_distance'] = abs(calls['strike'] - target_short_strike)
                short_call = calls.sort_values('strike_distance').iloc[0:1]
                
                if short_call.empty:
                    return {}
                
                # Calculate debit and max profit
                long_price = long_call['option_price'].values[0]
                short_price = short_call['option_price'].values[0]
                debit = long_price - short_price
                
                if debit <= 0:
                    logger.warning("Invalid bull call spread: debit <= 0")
                    return {}
                
                # Calculate max profit and risk/reward ratio
                max_profit = short_call['strike'].values[0] - long_call['strike'].values[0] - debit
                risk_reward_ratio = debit / max_profit if max_profit > 0 else float('inf')
                
                if risk_reward_ratio > self.max_risk_reward_ratio:
                    logger.warning(f"Risk/reward ratio too high: {risk_reward_ratio}")
                    return {}
                
                return {
                    'long_call': long_call,
                    'short_call': short_call,
                    'debit': debit,
                    'max_profit': max_profit,
                    'risk_reward_ratio': risk_reward_ratio
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error selecting bull call spread options: {e}")
            return {}
    
    def _select_bear_put_spread_options(self, options_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Select options for a bear put spread.
        
        Args:
            options_df: DataFrame containing filtered options data
            
        Returns:
            Dictionary with selected options for each leg
        """
        try:
            # Filter for puts
            puts = options_df[options_df['right'].isin(['P', 'p', 'PUT', 'put'])]
            
            if puts.empty:
                return {}
            
            # For bear put spread, we need:
            # 1. Buy higher strike put
            # 2. Sell lower strike put (spread_width away)
            
            # Find options with delta close to preferred delta (negative for puts)
            if 'delta' in puts.columns:
                puts['delta_distance'] = abs(puts['delta'] + self.preferred_delta)  # Note the + for puts
                puts = puts.sort_values('delta_distance')
                
                # Get the long put (closest to preferred delta)
                long_put = puts.iloc[0:1]
                
                if long_put.empty:
                    return {}
                
                long_strike = long_put['strike'].values[0]
                
                # Find short put with strike spread_width away
                target_short_strike = long_strike - self.spread_width
                
                # Find the closest strike
                puts['strike_distance'] = abs(puts['strike'] - target_short_strike)
                short_put = puts.sort_values('strike_distance').iloc[0:1]
                
                if short_put.empty:
                    return {}
                
                # Calculate debit and max profit
                long_price = long_put['option_price'].values[0]
                short_price = short_put['option_price'].values[0]
                debit = long_price - short_price
                
                if debit <= 0:
                    logger.warning("Invalid bear put spread: debit <= 0")
                    return {}
                
                # Calculate max profit and risk/reward ratio
                max_profit = long_put['strike'].values[0] - short_put['strike'].values[0] - debit
                risk_reward_ratio = debit / max_profit if max_profit > 0 else float('inf')
                
                if risk_reward_ratio > self.max_risk_reward_ratio:
                    logger.warning(f"Risk/reward ratio too high: {risk_reward_ratio}")
                    return {}
                
                return {
                    'long_put': long_put,
                    'short_put': short_put,
                    'debit': debit,
                    'max_profit': max_profit,
                    'risk_reward_ratio': risk_reward_ratio
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error selecting bear put spread options: {e}")
            return {}
    
    def _select_bull_put_spread_options(self, options_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Select options for a bull put spread.
        
        Args:
            options_df: DataFrame containing filtered options data
            
        Returns:
            Dictionary with selected options for each leg
        """
        try:
            # Filter for puts
            puts = options_df[options_df['right'].isin(['P', 'p', 'PUT', 'put'])]
            
            if puts.empty:
                return {}
            
            # For bull put spread, we need:
            # 1. Sell higher strike put
            # 2. Buy lower strike put (spread_width away)
            
            # Find options with delta close to preferred delta (negative for puts)
            if 'delta' in puts.columns:
                puts['delta_distance'] = abs(puts['delta'] + self.preferred_delta)  # Note the + for puts
                puts = puts.sort_values('delta_distance')
                
                # Get the short put (closest to preferred delta)
                short_put = puts.iloc[0:1]
                
                if short_put.empty:
                    return {}
                
                short_strike = short_put['strike'].values[0]
                
                # Find long put with strike spread_width away
                target_long_strike = short_strike - self.spread_width
                
                # Find the closest strike
                puts['strike_distance'] = abs(puts['strike'] - target_long_strike)
                long_put = puts.sort_values('strike_distance').iloc[0:1]
                
                if long_put.empty:
                    return {}
                
                # Calculate credit and max profit
                short_price = short_put['option_price'].values[0]
                long_price = long_put['option_price'].values[0]
                credit = short_price - long_price
                
                if credit <= 0:
                    logger.warning("Invalid bull put spread: credit <= 0")
                    return {}
                
                # Calculate max risk and credit/risk ratio
                max_risk = short_put['strike'].values[0] - long_put['strike'].values[0] - credit
                credit_risk_ratio = credit / max_risk if max_risk > 0 else float('inf')
                
                if credit_risk_ratio < self.min_credit_debit_ratio:
                    logger.warning(f"Credit/risk ratio too low: {credit_risk_ratio}")
                    return {}
                
                return {
                    'short_put': short_put,
                    'long_put': long_put,
                    'credit': credit,
                    'max_risk': max_risk,
                    'credit_risk_ratio': credit_risk_ratio
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error selecting bull put spread options: {e}")
            return {}
    
    def _select_bear_call_spread_options(self, options_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Select options for a bear call spread.
        
        Args:
            options_df: DataFrame containing filtered options data
            
        Returns:
            Dictionary with selected options for each leg
        """
        try:
            # Filter for calls
            calls = options_df[options_df['right'].isin(['C', 'c', 'CALL', 'call'])]
            
            if calls.empty:
                return {}
            
            # For bear call spread, we need:
            # 1. Sell lower strike call
            # 2. Buy higher strike call (spread_width away)
            
            # Find options with delta close to preferred delta
            if 'delta' in calls.columns:
                calls['delta_distance'] = abs(calls['delta'] - self.preferred_delta)
                calls = calls.sort_values('delta_distance')
                
                # Get the short call (closest to preferred delta)
                short_call = calls.iloc[0:1]
                
                if short_call.empty:
                    return {}
                
                short_strike = short_call['strike'].values[0]
                
                # Find long call with strike spread_width away
                target_long_strike = short_strike + self.spread_width
                
                # Find the closest strike
                calls['strike_distance'] = abs(calls['strike'] - target_long_strike)
                long_call = calls.sort_values('strike_distance').iloc[0:1]
                
                if long_call.empty:
                    return {}
                
                # Calculate credit and max profit
                short_price = short_call['option_price'].values[0]
                long_price = long_call['option_price'].values[0]
                credit = short_price - long_price
                
                if credit <= 0:
                    logger.warning("Invalid bear call spread: credit <= 0")
                    return {}
                
                # Calculate max risk and credit/risk ratio
                max_risk = long_call['strike'].values[0] - short_call['strike'].values[0] - credit
                credit_risk_ratio = credit / max_risk if max_risk > 0 else float('inf')
                
                if credit_risk_ratio < self.min_credit_debit_ratio:
                    logger.warning(f"Credit/risk ratio too low: {credit_risk_ratio}")
                    return {}
                
                return {
                    'short_call': short_call,
                    'long_call': long_call,
                    'credit': credit,
                    'max_risk': max_risk,
                    'credit_risk_ratio': credit_risk_ratio
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error selecting bear call spread options: {e}")
            return {}
    
    def _select_iron_condor_options(self, options_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Select options for an iron condor.
        
        Args:
            options_df: DataFrame containing filtered options data
            
        Returns:
            Dictionary with selected options for each leg
        """
        try:
            # Filter for calls and puts
            calls = options_df[options_df['right'].isin(['C', 'c', 'CALL', 'call'])]
            puts = options_df[options_df['right'].isin(['P', 'p', 'PUT', 'put'])]
            
            if calls.empty or puts.empty:
                return {}
            
            # For iron condor, we need:
            # 1. Sell OTM put (bull put spread)
            # 2. Buy further OTM put (bull put spread)
            # 3. Sell OTM call (bear call spread)
            # 4. Buy further OTM call (bear call spread)
            
            # Find options with delta close to target delta
            if 'delta' in calls.columns and 'delta' in puts.columns:
                # For short call, target delta around 0.2-0.3
                calls['delta_distance'] = abs(calls['delta'] - 0.25)
                calls = calls.sort_values('delta_distance')
                
                # For short put, target delta around -0.2 to -0.3
                puts['delta_distance'] = abs(puts['delta'] + 0.25)  # Note the + for puts
                puts = puts.sort_values('delta_distance')
                
                # Get the short options
                short_call = calls.iloc[0:1]
                short_put = puts.iloc[0:1]
                
                if short_call.empty or short_put.empty:
                    return {}
                
                short_call_strike = short_call['strike'].values[0]
                short_put_strike = short_put['strike'].values[0]
                
                # Find long options with strike spread_width away
                target_long_call_strike = short_call_strike + self.spread_width
                target_long_put_strike = short_put_strike - self.spread_width
                
                # Find the closest strikes
                calls['strike_distance'] = abs(calls['strike'] - target_long_call_strike)
                puts['strike_distance'] = abs(puts['strike'] - target_long_put_strike)
                
                long_call = calls.sort_values('strike_distance').iloc[0:1]
                long_put = puts.sort_values('strike_distance').iloc[0:1]
                
                if long_call.empty or long_put.empty:
                    return {}
                
                # Calculate credit and max profit
                short_call_price = short_call['option_price'].values[0]
                long_call_price = long_call['option_price'].values[0]
                short_put_price = short_put['option_price'].values[0]
                long_put_price = long_put['option_price'].values[0]
                
                total_credit = (short_call_price - long_call_price) + (short_put_price - long_put_price)
                
                if total_credit <= 0:
                    logger.warning("Invalid iron condor: total credit <= 0")
                    return {}
                
                # Calculate max risk and credit/risk ratio
                call_spread_width = long_call['strike'].values[0] - short_call['strike'].values[0]
                put_spread_width = short_put['strike'].values[0] - long_put['strike'].values[0]
                
                max_risk = max(call_spread_width, put_spread_width) - total_credit
                credit_risk_ratio = total_credit / max_risk if max_risk > 0 else float('inf')
                
                if credit_risk_ratio < self.min_credit_debit_ratio:
                    logger.warning(f"Credit/risk ratio too low: {credit_risk_ratio}")
                    return {}
                
                return {
                    'short_call': short_call,
                    'long_call': long_call,
                    'short_put': short_put,
                    'long_put': long_put,
                    'total_credit': total_credit,
                    'max_risk': max_risk,
                    'credit_risk_ratio': credit_risk_ratio
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error selecting iron condor options: {e}")
            return {}
    
    def _select_butterfly_options(self, options_df: pd.DataFrame, spread_type: str) -> Dict[str, pd.DataFrame]:
        """
        Select options for a butterfly spread.
        
        Args:
            options_df: DataFrame containing filtered options data
            spread_type: Type of butterfly (call_butterfly, put_butterfly, iron_butterfly)
            
        Returns:
            Dictionary with selected options for each leg
        """
        try:
            # Determine option type based on spread type
            if spread_type == 'call_butterfly':
                option_type = ['C', 'c', 'CALL', 'call']
            elif spread_type == 'put_butterfly':
                option_type = ['P', 'p', 'PUT', 'put']
            elif spread_type == 'iron_butterfly':
                # For iron butterfly, we'll handle calls and puts separately
                return self._select_iron_butterfly_options(options_df)
            else:
                logger.warning(f"Unknown butterfly type: {spread_type}")
                return {}
            
            # Filter for the right option type
            options = options_df[options_df['right'].isin(option_type)]
            
            if options.empty:
                return {}
            
            # For butterfly, we need:
            # 1. Buy lower strike option
            # 2. Sell 2x middle strike option
            # 3. Buy higher strike option
            
            # Find ATM options for middle strike
            if 'delta' in options.columns:
                if option_type[0] in ['C', 'c']:  # Calls
                    options['delta_distance'] = abs(options['delta'] - 0.5)
                else:  # Puts
                    options['delta_distance'] = abs(options['delta'] + 0.5)
                
                options = options.sort_values('delta_distance')
                
                # Get the middle strike option (closest to ATM)
                middle_option = options.iloc[0:1]
                
                if middle_option.empty:
                    return {}
                
                middle_strike = middle_option['strike'].values[0]
                
                # Find wing options with strikes spread_width away
                target_lower_strike = middle_strike - self.spread_width
                target_higher_strike = middle_strike + self.spread_width
                
                # Find the closest strikes
                options['lower_strike_distance'] = abs(options['strike'] - target_lower_strike)
                options['higher_strike_distance'] = abs(options['strike'] - target_higher_strike)
                
                lower_option = options.sort_values('lower_strike_distance').iloc[0:1]
                higher_option = options.sort_values('higher_strike_distance').iloc[0:1]
                
                if lower_option.empty or higher_option.empty:
                    return {}
                
                # Calculate debit and max profit
                lower_price = lower_option['option_price'].values[0]
                middle_price = middle_option['option_price'].values[0]
                higher_price = higher_option['option_price'].values[0]
                
                # Butterfly is long 1 lower, short 2 middle, long 1 higher
                debit = lower_price - 2 * middle_price + higher_price
                
                # For a butterfly, max profit is at middle strike at expiration
                # Max profit = distance between strikes - debit
                max_profit = self.spread_width - debit
                
                if max_profit <= 0:
                    logger.warning("Invalid butterfly: max profit <= 0")
                    return {}
                
                # Calculate risk/reward ratio
                risk_reward_ratio = debit / max_profit if max_profit > 0 else float('inf')
                
                if risk_reward_ratio > self.max_risk_reward_ratio:
                    logger.warning(f"Risk/reward ratio too high: {risk_reward_ratio}")
                    return {}
                
                return {
                    'lower_option': lower_option,
                    'middle_option': middle_option,
                    'higher_option': higher_option,
                    'debit': debit,
                    'max_profit': max_profit,
                    'risk_reward_ratio': risk_reward_ratio
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error selecting butterfly options: {e}")
            return {}
    
    def _select_iron_butterfly_options(self, options_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Select options for an iron butterfly.
        
        Args:
            options_df: DataFrame containing filtered options data
            
        Returns:
            Dictionary with selected options for each leg
        """
        try:
            # Filter for calls and puts
            calls = options_df[options_df['right'].isin(['C', 'c', 'CALL', 'call'])]
            puts = options_df[options_df['right'].isin(['P', 'p', 'PUT', 'put'])]
            
            if calls.empty or puts.empty:
                return {}
            
            # For iron butterfly, we need:
            # 1. Sell ATM put
            # 2. Buy OTM put
            # 3. Sell ATM call
            # 4. Buy OTM call
            
            # Find ATM options
            if 'delta' in calls.columns and 'delta' in puts.columns:
                # For ATM call, target delta around 0.5
                calls['delta_distance'] = abs(calls['delta'] - 0.5)
                calls = calls.sort_values('delta_distance')
                
                # For ATM put, target delta around -0.5
                puts['delta_distance'] = abs(puts['delta'] + 0.5)  # Note the + for puts
                puts = puts.sort_values('delta_distance')
                
                # Get the ATM options
                atm_call = calls.iloc[0:1]
                atm_put = puts.iloc[0:1]
                
                if atm_call.empty or atm_put.empty:
                    return {}
                
                # Ensure we have the same strike for both ATM options
                # In a true iron butterfly, the short call and short put have the same strike
                atm_call_strike = atm_call['strike'].values[0]
                atm_put_strike = atm_put['strike'].values[0]
                
                # Use the average if they're different
                middle_strike = (atm_call_strike + atm_put_strike) / 2
                
                # Find the closest strikes to the middle
                calls['middle_distance'] = abs(calls['strike'] - middle_strike)
                puts['middle_distance'] = abs(puts['strike'] - middle_strike)
                
                short_call = calls.sort_values('middle_distance').iloc[0:1]
                short_put = puts.sort_values('middle_distance').iloc[0:1]
                
                if short_call.empty or short_put.empty:
                    return {}
                
                short_call_strike = short_call['strike'].values[0]
                short_put_strike = short_put['strike'].values[0]
                
                # Find wing options with strikes spread_width away
                target_long_call_strike = short_call_strike + self.spread_width
                target_long_put_strike = short_put_strike - self.spread_width
                
                # Find the closest strikes
                calls['strike_distance'] = abs(calls['strike'] - target_long_call_strike)
                puts['strike_distance'] = abs(puts['strike'] - target_long_put_strike)
                
                long_call = calls.sort_values('strike_distance').iloc[0:1]
                long_put = puts.sort_values('strike_distance').iloc[0:1]
                
                if long_call.empty or long_put.empty:
                    return {}
                
                # Calculate credit and max profit
                short_call_price = short_call['option_price'].values[0]
                long_call_price = long_call['option_price'].values[0]
                short_put_price = short_put['option_price'].values[0]
                long_put_price = long_put['option_price'].values[0]
                
                total_credit = (short_call_price - long_call_price) + (short_put_price - long_put_price)
                
                if total_credit <= 0:
                    logger.warning("Invalid iron butterfly: total credit <= 0")
                    return {}
                
                # Calculate max risk and credit/risk ratio
                max_risk = self.spread_width - total_credit
                credit_risk_ratio = total_credit / max_risk if max_risk > 0 else float('inf')
                
                if credit_risk_ratio < self.min_credit_debit_ratio:
                    logger.warning(f"Credit/risk ratio too low: {credit_risk_ratio}")
                    return {}
                
                return {
                    'short_call': short_call,
                    'long_call': long_call,
                    'short_put': short_put,
                    'long_put': long_put,
                    'total_credit': total_credit,
                    'max_risk': max_risk,
                    'credit_risk_ratio': credit_risk_ratio
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error selecting iron butterfly options: {e}")
            return {}
    
    def _select_calendar_spread_options(self, options_df: pd.DataFrame, spread_type: str) -> Dict[str, pd.DataFrame]:
        """
        Select options for a calendar spread.
        
        Args:
            options_df: DataFrame containing filtered options data
            spread_type: Type of calendar spread (call_calendar, put_calendar)
            
        Returns:
            Dictionary with selected options for each leg
        """
        try:
            # Determine option type based on spread type
            if spread_type == 'call_calendar':
                option_type = ['C', 'c', 'CALL', 'call']
            elif spread_type == 'put_calendar':
                option_type = ['P', 'p', 'PUT', 'put']
            else:
                logger.warning(f"Unknown calendar spread type: {spread_type}")
                return {}
            
            # Filter for the right option type
            options = options_df[options_df['right'].isin(option_type)]
            
            if options.empty:
                return {}
            
            # For calendar spread, we need:
            # 1. Sell near-term option
            # 2. Buy longer-term option
            # Both at the same strike, preferably ATM
            
            # Group by expiration
            if 'expiration' in options.columns and 'dte' in options.columns:
                # Sort by days to expiration
                expirations = options['expiration'].unique()
                if len(expirations) < 2:
                    logger.warning("Not enough expirations for calendar spread")
                    return {}
                
                # Get the two closest expirations
                options = options.sort_values('dte')
                near_term_options = options.iloc[0:len(options)//2]
                longer_term_options = options.iloc[len(options)//2:]
                
                if near_term_options.empty or longer_term_options.empty:
                    return {}
                
                # Find ATM options
                if 'delta' in options.columns:
                    if option_type[0] in ['C', 'c']:  # Calls
                        near_term_options['delta_distance'] = abs(near_term_options['delta'] - 0.5)
                        longer_term_options['delta_distance'] = abs(longer_term_options['delta'] - 0.5)
                    else:  # Puts
                        near_term_options['delta_distance'] = abs(near_term_options['delta'] + 0.5)
                        longer_term_options['delta_distance'] = abs(longer_term_options['delta'] + 0.5)
                    
                    # Get the ATM options
                    near_term_option = near_term_options.sort_values('delta_distance').iloc[0:1]
                    
                    if near_term_option.empty:
                        return {}
                    
                    # Find longer-term option with same strike
                    near_term_strike = near_term_option['strike'].values[0]
                    
                    longer_term_options['strike_distance'] = abs(longer_term_options['strike'] - near_term_strike)
                    longer_term_option = longer_term_options.sort_values('strike_distance').iloc[0:1]
                    
                    if longer_term_option.empty:
                        return {}
                    
                    # Calculate debit
                    near_term_price = near_term_option['option_price'].values[0]
                    longer_term_price = longer_term_option['option_price'].values[0]
                    
                    debit = longer_term_price - near_term_price
                    
                    if debit <= 0:
                        logger.warning("Invalid calendar spread: debit <= 0")
                        return {}
                    
                    # For calendar spreads, max profit is harder to estimate
                    # It depends on IV changes and time decay
                    # We'll use a simple heuristic: max profit is 2x debit
                    estimated_max_profit = debit * 2
                    
                    return {
                        'short_option': near_term_option,
                        'long_option': longer_term_option,
                        'debit': debit,
                        'estimated_max_profit': estimated_max_profit
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error selecting calendar spread options: {e}")
            return {}
    
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
                spread_type = row.get('spread_type', '')
                
                # Check if we have options data for this symbol
                if symbol not in options_data or options_data[symbol].empty:
                    continue
                
                # Select options for this strategy and signal
                selected_options = self.select_strategy_options(options_data[symbol], signal, spread_type)
                
                if not selected_options:
                    continue
                
                # Create trade plan based on strategy type
                if spread_type in ['bull_call', 'bear_put']:
                    trade_plan = self._create_debit_spread_trade_plan(selected_options, row, account_value, spread_type)
                elif spread_type in ['bull_put', 'bear_call']:
                    trade_plan = self._create_credit_spread_trade_plan(selected_options, row, account_value, spread_type)
                elif spread_type == 'iron_condor':
                    trade_plan = self._create_iron_condor_trade_plan(selected_options, row, account_value)
                elif spread_type in ['call_butterfly', 'put_butterfly']:
                    trade_plan = self._create_butterfly_trade_plan(selected_options, row, account_value, spread_type)
                elif spread_type == 'iron_butterfly':
                    trade_plan = self._create_iron_butterfly_trade_plan(selected_options, row, account_value)
                elif spread_type in ['call_calendar', 'put_calendar']:
                    trade_plan = self._create_calendar_spread_trade_plan(selected_options, row, account_value, spread_type)
                else:
                    logger.warning(f"Unknown spread type: {spread_type}")
                    continue
                
                if trade_plan:
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
            logger.error(f"Error generating multi-leg trade plan: {e}")
            return pd.DataFrame()
    
    def _create_debit_spread_trade_plan(self, 
                                      selected_options: Dict, 
                                      signal_row: pd.Series, 
                                      account_value: float, 
                                      spread_type: str) -> Dict:
        """
        Create a trade plan for a debit spread.
        
        Args:
            selected_options: Dictionary with selected options
            signal_row: Series containing signal data
            account_value: Current account value
            spread_type: Type of spread
            
        Returns:
            Dictionary with trade plan
        """
        try:
            # Extract data based on spread type
            if spread_type == 'bull_call':
                long_option = selected_options.get('long_call')
                short_option = selected_options.get('short_call')
                long_action = 'BUY'
                short_action = 'SELL'
            elif spread_type == 'bear_put':
                long_option = selected_options.get('long_put')
                short_option = selected_options.get('short_put')
                long_action = 'BUY'
                short_action = 'SELL'
            else:
                logger.warning(f"Unknown debit spread type: {spread_type}")
                return None
            
            if long_option is None or short_option is None:
                return None
            
            # Extract as Series
            long_option = long_option.iloc[0] if not long_option.empty else None
            short_option = short_option.iloc[0] if not short_option.empty else None
            
            if long_option is None or short_option is None:
                return None
            
            # Calculate position size based on risk
            debit = selected_options.get('debit', 0)
            max_profit = selected_options.get('max_profit', 0)
            risk_reward_ratio = selected_options.get('risk_reward_ratio', float('inf'))
            
            # Calculate position size
            risk_amount = account_value * self.risk_per_trade
            position_size = int(risk_amount / (debit * 100))
            position_size = max(1, position_size)
            
            # Create trade plan
            trade_plan = {
                'symbol': signal_row['symbol'],
                'underlying': signal_row.get('underlying', signal_row['symbol']),
                'signal': signal_row['signal'],
                'strategy_type': spread_type,
                'legs': [
                    {
                        'option_symbol': long_option.get('option_symbol', ''),
                        'strike': long_option.get('strike', 0),
                        'expiration': long_option.get('expiration', ''),
                        'right': long_option.get('right', ''),
                        'action': long_action,
                        'option_price': long_option.get('option_price', 0),
                        'delta': long_option.get('delta', 0),
                        'gamma': long_option.get('gamma', 0),
                        'theta': long_option.get('theta', 0),
                        'vega': long_option.get('vega', 0)
                    },
                    {
                        'option_symbol': short_option.get('option_symbol', ''),
                        'strike': short_option.get('strike', 0),
                        'expiration': short_option.get('expiration', ''),
                        'right': short_option.get('right', ''),
                        'action': short_action,
                        'option_price': short_option.get('option_price', 0),
                        'delta': short_option.get('delta', 0),
                        'gamma': short_option.get('gamma', 0),
                        'theta': short_option.get('theta', 0),
                        'vega': short_option.get('vega', 0)
                    }
                ],
                'position_size': position_size,
                'debit': debit * position_size * 100,
                'max_profit': max_profit * position_size * 100,
                'risk_reward_ratio': risk_reward_ratio,
                'expected_return': max_profit / debit if debit > 0 else 0,
                'signal_strength': signal_row.get('signal_strength', 0),
                'stop_loss': debit * (1 - self.stop_loss_pct),
                'take_profit': debit + (max_profit * self.take_profit_pct)
            }
            
            return trade_plan
            
        except Exception as e:
            logger.error(f"Error creating debit spread trade plan: {e}")
            return None
    
    def _create_credit_spread_trade_plan(self, 
                                       selected_options: Dict, 
                                       signal_row: pd.Series, 
                                       account_value: float, 
                                       spread_type: str) -> Dict:
        """
        Create a trade plan for a credit spread.
        
        Args:
            selected_options: Dictionary with selected options
            signal_row: Series containing signal data
            account_value: Current account value
            spread_type: Type of spread
            
        Returns:
            Dictionary with trade plan
        """
        try:
            # Extract data based on spread type
            if spread_type == 'bull_put':
                short_option = selected_options.get('short_put')
                long_option = selected_options.get('long_put')
                short_action = 'SELL'
                long_action = 'BUY'
            elif spread_type == 'bear_call':
                short_option = selected_options.get('short_call')
                long_option = selected_options.get('long_call')
                short_action = 'SELL'
                long_action = 'BUY'
            else:
                logger.warning(f"Unknown credit spread type: {spread_type}")
                return None
            
            if short_option is None or long_option is None:
                return None
            
            # Extract as Series
            short_option = short_option.iloc[0] if not short_option.empty else None
            long_option = long_option.iloc[0] if not long_option.empty else None
            
            if short_option is None or long_option is None:
                return None
            
            # Calculate position size based on risk
            credit = selected_options.get('credit', 0)
            max_risk = selected_options.get('max_risk', 0)
            credit_risk_ratio = selected_options.get('credit_risk_ratio', 0)
            
            # Calculate position size
            risk_amount = account_value * self.risk_per_trade
            position_size = int(risk_amount / (max_risk * 100))
            position_size = max(1, position_size)
            
            # Create trade plan
            trade_plan = {
                'symbol': signal_row['symbol'],
                'underlying': signal_row.get('underlying', signal_row['symbol']),
                'signal': signal_row['signal'],
                'strategy_type': spread_type,
                'legs': [
                    {
                        'option_symbol': short_option.get('option_symbol', ''),
                        'strike': short_option.get('strike', 0),
                        'expiration': short_option.get('expiration', ''),
                        'right': short_option.get('right', ''),
                        'action': short_action,
                        'option_price': short_option.get('option_price', 0),
                        'delta': short_option.get('delta', 0),
                        'gamma': short_option.get('gamma', 0),
                        'theta': short_option.get('theta', 0),
                        'vega': short_option.get('vega', 0)
                    },
                    {
                        'option_symbol': long_option.get('option_symbol', ''),
                        'strike': long_option.get('strike', 0),
                        'expiration': long_option.get('expiration', ''),
                        'right': long_option.get('right', ''),
                        'action': long_action,
                        'option_price': long_option.get('option_price', 0),
                        'delta': long_option.get('delta', 0),
                        'gamma': long_option.get('gamma', 0),
                        'theta': long_option.get('theta', 0),
                        'vega': long_option.get('vega', 0)
                    }
                ],
                'position_size': position_size,
                'credit': credit * position_size * 100,
                'max_risk': max_risk * position_size * 100,
                'credit_risk_ratio': credit_risk_ratio,
                'expected_return': credit / max_risk if max_risk > 0 else 0,
                'signal_strength': signal_row.get('signal_strength', 0),
                'stop_loss': max_risk * 0.8,  # 80% of max risk
                'take_profit': credit * 0.5  # 50% of credit
            }
            
            return trade_plan
            
        except Exception as e:
            logger.error(f"Error creating credit spread trade plan: {e}")
            return None
    
    def _create_iron_condor_trade_plan(self, 
                                     selected_options: Dict, 
                                     signal_row: pd.Series, 
                                     account_value: float) -> Dict:
        """
        Create a trade plan for an iron condor.
        
        Args:
            selected_options: Dictionary with selected options
            signal_row: Series containing signal data
            account_value: Current account value
            
        Returns:
            Dictionary with trade plan
        """
        try:
            # Extract options
            short_call = selected_options.get('short_call')
            long_call = selected_options.get('long_call')
            short_put = selected_options.get('short_put')
            long_put = selected_options.get('long_put')
            
            if short_call is None or long_call is None or short_put is None or long_put is None:
                return None
            
            # Extract as Series
            short_call = short_call.iloc[0] if not short_call.empty else None
            long_call = long_call.iloc[0] if not long_call.empty else None
            short_put = short_put.iloc[0] if not short_put.empty else None
            long_put = long_put.iloc[0] if not long_put.empty else None
            
            if short_call is None or long_call is None or short_put is None or long_put is None:
                return None
            
            # Calculate position size based on risk
            total_credit = selected_options.get('total_credit', 0)
            max_risk = selected_options.get('max_risk', 0)
            credit_risk_ratio = selected_options.get('credit_risk_ratio', 0)
            
            # Calculate position size
            risk_amount = account_value * self.risk_per_trade
            position_size = int(risk_amount / (max_risk * 100))
            position_size = max(1, position_size)
            
            # Create trade plan
            trade_plan = {
                'symbol': signal_row['symbol'],
                'underlying': signal_row.get('underlying', signal_row['symbol']),
                'signal': signal_row['signal'],
                'strategy_type': 'iron_condor',
                'legs': [
                    {
                        'option_symbol': short_call.get('option_symbol', ''),
                        'strike': short_call.get('strike', 0),
                        'expiration': short_call.get('expiration', ''),
                        'right': short_call.get('right', ''),
                        'action': 'SELL',
                        'option_price': short_call.get('option_price', 0),
                        'delta': short_call.get('delta', 0)
                    },
                    {
                        'option_symbol': long_call.get('option_symbol', ''),
                        'strike': long_call.get('strike', 0),
                        'expiration': long_call.get('expiration', ''),
                        'right': long_call.get('right', ''),
                        'action': 'BUY',
                        'option_price': long_call.get('option_price', 0),
                        'delta': long_call.get('delta', 0)
                    },
                    {
                        'option_symbol': short_put.get('option_symbol', ''),
                        'strike': short_put.get('strike', 0),
                        'expiration': short_put.get('expiration', ''),
                        'right': short_put.get('right', ''),
                        'action': 'SELL',
                        'option_price': short_put.get('option_price', 0),
                        'delta': short_put.get('delta', 0)
                    },
                    {
                        'option_symbol': long_put.get('option_symbol', ''),
                        'strike': long_put.get('strike', 0),
                        'expiration': long_put.get('expiration', ''),
                        'right': long_put.get('right', ''),
                        'action': 'BUY',
                        'option_price': long_put.get('option_price', 0),
                        'delta': long_put.get('delta', 0)
                    }
                ],
                'position_size': position_size,
                'credit': total_credit * position_size * 100,
                'max_risk': max_risk * position_size * 100,
                'credit_risk_ratio': credit_risk_ratio,
                'expected_return': total_credit / max_risk if max_risk > 0 else 0,
                'signal_strength': signal_row.get('signal_strength', 0),
                'stop_loss': max_risk * 0.8,  # 80% of max risk
                'take_profit': total_credit * 0.5  # 50% of credit
            }
            
            return trade_plan
            
        except Exception as e:
            logger.error(f"Error creating iron condor trade plan: {e}")
            return None
    
    def _create_butterfly_trade_plan(self, 
                                   selected_options: Dict, 
                                   signal_row: pd.Series, 
                                   account_value: float, 
                                   spread_type: str) -> Dict:
        """
        Create a trade plan for a butterfly spread.
        
        Args:
            selected_options: Dictionary with selected options
            signal_row: Series containing signal data
            account_value: Current account value
            spread_type: Type of butterfly
            
        Returns:
            Dictionary with trade plan
        """
        try:
            # Extract options
            lower_option = selected_options.get('lower_option')
            middle_option = selected_options.get('middle_option')
            higher_option = selected_options.get('higher_option')
            
            if lower_option is None or middle_option is None or higher_option is None:
                return None
            
            # Extract as Series
            lower_option = lower_option.iloc[0] if not lower_option.empty else None
            middle_option = middle_option.iloc[0] if not middle_option.empty else None
            higher_option = higher_option.iloc[0] if not higher_option.empty else None
            
            if lower_option is None or middle_option is None or higher_option is None:
                return None
            
            # Calculate position size based on risk
            debit = selected_options.get('debit', 0)
            max_profit = selected_options.get('max_profit', 0)
            risk_reward_ratio = selected_options.get('risk_reward_ratio', float('inf'))
            
            # Calculate position size
            risk_amount = account_value * self.risk_per_trade
            position_size = int(risk_amount / (debit * 100))
            position_size = max(1, position_size)
            
            # Create trade plan
            trade_plan = {
                'symbol': signal_row['symbol'],
                'underlying': signal_row.get('underlying', signal_row['symbol']),
                'signal': signal_row['signal'],
                'strategy_type': spread_type,
                'legs': [
                    {
                        'option_symbol': lower_option.get('option_symbol', ''),
                        'strike': lower_option.get('strike', 0),
                        'expiration': lower_option.get('expiration', ''),
                        'right': lower_option.get('right', ''),
                        'action': 'BUY',
                        'option_price': lower_option.get('option_price', 0),
                        'delta': lower_option.get('delta', 0)
                    },
                    {
                        'option_symbol': middle_option.get('option_symbol', ''),
                        'strike': middle_option.get('strike', 0),
                        'expiration': middle_option.get('expiration', ''),
                        'right': middle_option.get('right', ''),
                        'action': 'SELL',
                        'quantity': 2,  # Sell 2x
                        'option_price': middle_option.get('option_price', 0),
                        'delta': middle_option.get('delta', 0)
                    },
                    {
                        'option_symbol': higher_option.get('option_symbol', ''),
                        'strike': higher_option.get('strike', 0),
                        'expiration': higher_option.get('expiration', ''),
                        'right': higher_option.get('right', ''),
                        'action': 'BUY',
                        'option_price': higher_option.get('option_price', 0),
                        'delta': higher_option.get('delta', 0)
                    }
                ],
                'position_size': position_size,
                'debit': debit * position_size * 100,
                'max_profit': max_profit * position_size * 100,
                'risk_reward_ratio': risk_reward_ratio,
                'expected_return': max_profit / debit if debit > 0 else 0,
                'signal_strength': signal_row.get('signal_strength', 0),
                'stop_loss': debit * (1 - self.stop_loss_pct),
                'take_profit': debit + (max_profit * self.take_profit_pct)
            }
            
            return trade_plan
            
        except Exception as e:
            logger.error(f"Error creating butterfly trade plan: {e}")
            return None
    
    def _create_iron_butterfly_trade_plan(self, 
                                        selected_options: Dict, 
                                        signal_row: pd.Series, 
                                        account_value: float) -> Dict:
        """
        Create a trade plan for an iron butterfly.
        
        Args:
            selected_options: Dictionary with selected options
            signal_row: Series containing signal data
            account_value: Current account value
            
        Returns:
            Dictionary with trade plan
        """
        try:
            # Extract options
            short_call = selected_options.get('short_call')
            long_call = selected_options.get('long_call')
            short_put = selected_options.get('short_put')
            long_put = selected_options.get('long_put')
            
            if short_call is None or long_call is None or short_put is None or long_put is None:
                return None
            
            # Extract as Series
            short_call = short_call.iloc[0] if not short_call.empty else None
            long_call = long_call.iloc[0] if not long_call.empty else None
            short_put = short_put.iloc[0] if not short_put.empty else None
            long_put = long_put.iloc[0] if not long_put.empty else None
            
            if short_call is None or long_call is None or short_put is None or long_put is None:
                return None
            
            # Calculate position size based on risk
            total_credit = selected_options.get('total_credit', 0)
            max_risk = selected_options.get('max_risk', 0)
            credit_risk_ratio = selected_options.get('credit_risk_ratio', 0)
            
            # Calculate position size
            risk_amount = account_value * self.risk_per_trade
            position_size = int(risk_amount / (max_risk * 100))
            position_size = max(1, position_size)
            
            # Create trade plan
            trade_plan = {
                'symbol': signal_row['symbol'],
                'underlying': signal_row.get('underlying', signal_row['symbol']),
                'signal': signal_row['signal'],
                'strategy_type': 'iron_butterfly',
                'legs': [
                    {
                        'option_symbol': short_call.get('option_symbol', ''),
                        'strike': short_call.get('strike', 0),
                        'expiration': short_call.get('expiration', ''),
                        'right': short_call.get('right', ''),
                        'action': 'SELL',
                        'option_price': short_call.get('option_price', 0),
                        'delta': short_call.get('delta', 0)
                    },
                    {
                        'option_symbol': long_call.get('option_symbol', ''),
                        'strike': long_call.get('strike', 0),
                        'expiration': long_call.get('expiration', ''),
                        'right': long_call.get('right', ''),
                        'action': 'BUY',
                        'option_price': long_call.get('option_price', 0),
                        'delta': long_call.get('delta', 0)
                    },
                    {
                        'option_symbol': short_put.get('option_symbol', ''),
                        'strike': short_put.get('strike', 0),
                        'expiration': short_put.get('expiration', ''),
                        'right': short_put.get('right', ''),
                        'action': 'SELL',
                        'option_price': short_put.get('option_price', 0),
                        'delta': short_put.get('delta', 0)
                    },
                    {
                        'option_symbol': long_put.get('option_symbol', ''),
                        'strike': long_put.get('strike', 0),
                        'expiration': long_put.get('expiration', ''),
                        'right': long_put.get('right', ''),
                        'action': 'BUY',
                        'option_price': long_put.get('option_price', 0),
                        'delta': long_put.get('delta', 0)
                    }
                ],
                'position_size': position_size,
                'credit': total_credit * position_size * 100,
                'max_risk': max_risk * position_size * 100,
                'credit_risk_ratio': credit_risk_ratio,
                'expected_return': total_credit / max_risk if max_risk > 0 else 0,
                'signal_strength': signal_row.get('signal_strength', 0),
                'stop_loss': max_risk * 0.8,  # 80% of max risk
                'take_profit': total_credit * 0.5  # 50% of credit
            }
            
            return trade_plan
            
        except Exception as e:
            logger.error(f"Error creating iron butterfly trade plan: {e}")
            return None
    
    def _create_calendar_spread_trade_plan(self, 
                                         selected_options: Dict, 
                                         signal_row: pd.Series, 
                                         account_value: float, 
                                         spread_type: str) -> Dict:
        """
        Create a trade plan for a calendar spread.
        
        Args:
            selected_options: Dictionary with selected options
            signal_row: Series containing signal data
            account_value: Current account value
            spread_type: Type of calendar spread
            
        Returns:
            Dictionary with trade plan
        """
        try:
            # Extract options
            short_option = selected_options.get('short_option')
            long_option = selected_options.get('long_option')
            
            if short_option is None or long_option is None:
                return None
            
            # Extract as Series
            short_option = short_option.iloc[0] if not short_option.empty else None
            long_option = long_option.iloc[0] if not long_option.empty else None
            
            if short_option is None or long_option is None:
                return None
            
            # Calculate position size based on risk
            debit = selected_options.get('debit', 0)
            estimated_max_profit = selected_options.get('estimated_max_profit', 0)
            
            # Calculate position size
            risk_amount = account_value * self.risk_per_trade
            position_size = int(risk_amount / (debit * 100))
            position_size = max(1, position_size)
            
            # Create trade plan
            trade_plan = {
                'symbol': signal_row['symbol'],
                'underlying': signal_row.get('underlying', signal_row['symbol']),
                'signal': signal_row['signal'],
                'strategy_type': spread_type,
                'legs': [
                    {
                        'option_symbol': short_option.get('option_symbol', ''),
                        'strike': short_option.get('strike', 0),
                        'expiration': short_option.get('expiration', ''),
                        'right': short_option.get('right', ''),
                        'action': 'SELL',
                        'option_price': short_option.get('option_price', 0),
                        'delta': short_option.get('delta', 0),
                        'gamma': short_option.get('gamma', 0),
                        'theta': short_option.get('theta', 0),
                        'vega': short_option.get('vega', 0)
                    },
                    {
                        'option_symbol': long_option.get('option_symbol', ''),
                        'strike': long_option.get('strike', 0),
                        'expiration': long_option.get('expiration', ''),
                        'right': long_option.get('right', ''),
                        'action': 'BUY',
                        'option_price': long_option.get('option_price', 0),
                        'delta': long_option.get('delta', 0),
                        'gamma': long_option.get('gamma', 0),
                        'theta': long_option.get('theta', 0),
                        'vega': long_option.get('vega', 0)
                    }
                ],
                'position_size': position_size,
                'debit': debit * position_size * 100,
                'estimated_max_profit': estimated_max_profit * position_size * 100,
                'expected_return': estimated_max_profit / debit if debit > 0 else 0,
                'signal_strength': signal_row.get('signal_strength', 0),
                'stop_loss': debit * (1 - self.stop_loss_pct),
                'take_profit': debit + (estimated_max_profit * self.take_profit_pct)
            }
            
            return trade_plan
            
        except Exception as e:
            logger.error(f"Error creating calendar spread trade plan: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    strategy = MultiLegStrategy()
    
    # Example: Create sample data
    data = {
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        'underlying_price': [150.0, 280.0, 2800.0, 3200.0],
        'predicted_up_probability': [0.75, 0.45, 0.25, 0.55],
        'predicted_iv_change': [-0.02, 0.01, -0.01, 0.03],
        'iv_rank': [0.85, 0.35, 0.75, 0.25]
    }
    
    df = pd.DataFrame(data)
    
    # Generate signals
    signals = strategy.generate_signals(df)
    
    print("Generated Multi-Leg Signals:")
    print(signals[['symbol', 'predicted_up_probability', 'signal', 'spread_type']])
