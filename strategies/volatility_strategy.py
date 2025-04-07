"""
Volatility Strategy Module

This module implements volatility-based options trading strategies including
IV mean reversion, volatility skew, and earnings volatility plays.
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


class VolatilityStrategy(StrategyBase):
    """
    Strategy for volatility-based options trading.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the volatility strategy with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        # Initialize base strategy
        super().__init__(config_path)
        
        # Load strategy-specific configuration
        self.config = self._load_config(config_path)
        self.strategy_config = self.config.get('strategies', {}).get('volatility', {})
        
        # Strategy metadata
        self.name = "VolatilityStrategy"
        self.description = "Volatility-based options trading strategy"
        
        # Strategy parameters
        self.iv_rank_high = self.strategy_config.get('iv_rank_high', 0.8)
        self.iv_rank_low = self.strategy_config.get('iv_rank_low', 0.2)
        self.iv_hv_spread_threshold = self.strategy_config.get('iv_hv_spread_threshold', 0.2)
        self.skew_threshold = self.strategy_config.get('skew_threshold', 0.1)
        self.earnings_days_threshold = self.strategy_config.get('earnings_days_threshold', 10)
        self.min_expected_return = self.strategy_config.get('min_expected_return', 0.15)
        self.strategy_type = self.strategy_config.get('strategy_type', 'mean_reversion')
        self.use_ml_predictions = self.strategy_config.get('use_ml_predictions', True)
    
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
                    'volatility': {
                        'iv_rank_high': 0.8,
                        'iv_rank_low': 0.2,
                        'iv_hv_spread_threshold': 0.2,
                        'skew_threshold': 0.1,
                        'earnings_days_threshold': 10,
                        'min_expected_return': 0.15,
                        'strategy_type': 'mean_reversion',
                        'use_ml_predictions': True
                    }
                }
            }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on volatility metrics.
        
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
            
            # Initialize signal column (0 = neutral, 1 = long vol, -1 = short vol)
            result['signal'] = 0
            
            # Check strategy type
            if self.strategy_type == 'mean_reversion':
                # Generate mean reversion signals
                result = self._generate_mean_reversion_signals(result)
            elif self.strategy_type == 'skew':
                # Generate volatility skew signals
                result = self._generate_skew_signals(result)
            elif self.strategy_type == 'earnings':
                # Generate earnings volatility signals
                result = self._generate_earnings_signals(result)
            else:
                # Default to mean reversion
                result = self._generate_mean_reversion_signals(result)
            
            # Apply ML predictions if enabled
            if self.use_ml_predictions and 'predicted_iv_change' in result.columns:
                # Adjust signals based on predicted IV change
                # For long vol signals (1), we want predicted_iv_change > 0
                result.loc[
                    (result['signal'] == 1) & (result['predicted_iv_change'] <= 0),
                    'signal'
                ] = 0
                
                # For short vol signals (-1), we want predicted_iv_change < 0
                result.loc[
                    (result['signal'] == -1) & (result['predicted_iv_change'] >= 0),
                    'signal'
                ] = 0
            
            # Calculate signal strength
            result['signal_strength'] = 0.0
            
            # For mean reversion strategy
            if self.strategy_type == 'mean_reversion' and 'iv_rank' in result.columns:
                # For long vol signals (1), strength is proportional to how low IV rank is
                result.loc[result['signal'] == 1, 'signal_strength'] = 1 - result.loc[result['signal'] == 1, 'iv_rank']
                
                # For short vol signals (-1), strength is proportional to how high IV rank is
                result.loc[result['signal'] == -1, 'signal_strength'] = result.loc[result['signal'] == -1, 'iv_rank']
            
            # For skew strategy
            elif self.strategy_type == 'skew' and 'skew_ratio' in result.columns:
                # Signal strength is proportional to skew deviation
                result.loc[result['signal'] != 0, 'signal_strength'] = abs(
                    result.loc[result['signal'] != 0, 'skew_ratio'] - 1
                )
            
            # For earnings strategy
            elif self.strategy_type == 'earnings' and 'days_to_earnings' in result.columns:
                # Signal strength is inversely proportional to days to earnings
                max_days = self.earnings_days_threshold
                result.loc[result['signal'] != 0, 'signal_strength'] = 1 - (
                    result.loc[result['signal'] != 0, 'days_to_earnings'] / max_days
                )
            
            # Normalize signal strength to 0-1 range
            if result['signal_strength'].max() > 0:
                result['signal_strength'] = result['signal_strength'] / result['signal_strength'].max()
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating volatility signals: {e}")
            # Return original data with neutral signal
            data['signal'] = 0
            return data
    
    def _generate_mean_reversion_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on IV mean reversion strategy.
        
        Args:
            data: DataFrame containing market data and features
            
        Returns:
            DataFrame with added signal columns
        """
        result = data.copy()
        
        # Check if we have the required columns
        required_cols = ['iv_rank', 'iv_hv_spread']
        missing_cols = [col for col in required_cols if col not in result.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for mean reversion strategy: {missing_cols}")
            return result
        
        # Long volatility when IV is low (buy options)
        result.loc[
            (result['iv_rank'] < self.iv_rank_low) &
            (result['iv_hv_spread'] < -self.iv_hv_spread_threshold),
            'signal'
        ] = 1
        
        # Short volatility when IV is high (sell options)
        result.loc[
            (result['iv_rank'] > self.iv_rank_high) &
            (result['iv_hv_spread'] > self.iv_hv_spread_threshold),
            'signal'
        ] = -1
        
        return result
    
    def _generate_skew_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on volatility skew strategy.
        
        Args:
            data: DataFrame containing market data and features
            
        Returns:
            DataFrame with added signal columns
        """
        result = data.copy()
        
        # Check if we have the required columns
        required_cols = ['skew_ratio', 'atm_skew']
        missing_cols = [col for col in required_cols if col not in result.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for skew strategy: {missing_cols}")
            return result
        
        # Long put volatility when put skew is low (buy puts)
        result.loc[
            (result['skew_ratio'] < (1 - self.skew_threshold)) &
            (result['atm_skew'] < 0),
            'signal'
        ] = 1
        
        # Long call volatility when call skew is low (buy calls)
        result.loc[
            (result['skew_ratio'] > (1 + self.skew_threshold)) &
            (result['atm_skew'] > 0),
            'signal'
        ] = 1
        
        # Short put volatility when put skew is high (sell puts)
        result.loc[
            (result['skew_ratio'] > (1 + self.skew_threshold)) &
            (result['atm_skew'] < 0),
            'signal'
        ] = -1
        
        # Short call volatility when call skew is high (sell calls)
        result.loc[
            (result['skew_ratio'] < (1 - self.skew_threshold)) &
            (result['atm_skew'] > 0),
            'signal'
        ] = -1
        
        return result
    
    def _generate_earnings_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on earnings volatility strategy.
        
        Args:
            data: DataFrame containing market data and features
            
        Returns:
            DataFrame with added signal columns
        """
        result = data.copy()
        
        # Check if we have the required columns
        required_cols = ['days_to_earnings', 'iv_rank']
        missing_cols = [col for col in required_cols if col not in result.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for earnings strategy: {missing_cols}")
            return result
        
        # Long volatility before earnings (buy straddles/strangles)
        result.loc[
            (result['days_to_earnings'] <= self.earnings_days_threshold) &
            (result['days_to_earnings'] > 1) &
            (result['iv_rank'] < 0.5),  # Only if IV isn't already too high
            'signal'
        ] = 1
        
        # Short volatility after earnings (sell straddles/strangles)
        result.loc[
            (result['days_to_earnings'] == 0) &  # Earnings day or just reported
            (result['iv_rank'] > 0.7),  # High IV due to earnings
            'signal'
        ] = -1
        
        return result
    
    def select_strategy_options(self, options_df: pd.DataFrame, signal: int) -> pd.DataFrame:
        """
        Select the appropriate options for the volatility strategy.
        
        Args:
            options_df: DataFrame containing options data
            signal: Signal direction (1 for long vol, -1 for short vol)
            
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
            
            # Strategy-specific filtering
            if self.strategy_type == 'mean_reversion':
                return self._select_mean_reversion_options(filtered_options, signal)
            elif self.strategy_type == 'skew':
                return self._select_skew_options(filtered_options, signal)
            elif self.strategy_type == 'earnings':
                return self._select_earnings_options(filtered_options, signal)
            else:
                # Default to mean reversion
                return self._select_mean_reversion_options(filtered_options, signal)
            
        except Exception as e:
            logger.error(f"Error selecting strategy options: {e}")
            return options_df
    
    def _select_mean_reversion_options(self, options_df: pd.DataFrame, signal: int) -> pd.DataFrame:
        """
        Select options for mean reversion strategy.
        
        Args:
            options_df: DataFrame containing options data
            signal: Signal direction (1 for long vol, -1 for short vol)
            
        Returns:
            DataFrame with selected options
        """
        if options_df.empty:
            return options_df
            
        # For long volatility (1), we want to buy straddles or strangles
        # For short volatility (-1), we want to sell iron condors or strangles
        
        # Filter by delta for ATM options
        if 'delta' in options_df.columns:
            # For straddles, we want options close to ATM (delta around 0.5 for calls, -0.5 for puts)
            if signal == 1:  # Long vol
                # Find ATM calls
                calls = options_df[options_df['right'].isin(['C', 'c', 'CALL', 'call'])]
                calls = calls[(calls['delta'] >= 0.4) & (calls['delta'] <= 0.6)]
                
                # Find ATM puts
                puts = options_df[options_df['right'].isin(['P', 'p', 'PUT', 'put'])]
                puts = puts[(puts['delta'] >= -0.6) & (puts['delta'] <= -0.4)]
                
                # Combine and sort by closest to ATM
                atm_options = pd.concat([calls, puts])
                if not atm_options.empty:
                    atm_options['delta_distance'] = abs(abs(atm_options['delta']) - 0.5)
                    atm_options = atm_options.sort_values('delta_distance')
                
                return atm_options
                
            elif signal == -1:  # Short vol
                # For iron condors, we want OTM options (delta around 0.2/-0.2)
                # Find OTM calls
                calls = options_df[options_df['right'].isin(['C', 'c', 'CALL', 'call'])]
                calls = calls[(calls['delta'] >= 0.15) & (calls['delta'] <= 0.25)]
                
                # Find OTM puts
                puts = options_df[options_df['right'].isin(['P', 'p', 'PUT', 'put'])]
                puts = puts[(puts['delta'] >= -0.25) & (puts['delta'] <= -0.15)]
                
                # Combine and sort by highest premium
                otm_options = pd.concat([calls, puts])
                if not otm_options.empty and 'option_price' in otm_options.columns:
                    otm_options = otm_options.sort_values('option_price', ascending=False)
                
                return otm_options
        
        return options_df
    
    def _select_skew_options(self, options_df: pd.DataFrame, signal: int) -> pd.DataFrame:
        """
        Select options for volatility skew strategy.
        
        Args:
            options_df: DataFrame containing options data
            signal: Signal direction (1 for long vol, -1 for short vol)
            
        Returns:
            DataFrame with selected options
        """
        if options_df.empty:
            return options_df
            
        # For skew strategy, we focus on the side with mispriced volatility
        
        # Check if we have skew information
        if 'atm_skew' in options_df.columns:
            skew = options_df['atm_skew'].iloc[0] if not options_df.empty else 0
            
            # Determine which side (calls or puts) to trade based on skew
            if skew < 0:  # Put skew
                # Filter for puts
                options = options_df[options_df['right'].isin(['P', 'p', 'PUT', 'put'])]
                
                if signal == 1:  # Long put vol when put skew is low
                    # Look for OTM puts
                    if 'delta' in options.columns:
                        options = options[(options['delta'] >= -0.3) & (options['delta'] <= -0.1)]
                elif signal == -1:  # Short put vol when put skew is high
                    # Look for OTM puts
                    if 'delta' in options.columns:
                        options = options[(options['delta'] >= -0.3) & (options['delta'] <= -0.1)]
            else:  # Call skew or neutral
                # Filter for calls
                options = options_df[options_df['right'].isin(['C', 'c', 'CALL', 'call'])]
                
                if signal == 1:  # Long call vol when call skew is low
                    # Look for OTM calls
                    if 'delta' in options.columns:
                        options = options[(options['delta'] >= 0.1) & (options['delta'] <= 0.3)]
                elif signal == -1:  # Short call vol when call skew is high
                    # Look for OTM calls
                    if 'delta' in options.columns:
                        options = options[(options['delta'] >= 0.1) & (options['delta'] <= 0.3)]
            
            # Sort by expected return if available
            if not options.empty and 'expected_return' in options.columns:
                options = options.sort_values('expected_return', ascending=False)
            
            return options
        
        return options_df
    
    def _select_earnings_options(self, options_df: pd.DataFrame, signal: int) -> pd.DataFrame:
        """
        Select options for earnings volatility strategy.
        
        Args:
            options_df: DataFrame containing options data
            signal: Signal direction (1 for long vol, -1 for short vol)
            
        Returns:
            DataFrame with selected options
        """
        if options_df.empty:
            return options_df
            
        # For earnings strategy, we want options that expire after earnings
        
        # Check if we have earnings date information
        if 'days_to_earnings' in options_df.columns:
            days_to_earnings = options_df['days_to_earnings'].iloc[0] if not options_df.empty else 0
            
            # Filter for options that expire after earnings
            if 'dte' in options_df.columns:
                options_df = options_df[options_df['dte'] > days_to_earnings + 2]  # +2 days buffer
            
            if options_df.empty:
                return options_df
            
            if signal == 1:  # Long vol before earnings (straddles/strangles)
                # Find ATM options for straddle
                if 'delta' in options_df.columns:
                    # Find ATM calls
                    calls = options_df[options_df['right'].isin(['C', 'c', 'CALL', 'call'])]
                    calls = calls[(calls['delta'] >= 0.4) & (calls['delta'] <= 0.6)]
                    
                    # Find ATM puts
                    puts = options_df[options_df['right'].isin(['P', 'p', 'PUT', 'put'])]
                    puts = puts[(puts['delta'] >= -0.6) & (puts['delta'] <= -0.4)]
                    
                    # Combine and sort by closest to ATM
                    atm_options = pd.concat([calls, puts])
                    if not atm_options.empty:
                        atm_options['delta_distance'] = abs(abs(atm_options['delta']) - 0.5)
                        atm_options = atm_options.sort_values('delta_distance')
                    
                    return atm_options
                
            elif signal == -1:  # Short vol after earnings (iron condors)
                # For iron condors, we want OTM options
                if 'delta' in options_df.columns:
                    # Find OTM calls
                    calls = options_df[options_df['right'].isin(['C', 'c', 'CALL', 'call'])]
                    calls = calls[(calls['delta'] >= 0.15) & (calls['delta'] <= 0.25)]
                    
                    # Find OTM puts
                    puts = options_df[options_df['right'].isin(['P', 'p', 'PUT', 'put'])]
                    puts = puts[(puts['delta'] >= -0.25) & (puts['delta'] <= -0.15)]
                    
                    # Combine and sort by highest premium
                    otm_options = pd.concat([calls, puts])
                    if not otm_options.empty and 'option_price' in otm_options.columns:
                        otm_options = otm_options.sort_values('option_price', ascending=False)
                    
                    return otm_options
        
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
                
                # Select options for this strategy and signal
                options = self.select_strategy_options(options_data[symbol], signal)
                
                if options.empty:
                    continue
                
                # For volatility strategies, we often need multiple options
                if self.strategy_type == 'mean_reversion':
                    if signal == 1:  # Long vol (straddle)
                        trade_plan = self._create_straddle_trade(options, row, account_value)
                    else:  # Short vol (iron condor)
                        trade_plan = self._create_iron_condor_trade(options, row, account_value)
                elif self.strategy_type == 'skew':
                    # For skew, we typically trade single options
                    best_option = options.iloc[0] if not options.empty else None
                    if best_option is not None:
                        trade_plan = self._create_single_option_trade(best_option, row, account_value)
                    else:
                        continue
                elif self.strategy_type == 'earnings':
                    if signal == 1:  # Long vol (straddle)
                        trade_plan = self._create_straddle_trade(options, row, account_value)
                    else:  # Short vol (iron condor)
                        trade_plan = self._create_iron_condor_trade(options, row, account_value)
                else:
                    # Default to single option
                    best_option = options.iloc[0] if not options.empty else None
                    if best_option is not None:
                        trade_plan = self._create_single_option_trade(best_option, row, account_value)
                    else:
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
            logger.error(f"Error generating volatility trade plan: {e}")
            return pd.DataFrame()
    
    def _create_straddle_trade(self, options: pd.DataFrame, signal_row: pd.Series, account_value: float) -> Dict:
        """
        Create a straddle trade plan.
        
        Args:
            options: DataFrame containing filtered options
            signal_row: Series containing signal data
            account_value: Current account value
            
        Returns:
            Dictionary with trade plan
        """
        try:
            # Find ATM call and put
            calls = options[options['right'].isin(['C', 'c', 'CALL', 'call'])]
            puts = options[options['right'].isin(['P', 'p', 'PUT', 'put'])]
            
            if calls.empty or puts.empty:
                return None
            
            # Get the best call and put
            call = calls.iloc[0]
            put = puts.iloc[0]
            
            # Calculate total cost
            total_price = call['option_price'] + put['option_price']
            
            # Calculate position size
            position_size = self.calculate_position_size(account_value, total_price)
            
            # Create trade plan
            trade_plan = {
                'symbol': signal_row['symbol'],
                'underlying': signal_row.get('underlying', signal_row['symbol']),
                'signal': signal_row['signal'],
                'strategy_type': 'straddle',
                'legs': [
                    {
                        'option_symbol': call.get('option_symbol', ''),
                        'strike': call.get('strike', 0),
                        'expiration': call.get('expiration', ''),
                        'right': 'CALL',
                        'action': 'BUY',
                        'option_price': call.get('option_price', 0),
                        'delta': call.get('delta', 0),
                        'gamma': call.get('gamma', 0),
                        'theta': call.get('theta', 0),
                        'vega': call.get('vega', 0)
                    },
                    {
                        'option_symbol': put.get('option_symbol', ''),
                        'strike': put.get('strike', 0),
                        'expiration': put.get('expiration', ''),
                        'right': 'PUT',
                        'action': 'BUY',
                        'option_price': put.get('option_price', 0),
                        'delta': put.get('delta', 0),
                        'gamma': put.get('gamma', 0),
                        'theta': put.get('theta', 0),
                        'vega': put.get('vega', 0)
                    }
                ],
                'position_size': position_size,
                'total_cost': total_price * position_size * 100,
                'expected_return': signal_row.get('expected_return', 0),
                'signal_strength': signal_row.get('signal_strength', 0),
                'stop_loss': total_price * (1 - self.stop_loss_pct),
                'take_profit': total_price * (1 + self.take_profit_pct)
            }
            
            return trade_plan
            
        except Exception as e:
            logger.error(f"Error creating straddle trade: {e}")
            return None
    
    def _create_iron_condor_trade(self, options: pd.DataFrame, signal_row: pd.Series, account_value: float) -> Dict:
        """
        Create an iron condor trade plan.
        
        Args:
            options: DataFrame containing filtered options
            signal_row: Series containing signal data
            account_value: Current account value
            
        Returns:
            Dictionary with trade plan
        """
        try:
            # Find OTM calls and puts
            calls = options[options['right'].isin(['C', 'c', 'CALL', 'call'])]
            puts = options[options['right'].isin(['P', 'p', 'PUT', 'put'])]
            
            if calls.empty or puts.empty:
                return None
            
            # Get the best call and put for short legs
            short_call = calls.iloc[0]
            short_put = puts.iloc[0]
            
            # Find further OTM options for long legs
            if 'strike' in short_call and 'underlying_price' in short_call:
                # Long call should be higher strike
                long_calls = calls[calls['strike'] > short_call['strike']]
                long_call = long_calls.iloc[0] if not long_calls.empty else None
                
                # Long put should be lower strike
                long_puts = puts[puts['strike'] < short_put['strike']]
                long_put = long_puts.iloc[0] if not long_puts.empty else None
                
                if long_call is None or long_put is None:
                    return None
                
                # Calculate net credit
                net_credit = (short_call['option_price'] + short_put['option_price'] - 
                             long_call['option_price'] - long_put['option_price'])
                
                if net_credit <= 0:
                    return None
                
                # Calculate max risk (width of spread - net credit)
                call_spread_width = long_call['strike'] - short_call['strike']
                put_spread_width = short_put['strike'] - long_put['strike']
                max_risk = min(call_spread_width, put_spread_width) - net_credit
                
                # Calculate position size based on risk
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
                            'right': 'CALL',
                            'action': 'SELL',
                            'option_price': short_call.get('option_price', 0),
                            'delta': short_call.get('delta', 0)
                        },
                        {
                            'option_symbol': long_call.get('option_symbol', ''),
                            'strike': long_call.get('strike', 0),
                            'expiration': long_call.get('expiration', ''),
                            'right': 'CALL',
                            'action': 'BUY',
                            'option_price': long_call.get('option_price', 0),
                            'delta': long_call.get('delta', 0)
                        },
                        {
                            'option_symbol': short_put.get('option_symbol', ''),
                            'strike': short_put.get('strike', 0),
                            'expiration': short_put.get('expiration', ''),
                            'right': 'PUT',
                            'action': 'SELL',
                            'option_price': short_put.get('option_price', 0),
                            'delta': short_put.get('delta', 0)
                        },
                        {
                            'option_symbol': long_put.get('option_symbol', ''),
                            'strike': long_put.get('strike', 0),
                            'expiration': long_put.get('expiration', ''),
                            'right': 'PUT',
                            'action': 'BUY',
                            'option_price': long_put.get('option_price', 0),
                            'delta': long_put.get('delta', 0)
                        }
                    ],
                    'position_size': position_size,
                    'net_credit': net_credit * position_size * 100,
                    'max_risk': max_risk * position_size * 100,
                    'expected_return': net_credit / max_risk if max_risk > 0 else 0,
                    'signal_strength': signal_row.get('signal_strength', 0),
                    'stop_loss': max_risk * 0.8,  # 80% of max risk
                    'take_profit': net_credit * 0.5  # 50% of max credit
                }
                
                return trade_plan
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating iron condor trade: {e}")
            return None
    
    def _create_single_option_trade(self, option: pd.Series, signal_row: pd.Series, account_value: float) -> Dict:
        """
        Create a single option trade plan.
        
        Args:
            option: Series containing option data
            signal_row: Series containing signal data
            account_value: Current account value
            
        Returns:
            Dictionary with trade plan
        """
        try:
            # Calculate position size
            position_size = self.calculate_position_size(account_value, option['option_price'])
            
            # Determine action based on signal
            action = "BUY" if signal_row['signal'] == 1 else "SELL"
            
            # Create trade plan
            trade_plan = {
                'symbol': signal_row['symbol'],
                'underlying': signal_row.get('underlying', signal_row['symbol']),
                'signal': signal_row['signal'],
                'strategy_type': 'single_option',
                'legs': [
                    {
                        'option_symbol': option.get('option_symbol', ''),
                        'strike': option.get('strike', 0),
                        'expiration': option.get('expiration', ''),
                        'right': option.get('right', ''),
                        'action': action,
                        'option_price': option.get('option_price', 0),
                        'delta': option.get('delta', 0),
                        'gamma': option.get('gamma', 0),
                        'theta': option.get('theta', 0),
                        'vega': option.get('vega', 0)
                    }
                ],
                'position_size': position_size,
                'total_cost': option['option_price'] * position_size * 100 if action == "BUY" else 0,
                'net_credit': option['option_price'] * position_size * 100 if action == "SELL" else 0,
                'expected_return': signal_row.get('expected_return', 0),
                'signal_strength': signal_row.get('signal_strength', 0),
                'stop_loss': option['option_price'] * (1 - self.stop_loss_pct) if action == "BUY" else option['option_price'] * (1 + self.stop_loss_pct),
                'take_profit': option['option_price'] * (1 + self.take_profit_pct) if action == "BUY" else option['option_price'] * (1 - self.take_profit_pct)
            }
            
            return trade_plan
            
        except Exception as e:
            logger.error(f"Error creating single option trade: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    strategy = VolatilityStrategy()
    
    # Example: Create sample data
    data = {
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        'underlying_price': [150.0, 280.0, 2800.0, 3200.0],
        'iv_rank': [0.85, 0.15, 0.75, 0.25],
        'iv_hv_spread': [0.25, -0.15, 0.20, -0.10],
        'days_to_earnings': [20, 5, 15, 30],
        'predicted_iv_change': [-0.02, 0.03, -0.01, 0.02]
    }
    
    df = pd.DataFrame(data)
    
    # Generate signals
    signals = strategy.generate_signals(df)
    
    print("Generated Volatility Signals:")
    print(signals[['symbol', 'iv_rank', 'signal', 'signal_strength']])
