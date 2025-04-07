"""
Order Execution Module

This module provides functionality for executing orders based on strategy signals.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import time
import threading
import queue

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from execution.ibkr_connection import create_ibkr_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Class for executing orders based on strategy signals.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the order executor.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        self.execution_config = self.config.get('execution', {})
        
        # Create IBKR connection
        self.ibkr = create_ibkr_connection(config_path)
        
        # Connect to IBKR
        self.connected = False
        
        # Order tracking
        self.orders = {}
        self.positions = {}
        self.account_summary = {}
        
        # Risk management
        self.max_position_size = self.execution_config.get('max_position_size', 100)
        self.max_loss_per_trade = self.execution_config.get('max_loss_per_trade', 1000.0)
        self.max_daily_loss = self.execution_config.get('max_daily_loss', 5000.0)
        self.daily_pnl = 0.0
        
        # Order execution settings
        self.default_order_type = self.execution_config.get('default_order_type', 'LIMIT')
        self.limit_price_buffer = self.execution_config.get('limit_price_buffer', 0.01)  # 1% buffer for limit orders
        self.use_stop_loss = self.execution_config.get('use_stop_loss', True)
        self.stop_loss_pct = self.execution_config.get('stop_loss_pct', 0.05)  # 5% stop loss
        self.use_take_profit = self.execution_config.get('use_take_profit', False)
        self.take_profit_pct = self.execution_config.get('take_profit_pct', 0.10)  # 10% take profit
    
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
                'execution': {
                    'max_position_size': 100,
                    'max_loss_per_trade': 1000.0,
                    'max_daily_loss': 5000.0,
                    'default_order_type': 'LIMIT',
                    'limit_price_buffer': 0.01,
                    'use_stop_loss': True,
                    'stop_loss_pct': 0.05,
                    'use_take_profit': False,
                    'take_profit_pct': 0.10
                }
            }
    
    def connect(self) -> bool:
        """
        Connect to IBKR.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.connected:
            logger.info("Already connected to IBKR")
            return True
        
        # Connect to IBKR
        if self.ibkr.connect():
            self.connected = True
            
            # Get account summary
            self.account_summary = self.ibkr.get_account_summary()
            
            # Get positions
            self.positions = self.ibkr.get_positions()
            
            logger.info(f"Connected to IBKR. Account value: ${self.account_summary.get('NetLiquidation', 0.0):.2f}")
            return True
        else:
            logger.error("Failed to connect to IBKR")
            return False
    
    def disconnect(self):
        """
        Disconnect from IBKR.
        """
        if self.connected:
            self.ibkr.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def check_connection(self) -> bool:
        """
        Check if connected to IBKR.
        
        Returns:
            True if connected, False otherwise
        """
        if not self.connected:
            return False
        
        return self.ibkr.check_connection()
    
    def update_account_info(self):
        """
        Update account information.
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return
        
        # Get account summary
        self.account_summary = self.ibkr.get_account_summary()
        
        # Get positions
        self.positions = self.ibkr.get_positions()
        
        logger.info(f"Updated account info. Account value: ${self.account_summary.get('NetLiquidation', 0.0):.2f}")
    
    def execute_signals(self, signals: List[Dict]) -> Dict:
        """
        Execute trading signals.
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            Dict containing execution results
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {'success': False, 'error': 'Not connected to IBKR'}
        
        # Update account info
        self.update_account_info()
        
        # Check risk limits
        if self.daily_pnl <= -self.max_daily_loss:
            logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return {'success': False, 'error': 'Daily loss limit reached'}
        
        # Process signals
        results = []
        
        for signal in signals:
            # Validate signal
            if not self._validate_signal(signal):
                logger.warning(f"Invalid signal: {signal}")
                results.append({
                    'signal': signal,
                    'success': False,
                    'error': 'Invalid signal format'
                })
                continue
            
            # Extract signal details
            symbol = signal['symbol']
            direction = signal['direction']  # BUY or SELL
            quantity = signal.get('quantity', 1)
            signal_type = signal.get('type', 'stock')  # stock, option, spread
            
            # Apply position size limits
            quantity = min(quantity, self.max_position_size)
            
            # Check if we already have a position
            current_position = 0
            if symbol in self.positions:
                current_position = self.positions[symbol]['position']
            
            # Determine action based on direction and current position
            if direction == 'BUY':
                if current_position >= 0:
                    # Adding to long position or opening new long
                    action = 'BUY'
                    trade_quantity = quantity
                else:
                    # Reducing short position
                    action = 'BUY'
                    trade_quantity = min(quantity, abs(current_position))
            else:  # SELL
                if current_position <= 0:
                    # Adding to short position or opening new short
                    action = 'SELL'
                    trade_quantity = quantity
                else:
                    # Reducing long position
                    action = 'SELL'
                    trade_quantity = min(quantity, current_position)
            
            # Skip if quantity is zero
            if trade_quantity <= 0:
                logger.info(f"Skipping signal for {symbol}: zero quantity after adjustment")
                results.append({
                    'signal': signal,
                    'success': False,
                    'error': 'Zero quantity after adjustment'
                })
                continue
            
            # Execute based on signal type
            if signal_type == 'stock':
                result = self._execute_stock_signal(symbol, action, trade_quantity, signal)
            elif signal_type == 'option':
                result = self._execute_option_signal(symbol, action, trade_quantity, signal)
            elif signal_type == 'spread':
                result = self._execute_spread_signal(symbol, action, trade_quantity, signal)
            else:
                logger.warning(f"Unsupported signal type: {signal_type}")
                result = {
                    'success': False,
                    'error': f"Unsupported signal type: {signal_type}"
                }
            
            # Add signal to result
            result['signal'] = signal
            results.append(result)
        
        # Update account info after executions
        self.update_account_info()
        
        return {
            'success': True,
            'results': results
        }
    
    def _validate_signal(self, signal: Dict) -> bool:
        """
        Validate a trading signal.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required_fields = ['symbol', 'direction']
        for field in required_fields:
            if field not in signal:
                logger.warning(f"Missing required field in signal: {field}")
                return False
        
        # Check direction
        if signal['direction'] not in ['BUY', 'SELL']:
            logger.warning(f"Invalid direction in signal: {signal['direction']}")
            return False
        
        # Check signal type
        if 'type' in signal and signal['type'] not in ['stock', 'option', 'spread']:
            logger.warning(f"Invalid signal type: {signal['type']}")
            return False
        
        # Check option-specific fields
        if signal.get('type') == 'option':
            option_fields = ['expiration', 'strike', 'option_type']
            for field in option_fields:
                if field not in signal:
                    logger.warning(f"Missing required field for option signal: {field}")
                    return False
            
            # Check option type
            if signal['option_type'] not in ['call', 'put', 'C', 'P']:
                logger.warning(f"Invalid option type: {signal['option_type']}")
                return False
        
        # Check spread-specific fields
        if signal.get('type') == 'spread':
            if 'legs' not in signal or not isinstance(signal['legs'], list) or len(signal['legs']) < 2:
                logger.warning(f"Invalid spread legs: {signal.get('legs')}")
                return False
            
            # Check each leg
            for leg in signal['legs']:
                if not isinstance(leg, dict):
                    logger.warning(f"Invalid spread leg format: {leg}")
                    return False
                
                leg_fields = ['symbol', 'direction', 'quantity']
                for field in leg_fields:
                    if field not in leg:
                        logger.warning(f"Missing required field for spread leg: {field}")
                        return False
                
                if leg['direction'] not in ['BUY', 'SELL']:
                    logger.warning(f"Invalid direction in spread leg: {leg['direction']}")
                    return False
        
        return True
    
    def _execute_stock_signal(self, symbol: str, action: str, quantity: float, signal: Dict) -> Dict:
        """
        Execute a stock signal.
        
        Args:
            symbol: Stock symbol
            action: Action (BUY or SELL)
            quantity: Quantity to trade
            signal: Original signal dictionary
            
        Returns:
            Dict containing execution result
        """
        try:
            # Get market data
            market_data = self.ibkr.get_market_data(symbol)
            
            if not market_data:
                logger.warning(f"Failed to get market data for {symbol}")
                return {
                    'success': False,
                    'error': f"Failed to get market data for {symbol}"
                }
            
            # Determine price
            if action == 'BUY':
                market_price = market_data.get('ask', market_data.get('last', 0.0))
            else:  # SELL
                market_price = market_data.get('bid', market_data.get('last', 0.0))
            
            if market_price <= 0:
                logger.warning(f"Invalid market price for {symbol}: {market_price}")
                return {
                    'success': False,
                    'error': f"Invalid market price for {symbol}: {market_price}"
                }
            
            # Create contract
            contract = self.ibkr.create_stock_contract(symbol)
            
            # Create order
            order_type = signal.get('order_type', self.default_order_type)
            
            if order_type == 'MARKET':
                order = self.ibkr.create_market_order(action, quantity)
            elif order_type == 'LIMIT':
                # Calculate limit price with buffer
                if action == 'BUY':
                    limit_price = market_price * (1 + self.limit_price_buffer)
                else:  # SELL
                    limit_price = market_price * (1 - self.limit_price_buffer)
                
                # Override with signal limit price if provided
                if 'limit_price' in signal:
                    limit_price = signal['limit_price']
                
                order = self.ibkr.create_limit_order(action, quantity, limit_price)
            else:
                logger.warning(f"Unsupported order type: {order_type}")
                return {
                    'success': False,
                    'error': f"Unsupported order type: {order_type}"
                }
            
            # Place order
            order_id = self.ibkr.place_order(contract, order)
            
            if order_id <= 0:
                logger.warning(f"Failed to place order for {symbol}")
                return {
                    'success': False,
                    'error': f"Failed to place order for {symbol}"
                }
            
            # Track order
            self.orders[order_id] = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type,
                'market_price': market_price,
                'timestamp': datetime.now(),
                'status': 'Submitted'
            }
            
            # Add stop loss if enabled
            if self.use_stop_loss and action == 'BUY':
                stop_price = market_price * (1 - self.stop_loss_pct)
                stop_order = self.ibkr.create_stop_order('SELL', quantity, stop_price)
                stop_order_id = self.ibkr.place_order(contract, stop_order)
                
                if stop_order_id > 0:
                    self.orders[stop_order_id] = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity,
                        'order_type': 'STOP',
                        'stop_price': stop_price,
                        'timestamp': datetime.now(),
                        'status': 'Submitted',
                        'parent_order_id': order_id
                    }
                    
                    logger.info(f"Placed stop loss order {stop_order_id} for {symbol} at {stop_price:.2f}")
            
            # Add take profit if enabled
            if self.use_take_profit and action == 'BUY':
                take_profit_price = market_price * (1 + self.take_profit_pct)
                take_profit_order = self.ibkr.create_limit_order('SELL', quantity, take_profit_price)
                take_profit_order_id = self.ibkr.place_order(contract, take_profit_order)
                
                if take_profit_order_id > 0:
                    self.orders[take_profit_order_id] = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity,
                        'order_type': 'LIMIT',
                        'limit_price': take_profit_price,
                        'timestamp': datetime.now(),
                        'status': 'Submitted',
                        'parent_order_id': order_id
                    }
                    
                    logger.info(f"Placed take profit order {take_profit_order_id} for {symbol} at {take_profit_price:.2f}")
            
            logger.info(f"Placed order {order_id} for {symbol}: {action} {quantity} @ {market_price:.2f}")
            
            return {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': market_price
            }
        except Exception as e:
            logger.error(f"Error executing stock signal for {symbol}: {e}")
            return {
                'success': False,
                'error': f"Error executing stock signal: {e}"
            }
    
    def _execute_option_signal(self, symbol: str, action: str, quantity: float, signal: Dict) -> Dict:
        """
        Execute an option signal.
        
        Args:
            symbol: Underlying symbol
            action: Action (BUY or SELL)
            quantity: Quantity to trade
            signal: Original signal dictionary
            
        Returns:
            Dict containing execution result
        """
        try:
            # Extract option details
            expiration = signal['expiration']
            strike = signal['strike']
            option_type = signal['option_type']
            
            # Convert option type to IB format
            if option_type.upper() in ['CALL', 'C']:
                right = 'C'
            else:
                right = 'P'
            
            # Get option market data
            option_data = self.ibkr.get_option_market_data(symbol, expiration, strike, right)
            
            if not option_data:
                logger.warning(f"Failed to get option data for {symbol} {expiration} {strike} {right}")
                return {
                    'success': False,
                    'error': f"Failed to get option data for {symbol} {expiration} {strike} {right}"
                }
            
            # Determine price
            if action == 'BUY':
                market_price = option_data.get('ask', option_data.get('last', option_data.get('mid', 0.0)))
            else:  # SELL
                market_price = option_data.get('bid', option_data.get('last', option_data.get('mid', 0.0)))
            
            if market_price <= 0:
                logger.warning(f"Invalid option price: {market_price}")
                return {
                    'success': False,
                    'error': f"Invalid option price: {market_price}"
                }
            
            # Create contract
            contract = self.ibkr.create_option_contract(symbol, expiration, strike, right)
            
            # Create order
            order_type = signal.get('order_type', self.default_order_type)
            
            if order_type == 'MARKET':
                order = self.ibkr.create_market_order(action, quantity)
            elif order_type == 'LIMIT':
                # Calculate limit price with buffer
                if action == 'BUY':
                    limit_price = market_price * (1 + self.limit_price_buffer)
                else:  # SELL
                    limit_price = market_price * (1 - self.limit_price_buffer)
                
                # Override with signal limit price if provided
                if 'limit_price' in signal:
                    limit_price = signal['limit_price']
                
                order = self.ibkr.create_limit_order(action, quantity, limit_price)
            else:
                logger.warning(f"Unsupported order type: {order_type}")
                return {
                    'success': False,
                    'error': f"Unsupported order type: {order_type}"
                }
            
            # Place order
            order_id = self.ibkr.place_order(contract, order)
            
            if order_id <= 0:
                logger.warning(f"Failed to place option order")
                return {
                    'success': False,
                    'error': f"Failed to place option order"
                }
            
            # Track order
            option_symbol = f"{symbol}_{expiration}_{strike}_{right}"
            self.orders[order_id] = {
                'symbol': option_symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type,
                'market_price': market_price,
                'timestamp': datetime.now(),
                'status': 'Submitted',
                'underlying': symbol,
                'expiration': expiration,
                'strike': strike,
                'right': right
            }
            
            logger.info(f"Placed option order {order_id} for {option_symbol}: {action} {quantity} @ {market_price:.2f}")
            
            return {
                'success': True,
                'order_id': order_id,
                'symbol': option_symbol,
                'action': action,
                'quantity': quantity,
                'price': market_price,
                'underlying': symbol,
                'expiration': expiration,
                'strike': strike,
                'right': right
            }
        except Exception as e:
            logger.error(f"Error executing option signal for {symbol}: {e}")
            return {
                'success': False,
                'error': f"Error executing option signal: {e}"
            }
    
    def _execute_spread_signal(self, symbol: str, action: str, quantity: float, signal: Dict) -> Dict:
        """
        Execute a spread signal.
        
        Args:
            symbol: Symbol (ignored, using legs)
            action: Action (ignored, using legs)
            quantity: Quantity (ignored, using legs)
            signal: Original signal dictionary
            
        Returns:
            Dict containing execution result
        """
        try:
            # Extract legs
            legs = signal['legs']
            
            # Execute each leg
            leg_results = []
            
            for leg in legs:
                leg_symbol = leg['symbol']
                leg_action = leg['direction']
                leg_quantity = leg['quantity']
                
                # Determine leg type
                if 'type' not in leg:
                    # Try to determine type from symbol format
                    if '_' in leg_symbol and len(leg_symbol.split('_')) == 4:
                        # Looks like an option symbol: AAPL_20230721_150_C
                        leg['type'] = 'option'
                        parts = leg_symbol.split('_')
                        leg['symbol'] = parts[0]
                        leg['expiration'] = parts[1]
                        leg['strike'] = float(parts[2])
                        leg['option_type'] = parts[3]
                    else:
                        leg['type'] = 'stock'
                
                # Execute leg
                if leg.get('type') == 'option':
                    result = self._execute_option_signal(
                        leg['symbol'], leg_action, leg_quantity, leg
                    )
                else:
                    result = self._execute_stock_signal(
                        leg_symbol, leg_action, leg_quantity, leg
                    )
                
                leg_results.append(result)
            
            # Check if all legs were successful
            all_success = all(result['success'] for result in leg_results)
            
            if all_success:
                logger.info(f"Successfully executed spread with {len(legs)} legs")
                return {
                    'success': True,
                    'spread_type': signal.get('spread_type', 'custom'),
                    'legs': leg_results
                }
            else:
                # If any leg failed, try to cancel successful legs
                for result in leg_results:
                    if result['success'] and 'order_id' in result:
                        self.ibkr.cancel_order(result['order_id'])
                
                logger.warning(f"Failed to execute all legs of spread")
                return {
                    'success': False,
                    'error': f"Failed to execute all legs of spread",
                    'legs': leg_results
                }
        except Exception as e:
            logger.error(f"Error executing spread signal: {e}")
            return {
                'success': False,
                'error': f"Error executing spread signal: {e}"
            }
    
    def cancel_all_orders(self) -> Dict:
        """
        Cancel all open orders.
        
        Returns:
            Dict containing cancellation results
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {'success': False, 'error': 'Not connected to IBKR'}
        
        results = []
        
        for order_id, order_info in self.orders.items():
            if order_info['status'] in ['Submitted', 'PreSubmitted']:
                if self.ibkr.cancel_order(order_id):
                    order_info['status'] = 'Cancelled'
                    results.append({
                        'order_id': order_id,
                        'symbol': order_info['symbol'],
                        'success': True
                    })
                else:
                    results.append({
                        'order_id': order_id,
                        'symbol': order_info['symbol'],
                        'success': False,
                        'error': 'Failed to cancel order'
                    })
        
        return {
            'success': True,
            'results': results
        }
    
    def get_order_status(self, order_id: int) -> Dict:
        """
        Get status of an order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Dict containing order status
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {'success': False, 'error': 'Not connected to IBKR'}
        
        # Get status from IBKR
        ibkr_status = self.ibkr.get_order_status(order_id)
        
        if not ibkr_status:
            logger.warning(f"Order {order_id} not found")
            return {'success': False, 'error': f"Order {order_id} not found"}
        
        # Update our tracking
        if order_id in self.orders:
            self.orders[order_id]['status'] = ibkr_status['status']
            self.orders[order_id]['filled'] = ibkr_status['filled']
            self.orders[order_id]['remaining'] = ibkr_status['remaining']
            self.orders[order_id]['avg_fill_price'] = ibkr_status['avg_fill_price']
            self.orders[order_id]['last_update'] = datetime.now()
        
        return {
            'success': True,
            'order_id': order_id,
            'status': ibkr_status
        }
    
    def get_all_orders(self) -> Dict:
        """
        Get all orders.
        
        Returns:
            Dict containing all orders
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {'success': False, 'error': 'Not connected to IBKR'}
        
        # Update status of all orders
        for order_id in list(self.orders.keys()):
            self.get_order_status(order_id)
        
        return {
            'success': True,
            'orders': self.orders
        }
    
    def get_positions(self) -> Dict:
        """
        Get current positions.
        
        Returns:
            Dict containing positions
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {'success': False, 'error': 'Not connected to IBKR'}
        
        # Update positions
        self.positions = self.ibkr.get_positions()
        
        return {
            'success': True,
            'positions': self.positions
        }
    
    def get_account_summary(self) -> Dict:
        """
        Get account summary.
        
        Returns:
            Dict containing account summary
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {'success': False, 'error': 'Not connected to IBKR'}
        
        # Update account summary
        self.account_summary = self.ibkr.get_account_summary()
        
        return {
            'success': True,
            'account_summary': self.account_summary
        }
    
    def calculate_daily_pnl(self) -> float:
        """
        Calculate daily P&L.
        
        Returns:
            Daily P&L
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return 0.0
        
        # Get executions
        executions = self.ibkr.get_executions()
        
        # Filter today's executions
        today = datetime.now().strftime('%Y%m%d')
        today_executions = {
            exec_id: exec_info for exec_id, exec_info in executions.items()
            if exec_info['time'].startswith(today)
        }
        
        # Calculate P&L from executions
        pnl = 0.0
        
        for exec_info in today_executions.values():
            if exec_info['side'] == 'BOT':
                pnl -= exec_info['price'] * exec_info['shares']
            else:  # SLD
                pnl += exec_info['price'] * exec_info['shares']
        
        # Update daily P&L
        self.daily_pnl = pnl
        
        return pnl
    
    def reset_daily_pnl(self):
        """
        Reset daily P&L.
        """
        self.daily_pnl = 0.0
        logger.info("Reset daily P&L")


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    executor = OrderExecutor()
    
    # Connect to IBKR
    if executor.connect():
        # Get account summary
        account_info = executor.get_account_summary()
        print("Account Summary:")
        print(account_info)
        
        # Get positions
        positions = executor.get_positions()
        print("\nPositions:")
        print(positions)
        
        # Execute a test signal
        test_signal = {
            'symbol': 'AAPL',
            'direction': 'BUY',
            'quantity': 10,
            'type': 'stock',
            'order_type': 'LIMIT'
        }
        
        result = executor.execute_signals([test_signal])
        print("\nExecution Result:")
        print(result)
        
        # Get order status
        if result['success'] and 'results' in result and result['results'][0]['success']:
            order_id = result['results'][0]['order_id']
            status = executor.get_order_status(order_id)
            print("\nOrder Status:")
            print(status)
        
        # Disconnect
        executor.disconnect()
    else:
        print("Failed to connect to IBKR")
