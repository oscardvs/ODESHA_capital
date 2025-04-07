"""
IBKR Connection Module

This module provides functionality for connecting to Interactive Brokers
and executing trades.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import IB API
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.execution import Execution
    from ibapi.common import BarData
    from ibapi.ticktype import TickTypeEnum
    HAS_IBAPI = True
except ImportError:
    logger.warning("IB API not found. Using simulation mode.")
    HAS_IBAPI = False


class IBKRConnection(EWrapper, EClient):
    """
    Connection to Interactive Brokers API.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the IBKR connection.
        
        Args:
            config_path: Path to the configuration file
        """
        # Initialize EWrapper and EClient
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.ibkr_config = self.config.get('ibkr', {})
        
        # Connection parameters
        self.host = self.ibkr_config.get('host', '127.0.0.1')
        self.port = self.ibkr_config.get('port', 7497)  # 7497 for TWS Paper, 7496 for TWS Live, 4002 for Gateway
        self.client_id = self.ibkr_config.get('client_id', 1)
        self.account = self.ibkr_config.get('account', '')
        
        # State variables
        self.connected = False
        self.next_order_id = None
        self.next_request_id = 1
        
        # Data storage
        self.positions = {}
        self.account_summary = {}
        self.market_data = {}
        self.options_chains = {}
        self.order_status = {}
        self.executions = {}
        
        # Synchronization
        self.data_events = {}
        self.data_queues = {}
    
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
                'ibkr': {
                    'host': '127.0.0.1',
                    'port': 7497,
                    'client_id': 1,
                    'account': ''
                }
            }
    
    def connect(self) -> bool:
        """
        Connect to IBKR TWS or Gateway.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not HAS_IBAPI:
            logger.error("IB API not available. Cannot connect.")
            return False
        
        try:
            # Connect to TWS
            logger.info(f"Connecting to IBKR at {self.host}:{self.port} with client ID {self.client_id}")
            self.connect(self.host, self.port, self.client_id)
            
            # Wait for nextValidId
            timeout = 10  # seconds
            wait_time = 0
            while self.next_order_id is None and wait_time < timeout:
                time.sleep(0.1)
                wait_time += 0.1
            
            if self.next_order_id is None:
                logger.error("Failed to receive next valid order ID")
                return False
            
            self.connected = True
            logger.info(f"Connected to IBKR. Next valid order ID: {self.next_order_id}")
            
            # Request account updates
            self.reqAccountUpdates(True, self.account)
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to IBKR: {e}")
            return False
    
    def disconnect(self):
        """
        Disconnect from IBKR TWS or Gateway.
        """
        if self.connected:
            # Cancel account updates
            self.reqAccountUpdates(False, self.account)
            
            # Disconnect
            self.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def check_connection(self) -> bool:
        """
        Check if connected to IBKR.
        
        Returns:
            True if connected, False otherwise
        """
        return self.connected
    
    def get_account_summary(self) -> Dict:
        """
        Get account summary.
        
        Returns:
            Dict containing account summary
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {}
        
        # Request account summary
        req_id = self._get_next_request_id()
        self.data_events[req_id] = threading.Event()
        self.data_queues[req_id] = queue.Queue()
        
        self.reqAccountSummary(req_id, "All", "NetLiquidation,TotalCashValue,AvailableFunds,BuyingPower")
        
        # Wait for data
        timeout = 5  # seconds
        if not self.data_events[req_id].wait(timeout):
            logger.warning(f"Timeout waiting for account summary")
            del self.data_events[req_id]
            del self.data_queues[req_id]
            return self.account_summary
        
        # Process data
        while not self.data_queues[req_id].empty():
            data = self.data_queues[req_id].get()
            self.account_summary[data['tag']] = float(data['value'])
        
        # Clean up
        del self.data_events[req_id]
        del self.data_queues[req_id]
        
        return self.account_summary
    
    def get_positions(self) -> Dict:
        """
        Get current positions.
        
        Returns:
            Dict containing positions
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {}
        
        # Request positions
        self.positions = {}
        req_id = self._get_next_request_id()
        self.data_events[req_id] = threading.Event()
        self.data_queues[req_id] = queue.Queue()
        
        self.reqPositions()
        
        # Wait for data
        timeout = 5  # seconds
        if not self.data_events[req_id].wait(timeout):
            logger.warning(f"Timeout waiting for positions")
            del self.data_events[req_id]
            del self.data_queues[req_id]
            return self.positions
        
        # Process data
        while not self.data_queues[req_id].empty():
            data = self.data_queues[req_id].get()
            contract = data['contract']
            position = data['position']
            avg_cost = data['avgCost']
            
            symbol = contract.symbol
            if contract.secType == "OPT":
                # Format: AAPL_20230721_150_C
                symbol = f"{contract.symbol}_{contract.lastTradeDateOrContractMonth}_{contract.strike}_{contract.right}"
            
            self.positions[symbol] = {
                'symbol': symbol,
                'position': position,
                'avg_cost': avg_cost,
                'contract': contract
            }
        
        # Clean up
        del self.data_events[req_id]
        del self.data_queues[req_id]
        
        return self.positions
    
    def get_market_data(self, symbol: str, sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD") -> Dict:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            sec_type: Security type (STK, OPT, FUT, etc.)
            exchange: Exchange
            currency: Currency
            
        Returns:
            Dict containing market data
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {}
        
        # Create contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        
        # Request market data
        req_id = self._get_next_request_id()
        self.data_events[req_id] = threading.Event()
        self.data_queues[req_id] = queue.Queue()
        
        self.reqMktData(req_id, contract, "", False, False, [])
        
        # Wait for data
        timeout = 5  # seconds
        if not self.data_events[req_id].wait(timeout):
            logger.warning(f"Timeout waiting for market data for {symbol}")
            self.cancelMktData(req_id)
            del self.data_events[req_id]
            del self.data_queues[req_id]
            return {}
        
        # Process data
        market_data = {}
        while not self.data_queues[req_id].empty():
            data = self.data_queues[req_id].get()
            field = data['field']
            value = data['value']
            
            if field == TickTypeEnum.BID:
                market_data['bid'] = value
            elif field == TickTypeEnum.ASK:
                market_data['ask'] = value
            elif field == TickTypeEnum.LAST:
                market_data['last'] = value
            elif field == TickTypeEnum.HIGH:
                market_data['high'] = value
            elif field == TickTypeEnum.LOW:
                market_data['low'] = value
            elif field == TickTypeEnum.VOLUME:
                market_data['volume'] = value
            elif field == TickTypeEnum.BID_SIZE:
                market_data['bid_size'] = value
            elif field == TickTypeEnum.ASK_SIZE:
                market_data['ask_size'] = value
            elif field == TickTypeEnum.LAST_SIZE:
                market_data['last_size'] = value
        
        # Cancel market data
        self.cancelMktData(req_id)
        
        # Clean up
        del self.data_events[req_id]
        del self.data_queues[req_id]
        
        # Store data
        self.market_data[symbol] = market_data
        
        return market_data
    
    def get_options_chain(self, symbol: str, exchange: str = "SMART", currency: str = "USD") -> Dict:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Symbol to get options chain for
            exchange: Exchange
            currency: Currency
            
        Returns:
            Dict containing options chain
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {}
        
        # Create contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = currency
        
        # Request options chain
        req_id = self._get_next_request_id()
        self.data_events[req_id] = threading.Event()
        self.data_queues[req_id] = queue.Queue()
        
        self.reqSecDefOptParams(req_id, symbol, exchange, "STK", contract.conId)
        
        # Wait for data
        timeout = 10  # seconds
        if not self.data_events[req_id].wait(timeout):
            logger.warning(f"Timeout waiting for options chain for {symbol}")
            del self.data_events[req_id]
            del self.data_queues[req_id]
            return {}
        
        # Process data
        options_chain = {}
        while not self.data_queues[req_id].empty():
            data = self.data_queues[req_id].get()
            
            # Extract expirations and strikes
            expirations = data['expirations']
            strikes = data['strikes']
            
            # Store data
            options_chain = {
                'expirations': expirations,
                'strikes': strikes
            }
        
        # Clean up
        del self.data_events[req_id]
        del self.data_queues[req_id]
        
        # Store data
        self.options_chains[symbol] = options_chain
        
        return options_chain
    
    def get_option_market_data(self, symbol: str, expiration: str, strike: float, right: str, exchange: str = "SMART", currency: str = "USD") -> Dict:
        """
        Get market data for an option.
        
        Args:
            symbol: Symbol of the underlying
            expiration: Expiration date (YYYYMMDD)
            strike: Strike price
            right: Right (C or P)
            exchange: Exchange
            currency: Currency
            
        Returns:
            Dict containing option market data
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {}
        
        # Create contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = exchange
        contract.currency = currency
        contract.lastTradeDateOrContractMonth = expiration
        contract.strike = strike
        contract.right = right
        
        # Request market data
        req_id = self._get_next_request_id()
        self.data_events[req_id] = threading.Event()
        self.data_queues[req_id] = queue.Queue()
        
        self.reqMktData(req_id, contract, "106,100,101,104,106,291,293", False, False, [])
        
        # Wait for data
        timeout = 5  # seconds
        if not self.data_events[req_id].wait(timeout):
            logger.warning(f"Timeout waiting for option market data for {symbol} {expiration} {strike} {right}")
            self.cancelMktData(req_id)
            del self.data_events[req_id]
            del self.data_queues[req_id]
            return {}
        
        # Process data
        option_data = {}
        while not self.data_queues[req_id].empty():
            data = self.data_queues[req_id].get()
            field = data['field']
            value = data['value']
            
            if field == TickTypeEnum.BID:
                option_data['bid'] = value
            elif field == TickTypeEnum.ASK:
                option_data['ask'] = value
            elif field == TickTypeEnum.LAST:
                option_data['last'] = value
            elif field == TickTypeEnum.MODEL_OPTION:
                option_data['model_price'] = value
            elif field == TickTypeEnum.OPTION_HISTORICAL_VOL:
                option_data['historical_volatility'] = value
            elif field == TickTypeEnum.OPTION_IMPLIED_VOL:
                option_data['implied_volatility'] = value
            elif field == TickTypeEnum.OPTION_CALL_OPEN_INTEREST:
                option_data['call_open_interest'] = value
            elif field == TickTypeEnum.OPTION_PUT_OPEN_INTEREST:
                option_data['put_open_interest'] = value
            elif field == TickTypeEnum.OPTION_CALL_VOLUME:
                option_data['call_volume'] = value
            elif field == TickTypeEnum.OPTION_PUT_VOLUME:
                option_data['put_volume'] = value
            elif field == TickTypeEnum.BID_SIZE:
                option_data['bid_size'] = value
            elif field == TickTypeEnum.ASK_SIZE:
                option_data['ask_size'] = value
            elif field == TickTypeEnum.LAST_SIZE:
                option_data['last_size'] = value
        
        # Calculate mid price
        if 'bid' in option_data and 'ask' in option_data:
            option_data['mid'] = (option_data['bid'] + option_data['ask']) / 2
        
        # Cancel market data
        self.cancelMktData(req_id)
        
        # Clean up
        del self.data_events[req_id]
        del self.data_queues[req_id]
        
        # Format option symbol
        option_symbol = f"{symbol}_{expiration}_{strike}_{right}"
        
        # Store data
        self.market_data[option_symbol] = option_data
        
        return option_data
    
    def place_order(self, contract: Contract, order: Order) -> int:
        """
        Place an order.
        
        Args:
            contract: Contract to trade
            order: Order details
            
        Returns:
            Order ID if successful, -1 otherwise
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return -1
        
        try:
            # Get next order ID
            order_id = self.next_order_id
            self.next_order_id += 1
            
            # Place order
            logger.info(f"Placing order {order_id}: {order.action} {order.totalQuantity} {contract.symbol} {contract.secType} @ {order.lmtPrice if order.orderType == 'LMT' else 'MKT'}")
            self.placeOrder(order_id, contract, order)
            
            # Initialize order status
            self.order_status[order_id] = {
                'status': 'Submitted',
                'filled': 0,
                'remaining': order.totalQuantity,
                'avg_fill_price': 0.0,
                'last_fill_price': 0.0,
                'client_id': self.client_id,
                'perm_id': 0,
                'parent_id': 0,
                'last_update': datetime.now()
            }
            
            return order_id
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return -1
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation request successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return False
        
        try:
            # Cancel order
            logger.info(f"Cancelling order {order_id}")
            self.cancelOrder(order_id)
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
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
            return {}
        
        if order_id not in self.order_status:
            logger.warning(f"Order {order_id} not found")
            return {}
        
        return self.order_status[order_id]
    
    def get_executions(self) -> Dict:
        """
        Get all executions.
        
        Returns:
            Dict containing executions
        """
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {}
        
        # Request executions
        req_id = self._get_next_request_id()
        self.data_events[req_id] = threading.Event()
        self.data_queues[req_id] = queue.Queue()
        
        self.reqExecutions(req_id, ExecutionFilter())
        
        # Wait for data
        timeout = 5  # seconds
        if not self.data_events[req_id].wait(timeout):
            logger.warning(f"Timeout waiting for executions")
            del self.data_events[req_id]
            del self.data_queues[req_id]
            return self.executions
        
        # Process data
        while not self.data_queues[req_id].empty():
            data = self.data_queues[req_id].get()
            contract = data['contract']
            execution = data['execution']
            
            exec_id = execution.execId
            self.executions[exec_id] = {
                'exec_id': exec_id,
                'order_id': execution.orderId,
                'symbol': contract.symbol,
                'sec_type': contract.secType,
                'side': execution.side,
                'shares': execution.shares,
                'price': execution.price,
                'time': execution.time,
                'commission': execution.commission
            }
        
        # Clean up
        del self.data_events[req_id]
        del self.data_queues[req_id]
        
        return self.executions
    
    def create_stock_contract(self, symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
        """
        Create a stock contract.
        
        Args:
            symbol: Symbol
            exchange: Exchange
            currency: Currency
            
        Returns:
            Contract object
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = currency
        return contract
    
    def create_option_contract(self, symbol: str, expiration: str, strike: float, right: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
        """
        Create an option contract.
        
        Args:
            symbol: Symbol of the underlying
            expiration: Expiration date (YYYYMMDD)
            strike: Strike price
            right: Right (C or P)
            exchange: Exchange
            currency: Currency
            
        Returns:
            Contract object
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = exchange
        contract.currency = currency
        contract.lastTradeDateOrContractMonth = expiration
        contract.strike = strike
        contract.right = right
        return contract
    
    def create_market_order(self, action: str, quantity: float) -> Order:
        """
        Create a market order.
        
        Args:
            action: Action (BUY or SELL)
            quantity: Quantity
            
        Returns:
            Order object
        """
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity
        return order
    
    def create_limit_order(self, action: str, quantity: float, limit_price: float) -> Order:
        """
        Create a limit order.
        
        Args:
            action: Action (BUY or SELL)
            quantity: Quantity
            limit_price: Limit price
            
        Returns:
            Order object
        """
        order = Order()
        order.action = action
        order.orderType = "LMT"
        order.totalQuantity = quantity
        order.lmtPrice = limit_price
        return order
    
    def create_stop_order(self, action: str, quantity: float, stop_price: float) -> Order:
        """
        Create a stop order.
        
        Args:
            action: Action (BUY or SELL)
            quantity: Quantity
            stop_price: Stop price
            
        Returns:
            Order object
        """
        order = Order()
        order.action = action
        order.orderType = "STP"
        order.totalQuantity = quantity
        order.auxPrice = stop_price
        return order
    
    def create_stop_limit_order(self, action: str, quantity: float, stop_price: float, limit_price: float) -> Order:
        """
        Create a stop-limit order.
        
        Args:
            action: Action (BUY or SELL)
            quantity: Quantity
            stop_price: Stop price
            limit_price: Limit price
            
        Returns:
            Order object
        """
        order = Order()
        order.action = action
        order.orderType = "STP LMT"
        order.totalQuantity = quantity
        order.auxPrice = stop_price
        order.lmtPrice = limit_price
        return order
    
    def _get_next_request_id(self) -> int:
        """
        Get next request ID.
        
        Returns:
            Next request ID
        """
        req_id = self.next_request_id
        self.next_request_id += 1
        return req_id
    
    # EWrapper callbacks
    
    def nextValidId(self, orderId: int):
        """
        Callback for next valid order ID.
        
        Args:
            orderId: Next valid order ID
        """
        self.next_order_id = orderId
    
    def error(self, reqId: int, errorCode: int, errorString: str):
        """
        Callback for errors.
        
        Args:
            reqId: Request ID
            errorCode: Error code
            errorString: Error message
        """
        if errorCode == 2104 or errorCode == 2106:  # Market data farm connection is OK
            return
        
        logger.error(f"Error {errorCode}: {errorString} (reqId: {reqId})")
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """
        Callback for account summary.
        
        Args:
            reqId: Request ID
            account: Account
            tag: Tag
            value: Value
            currency: Currency
        """
        if reqId in self.data_queues:
            self.data_queues[reqId].put({
                'account': account,
                'tag': tag,
                'value': value,
                'currency': currency
            })
    
    def accountSummaryEnd(self, reqId: int):
        """
        Callback for end of account summary.
        
        Args:
            reqId: Request ID
        """
        if reqId in self.data_events:
            self.data_events[reqId].set()
    
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """
        Callback for position.
        
        Args:
            account: Account
            contract: Contract
            position: Position
            avgCost: Average cost
        """
        if 1 in self.data_queues:  # Use a fixed reqId for positions
            self.data_queues[1].put({
                'account': account,
                'contract': contract,
                'position': position,
                'avgCost': avgCost
            })
    
    def positionEnd(self):
        """
        Callback for end of positions.
        """
        if 1 in self.data_events:  # Use a fixed reqId for positions
            self.data_events[1].set()
    
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib: int):
        """
        Callback for tick price.
        
        Args:
            reqId: Request ID
            tickType: Tick type
            price: Price
            attrib: Attributes
        """
        if reqId in self.data_queues:
            self.data_queues[reqId].put({
                'field': tickType,
                'value': price
            })
            self.data_events[reqId].set()
    
    def tickSize(self, reqId: int, tickType: int, size: int):
        """
        Callback for tick size.
        
        Args:
            reqId: Request ID
            tickType: Tick type
            size: Size
        """
        if reqId in self.data_queues:
            self.data_queues[reqId].put({
                'field': tickType,
                'value': size
            })
            self.data_events[reqId].set()
    
    def securityDefinitionOptionParameter(self, reqId: int, exchange: str, underlyingConId: int, tradingClass: str, multiplier: str, expirations: list, strikes: list):
        """
        Callback for security definition option parameters.
        
        Args:
            reqId: Request ID
            exchange: Exchange
            underlyingConId: Underlying contract ID
            tradingClass: Trading class
            multiplier: Multiplier
            expirations: List of expirations
            strikes: List of strikes
        """
        if reqId in self.data_queues:
            self.data_queues[reqId].put({
                'exchange': exchange,
                'underlyingConId': underlyingConId,
                'tradingClass': tradingClass,
                'multiplier': multiplier,
                'expirations': expirations,
                'strikes': strikes
            })
    
    def securityDefinitionOptionParameterEnd(self, reqId: int):
        """
        Callback for end of security definition option parameters.
        
        Args:
            reqId: Request ID
        """
        if reqId in self.data_events:
            self.data_events[reqId].set()
    
    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float, avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCapPrice: float):
        """
        Callback for order status.
        
        Args:
            orderId: Order ID
            status: Status
            filled: Filled quantity
            remaining: Remaining quantity
            avgFillPrice: Average fill price
            permId: Permanent ID
            parentId: Parent ID
            lastFillPrice: Last fill price
            clientId: Client ID
            whyHeld: Why held
            mktCapPrice: Market cap price
        """
        if orderId in self.order_status:
            self.order_status[orderId].update({
                'status': status,
                'filled': filled,
                'remaining': remaining,
                'avg_fill_price': avgFillPrice,
                'last_fill_price': lastFillPrice,
                'client_id': clientId,
                'perm_id': permId,
                'parent_id': parentId,
                'last_update': datetime.now()
            })
    
    def execDetails(self, reqId: int, contract: Contract, execution: Execution):
        """
        Callback for execution details.
        
        Args:
            reqId: Request ID
            contract: Contract
            execution: Execution
        """
        if reqId in self.data_queues:
            self.data_queues[reqId].put({
                'contract': contract,
                'execution': execution
            })
    
    def execDetailsEnd(self, reqId: int):
        """
        Callback for end of execution details.
        
        Args:
            reqId: Request ID
        """
        if reqId in self.data_events:
            self.data_events[reqId].set()


class IBKRSimulator:
    """
    Simulator for IBKR connection when API is not available.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the IBKR simulator.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        self.ibkr_config = self.config.get('ibkr', {})
        
        # Connection parameters
        self.host = self.ibkr_config.get('host', '127.0.0.1')
        self.port = self.ibkr_config.get('port', 7497)
        self.client_id = self.ibkr_config.get('client_id', 1)
        self.account = self.ibkr_config.get('account', '')
        
        # State variables
        self.connected = False
        self.next_order_id = 1
        
        # Data storage
        self.positions = {}
        self.account_summary = {
            'NetLiquidation': 100000.0,
            'TotalCashValue': 100000.0,
            'AvailableFunds': 100000.0,
            'BuyingPower': 200000.0
        }
        self.market_data = {}
        self.options_chains = {}
        self.order_status = {}
        self.executions = {}
        
        # Simulated market data
        self._init_simulated_data()
    
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
                'ibkr': {
                    'host': '127.0.0.1',
                    'port': 7497,
                    'client_id': 1,
                    'account': ''
                }
            }
    
    def _init_simulated_data(self):
        """
        Initialize simulated market data.
        """
        # Simulated stock data
        self.market_data = {
            'AAPL': {'bid': 175.25, 'ask': 175.35, 'last': 175.30, 'volume': 1000000},
            'MSFT': {'bid': 325.75, 'ask': 326.00, 'last': 325.90, 'volume': 800000},
            'GOOGL': {'bid': 142.50, 'ask': 142.65, 'last': 142.60, 'volume': 600000},
            'AMZN': {'bid': 178.80, 'ask': 179.00, 'last': 178.90, 'volume': 700000},
            'TSLA': {'bid': 172.30, 'ask': 172.50, 'last': 172.40, 'volume': 900000}
        }
        
        # Simulated options chains
        self.options_chains = {
            'AAPL': {
                'expirations': ['20230721', '20230818', '20230915'],
                'strikes': [160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0]
            },
            'MSFT': {
                'expirations': ['20230721', '20230818', '20230915'],
                'strikes': [300.0, 310.0, 320.0, 330.0, 340.0, 350.0]
            },
            'GOOGL': {
                'expirations': ['20230721', '20230818', '20230915'],
                'strikes': [130.0, 135.0, 140.0, 145.0, 150.0, 155.0]
            },
            'AMZN': {
                'expirations': ['20230721', '20230818', '20230915'],
                'strikes': [160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0]
            },
            'TSLA': {
                'expirations': ['20230721', '20230818', '20230915'],
                'strikes': [150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0]
            }
        }
        
        # Simulated option data
        for symbol, data in self.market_data.items():
            stock_price = data['last']
            
            for expiration in self.options_chains[symbol]['expirations']:
                days_to_expiration = (datetime.strptime(expiration, '%Y%m%d') - datetime.now()).days
                
                for strike in self.options_chains[symbol]['strikes']:
                    # Call option
                    call_symbol = f"{symbol}_{expiration}_{strike}_C"
                    call_itm = stock_price > strike
                    call_price = max(0.1, stock_price - strike) if call_itm else max(0.05, (stock_price / strike) * 5)
                    
                    self.market_data[call_symbol] = {
                        'bid': call_price * 0.95,
                        'ask': call_price * 1.05,
                        'last': call_price,
                        'implied_volatility': 0.3,
                        'delta': 0.6 if call_itm else 0.4,
                        'gamma': 0.05,
                        'theta': -0.02,
                        'vega': 0.1
                    }
                    
                    # Put option
                    put_symbol = f"{symbol}_{expiration}_{strike}_P"
                    put_itm = stock_price < strike
                    put_price = max(0.1, strike - stock_price) if put_itm else max(0.05, (strike / stock_price) * 5)
                    
                    self.market_data[put_symbol] = {
                        'bid': put_price * 0.95,
                        'ask': put_price * 1.05,
                        'last': put_price,
                        'implied_volatility': 0.35,
                        'delta': -0.6 if put_itm else -0.4,
                        'gamma': 0.05,
                        'theta': -0.02,
                        'vega': 0.1
                    }
    
    def connect(self) -> bool:
        """
        Connect to simulated IBKR.
        
        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"Connecting to simulated IBKR at {self.host}:{self.port} with client ID {self.client_id}")
        self.connected = True
        logger.info(f"Connected to simulated IBKR. Next valid order ID: {self.next_order_id}")
        return True
    
    def disconnect(self):
        """
        Disconnect from simulated IBKR.
        """
        if self.connected:
            self.connected = False
            logger.info("Disconnected from simulated IBKR")
    
    def check_connection(self) -> bool:
        """
        Check if connected to simulated IBKR.
        
        Returns:
            True if connected, False otherwise
        """
        return self.connected
    
    def get_account_summary(self) -> Dict:
        """
        Get account summary.
        
        Returns:
            Dict containing account summary
        """
        if not self.connected:
            logger.warning("Not connected to simulated IBKR")
            return {}
        
        return self.account_summary
    
    def get_positions(self) -> Dict:
        """
        Get current positions.
        
        Returns:
            Dict containing positions
        """
        if not self.connected:
            logger.warning("Not connected to simulated IBKR")
            return {}
        
        return self.positions
    
    def get_market_data(self, symbol: str, sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD") -> Dict:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            sec_type: Security type (STK, OPT, FUT, etc.)
            exchange: Exchange
            currency: Currency
            
        Returns:
            Dict containing market data
        """
        if not self.connected:
            logger.warning("Not connected to simulated IBKR")
            return {}
        
        if symbol in self.market_data:
            return self.market_data[symbol]
        else:
            # Generate random data
            price = 100.0 + np.random.normal(0, 10)
            return {
                'bid': price * 0.995,
                'ask': price * 1.005,
                'last': price,
                'volume': int(np.random.uniform(100000, 1000000))
            }
    
    def get_options_chain(self, symbol: str, exchange: str = "SMART", currency: str = "USD") -> Dict:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Symbol to get options chain for
            exchange: Exchange
            currency: Currency
            
        Returns:
            Dict containing options chain
        """
        if not self.connected:
            logger.warning("Not connected to simulated IBKR")
            return {}
        
        if symbol in self.options_chains:
            return self.options_chains[symbol]
        else:
            # Generate random data
            today = datetime.now()
            expirations = [
                (today + timedelta(days=30)).strftime('%Y%m%d'),
                (today + timedelta(days=60)).strftime('%Y%m%d'),
                (today + timedelta(days=90)).strftime('%Y%m%d')
            ]
            
            strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
            
            return {
                'expirations': expirations,
                'strikes': strikes
            }
    
    def get_option_market_data(self, symbol: str, expiration: str, strike: float, right: str, exchange: str = "SMART", currency: str = "USD") -> Dict:
        """
        Get market data for an option.
        
        Args:
            symbol: Symbol of the underlying
            expiration: Expiration date (YYYYMMDD)
            strike: Strike price
            right: Right (C or P)
            exchange: Exchange
            currency: Currency
            
        Returns:
            Dict containing option market data
        """
        if not self.connected:
            logger.warning("Not connected to simulated IBKR")
            return {}
        
        option_symbol = f"{symbol}_{expiration}_{strike}_{right}"
        
        if option_symbol in self.market_data:
            return self.market_data[option_symbol]
        else:
            # Get underlying price
            if symbol in self.market_data:
                stock_price = self.market_data[symbol]['last']
            else:
                stock_price = 100.0
            
            # Calculate days to expiration
            days_to_expiration = (datetime.strptime(expiration, '%Y%m%d') - datetime.now()).days
            
            # Calculate option price
            if right in ['C', 'c', 'CALL', 'call']:
                itm = stock_price > strike
                price = max(0.1, stock_price - strike) if itm else max(0.05, (stock_price / strike) * 5)
                delta = 0.6 if itm else 0.4
            else:  # Put
                itm = stock_price < strike
                price = max(0.1, strike - stock_price) if itm else max(0.05, (strike / stock_price) * 5)
                delta = -0.6 if itm else -0.4
            
            # Generate option data
            option_data = {
                'bid': price * 0.95,
                'ask': price * 1.05,
                'last': price,
                'implied_volatility': 0.3,
                'delta': delta,
                'gamma': 0.05,
                'theta': -0.02,
                'vega': 0.1
            }
            
            # Store data
            self.market_data[option_symbol] = option_data
            
            return option_data
    
    def place_order(self, contract, order) -> int:
        """
        Place an order.
        
        Args:
            contract: Contract to trade
            order: Order details
            
        Returns:
            Order ID if successful, -1 otherwise
        """
        if not self.connected:
            logger.warning("Not connected to simulated IBKR")
            return -1
        
        try:
            # Get next order ID
            order_id = self.next_order_id
            self.next_order_id += 1
            
            # Extract contract details
            symbol = contract.symbol
            sec_type = contract.secType
            
            if sec_type == "OPT":
                # Format option symbol
                symbol = f"{symbol}_{contract.lastTradeDateOrContractMonth}_{contract.strike}_{contract.right}"
            
            # Get market data
            if symbol in self.market_data:
                market_data = self.market_data[symbol]
                price = market_data.get('last', 100.0)
            else:
                price = 100.0
            
            # Determine fill price
            if order.orderType == "MKT":
                fill_price = price
            elif order.orderType == "LMT":
                if order.action == "BUY" and order.lmtPrice >= price:
                    fill_price = price
                elif order.action == "SELL" and order.lmtPrice <= price:
                    fill_price = price
                else:
                    fill_price = order.lmtPrice
            else:
                fill_price = price
            
            # Create order status
            self.order_status[order_id] = {
                'status': 'Filled',
                'filled': order.totalQuantity,
                'remaining': 0,
                'avg_fill_price': fill_price,
                'last_fill_price': fill_price,
                'client_id': self.client_id,
                'perm_id': order_id,
                'parent_id': 0,
                'last_update': datetime.now()
            }
            
            # Create execution
            exec_id = f"sim_{order_id}"
            self.executions[exec_id] = {
                'exec_id': exec_id,
                'order_id': order_id,
                'symbol': symbol,
                'sec_type': sec_type,
                'side': order.action,
                'shares': order.totalQuantity,
                'price': fill_price,
                'time': datetime.now().strftime('%Y%m%d %H:%M:%S'),
                'commission': 1.0
            }
            
            # Update positions
            if symbol in self.positions:
                if order.action == "BUY":
                    self.positions[symbol]['position'] += order.totalQuantity
                else:  # SELL
                    self.positions[symbol]['position'] -= order.totalQuantity
                
                # Update average cost
                if self.positions[symbol]['position'] != 0:
                    self.positions[symbol]['avg_cost'] = (
                        (self.positions[symbol]['avg_cost'] * self.positions[symbol]['position']) +
                        (fill_price * order.totalQuantity)
                    ) / self.positions[symbol]['position']
                else:
                    # Position closed
                    del self.positions[symbol]
            else:
                # New position
                if order.action == "BUY":
                    position = order.totalQuantity
                else:  # SELL
                    position = -order.totalQuantity
                
                self.positions[symbol] = {
                    'symbol': symbol,
                    'position': position,
                    'avg_cost': fill_price
                }
            
            # Update account
            if order.action == "BUY":
                self.account_summary['TotalCashValue'] -= fill_price * order.totalQuantity
            else:  # SELL
                self.account_summary['TotalCashValue'] += fill_price * order.totalQuantity
            
            self.account_summary['AvailableFunds'] = self.account_summary['TotalCashValue']
            self.account_summary['BuyingPower'] = self.account_summary['TotalCashValue'] * 2
            
            # Calculate portfolio value
            portfolio_value = self.account_summary['TotalCashValue']
            for pos_symbol, pos_data in self.positions.items():
                if pos_symbol in self.market_data:
                    pos_price = self.market_data[pos_symbol].get('last', pos_data['avg_cost'])
                else:
                    pos_price = pos_data['avg_cost']
                
                portfolio_value += pos_price * pos_data['position']
            
            self.account_summary['NetLiquidation'] = portfolio_value
            
            logger.info(f"Simulated order {order_id} filled: {order.action} {order.totalQuantity} {symbol} @ {fill_price}")
            
            return order_id
        except Exception as e:
            logger.error(f"Error placing simulated order: {e}")
            return -1
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation request successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to simulated IBKR")
            return False
        
        if order_id not in self.order_status:
            logger.warning(f"Order {order_id} not found")
            return False
        
        # Check if order can be cancelled
        status = self.order_status[order_id]['status']
        if status in ['Filled', 'Cancelled', 'Inactive']:
            logger.warning(f"Cannot cancel order {order_id} with status {status}")
            return False
        
        # Cancel order
        self.order_status[order_id]['status'] = 'Cancelled'
        logger.info(f"Simulated order {order_id} cancelled")
        
        return True
    
    def get_order_status(self, order_id: int) -> Dict:
        """
        Get status of an order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Dict containing order status
        """
        if not self.connected:
            logger.warning("Not connected to simulated IBKR")
            return {}
        
        if order_id not in self.order_status:
            logger.warning(f"Order {order_id} not found")
            return {}
        
        return self.order_status[order_id]
    
    def get_executions(self) -> Dict:
        """
        Get all executions.
        
        Returns:
            Dict containing executions
        """
        if not self.connected:
            logger.warning("Not connected to simulated IBKR")
            return {}
        
        return self.executions
    
    def create_stock_contract(self, symbol: str, exchange: str = "SMART", currency: str = "USD"):
        """
        Create a stock contract.
        
        Args:
            symbol: Symbol
            exchange: Exchange
            currency: Currency
            
        Returns:
            Contract object
        """
        class SimContract:
            def __init__(self):
                self.symbol = symbol
                self.secType = "STK"
                self.exchange = exchange
                self.currency = currency
        
        return SimContract()
    
    def create_option_contract(self, symbol: str, expiration: str, strike: float, right: str, exchange: str = "SMART", currency: str = "USD"):
        """
        Create an option contract.
        
        Args:
            symbol: Symbol of the underlying
            expiration: Expiration date (YYYYMMDD)
            strike: Strike price
            right: Right (C or P)
            exchange: Exchange
            currency: Currency
            
        Returns:
            Contract object
        """
        class SimContract:
            def __init__(self):
                self.symbol = symbol
                self.secType = "OPT"
                self.exchange = exchange
                self.currency = currency
                self.lastTradeDateOrContractMonth = expiration
                self.strike = strike
                self.right = right
        
        return SimContract()
    
    def create_market_order(self, action: str, quantity: float):
        """
        Create a market order.
        
        Args:
            action: Action (BUY or SELL)
            quantity: Quantity
            
        Returns:
            Order object
        """
        class SimOrder:
            def __init__(self):
                self.action = action
                self.orderType = "MKT"
                self.totalQuantity = quantity
        
        return SimOrder()
    
    def create_limit_order(self, action: str, quantity: float, limit_price: float):
        """
        Create a limit order.
        
        Args:
            action: Action (BUY or SELL)
            quantity: Quantity
            limit_price: Limit price
            
        Returns:
            Order object
        """
        class SimOrder:
            def __init__(self):
                self.action = action
                self.orderType = "LMT"
                self.totalQuantity = quantity
                self.lmtPrice = limit_price
        
        return SimOrder()
    
    def create_stop_order(self, action: str, quantity: float, stop_price: float):
        """
        Create a stop order.
        
        Args:
            action: Action (BUY or SELL)
            quantity: Quantity
            stop_price: Stop price
            
        Returns:
            Order object
        """
        class SimOrder:
            def __init__(self):
                self.action = action
                self.orderType = "STP"
                self.totalQuantity = quantity
                self.auxPrice = stop_price
        
        return SimOrder()
    
    def create_stop_limit_order(self, action: str, quantity: float, stop_price: float, limit_price: float):
        """
        Create a stop-limit order.
        
        Args:
            action: Action (BUY or SELL)
            quantity: Quantity
            stop_price: Stop price
            limit_price: Limit price
            
        Returns:
            Order object
        """
        class SimOrder:
            def __init__(self):
                self.action = action
                self.orderType = "STP LMT"
                self.totalQuantity = quantity
                self.auxPrice = stop_price
                self.lmtPrice = limit_price
        
        return SimOrder()


# Factory function to create appropriate connection
def create_ibkr_connection(config_path: str = "../config/settings.yaml") -> Union[IBKRConnection, IBKRSimulator]:
    """
    Create an IBKR connection or simulator.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        IBKR connection or simulator
    """
    if HAS_IBAPI:
        return IBKRConnection(config_path)
    else:
        return IBKRSimulator(config_path)


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    ibkr = create_ibkr_connection()
    
    # Connect to IBKR
    if ibkr.connect():
        # Get account summary
        account_summary = ibkr.get_account_summary()
        print("Account Summary:")
        print(account_summary)
        
        # Get positions
        positions = ibkr.get_positions()
        print("\nPositions:")
        print(positions)
        
        # Get market data
        market_data = ibkr.get_market_data("AAPL")
        print("\nAAPL Market Data:")
        print(market_data)
        
        # Get options chain
        options_chain = ibkr.get_options_chain("AAPL")
        print("\nAAPL Options Chain:")
        print(options_chain)
        
        # Place a simulated order
        contract = ibkr.create_stock_contract("AAPL")
        order = ibkr.create_market_order("BUY", 100)
        order_id = ibkr.place_order(contract, order)
        
        if order_id > 0:
            # Get order status
            order_status = ibkr.get_order_status(order_id)
            print("\nOrder Status:")
            print(order_status)
            
            # Get executions
            executions = ibkr.get_executions()
            print("\nExecutions:")
            print(executions)
        
        # Disconnect
        ibkr.disconnect()
    else:
        print("Failed to connect to IBKR")
