"""
Data Storage Module

This module handles the storage and retrieval of market data, including options chains,
historical prices, and derived features. It supports multiple storage backends including
SQLite, Parquet files, and potentially PostgreSQL or MongoDB.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    logging.warning("pyarrow not installed. Parquet storage will be limited.")

try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Integer, DateTime, Boolean
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    logging.warning("sqlalchemy not installed. SQL database storage will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataStorage:
    """
    Class to handle data storage and retrieval.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the data storage with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.storage_type = self.config['storage']['type']
        self.engine = None
        self.connection = None
        
        # Initialize storage based on configuration
        self._initialize_storage()
    
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
                'storage': {
                    'type': 'sqlite',
                    'sqlite': {
                        'db_path': '../data/market_data.db'
                    },
                    'parquet': {
                        'dir_path': '../data/parquet'
                    }
                }
            }
    
    def _initialize_storage(self):
        """Initialize the appropriate storage backend based on configuration."""
        if self.storage_type == 'sqlite':
            self._initialize_sqlite()
        elif self.storage_type == 'parquet':
            self._initialize_parquet()
        elif self.storage_type == 'postgresql':
            self._initialize_postgresql()
        else:
            logger.warning(f"Unsupported storage type: {self.storage_type}. Defaulting to SQLite.")
            self.storage_type = 'sqlite'
            self._initialize_sqlite()
    
    def _initialize_sqlite(self):
        """Initialize SQLite database and create tables if they don't exist."""
        if not HAS_SQLALCHEMY:
            logger.error("SQLAlchemy is required for SQLite storage")
            return
            
        try:
            db_path = self.config['storage']['sqlite']['db_path']
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            
            # Create engine and connection
            self.engine = create_engine(f'sqlite:///{db_path}')
            self.connection = self.engine.connect()
            
            # Create tables if they don't exist
            self._create_sqlite_tables()
            
            logger.info(f"Initialized SQLite storage at {db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing SQLite storage: {e}")
    
    def _create_sqlite_tables(self):
        """Create SQLite tables for storing market data."""
        if not self.engine:
            return
            
        metadata = MetaData()
        
        # Options Chain Table
        options_chain = Table(
            'options_chain', metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String, nullable=False),
            Column('expiration', String, nullable=False),
            Column('strike', Float, nullable=False),
            Column('right', String, nullable=False),
            Column('dte', Integer),
            Column('bid', Float),
            Column('ask', Float),
            Column('last', Float),
            Column('volume', Integer),
            Column('open_interest', Integer),
            Column('implied_volatility', Float),
            Column('delta', Float),
            Column('gamma', Float),
            Column('vega', Float),
            Column('theta', Float),
            Column('underlying_price', Float),
            Column('timestamp', DateTime, default=datetime.now),
        )
        
        # Underlying Prices Table
        underlying_prices = Table(
            'underlying_prices', metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String, nullable=False),
            Column('date', DateTime, nullable=False),
            Column('open', Float),
            Column('high', Float),
            Column('low', Float),
            Column('close', Float),
            Column('volume', Integer),
            Column('adjusted_close', Float),
        )
        
        # Sentiment Scores Table
        sentiment_scores = Table(
            'sentiment_scores', metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String, nullable=False),
            Column('date', DateTime, nullable=False),
            Column('source', String),
            Column('sentiment_score', Float),
            Column('sentiment_label', String),
            Column('confidence', Float),
        )
        
        # Feature Cache Table
        feature_cache = Table(
            'feature_cache', metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String, nullable=False),
            Column('date', DateTime, nullable=False),
            Column('feature_name', String, nullable=False),
            Column('feature_value', Float),
            Column('feature_group', String),
        )
        
        # Executed Trades Table
        executed_trades = Table(
            'executed_trades', metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String, nullable=False),
            Column('strategy', String),
            Column('entry_date', DateTime),
            Column('exit_date', DateTime),
            Column('entry_price', Float),
            Column('exit_price', Float),
            Column('quantity', Integer),
            Column('pnl', Float),
            Column('commission', Float),
            Column('trade_details', String),  # JSON string with additional details
        )
        
        # Create all tables
        metadata.create_all(self.engine)
    
    def _initialize_parquet(self):
        """Initialize Parquet storage directory."""
        if not HAS_PYARROW:
            logger.error("PyArrow is required for Parquet storage")
            return
            
        try:
            dir_path = self.config['storage']['parquet']['dir_path']
            
            # Ensure directory exists
            os.makedirs(os.path.abspath(dir_path), exist_ok=True)
            
            # Create subdirectories for different data types
            os.makedirs(os.path.join(dir_path, 'options_chain'), exist_ok=True)
            os.makedirs(os.path.join(dir_path, 'underlying_prices'), exist_ok=True)
            os.makedirs(os.path.join(dir_path, 'sentiment_scores'), exist_ok=True)
            os.makedirs(os.path.join(dir_path, 'feature_cache'), exist_ok=True)
            os.makedirs(os.path.join(dir_path, 'executed_trades'), exist_ok=True)
            
            logger.info(f"Initialized Parquet storage at {dir_path}")
            
        except Exception as e:
            logger.error(f"Error initializing Parquet storage: {e}")
    
    def _initialize_postgresql(self):
        """Initialize PostgreSQL database connection."""
        if not HAS_SQLALCHEMY:
            logger.error("SQLAlchemy is required for PostgreSQL storage")
            return
            
        try:
            # This would be implemented for PostgreSQL support
            # Similar to SQLite but with PostgreSQL connection string
            logger.warning("PostgreSQL storage not fully implemented yet")
            
        except Exception as e:
            logger.error(f"Error initializing PostgreSQL storage: {e}")
    
    def store_options_chain(self, options_df: pd.DataFrame, symbol: str = None, timestamp: datetime = None):
        """
        Store options chain data.
        
        Args:
            options_df: DataFrame containing options chain data
            symbol: Symbol override (if not in DataFrame)
            timestamp: Timestamp override (if not in DataFrame)
        """
        if options_df.empty:
            logger.warning("Empty options chain data, nothing to store")
            return
            
        try:
            # Add symbol if provided and not in DataFrame
            if symbol and 'symbol' not in options_df.columns:
                options_df['symbol'] = symbol
                
            # Add timestamp if provided and not in DataFrame
            if timestamp and 'timestamp' not in options_df.columns:
                options_df['timestamp'] = timestamp
            elif 'timestamp' not in options_df.columns:
                options_df['timestamp'] = datetime.now()
                
            if self.storage_type == 'sqlite':
                self._store_options_chain_sqlite(options_df)
            elif self.storage_type == 'parquet':
                self._store_options_chain_parquet(options_df)
                
        except Exception as e:
            logger.error(f"Error storing options chain data: {e}")
    
    def _store_options_chain_sqlite(self, options_df: pd.DataFrame):
        """Store options chain data in SQLite database."""
        if not self.engine:
            logger.error("SQLite engine not initialized")
            return
            
        try:
            # Store in database
            options_df.to_sql('options_chain', self.engine, if_exists='append', index=False)
            logger.info(f"Stored {len(options_df)} options contracts in SQLite")
            
        except Exception as e:
            logger.error(f"Error storing options chain in SQLite: {e}")
    
    def _store_options_chain_parquet(self, options_df: pd.DataFrame):
        """Store options chain data in Parquet files."""
        if not HAS_PYARROW:
            logger.error("PyArrow is required for Parquet storage")
            return
            
        try:
            dir_path = self.config['storage']['parquet']['dir_path']
            symbol = options_df['symbol'].iloc[0]
            timestamp = options_df['timestamp'].iloc[0]
            date_str = timestamp.strftime('%Y%m%d')
            
            # Create filename
            filename = f"{symbol}_options_{date_str}.parquet"
            filepath = os.path.join(dir_path, 'options_chain', filename)
            
            # Write to Parquet file
            pq.write_table(pa.Table.from_pandas(options_df), filepath)
            logger.info(f"Stored {len(options_df)} options contracts in Parquet file: {filepath}")
            
        except Exception as e:
            logger.error(f"Error storing options chain in Parquet: {e}")
    
    def store_historical_prices(self, prices_df: pd.DataFrame, symbol: str = None):
        """
        Store historical price data.
        
        Args:
            prices_df: DataFrame containing historical price data
            symbol: Symbol override (if not in DataFrame)
        """
        if prices_df.empty:
            logger.warning("Empty historical price data, nothing to store")
            return
            
        try:
            # Add symbol if provided and not in DataFrame
            if symbol and 'symbol' not in prices_df.columns:
                prices_df['symbol'] = symbol
                
            if self.storage_type == 'sqlite':
                self._store_historical_prices_sqlite(prices_df)
            elif self.storage_type == 'parquet':
                self._store_historical_prices_parquet(prices_df)
                
        except Exception as e:
            logger.error(f"Error storing historical price data: {e}")
    
    def _store_historical_prices_sqlite(self, prices_df: pd.DataFrame):
        """Store historical price data in SQLite database."""
        if not self.engine:
            logger.error("SQLite engine not initialized")
            return
            
        try:
            # Ensure date column is properly formatted
            if 'date' not in prices_df.columns and 'Date' in prices_df.columns:
                prices_df = prices_df.rename(columns={'Date': 'date'})
                
            # Store in database
            prices_df.to_sql('underlying_prices', self.engine, if_exists='append', index=False)
            logger.info(f"Stored {len(prices_df)} price records in SQLite")
            
        except Exception as e:
            logger.error(f"Error storing historical prices in SQLite: {e}")
    
    def _store_historical_prices_parquet(self, prices_df: pd.DataFrame):
        """Store historical price data in Parquet files."""
        if not HAS_PYARROW:
            logger.error("PyArrow is required for Parquet storage")
            return
            
        try:
            dir_path = self.config['storage']['parquet']['dir_path']
            symbol = prices_df['symbol'].iloc[0]
            
            # Create filename
            filename = f"{symbol}_prices.parquet"
            filepath = os.path.join(dir_path, 'underlying_prices', filename)
            
            # Write to Parquet file
            pq.write_table(pa.Table.from_pandas(prices_df), filepath)
            logger.info(f"Stored {len(prices_df)} price records in Parquet file: {filepath}")
            
        except Exception as e:
            logger.error(f"Error storing historical prices in Parquet: {e}")
    
    def store_sentiment_data(self, sentiment_df: pd.DataFrame, symbol: str = None):
        """
        Store sentiment data.
        
        Args:
            sentiment_df: DataFrame containing sentiment data
            symbol: Symbol override (if not in DataFrame)
        """
        if sentiment_df.empty:
            logger.warning("Empty sentiment data, nothing to store")
            return
            
        try:
            # Add symbol if provided and not in DataFrame
            if symbol and 'symbol' not in sentiment_df.columns:
                sentiment_df['symbol'] = symbol
                
            if self.storage_type == 'sqlite':
                self._store_sentiment_data_sqlite(sentiment_df)
            elif self.storage_type == 'parquet':
                self._store_sentiment_data_parquet(sentiment_df)
                
        except Exception as e:
            logger.error(f"Error storing sentiment data: {e}")
    
    def _store_sentiment_data_sqlite(self, sentiment_df: pd.DataFrame):
        """Store sentiment data in SQLite database."""
        if not self.engine:
            logger.error("SQLite engine not initialized")
            return
            
        try:
            # Store in database
            sentiment_df.to_sql('sentiment_scores', self.engine, if_exists='append', index=False)
            logger.info(f"Stored {len(sentiment_df)} sentiment records in SQLite")
            
        except Exception as e:
            logger.error(f"Error storing sentiment data in SQLite: {e}")
    
    def _store_sentiment_data_parquet(self, sentiment_df: pd.DataFrame):
        """Store sentiment data in Parquet files."""
        if not HAS_PYARROW:
            logger.error("PyArrow is required for Parquet storage")
            return
            
        try:
            dir_path = self.config['storage']['parquet']['dir_path']
            symbol = sentiment_df['symbol'].iloc[0]
            
            # Create filename
            filename = f"{symbol}_sentiment.parquet"
            filepath = os.path.join(dir_path, 'sentiment_scores', filename)
            
            # Write to Parquet file
            pq.write_table(pa.Table.from_pandas(sentiment_df), filepath)
            logger.info(f"Stored {len(sentiment_df)} sentiment records in Parquet file: {filepath}")
            
        except Exception as e:
            logger.error(f"Error storing sentiment data in Parquet: {e}")
    
    def store_feature_data(self, feature_df: pd.DataFrame, symbol: str = None):
        """
        Store feature data.
        
        Args:
            feature_df: DataFrame containing feature data
            symbol: Symbol override (if not in DataFrame)
        """
        if feature_df.empty:
            logger.warning("Empty feature data, nothing to store")
            return
            
        try:
            # Add symbol if provided and not in DataFrame
            if symbol and 'symbol' not in feature_df.columns:
                feature_df['symbol'] = symbol
                
            if self.storage_type == 'sqlite':
                self._store_feature_data_sqlite(feature_df)
            elif self.storage_type == 'parquet':
                self._store_feature_data_parquet(feature_df)
                
        except Exception as e:
            logger.error(f"Error storing feature data: {e}")
    
    def _store_feature_data_sqlite(self, feature_df: pd.DataFrame):
        """Store feature data in SQLite database."""
        if not self.engine:
            logger.error("SQLite engine not initialized")
            return
            
        try:
            # Store in database
            feature_df.to_sql('feature_cache', self.engine, if_exists='append', index=False)
            logger.info(f"Stored {len(feature_df)} feature records in SQLite")
            
        except Exception as e:
            logger.error(f"Error storing feature data in SQLite: {e}")
    
    def _store_feature_data_parquet(self, feature_df: pd.DataFrame):
        """Store feature data in Parquet files."""
        if not HAS_PYARROW:
            logger.error("PyArrow is required for Parquet storage")
            return
            
        try:
            dir_path = self.config['storage']['parquet']['dir_path']
            symbol = feature_df['symbol'].iloc[0]
            feature_group = feature_df.get('feature_group', ['general'])[0]
            
            # Create filename
            filename = f"{symbol}_{feature_group}_features.parquet"
            filepath = os.path.join(dir_path, 'feature_cache', filename)
            
            # Write to Parquet file
            pq.write_table(pa.Table.from_pandas(feature_df), filepath)
            logger.info(f"Stored {len(feature_df)} feature records in Parquet file: {filepath}")
            
        except Exception as e:
            logger.error(f"Error storing feature data in Parquet: {e}")
    
    def query_options_chain(self, symbol: str, expiration: str = None, min_dte: int = None, max_dte: int = None) -> pd.DataFrame:
        """
        Query options chain data.
        
        Args:
            symbol: Stock symbol
            expiration: Specific expiration date (optional)
            min_dte: Minimum days to expiration (optional)
            max_dte: Maximum days to expiration (optional)
            
        Returns:
            DataFrame containing options chain data
        """
        try:
            if self.storage_type == 'sqlite':
                return self._query_options_chain_sqlite(symbol, expiration, min_dte, max_dte)
            elif self.storage_type == 'parquet':
                return self._query_options_chain_parquet(symbol, expiration, min_dte, max_dte)
            else:
                logger.error(f"Unsupported storage type for query: {self.storage_type}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error querying options chain data: {e}")
            return pd.DataFrame()
    
    def _query_options_chain_sqlite(self, symbol: str, expiration: str = None, min_dte: int = None, max_dte: int = None) -> pd.DataFrame:
        """Query options chain data from SQLite database."""
        if not self.engine:
            logger.error("SQLite engine not initialized")
            return pd.DataFrame()
            
        try:
            # Build query
            query = f"SELECT * FROM options_chain WHERE symbol = '{symbol}'"
            
            if expiration:
                query += f" AND expiration = '{expiration}'"
                
            if min_dte is not None:
                query += f" AND dte >= {min_dte}"
                
            if max_dte is not None:
                query += f" AND dte <= {max_dte}"
                
            # Execute query
            df = pd.read_sql(query, self.engine)
            logger.info(f"Retrieved {len(df)} options contracts for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error querying options chain from SQLite: {e}")
            return pd.DataFrame()
    
    def _query_options_chain_parquet(self, symbol: str, expiration: str = None, min_dte: int = None, max_dte: int = None) -> pd.DataFrame:
        """Query options chain data from Parquet files."""
        if not HAS_PYARROW:
            logger.error("PyArrow is required for Parquet queries")
            return pd.DataFrame()
            
        try:
            dir_path = self.config['storage']['parquet']['dir_path']
            options_dir = os.path.join(dir_path, 'options_chain')
            
            # Find all files for this symbol
            files = [f for f in os.listdir(options_dir) if f.startswith(f"{symbol}_options_")]
            
            if not files:
                logger.warning(f"No options data files found for {symbol}")
                return pd.DataFrame()
                
            # Read and filter data
            dfs = []
            for file in files:
                filepath = os.path.join(options_dir, file)
                df = pd.read_parquet(filepath)
                
                # Apply filters
                if expiration:
                    df = df[df['expiration'] == expiration]
                    
                if min_dte is not None:
                    df = df[df['dte'] >= min_dte]
                    
                if max_dte is not None:
                    df = df[df['dte'] <= max_dte]
                    
                dfs.append(df)
                
            # Combine results
            if dfs:
                result = pd.concat(dfs, ignore_index=True)
                logger.info(f"Retrieved {len(result)} options contracts for {symbol}")
                return result
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error querying options chain from Parquet: {e}")
            return pd.DataFrame()
    
    def query_historical_prices(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Query historical price data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for query (optional)
            end_date: End date for query (optional)
            
        Returns:
            DataFrame containing historical price data
        """
        try:
            if self.storage_type == 'sqlite':
                return self._query_historical_prices_sqlite(symbol, start_date, end_date)
            elif self.storage_type == 'parquet':
                return self._query_historical_prices_parquet(symbol, start_date, end_date)
            else:
                logger.error(f"Unsupported storage type for query: {self.storage_type}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error querying historical price data: {e}")
            return pd.DataFrame()
    
    def _query_historical_prices_sqlite(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Query historical price data from SQLite database."""
        if not self.engine:
            logger.error("SQLite engine not initialized")
            return pd.DataFrame()
            
        try:
            # Build query
            query = f"SELECT * FROM underlying_prices WHERE symbol = '{symbol}'"
            
            if start_date:
                query += f" AND date >= '{start_date}'"
                
            if end_date:
                query += f" AND date <= '{end_date}'"
                
            query += " ORDER BY date"
                
            # Execute query
            df = pd.read_sql(query, self.engine)
            logger.info(f"Retrieved {len(df)} price records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error querying historical prices from SQLite: {e}")
            return pd.DataFrame()
    
    def _query_historical_prices_parquet(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Query historical price data from Parquet files."""
        if not HAS_PYARROW:
            logger.error("PyArrow is required for Parquet queries")
            return pd.DataFrame()
            
        try:
            dir_path = self.config['storage']['parquet']['dir_path']
            prices_dir = os.path.join(dir_path, 'underlying_prices')
            
            # Find file for this symbol
            filepath = os.path.join(prices_dir, f"{symbol}_prices.parquet")
            
            if not os.path.exists(filepath):
                logger.warning(f"No price data file found for {symbol}")
                return pd.DataFrame()
                
            # Read data
            df = pd.read_parquet(filepath)
            
            # Apply date filters
            if start_date:
                df = df[df['date'] >= start_date]
                
            if end_date:
                df = df[df['date'] <= end_date]
                
            # Sort by date
            df = df.sort_values('date')
            
            logger.info(f"Retrieved {len(df)} price records for {symbol}")
            return df
                
        except Exception as e:
            logger.error(f"Error querying historical prices from Parquet: {e}")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    storage = DataStorage()
    
    # Example: Create sample data
    import numpy as np
    
    # Sample options data
    options_data = {
        'symbol': ['AAPL'] * 10,
        'expiration': ['2023-06-16'] * 10,
        'strike': np.linspace(150, 200, 10),
        'right': ['C'] * 5 + ['P'] * 5,
        'dte': [30] * 10,
        'bid': np.random.uniform(1, 10, 10),
        'ask': np.random.uniform(1, 10, 10),
        'volume': np.random.randint(100, 1000, 10),
        'open_interest': np.random.randint(500, 5000, 10),
        'implied_volatility': np.random.uniform(0.2, 0.5, 10),
        'delta': np.random.uniform(-1, 1, 10),
        'underlying_price': [175.0] * 10,
        'timestamp': [datetime.now()] * 10
    }
    
    options_df = pd.DataFrame(options_data)
    
    # Store sample data
    storage.store_options_chain(options_df)
    
    # Query data
    result = storage.query_options_chain('AAPL')
    print(f"Retrieved {len(result)} options contracts")
