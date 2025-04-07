"""
Dashboard Components for Quant ML Options Trading System

This module provides reusable UI components for the dashboard.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketOverviewComponents:
    """
    Components for the Market Overview section of the dashboard.
    """
    
    @staticmethod
    def render_market_summary(symbols: List[str], market_data: Dict):
        """
        Render market summary cards.
        
        Args:
            symbols: List of symbols to display
            market_data: Dictionary of market data
        """
        # Create columns for market data
        cols = st.columns(len(symbols))
        
        for i, symbol in enumerate(symbols):
            with cols[i]:
                if symbol in market_data:
                    data = market_data[symbol]
                    
                    # Display price and change
                    last_price = data.get('last', 0.0)
                    open_price = data.get('open', last_price)
                    change = last_price - open_price
                    change_pct = (change / open_price) * 100 if open_price > 0 else 0.0
                    
                    # Color based on change
                    color = "green" if change >= 0 else "red"
                    
                    st.metric(
                        label=symbol,
                        value=f"${last_price:.2f}",
                        delta=f"{change:.2f} ({change_pct:.2f}%)",
                        delta_color="normal" if change >= 0 else "inverse"
                    )
                    
                    # Display additional info
                    st.text(f"High: ${data.get('high', 0.0):.2f}")
                    st.text(f"Low: ${data.get('low', 0.0):.2f}")
                    st.text(f"Volume: {data.get('volume', 0):,}")
                    
                    # Display IV if available
                    if 'implied_volatility' in data:
                        iv = data['implied_volatility']
                        st.text(f"IV: {iv:.2%}")
                else:
                    st.metric(label=symbol, value="N/A")
                    st.text("No data available")
    
    @staticmethod
    def render_price_chart(symbol: str, historical_data: pd.DataFrame, signals: List[Dict] = None, show_signals: bool = True):
        """
        Render price chart for a symbol.
        
        Args:
            symbol: Symbol to display
            historical_data: DataFrame of historical price data
            signals: List of trading signals
            show_signals: Whether to show signals on chart
        """
        if historical_data.empty:
            st.warning(f"No historical data available for {symbol}")
            return
        
        # Create price chart
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=historical_data['date'],
            open=historical_data['open'],
            high=historical_data['high'],
            low=historical_data['low'],
            close=historical_data['close'],
            name=symbol
        ))
        
        # Add volume as bar chart
        fig.add_trace(go.Bar(
            x=historical_data['date'],
            y=historical_data['volume'],
            name='Volume',
            marker_color='rgba(0, 0, 255, 0.3)',
            opacity=0.3,
            yaxis='y2'
        ))
        
        # Add signals if available and enabled
        if show_signals and signals:
            # Filter signals for selected symbol
            symbol_signals = [s for s in signals if s['symbol'] == symbol]
            
            for signal in symbol_signals:
                # Get signal date
                signal_date = signal.get('date', datetime.now())
                
                # Determine color based on direction
                color = 'green' if signal['direction'] == 'BUY' else 'red'
                
                # Add signal marker
                fig.add_trace(go.Scatter(
                    x=[signal_date],
                    y=[historical_data.loc[historical_data['date'] == signal_date, 'high'].iloc[0] if not historical_data.loc[historical_data['date'] == signal_date].empty else 0],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down' if signal['direction'] == 'SELL' else 'triangle-up',
                        size=15,
                        color=color
                    ),
                    name=f"{signal['direction']} Signal"
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_options_chain(options_df: pd.DataFrame, show_greeks: bool = True):
        """
        Render options chain table.
        
        Args:
            options_df: DataFrame of options data
            show_greeks: Whether to show Greeks columns
        """
        if options_df.empty:
            st.warning("No options data available")
            return
        
        # Split into calls and puts
        calls = options_df[options_df['option_type'] == 'CALL'].copy()
        puts = options_df[options_df['option_type'] == 'PUT'].copy()
        
        # Sort by strike
        calls = calls.sort_values('strike')
        puts = puts.sort_values('strike')
        
        # Select columns to display
        display_columns = ['strike', 'bid', 'ask', 'mid', 'implied_volatility']
        
        if show_greeks:
            greek_columns = ['delta', 'gamma', 'theta', 'vega']
            for col in greek_columns:
                if col in options_df.columns:
                    display_columns.append(col)
        
        # Create merged table
        merged = pd.DataFrame()
        
        for col in display_columns:
            if col in calls.columns:
                merged[f'call_{col}'] = calls[col].values
            
            if col == 'strike':
                merged['strike'] = calls['strike'].values
            
            if col in puts.columns:
                merged[f'put_{col}'] = puts[col].values
        
        # Format columns
        for col in merged.columns:
            if 'price' in col or 'bid' in col or 'ask' in col or 'mid' in col:
                merged[col] = merged[col].map('${:,.2f}'.format)
            elif 'implied_volatility' in col:
                merged[col] = merged[col].map('{:,.2%}'.format)
            elif 'delta' in col or 'gamma' in col or 'theta' in col or 'vega' in col:
                merged[col] = merged[col].map('{:,.4f}'.format)
        
        # Display table
        st.dataframe(merged)


class StrategyComponents:
    """
    Components for the Strategy section of the dashboard.
    """
    
    @staticmethod
    def render_strategy_parameters(strategy, key_prefix: str = ""):
        """
        Render strategy parameters form.
        
        Args:
            strategy: Strategy object
            key_prefix: Prefix for form keys
            
        Returns:
            Dict of updated parameters if form submitted, None otherwise
        """
        # Get strategy parameters
        params = strategy.get_parameters()
        
        # Create form for parameters
        with st.form(key=f"{key_prefix}strategy_params"):
            new_params = {}
            
            for param_name, param_value in params.items():
                if isinstance(param_value, bool):
                    new_params[param_name] = st.checkbox(param_name, value=param_value, key=f"{key_prefix}{param_name}")
                elif isinstance(param_value, int):
                    new_params[param_name] = st.number_input(param_name, value=param_value, step=1, key=f"{key_prefix}{param_name}")
                elif isinstance(param_value, float):
                    new_params[param_name] = st.number_input(param_name, value=param_value, step=0.01, key=f"{key_prefix}{param_name}")
                elif isinstance(param_value, str):
                    new_params[param_name] = st.text_input(param_name, value=param_value, key=f"{key_prefix}{param_name}")
                elif isinstance(param_value, list):
                    if all(isinstance(x, str) for x in param_value):
                        new_params[param_name] = st.multiselect(param_name, options=param_value, default=param_value, key=f"{key_prefix}{param_name}")
                    else:
                        new_params[param_name] = st.text_input(param_name, value=str(param_value), key=f"{key_prefix}{param_name}")
            
            # Submit button
            if st.form_submit_button("Update Parameters"):
                return new_params
        
        return None
    
    @staticmethod
    def render_signals_table(signals: List[Dict]):
        """
        Render trading signals table.
        
        Args:
            signals: List of trading signals
        """
        if not signals:
            st.info("No signals available")
            return
        
        # Create DataFrame from signals
        signals_df = pd.DataFrame(signals)
        
        # Display signals table
        st.dataframe(signals_df)
    
    @staticmethod
    def render_performance_metrics(metrics: Dict):
        """
        Render performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        if not metrics:
            st.info("No performance metrics available")
            return
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0.0):.2%}")
            if 'annualized_return' in metrics:
                st.metric("Annualized Return", f"{metrics.get('annualized_return', 0.0):.2%}")
        
        with col2:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}")
            if 'sortino_ratio' in metrics:
                st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0.0):.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0.0):.2%}")
            if 'annualized_volatility' in metrics:
                st.metric("Volatility", f"{metrics.get('annualized_volatility', 0.0):.2%}")
        
        with col4:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0.0):.2%}")
            if 'profit_factor' in metrics:
                st.metric("Profit Factor", f"{metrics.get('profit_factor', 0.0):.2f}")
    
    @staticmethod
    def render_equity_curve(equity_curve: pd.DataFrame, title: str = "Equity Curve"):
        """
        Render equity curve chart.
        
        Args:
            equity_curve: DataFrame with equity curve data
            title: Chart title
        """
        if equity_curve.empty:
            st.info("No equity curve data available")
            return
        
        fig = px.line(
            equity_curve,
            x='date',
            y='equity',
            title=title
        )
        
        st.plotly_chart(fig, use_container_width=True)


class PortfolioComponents:
    """
    Components for the Portfolio section of the dashboard.
    """
    
    @staticmethod
    def render_account_summary(account_summary: Dict):
        """
        Render account summary.
        
        Args:
            account_summary: Dictionary of account summary data
        """
        if not account_summary:
            st.info("No account data available")
            return
        
        # Create columns for account metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Net Liquidation Value",
                f"${account_summary.get('NetLiquidation', 0.0):,.2f}"
            )
        
        with col2:
            st.metric(
                "Cash Balance",
                f"${account_summary.get('TotalCashValue', 0.0):,.2f}"
            )
        
        with col3:
            st.metric(
                "Available Funds",
                f"${account_summary.get('AvailableFunds', 0.0):,.2f}"
            )
        
        with col4:
            st.metric(
                "Buying Power",
                f"${account_summary.get('BuyingPower', 0.0):,.2f}"
            )
    
    @staticmethod
    def render_positions_table(positions: Dict, market_data: Dict):
        """
        Render positions table.
        
        Args:
            positions: Dictionary of positions
            market_data: Dictionary of market data
        """
        if not positions:
            st.info("No positions available")
            return
        
        # Create DataFrame from positions
        positions_data = []
        
        for symbol, position in positions.items():
            # Get market data for current price
            current_price = 0.0
            if symbol in market_data:
                current_price = market_data[symbol].get('last', 0.0)
            
            # Calculate P&L
            position_size = position.get('position', 0.0)
            avg_cost = position.get('avg_cost', 0.0)
            market_value = position_size * current_price
            cost_basis = position_size * avg_cost
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis != 0 else 0.0
            
            positions_data.append({
                'Symbol': symbol,
                'Position': position_size,
                'Avg Cost': avg_cost,
                'Current Price': current_price,
                'Market Value': market_value,
                'Unrealized P&L': unrealized_pnl,
                'Unrealized P&L %': unrealized_pnl_pct
            })
        
        positions_df = pd.DataFrame(positions_data)
        
        # Format DataFrame
        positions_df['Avg Cost'] = positions_df['Avg Cost'].map('${:,.2f}'.format)
        positions_df['Current Price'] = positions_df['Current Price'].map('${:,.2f}'.format)
        positions_df['Market Value'] = positions_df['Market Value'].map('${:,.2f}'.format)
        positions_df['Unrealized P&L'] = positions_df['Unrealized P&L'].map('${:,.2f}'.format)
        positions_df['Unrealized P&L %'] = positions_df['Unrealized P&L %'].map('{:,.2f}%'.format)
        
        # Display positions table
        st.dataframe(positions_df)
    
    @staticmethod
    def render_orders_table(orders: Dict):
        """
        Render orders table.
        
        Args:
            orders: Dictionary of orders
        """
        if not orders:
            st.info("No orders available")
            return
        
        # Filter open orders
        open_orders = {
            order_id: order_info for order_id, order_info in orders.items()
            if order_info['status'] in ['Submitted', 'PreSubmitted', 'PendingSubmit']
        }
        
        if not open_orders:
            st.info("No open orders")
            return
        
        # Create DataFrame from orders
        orders_data = []
        
        for order_id, order_info in open_orders.items():
            orders_data.append({
                'Order ID': order_id,
                'Symbol': order_info.get('symbol', 'N/A'),
                'Action': order_info.get('action', 'N/A'),
                'Quantity': order_info.get('quantity', 0),
                'Order Type': order_info.get('order_type', 'N/A'),
                'Price': order_info.get('market_price', order_info.get('limit_price', 0.0)),
                'Status': order_info.get('status', 'N/A'),
                'Time': order_info.get('timestamp', datetime.now()).strftime('%H:%M:%S')
            })
        
        orders_df = pd.DataFrame(orders_data)
        
        # Format DataFrame
        orders_df['Price'] = orders_df['Price'].map('${:,.2f}'.format)
        
        # Display orders table
        st.dataframe(orders_df)


class BacktestComponents:
    """
    Components for the Backtesting section of the dashboard.
    """
    
    @staticmethod
    def render_backtest_form(strategies: Dict, default_strategy: str = None):
        """
        Render backtest settings form.
        
        Args:
            strategies: Dictionary of strategy objects
            default_strategy: Default strategy name
            
        Returns:
            Dict of backtest settings if form submitted, None otherwise
        """
        with st.form(key="backtest_settings"):
            # Strategy selection
            strategy_name = st.selectbox(
                "Select Strategy",
                list(strategies.keys()),
                index=list(strategies.keys()).index(default_strategy) if default_strategy in strategies else 0
            )
            
            # Date range
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365)
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now()
                )
            
            # Initial capital
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=1000.0,
                max_value=10000000.0,
                value=100000.0,
                step=10000.0
            )
            
            # Submit button
            if st.form_submit_button("Run Backtest"):
                return {
                    'strategy_name': strategy_name,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'initial_capital': initial_capital
                }
        
        return None
    
    @staticmethod
    def render_drawdown_chart(drawdowns: pd.DataFrame, title: str = "Drawdowns"):
        """
        Render drawdown chart.
        
        Args:
            drawdowns: DataFrame with drawdown data
            title: Chart title
        """
        if drawdowns.empty:
            st.info("No drawdown data available")
            return
        
        fig = px.area(
            drawdowns,
            x='date',
            y='drawdown',
            title=title
        )
        
        fig.update_traces(line_color='red')
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_monthly_returns_heatmap(monthly_returns: pd.DataFrame, title: str = "Monthly Returns"):
        """
        Render monthly returns heatmap.
        
        Args:
            monthly_returns: DataFrame with monthly returns data
            title: Chart title
        """
        if monthly_returns.empty:
            st.info("No monthly returns data available")
            return
        
        # Create heatmap
        fig = px.imshow(
            monthly_returns,
            labels=dict(x="Month", y="Year", color="Return"),
            x=monthly_returns.columns,
            y=monthly_returns.index,
            color_continuous_scale='RdYlGn',
            title=title
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_trade_statistics(trades: List[Dict]):
        """
        Render trade statistics.
        
        Args:
            trades: List of trade dictionaries
        """
        if not trades:
            st.info("No trade data available")
            return
        
        # Create DataFrame from trades
        trades_df = pd.DataFrame(trades)
        
        # Display trade metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(trades))
        
        with col2:
            st.metric("Avg Trade", f"${trades_df['pnl'].mean():.2f}")
        
        with col3:
            st.metric("Best Trade", f"${trades_df['pnl'].max():.2f}")
        
        with col4:
            st.metric("Worst Trade", f"${trades_df['pnl'].min():.2f}")
        
        # Display trades table
        st.dataframe(trades_df)
