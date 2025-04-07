"""
Stress Testing Module

This module provides functionality for stress testing options trading strategies
under various market scenarios and extreme conditions.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to import backtest engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtests.backtest_engine import BacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StressTester:
    """
    Class for stress testing options trading strategies.
    """
    
    def __init__(self, backtest_engine: BacktestEngine):
        """
        Initialize the stress tester with a backtest engine.
        
        Args:
            backtest_engine: Initialized backtest engine
        """
        self.backtest_engine = backtest_engine
        self.original_market_data = {}
        self.original_options_data = {}
        self.stress_test_results = {}
    
    def save_original_data(self):
        """
        Save the original data for later restoration.
        """
        self.original_market_data = self.backtest_engine.market_data.copy()
        self.original_options_data = self.backtest_engine.options_data.copy()
    
    def restore_original_data(self):
        """
        Restore the original data.
        """
        self.backtest_engine.market_data = self.original_market_data.copy()
        self.backtest_engine.options_data = self.original_options_data.copy()
    
    def apply_price_shock(self, shock_pct: float):
        """
        Apply a price shock to the market data.
        
        Args:
            shock_pct: Percentage shock to apply (e.g., -0.10 for -10%)
        """
        try:
            # Save original data if not already saved
            if not self.original_market_data:
                self.save_original_data()
            
            # Apply shock to market data
            for symbol, df in self.backtest_engine.market_data.items():
                shocked_df = df.copy()
                
                # Apply shock to price columns
                for col in ['open', 'high', 'low', 'close']:
                    if col in shocked_df.columns:
                        shocked_df[col] = shocked_df[col] * (1 + shock_pct)
                
                # Update market data
                self.backtest_engine.market_data[symbol] = shocked_df
            
            # Apply shock to options data
            for symbol, data in self.backtest_engine.options_data.items():
                if isinstance(data, dict):  # Dictionary of dates
                    for date, options_df in data.items():
                        shocked_options = options_df.copy()
                        
                        # Apply shock to underlying price
                        if 'underlying_price' in shocked_options.columns:
                            shocked_options['underlying_price'] = shocked_options['underlying_price'] * (1 + shock_pct)
                        
                        # Update options data
                        self.backtest_engine.options_data[symbol][date] = shocked_options
                else:  # DataFrame
                    shocked_options = data.copy()
                    
                    # Apply shock to underlying price
                    if 'underlying_price' in shocked_options.columns:
                        shocked_options['underlying_price'] = shocked_options['underlying_price'] * (1 + shock_pct)
                    
                    # Update options data
                    self.backtest_engine.options_data[symbol] = shocked_options
            
            logger.info(f"Applied price shock of {shock_pct:.2%}")
        except Exception as e:
            logger.error(f"Error applying price shock: {e}")
    
    def apply_volatility_shock(self, shock_pct: float):
        """
        Apply a volatility shock to the options data.
        
        Args:
            shock_pct: Percentage shock to apply (e.g., 0.50 for +50%)
        """
        try:
            # Save original data if not already saved
            if not self.original_options_data:
                self.save_original_data()
            
            # Apply shock to options data
            for symbol, data in self.backtest_engine.options_data.items():
                if isinstance(data, dict):  # Dictionary of dates
                    for date, options_df in data.items():
                        shocked_options = options_df.copy()
                        
                        # Apply shock to implied volatility
                        if 'implied_volatility' in shocked_options.columns:
                            shocked_options['implied_volatility'] = shocked_options['implied_volatility'] * (1 + shock_pct)
                        
                        # Update options data
                        self.backtest_engine.options_data[symbol][date] = shocked_options
                else:  # DataFrame
                    shocked_options = data.copy()
                    
                    # Apply shock to implied volatility
                    if 'implied_volatility' in shocked_options.columns:
                        shocked_options['implied_volatility'] = shocked_options['implied_volatility'] * (1 + shock_pct)
                    
                    # Update options data
                    self.backtest_engine.options_data[symbol] = shocked_options
            
            logger.info(f"Applied volatility shock of {shock_pct:.2%}")
        except Exception as e:
            logger.error(f"Error applying volatility shock: {e}")
    
    def apply_correlation_shock(self, correlation_factor: float):
        """
        Apply a correlation shock to the market data.
        
        Args:
            correlation_factor: Factor to increase correlation (e.g., 1.5 for 50% increase)
        """
        try:
            # Save original data if not already saved
            if not self.original_market_data:
                self.save_original_data()
            
            # Get all symbols
            symbols = list(self.backtest_engine.market_data.keys())
            
            if len(symbols) < 2:
                logger.warning("Need at least 2 symbols for correlation shock")
                return
            
            # Create a combined DataFrame of returns
            returns_df = pd.DataFrame()
            
            for symbol in symbols:
                df = self.backtest_engine.market_data[symbol]
                if 'date' in df.columns and 'close' in df.columns:
                    returns_df[symbol] = df.set_index('date')['close'].pct_change()
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Apply correlation shock
            shocked_corr = corr_matrix.copy()
            
            # Increase correlation (move values closer to 1 or -1)
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    sym1, sym2 = symbols[i], symbols[j]
                    orig_corr = corr_matrix.loc[sym1, sym2]
                    
                    # Apply shock (move closer to 1 or -1)
                    if orig_corr >= 0:
                        shocked_corr.loc[sym1, sym2] = shocked_corr.loc[sym2, sym1] = min(1, orig_corr * correlation_factor)
                    else:
                        shocked_corr.loc[sym1, sym2] = shocked_corr.loc[sym2, sym1] = max(-1, orig_corr * correlation_factor)
            
            # Generate shocked returns
            # This is a simplified approach - in practice, you would use a more sophisticated method
            # like Cholesky decomposition to generate correlated returns
            
            # Get mean and std of returns
            mean_returns = returns_df.mean()
            std_returns = returns_df.std()
            
            # Generate random returns with shocked correlation
            np.random.seed(42)  # For reproducibility
            num_days = len(returns_df)
            
            # Generate uncorrelated random variables
            uncorrelated = np.random.normal(0, 1, size=(num_days, len(symbols)))
            
            # Compute Cholesky decomposition
            L = np.linalg.cholesky(shocked_corr)
            
            # Generate correlated random variables
            correlated = np.dot(uncorrelated, L.T)
            
            # Convert to returns
            for i, symbol in enumerate(symbols):
                correlated[:, i] = correlated[:, i] * std_returns[symbol] + mean_returns[symbol]
            
            # Convert returns to prices
            for i, symbol in enumerate(symbols):
                df = self.backtest_engine.market_data[symbol].copy()
                if 'date' in df.columns and 'close' in df.columns:
                    # Get base price
                    base_price = df['close'].iloc[0]
                    
                    # Generate new prices
                    new_prices = [base_price]
                    for ret in correlated[1:, i]:
                        new_prices.append(new_prices[-1] * (1 + ret))
                    
                    # Update close prices
                    df['close'] = new_prices[:len(df)]
                    
                    # Update other price columns proportionally
                    for col in ['open', 'high', 'low']:
                        if col in df.columns:
                            ratio = df[col] / df['close']
                            df[col] = df['close'] * ratio
                    
                    # Update market data
                    self.backtest_engine.market_data[symbol] = df
            
            logger.info(f"Applied correlation shock with factor {correlation_factor:.2f}")
        except Exception as e:
            logger.error(f"Error applying correlation shock: {e}")
    
    def apply_liquidity_shock(self, spread_increase_factor: float):
        """
        Apply a liquidity shock by increasing bid-ask spreads.
        
        Args:
            spread_increase_factor: Factor to increase spreads (e.g., 2.0 for doubling)
        """
        try:
            # Save original data if not already saved
            if not self.original_options_data:
                self.save_original_data()
            
            # Apply shock to options data
            for symbol, data in self.backtest_engine.options_data.items():
                if isinstance(data, dict):  # Dictionary of dates
                    for date, options_df in data.items():
                        shocked_options = options_df.copy()
                        
                        # Apply shock to bid-ask spread
                        if all(col in shocked_options.columns for col in ['bid', 'ask']):
                            # Calculate mid price
                            mid_price = (shocked_options['bid'] + shocked_options['ask']) / 2
                            
                            # Calculate original spread
                            original_spread = shocked_options['ask'] - shocked_options['bid']
                            
                            # Calculate new spread
                            new_spread = original_spread * spread_increase_factor
                            
                            # Update bid and ask
                            shocked_options['bid'] = mid_price - new_spread / 2
                            shocked_options['ask'] = mid_price + new_spread / 2
                            
                            # Ensure bid is not negative
                            shocked_options['bid'] = shocked_options['bid'].clip(lower=0.01)
                            
                            # Update option price (mid price)
                            if 'option_price' in shocked_options.columns:
                                shocked_options['option_price'] = (shocked_options['bid'] + shocked_options['ask']) / 2
                        
                        # Update options data
                        self.backtest_engine.options_data[symbol][date] = shocked_options
                else:  # DataFrame
                    shocked_options = data.copy()
                    
                    # Apply shock to bid-ask spread
                    if all(col in shocked_options.columns for col in ['bid', 'ask']):
                        # Calculate mid price
                        mid_price = (shocked_options['bid'] + shocked_options['ask']) / 2
                        
                        # Calculate original spread
                        original_spread = shocked_options['ask'] - shocked_options['bid']
                        
                        # Calculate new spread
                        new_spread = original_spread * spread_increase_factor
                        
                        # Update bid and ask
                        shocked_options['bid'] = mid_price - new_spread / 2
                        shocked_options['ask'] = mid_price + new_spread / 2
                        
                        # Ensure bid is not negative
                        shocked_options['bid'] = shocked_options['bid'].clip(lower=0.01)
                        
                        # Update option price (mid price)
                        if 'option_price' in shocked_options.columns:
                            shocked_options['option_price'] = (shocked_options['bid'] + shocked_options['ask']) / 2
                    
                    # Update options data
                    self.backtest_engine.options_data[symbol] = shocked_options
            
            logger.info(f"Applied liquidity shock with spread increase factor {spread_increase_factor:.2f}")
        except Exception as e:
            logger.error(f"Error applying liquidity shock: {e}")
    
    def apply_slippage_shock(self, slippage_increase_factor: float):
        """
        Apply a slippage shock by increasing the slippage parameter.
        
        Args:
            slippage_increase_factor: Factor to increase slippage (e.g., 3.0 for tripling)
        """
        try:
            # Save original slippage
            original_slippage = self.backtest_engine.slippage_pct
            
            # Apply shock to slippage
            self.backtest_engine.slippage_pct = original_slippage * slippage_increase_factor
            
            logger.info(f"Applied slippage shock: {original_slippage:.2%} -> {self.backtest_engine.slippage_pct:.2%}")
        except Exception as e:
            logger.error(f"Error applying slippage shock: {e}")
    
    def apply_commission_shock(self, commission_increase_factor: float):
        """
        Apply a commission shock by increasing the commission parameter.
        
        Args:
            commission_increase_factor: Factor to increase commission (e.g., 2.0 for doubling)
        """
        try:
            # Save original commission
            original_commission = self.backtest_engine.commission_per_contract
            
            # Apply shock to commission
            self.backtest_engine.commission_per_contract = original_commission * commission_increase_factor
            
            logger.info(f"Applied commission shock: ${original_commission:.2f} -> ${self.backtest_engine.commission_per_contract:.2f}")
        except Exception as e:
            logger.error(f"Error applying commission shock: {e}")
    
    def run_stress_test(self, scenario_name: str, regenerate_features: bool = True):
        """
        Run a stress test with the current data.
        
        Args:
            scenario_name: Name of the stress test scenario
            regenerate_features: Whether to regenerate features after applying shocks
        """
        try:
            logger.info(f"Running stress test: {scenario_name}")
            
            # Regenerate features if needed
            if regenerate_features:
                self.backtest_engine.generate_features()
            
            # Generate signals
            self.backtest_engine.generate_signals()
            
            # Run backtest
            self.backtest_engine.run_backtest()
            
            # Store results
            self.stress_test_results[scenario_name] = {
                'metrics': self.backtest_engine.metrics.copy() if self.backtest_engine.metrics else None,
                'portfolio': {
                    'history': self.backtest_engine.portfolio['history'].copy() if self.backtest_engine.portfolio['history'] else [],
                    'trades': self.backtest_engine.portfolio['trades'].copy() if self.backtest_engine.portfolio['trades'] else []
                }
            }
            
            logger.info(f"Completed stress test: {scenario_name}")
        except Exception as e:
            logger.error(f"Error running stress test: {e}")
    
    def run_standard_stress_tests(self):
        """
        Run a set of standard stress tests.
        """
        try:
            # Save original data
            self.save_original_data()
            
            # Run baseline test
            self.restore_original_data()
            self.run_stress_test("Baseline")
            
            # Market crash scenario (-20% price shock)
            self.restore_original_data()
            self.apply_price_shock(-0.20)
            self.run_stress_test("Market Crash (-20%)")
            
            # Market rally scenario (+20% price shock)
            self.restore_original_data()
            self.apply_price_shock(0.20)
            self.run_stress_test("Market Rally (+20%)")
            
            # Volatility spike scenario (+100% vol shock)
            self.restore_original_data()
            self.apply_volatility_shock(1.0)
            self.run_stress_test("Volatility Spike (+100%)")
            
            # Volatility collapse scenario (-50% vol shock)
            self.restore_original_data()
            self.apply_volatility_shock(-0.5)
            self.run_stress_test("Volatility Collapse (-50%)")
            
            # Correlation shock scenario (1.5x correlation)
            self.restore_original_data()
            self.apply_correlation_shock(1.5)
            self.run_stress_test("Increased Correlation (1.5x)")
            
            # Liquidity crisis scenario (3x spreads)
            self.restore_original_data()
            self.apply_liquidity_shock(3.0)
            self.run_stress_test("Liquidity Crisis (3x Spreads)")
            
            # High slippage scenario (5x slippage)
            self.restore_original_data()
            self.apply_slippage_shock(5.0)
            self.run_stress_test("High Slippage (5x)")
            
            # High commission scenario (2x commission)
            self.restore_original_data()
            self.apply_commission_shock(2.0)
            self.run_stress_test("High Commission (2x)")
            
            # Combined stress scenario
            self.restore_original_data()
            self.apply_price_shock(-0.15)
            self.apply_volatility_shock(0.75)
            self.apply_liquidity_shock(2.0)
            self.apply_slippage_shock(3.0)
            self.run_stress_test("Combined Stress")
            
            # Restore original data
            self.restore_original_data()
            
            logger.info("Completed standard stress tests")
        except Exception as e:
            logger.error(f"Error running standard stress tests: {e}")
            # Restore original data
            self.restore_original_data()
    
    def generate_stress_test_report(self, save_path=None):
        """
        Generate a report comparing stress test results.
        
        Args:
            save_path: Path to save the report (if None, return the report as a string)
            
        Returns:
            Report as a string if save_path is None
        """
        try:
            # Check if we have results
            if not self.stress_test_results:
                logger.warning("No stress test results to generate report")
                return "No stress test results available"
            
            # Create report
            report = []
            report.append("# Stress Test Report")
            report.append("")
            
            # Create comparison table
            report.append("## Performance Comparison")
            report.append("")
            report.append("| Scenario | Final Value | Total Return | Max Drawdown | Sharpe Ratio | Win Rate |")
            report.append("|----------|-------------|--------------|--------------|--------------|----------|")
            
            for scenario, results in self.stress_test_results.items():
                if results['metrics']:
                    metrics = results['metrics']
                    report.append(f"| {scenario} | ${metrics['final_value']:.2f} | {metrics['total_return']:.2%} | {metrics['max_drawdown']:.2%} | {metrics['sharpe_ratio']:.2f} | {metrics['win_rate']:.2%} |")
            
            report.append("")
            
            # Add detailed results for each scenario
            report.append("## Detailed Results")
            report.append("")
            
            for scenario, results in self.stress_test_results.items():
                report.append(f"### {scenario}")
                report.append("")
                
                if results['metrics']:
                    metrics = results['metrics']
                    report.append(f"Initial Capital: ${metrics['initial_capital']:.2f}")
                    report.append(f"Final Value: ${metrics['final_value']:.2f}")
                    report.append(f"Total Return: {metrics['total_return']:.2%}")
                    report.append(f"Annualized Return: {metrics['annualized_return']:.2%}")
                    report.append(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
                    report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    report.append(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
                    report.append(f"Win Rate: {metrics['win_rate']:.2%}")
                    report.append(f"Number of Trades: {metrics['num_trades']}")
                    report.append("")
            
            # Join report
            report_str = "\n".join(report)
            
            # Save or return
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_str)
                logger.info(f"Stress test report saved to {save_path}")
                return None
            else:
                return report_str
        except Exception as e:
            logger.error(f"Error generating stress test report: {e}")
            return f"Error generating stress test report: {e}"
    
    def plot_stress_test_comparison(self, save_path=None):
        """
        Plot a comparison of stress test results.
        
        Args:
            save_path: Path to save the plot (if None, display the plot)
        """
        try:
            # Check if we have results
            if not self.stress_test_results:
                logger.warning("No stress test results to plot")
                return
            
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Plot equity curves
            plt.subplot(2, 1, 1)
            
            for scenario, results in self.stress_test_results.items():
                if results['portfolio']['history']:
                    history_df = pd.DataFrame(results['portfolio']['history'])
                    plt.plot(history_df['date'], history_df['total_value'], label=scenario)
            
            plt.title('Equity Curves Comparison')
            plt.xlabel('Date')
            plt.ylabel('Value ($)')
            plt.grid(True)
            plt.legend()
            
            # Plot drawdowns
            plt.subplot(2, 1, 2)
            
            for scenario, results in self.stress_test_results.items():
                if results['portfolio']['history']:
                    history_df = pd.DataFrame(results['portfolio']['history'])
                    history_df['peak'] = history_df['total_value'].cummax()
                    history_df['drawdown'] = (history_df['total_value'] - history_df['peak']) / history_df['peak']
                    plt.plot(history_df['date'], history_df['drawdown'], label=scenario)
            
            plt.title('Drawdown Comparison')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Stress test comparison plot saved to {save_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Error plotting stress test comparison: {e}")
    
    def plot_stress_test_metrics(self, save_path=None):
        """
        Plot key metrics from stress tests as a bar chart.
        
        Args:
            save_path: Path to save the plot (if None, display the plot)
        """
        try:
            # Check if we have results
            if not self.stress_test_results:
                logger.warning("No stress test results to plot")
                return
            
            # Extract metrics
            scenarios = []
            total_returns = []
            max_drawdowns = []
            sharpe_ratios = []
            
            for scenario, results in self.stress_test_results.items():
                if results['metrics']:
                    scenarios.append(scenario)
                    total_returns.append(results['metrics']['total_return'] * 100)  # Convert to percentage
                    max_drawdowns.append(results['metrics']['max_drawdown'] * 100)  # Convert to percentage
                    sharpe_ratios.append(results['metrics']['sharpe_ratio'])
            
            # Create figure
            plt.figure(figsize=(15, 12))
            
            # Plot total returns
            plt.subplot(3, 1, 1)
            plt.bar(scenarios, total_returns)
            plt.title('Total Return (%)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y')
            
            # Plot max drawdowns
            plt.subplot(3, 1, 2)
            plt.bar(scenarios, max_drawdowns)
            plt.title('Max Drawdown (%)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y')
            
            # Plot Sharpe ratios
            plt.subplot(3, 1, 3)
            plt.bar(scenarios, sharpe_ratios)
            plt.title('Sharpe Ratio')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Stress test metrics plot saved to {save_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Error plotting stress test metrics: {e}")


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    from strategies.directional_strategy import DirectionalOptionsStrategy
    
    # Create backtest engine
    backtest = BacktestEngine()
    
    # Set strategy
    strategy = DirectionalOptionsStrategy()
    backtest.set_strategy(strategy)
    
    # Load data
    backtest.load_market_data("../data/market_data.csv")
    backtest.load_options_data("../data/options_data.csv")
    
    # Generate features
    backtest.generate_features()
    
    # Create stress tester
    stress_tester = StressTester(backtest)
    
    # Run standard stress tests
    stress_tester.run_standard_stress_tests()
    
    # Generate report
    report = stress_tester.generate_stress_test_report("stress_test_report.md")
    
    # Plot results
    stress_tester.plot_stress_test_comparison("stress_test_comparison.png")
    stress_tester.plot_stress_test_metrics("stress_test_metrics.png")
