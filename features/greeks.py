"""
Greeks Calculation Module

This module handles the calculation of option Greeks (delta, gamma, vega, theta, rho)
using various option pricing models including Black-Scholes and binomial trees.
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

try:
    import py_vollib
    from py_vollib.black_scholes import black_scholes as bs
    from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
    HAS_PYVOLLIB = True
except ImportError:
    HAS_PYVOLLIB = False
    logging.warning("py_vollib not installed. Greeks calculations will use approximations.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GreeksCalculator:
    """
    Class to handle the calculation of option Greeks.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the Greeks calculator with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.risk_free_rate = self.config.get('features', {}).get('greeks', {}).get('risk_free_rate', 0.03)
    
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
                    'greeks': {
                        'risk_free_rate': 0.03
                    }
                }
            }
    
    def calculate_greeks(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Greeks for a DataFrame of options.
        
        Args:
            options_df: DataFrame containing options data
            
        Returns:
            DataFrame with added Greeks columns
        """
        if options_df.empty:
            return options_df
            
        try:
            # Make a copy to avoid modifying the original
            df = options_df.copy()
            
            # Check if we have the required columns
            required_cols = ['strike', 'right', 'dte', 'underlying_price', 'implied_volatility']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns for Greeks calculation: {missing_cols}")
                return df
            
            # Use py_vollib if available, otherwise use approximations
            if HAS_PYVOLLIB:
                logger.info("Calculating Greeks using py_vollib")
                df = self._calculate_greeks_pyvollib(df)
            else:
                logger.info("Calculating Greeks using approximations")
                df = self._calculate_greeks_approximation(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return options_df
    
    def _calculate_greeks_pyvollib(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Greeks using py_vollib library.
        
        Args:
            df: DataFrame containing options data
            
        Returns:
            DataFrame with added Greeks columns
        """
        # Convert days to expiration to years
        df['T'] = df['dte'] / 365.0
        
        # Ensure implied_volatility is positive
        df['iv'] = np.abs(df['implied_volatility'])
        
        # Calculate Greeks for each option
        for idx, row in df.iterrows():
            try:
                # Get option parameters
                S = row['underlying_price']  # Underlying price
                K = row['strike']            # Strike price
                T = row['T']                 # Time to expiration in years
                r = self.risk_free_rate      # Risk-free rate
                sigma = row['iv']            # Implied volatility
                flag = row['right'].lower()  # Option type ('c' for call, 'p' for put)
                
                # Ensure flag is 'c' or 'p'
                if flag not in ['c', 'p']:
                    flag = 'c' if row['right'] in ['C', 'CALL'] else 'p'
                
                # Calculate Greeks
                df.at[idx, 'delta'] = delta(flag, S, K, T, r, sigma)
                df.at[idx, 'gamma'] = gamma(flag, S, K, T, r, sigma)
                df.at[idx, 'vega'] = vega(flag, S, K, T, r, sigma) / 100  # Convert to $ per 1% change in IV
                df.at[idx, 'theta'] = theta(flag, S, K, T, r, sigma) / 365  # Convert to $ per day
                df.at[idx, 'rho'] = rho(flag, S, K, T, r, sigma) / 100  # Convert to $ per 1% change in interest rate
                
            except Exception as e:
                logger.warning(f"Error calculating Greeks for row {idx}: {e}")
                # Set to NaN if calculation fails
                df.at[idx, 'delta'] = np.nan
                df.at[idx, 'gamma'] = np.nan
                df.at[idx, 'vega'] = np.nan
                df.at[idx, 'theta'] = np.nan
                df.at[idx, 'rho'] = np.nan
        
        # Drop temporary columns
        df = df.drop(['T', 'iv'], axis=1, errors='ignore')
        
        return df
    
    def _calculate_greeks_approximation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Greeks using approximations when py_vollib is not available.
        
        Args:
            df: DataFrame containing options data
            
        Returns:
            DataFrame with added Greeks columns
        """
        # Convert days to expiration to years
        df['T'] = df['dte'] / 365.0
        
        # Ensure implied_volatility is positive
        df['iv'] = np.abs(df['implied_volatility'])
        
        # Calculate Greeks for each option
        for idx, row in df.iterrows():
            try:
                # Get option parameters
                S = row['underlying_price']  # Underlying price
                K = row['strike']            # Strike price
                T = row['T']                 # Time to expiration in years
                r = self.risk_free_rate      # Risk-free rate
                sigma = row['iv']            # Implied volatility
                flag = row['right'].lower()  # Option type ('c' for call, 'p' for put)
                
                # Ensure flag is 'c' or 'p'
                if flag not in ['c', 'p']:
                    flag = 'c' if row['right'] in ['C', 'CALL'] else 'p'
                
                # Calculate d1 and d2 for Black-Scholes
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                
                # Standard normal CDF and PDF
                from scipy.stats import norm
                N_d1 = norm.cdf(d1)
                N_d2 = norm.cdf(d2)
                N_neg_d1 = norm.cdf(-d1)
                N_neg_d2 = norm.cdf(-d2)
                n_d1 = norm.pdf(d1)
                
                # Calculate Greeks
                if flag == 'c':
                    # Call option
                    df.at[idx, 'delta'] = N_d1
                    df.at[idx, 'gamma'] = n_d1 / (S * sigma * np.sqrt(T))
                    df.at[idx, 'vega'] = S * n_d1 * np.sqrt(T) / 100
                    df.at[idx, 'theta'] = (-S * n_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2) / 365
                    df.at[idx, 'rho'] = K * T * np.exp(-r * T) * N_d2 / 100
                else:
                    # Put option
                    df.at[idx, 'delta'] = N_d1 - 1
                    df.at[idx, 'gamma'] = n_d1 / (S * sigma * np.sqrt(T))
                    df.at[idx, 'vega'] = S * n_d1 * np.sqrt(T) / 100
                    df.at[idx, 'theta'] = (-S * n_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * N_neg_d2) / 365
                    df.at[idx, 'rho'] = -K * T * np.exp(-r * T) * N_neg_d2 / 100
                
            except Exception as e:
                logger.warning(f"Error calculating Greeks for row {idx}: {e}")
                # Set to NaN if calculation fails
                df.at[idx, 'delta'] = np.nan
                df.at[idx, 'gamma'] = np.nan
                df.at[idx, 'vega'] = np.nan
                df.at[idx, 'theta'] = np.nan
                df.at[idx, 'rho'] = np.nan
        
        # Drop temporary columns
        df = df.drop(['T', 'iv'], axis=1, errors='ignore')
        
        return df
    
    def calculate_option_price(self, 
                              flag: str, 
                              S: float, 
                              K: float, 
                              T: float, 
                              r: float, 
                              sigma: float) -> float:
        """
        Calculate theoretical option price using Black-Scholes model.
        
        Args:
            flag: Option type ('c' for call, 'p' for put)
            S: Underlying price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Implied volatility
            
        Returns:
            Theoretical option price
        """
        if HAS_PYVOLLIB:
            try:
                price = bs(flag, S, K, T, r, sigma)
                return price
            except Exception as e:
                logger.error(f"Error calculating option price with py_vollib: {e}")
        
        # Fallback to manual calculation
        try:
            # Calculate d1 and d2 for Black-Scholes
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Standard normal CDF
            from scipy.stats import norm
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)
            N_neg_d1 = norm.cdf(-d1)
            N_neg_d2 = norm.cdf(-d2)
            
            # Calculate price
            if flag.lower() == 'c':
                # Call option
                price = S * N_d1 - K * np.exp(-r * T) * N_d2
            else:
                # Put option
                price = K * np.exp(-r * T) * N_neg_d2 - S * N_neg_d1
                
            return price
            
        except Exception as e:
            logger.error(f"Error calculating option price manually: {e}")
            return np.nan
    
    def calculate_implied_volatility(self, 
                                    flag: str, 
                                    S: float, 
                                    K: float, 
                                    T: float, 
                                    r: float, 
                                    market_price: float, 
                                    precision: float = 0.00001, 
                                    max_iterations: int = 100) -> float:
        """
        Calculate implied volatility using bisection method.
        
        Args:
            flag: Option type ('c' for call, 'p' for put)
            S: Underlying price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            market_price: Market price of the option
            precision: Desired precision for implied volatility
            max_iterations: Maximum number of iterations
            
        Returns:
            Implied volatility
        """
        if HAS_PYVOLLIB:
            try:
                from py_vollib.black_scholes.implied_volatility import implied_volatility
                iv = implied_volatility(market_price, S, K, T, r, flag)
                return iv
            except Exception as e:
                logger.error(f"Error calculating implied volatility with py_vollib: {e}")
        
        # Fallback to bisection method
        try:
            # Initial volatility range
            sigma_low = 0.001
            sigma_high = 5.0
            
            # Check if market price is within theoretical bounds
            intrinsic_value = max(0, S - K) if flag.lower() == 'c' else max(0, K - S)
            if market_price < intrinsic_value:
                logger.warning(f"Market price {market_price} is below intrinsic value {intrinsic_value}")
                return np.nan
            
            # Bisection method
            for i in range(max_iterations):
                sigma_mid = (sigma_low + sigma_high) / 2
                price_mid = self.calculate_option_price(flag, S, K, T, r, sigma_mid)
                
                if abs(price_mid - market_price) < precision:
                    return sigma_mid
                
                if price_mid > market_price:
                    sigma_high = sigma_mid
                else:
                    sigma_low = sigma_mid
            
            # Return the midpoint after max iterations
            return (sigma_low + sigma_high) / 2
            
        except Exception as e:
            logger.error(f"Error calculating implied volatility manually: {e}")
            return np.nan


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    calculator = GreeksCalculator()
    
    # Example: Create sample options data
    import numpy as np
    
    # Sample options data
    options_data = {
        'symbol': ['AAPL'] * 10,
        'expiration': ['2023-06-16'] * 10,
        'strike': np.linspace(150, 200, 10),
        'right': ['C'] * 5 + ['P'] * 5,
        'dte': [30] * 10,
        'underlying_price': [175.0] * 10,
        'implied_volatility': [0.3] * 10
    }
    
    options_df = pd.DataFrame(options_data)
    
    # Calculate Greeks
    result_df = calculator.calculate_greeks(options_df)
    
    # Print results
    print(result_df[['strike', 'right', 'delta', 'gamma', 'vega', 'theta', 'rho']])
