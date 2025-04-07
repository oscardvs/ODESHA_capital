"""
Event Proximity Module

This module handles the calculation of event proximity features such as
days to earnings, days to FOMC meetings, and other significant market events.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventProximity:
    """
    Class to handle the calculation of event proximity features.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the event proximity calculator with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.event_types = self.config.get('features', {}).get('events', {}).get('types', 
                                                                               ["earnings", "fomc", "cpi", "nfp"])
    
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
                    'events': {
                        'types': ["earnings", "fomc", "cpi", "nfp"]
                    }
                }
            }
    
    def calculate_days_to_earnings(self, 
                                  symbol: str, 
                                  current_date: datetime, 
                                  earnings_dates: pd.DataFrame) -> int:
        """
        Calculate days to next earnings announcement.
        
        Args:
            symbol: Stock symbol
            current_date: Current date
            earnings_dates: DataFrame containing earnings dates
            
        Returns:
            Days to next earnings announcement
        """
        try:
            if earnings_dates.empty:
                return None
                
            # Filter for future earnings dates
            future_earnings = earnings_dates[earnings_dates['date'] > current_date]
            
            if future_earnings.empty:
                return None
                
            # Get the next earnings date
            next_earnings = future_earnings.iloc[0]['date']
            
            # Calculate days difference
            days_to_earnings = (next_earnings - current_date).days
            
            return days_to_earnings
            
        except Exception as e:
            logger.error(f"Error calculating days to earnings: {e}")
            return None
    
    def calculate_days_to_fomc(self, current_date: datetime, fomc_dates: List[datetime]) -> int:
        """
        Calculate days to next FOMC meeting.
        
        Args:
            current_date: Current date
            fomc_dates: List of FOMC meeting dates
            
        Returns:
            Days to next FOMC meeting
        """
        try:
            if not fomc_dates:
                return None
                
            # Filter for future FOMC dates
            future_fomc = [date for date in fomc_dates if date > current_date]
            
            if not future_fomc:
                return None
                
            # Get the next FOMC date
            next_fomc = min(future_fomc)
            
            # Calculate days difference
            days_to_fomc = (next_fomc - current_date).days
            
            return days_to_fomc
            
        except Exception as e:
            logger.error(f"Error calculating days to FOMC: {e}")
            return None
    
    def calculate_days_to_economic_release(self, 
                                         current_date: datetime, 
                                         release_type: str, 
                                         release_dates: List[datetime]) -> int:
        """
        Calculate days to next economic release (CPI, NFP, etc.).
        
        Args:
            current_date: Current date
            release_type: Type of economic release
            release_dates: List of release dates
            
        Returns:
            Days to next economic release
        """
        try:
            if not release_dates:
                return None
                
            # Filter for future release dates
            future_releases = [date for date in release_dates if date > current_date]
            
            if not future_releases:
                return None
                
            # Get the next release date
            next_release = min(future_releases)
            
            # Calculate days difference
            days_to_release = (next_release - current_date).days
            
            return days_to_release
            
        except Exception as e:
            logger.error(f"Error calculating days to {release_type}: {e}")
            return None
    
    def get_fomc_dates(self, year: int) -> List[datetime]:
        """
        Get FOMC meeting dates for a given year.
        
        Args:
            year: Year to get FOMC dates for
            
        Returns:
            List of FOMC meeting dates
        """
        # FOMC meeting dates for 2023-2025
        fomc_dates = {
            2023: [
                datetime(2023, 1, 31), datetime(2023, 3, 22), 
                datetime(2023, 5, 3), datetime(2023, 6, 14), 
                datetime(2023, 7, 26), datetime(2023, 9, 20), 
                datetime(2023, 11, 1), datetime(2023, 12, 13)
            ],
            2024: [
                datetime(2024, 1, 31), datetime(2024, 3, 20), 
                datetime(2024, 5, 1), datetime(2024, 6, 12), 
                datetime(2024, 7, 31), datetime(2024, 9, 18), 
                datetime(2024, 11, 7), datetime(2024, 12, 18)
            ],
            2025: [
                datetime(2025, 1, 29), datetime(2025, 3, 19), 
                datetime(2025, 4, 30), datetime(2025, 6, 11), 
                datetime(2025, 7, 30), datetime(2025, 9, 17), 
                datetime(2025, 11, 5), datetime(2025, 12, 17)
            ]
        }
        
        return fomc_dates.get(year, [])
    
    def get_cpi_dates(self, year: int) -> List[datetime]:
        """
        Get CPI release dates for a given year.
        
        Args:
            year: Year to get CPI dates for
            
        Returns:
            List of CPI release dates
        """
        # CPI release dates for 2023-2025
        cpi_dates = {
            2023: [
                datetime(2023, 1, 12), datetime(2023, 2, 14), 
                datetime(2023, 3, 14), datetime(2023, 4, 12), 
                datetime(2023, 5, 10), datetime(2023, 6, 13), 
                datetime(2023, 7, 12), datetime(2023, 8, 10), 
                datetime(2023, 9, 13), datetime(2023, 10, 12), 
                datetime(2023, 11, 14), datetime(2023, 12, 12)
            ],
            2024: [
                datetime(2024, 1, 11), datetime(2024, 2, 13), 
                datetime(2024, 3, 12), datetime(2024, 4, 10), 
                datetime(2024, 5, 15), datetime(2024, 6, 12), 
                datetime(2024, 7, 11), datetime(2024, 8, 14), 
                datetime(2024, 9, 11), datetime(2024, 10, 10), 
                datetime(2024, 11, 13), datetime(2024, 12, 11)
            ],
            2025: [
                datetime(2025, 1, 14), datetime(2025, 2, 13), 
                datetime(2025, 3, 13), datetime(2025, 4, 15), 
                datetime(2025, 5, 14), datetime(2025, 6, 12), 
                datetime(2025, 7, 11), datetime(2025, 8, 13), 
                datetime(2025, 9, 11), datetime(2025, 10, 14), 
                datetime(2025, 11, 13), datetime(2025, 12, 11)
            ]
        }
        
        return cpi_dates.get(year, [])
    
    def get_nfp_dates(self, year: int) -> List[datetime]:
        """
        Get Non-Farm Payroll (NFP) release dates for a given year.
        
        Args:
            year: Year to get NFP dates for
            
        Returns:
            List of NFP release dates
        """
        # NFP release dates for 2023-2025
        nfp_dates = {
            2023: [
                datetime(2023, 1, 6), datetime(2023, 2, 3), 
                datetime(2023, 3, 10), datetime(2023, 4, 7), 
                datetime(2023, 5, 5), datetime(2023, 6, 2), 
                datetime(2023, 7, 7), datetime(2023, 8, 4), 
                datetime(2023, 9, 1), datetime(2023, 10, 6), 
                datetime(2023, 11, 3), datetime(2023, 12, 8)
            ],
            2024: [
                datetime(2024, 1, 5), datetime(2024, 2, 2), 
                datetime(2024, 3, 8), datetime(2024, 4, 5), 
                datetime(2024, 5, 3), datetime(2024, 6, 7), 
                datetime(2024, 7, 5), datetime(2024, 8, 2), 
                datetime(2024, 9, 6), datetime(2024, 10, 4), 
                datetime(2024, 11, 1), datetime(2024, 12, 6)
            ],
            2025: [
                datetime(2025, 1, 10), datetime(2025, 2, 7), 
                datetime(2025, 3, 7), datetime(2025, 4, 4), 
                datetime(2025, 5, 2), datetime(2025, 6, 6), 
                datetime(2025, 7, 3), datetime(2025, 8, 1), 
                datetime(2025, 9, 5), datetime(2025, 10, 3), 
                datetime(2025, 11, 7), datetime(2025, 12, 5)
            ]
        }
        
        return nfp_dates.get(year, [])
    
    def add_event_proximity_features(self, 
                                    df: pd.DataFrame, 
                                    symbol: str = None, 
                                    earnings_dates: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add event proximity features to DataFrame.
        
        Args:
            df: DataFrame to add features to
            symbol: Stock symbol
            earnings_dates: DataFrame containing earnings dates
            
        Returns:
            DataFrame with added event proximity features
        """
        if df.empty:
            return df
            
        try:
            # Make a copy to avoid modifying the original
            result_df = df.copy()
            
            # Get current date
            if 'date' in result_df.columns:
                current_date = pd.to_datetime(result_df['date'].iloc[-1])
            else:
                current_date = datetime.now()
            
            # Calculate days to earnings
            if "earnings" in self.event_types and symbol and earnings_dates is not None:
                days_to_earnings = self.calculate_days_to_earnings(symbol, current_date, earnings_dates)
                result_df['days_to_earnings'] = days_to_earnings
            
            # Calculate days to FOMC
            if "fomc" in self.event_types:
                current_year = current_date.year
                fomc_dates = self.get_fomc_dates(current_year) + self.get_fomc_dates(current_year + 1)
                days_to_fomc = self.calculate_days_to_fomc(current_date, fomc_dates)
                result_df['days_to_fomc'] = days_to_fomc
            
            # Calculate days to CPI
            if "cpi" in self.event_types:
                current_year = current_date.year
                cpi_dates = self.get_cpi_dates(current_year) + self.get_cpi_dates(current_year + 1)
                days_to_cpi = self.calculate_days_to_economic_release(current_date, "CPI", cpi_dates)
                result_df['days_to_cpi'] = days_to_cpi
            
            # Calculate days to NFP
            if "nfp" in self.event_types:
                current_year = current_date.year
                nfp_dates = self.get_nfp_dates(current_year) + self.get_nfp_dates(current_year + 1)
                days_to_nfp = self.calculate_days_to_economic_release(current_date, "NFP", nfp_dates)
                result_df['days_to_nfp'] = days_to_nfp
            
            # Add event proximity score (lower is closer to an event)
            event_columns = [col for col in result_df.columns if col.startswith('days_to_')]
            
            if event_columns:
                # Replace None with a large number
                for col in event_columns:
                    result_df[col] = result_df[col].fillna(365)
                
                # Calculate minimum days to any event
                result_df['min_days_to_event'] = result_df[event_columns].min(axis=1)
                
                # Calculate event proximity score (inverse of days, normalized to 0-1)
                result_df['event_proximity_score'] = 1 / (1 + result_df['min_days_to_event'])
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding event proximity features: {e}")
            return df


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    event_proximity = EventProximity()
    
    # Example: Create sample data
    data = {
        'date': [datetime.now()],
        'close': [100]
    }
    
    df = pd.DataFrame(data)
    
    # Example earnings dates
    earnings_data = {
        'date': [datetime.now() + timedelta(days=15)],
        'estimate': [1.5],
        'actual': [None]
    }
    
    earnings_df = pd.DataFrame(earnings_data)
    
    # Add event proximity features
    result_df = event_proximity.add_event_proximity_features(df, 'AAPL', earnings_df)
    
    # Print results
    print("Event Proximity Features:")
    print(result_df)
