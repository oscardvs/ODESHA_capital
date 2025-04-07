"""
IV Predictor Model

This module implements a regression model for predicting implied volatility changes
using LightGBM or other gradient boosting models.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
import joblib
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

# Check for ML libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("lightgbm not installed. LightGBM models will not be available.")

try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed. Model training and evaluation will be limited.")


class IVPredictor:
    """
    Class for training and using regression models for predicting IV changes.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the IV predictor with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.model_config = self.config.get('ml', {}).get('models', {}).get('iv_predictor', {})
        self.model_type = self.model_config.get('type', 'lightgbm')
        self.target = self.model_config.get('target', 'iv_change')
        self.features = self.model_config.get('features', [])
        self.train_test_split_ratio = self.model_config.get('train_test_split', 0.8)
        self.cv_folds = self.model_config.get('cv_folds', 5)
        
        # Initialize model
        self.model = None
        self.scaler = None
        self.feature_importance = None
    
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
                'ml': {
                    'models': {
                        'iv_predictor': {
                            'type': 'lightgbm',
                            'target': 'iv_change',
                            'features': ['iv_rank', 'iv_percentile', 'hv_10', 'hv_20', 'days_to_earnings'],
                            'train_test_split': 0.8,
                            'cv_folds': 5
                        }
                    }
                }
            }
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training or prediction.
        
        Args:
            df: DataFrame containing features and target
            
        Returns:
            Tuple of (X, y) where X is feature DataFrame and y is target Series
        """
        if df.empty:
            logger.error("Empty DataFrame provided")
            return pd.DataFrame(), pd.Series()
            
        try:
            # Check if we have the required columns
            missing_features = [f for f in self.features if f not in df.columns]
            
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                return pd.DataFrame(), pd.Series()
            
            # Extract features
            X = df[self.features].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Extract target if available
            if self.target in df.columns:
                y = df[self.target]
            else:
                y = pd.Series()
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def train(self, df: pd.DataFrame, hyperparams: Dict = None) -> Dict[str, Any]:
        """
        Train the IV predictor model.
        
        Args:
            df: DataFrame containing features and target
            hyperparams: Optional dictionary of hyperparameters
            
        Returns:
            Dictionary with training results
        """
        if df.empty:
            logger.error("Empty DataFrame provided for training")
            return {'success': False, 'error': 'Empty DataFrame'}
            
        if not HAS_SKLEARN:
            logger.error("scikit-learn is required for model training")
            return {'success': False, 'error': 'scikit-learn not installed'}
            
        try:
            # Prepare data
            X, y = self.prepare_data(df)
            
            if X.empty or y.empty:
                return {'success': False, 'error': 'Failed to prepare data'}
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=self.train_test_split_ratio, random_state=42
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize model based on type
            if self.model_type == 'lightgbm':
                if not HAS_LIGHTGBM:
                    logger.error("lightgbm is required for LightGBM models")
                    return {'success': False, 'error': 'lightgbm not installed'}
                
                # Default hyperparameters
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'n_estimators': 100,
                    'random_state': 42
                }
                
                # Update with provided hyperparameters
                if hyperparams:
                    params.update(hyperparams)
                
                # Create and train model
                self.model = lgb.LGBMRegressor(**params)
                
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return {'success': False, 'error': f'Unsupported model type: {self.model_type}'}
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Get feature importance
            if self.model_type == 'lightgbm':
                importance = self.model.feature_importances_
            else:
                importance = np.zeros(len(self.features))
            
            # Create feature importance dictionary
            self.feature_importance = dict(zip(self.features, importance))
            
            # Sort feature importance
            self.feature_importance = {k: v for k, v in sorted(
                self.feature_importance.items(), key=lambda item: item[1], reverse=True
            )}
            
            # Return results
            return {
                'success': True,
                'model_type': self.model_type,
                'metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                },
                'feature_importance': self.feature_importance,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the trained model.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            DataFrame with added prediction columns
        """
        if df.empty:
            logger.error("Empty DataFrame provided for prediction")
            return df
            
        if self.model is None:
            logger.error("Model not trained")
            return df
            
        try:
            # Prepare data
            X, _ = self.prepare_data(df)
            
            if X.empty:
                return df
            
            # Scale features
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Make predictions
            y_pred = self.model.predict(X_scaled)
            
            # Add predictions to DataFrame
            result_df = df.copy()
            result_df['predicted_iv_change'] = y_pred
            
            # Calculate predicted IV
            if 'implied_volatility' in result_df.columns:
                result_df['predicted_iv'] = result_df['implied_volatility'] * (1 + result_df['predicted_iv_change'])
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return df
    
    def hyperparameter_tuning(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            df: DataFrame containing features and target
            
        Returns:
            Dictionary with tuning results
        """
        if df.empty:
            logger.error("Empty DataFrame provided for hyperparameter tuning")
            return {'success': False, 'error': 'Empty DataFrame'}
            
        if not HAS_SKLEARN:
            logger.error("scikit-learn is required for hyperparameter tuning")
            return {'success': False, 'error': 'scikit-learn not installed'}
            
        try:
            # Prepare data
            X, y = self.prepare_data(df)
            
            if X.empty or y.empty:
                return {'success': False, 'error': 'Failed to prepare data'}
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Define parameter grid based on model type
            if self.model_type == 'lightgbm':
                if not HAS_LIGHTGBM:
                    logger.error("lightgbm is required for LightGBM models")
                    return {'success': False, 'error': 'lightgbm not installed'}
                
                model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)
                
                param_grid = {
                    'num_leaves': [15, 31, 63],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'feature_fraction': [0.7, 0.8, 0.9]
                }
                
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return {'success': False, 'error': f'Unsupported model type: {self.model_type}'}
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_scaled, y)
            
            # Get best parameters and score
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Convert back to MSE
            
            # Train model with best parameters
            training_result = self.train(df, best_params)
            
            # Return results
            return {
                'success': True,
                'best_params': best_params,
                'best_cv_score': best_score,
                'training_result': training_result
            }
            
        except Exception as e:
            logger.error(f"Error performing hyperparameter tuning: {e}")
            return {'success': False, 'error': str(e)}
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No trained model to save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Save model
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'features': self.features,
                'model_type': self.model_type,
                'target': self.target
            }, filepath)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
            
            # Load model
            model_data = joblib.load(filepath)
            
            # Set model attributes
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data['feature_importance']
            self.features = model_data['features']
            self.model_type = model_data['model_type']
            self.target = model_data['target']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    predictor = IVPredictor()
    
    # Example: Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = {
        'iv_rank': np.random.uniform(0, 1, n_samples),
        'iv_percentile': np.random.uniform(0, 1, n_samples),
        'hv_10': np.random.uniform(0.1, 0.5, n_samples),
        'hv_20': np.random.uniform(0.1, 0.4, n_samples),
        'days_to_earnings': np.random.randint(1, 60, n_samples),
        'implied_volatility': np.random.uniform(0.2, 0.6, n_samples)
    }
    
    # Generate synthetic target (iv_change)
    # Higher IV rank and IV percentile tend to predict IV decrease
    # Higher HV/IV ratio tends to predict IV increase
    # Proximity to earnings tends to predict IV increase
    iv_change = np.zeros(n_samples)
    
    for i in range(n_samples):
        iv_rank = data['iv_rank'][i]
        iv_percentile = data['iv_percentile'][i]
        hv_iv_ratio = data['hv_20'][i] / data['implied_volatility'][i]
        days_to_earnings = data['days_to_earnings'][i]
        
        # Logic for synthetic data generation
        iv_change[i] = (
            -0.2 * iv_rank +                          # Higher IV rank -> IV decrease
            -0.1 * iv_percentile +                    # Higher IV percentile -> IV decrease
            0.3 * (hv_iv_ratio - 0.5) +               # Higher HV/IV ratio -> IV increase
            -0.01 * days_to_earnings +                # Closer to earnings -> IV increase
            np.random.normal(0, 0.05)                 # Random noise
        )
    
    # Create DataFrame
    data['iv_change'] = iv_change
    df = pd.DataFrame(data)
    
    # Train model
    print("Training model...")
    result = predictor.train(df)
    
    if result['success']:
        print(f"Training successful. RMSE: {result['metrics']['rmse']:.4f}")
        print(f"Feature importance: {result['feature_importance']}")
        
        # Save model
        predictor.save_model("../models/iv_predictor.joblib")
        
        # Make predictions on new data
        new_data = df.drop('iv_change', axis=1).head(5)
        predictions = predictor.predict(new_data)
        
        print("\nPredictions:")
        print(predictions[['iv_rank', 'implied_volatility', 'predicted_iv_change', 'predicted_iv']])
    else:
        print(f"Training failed: {result['error']}")
