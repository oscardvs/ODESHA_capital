"""
Direction Classifier Model

This module implements a binary classifier for predicting directional price moves
using XGBoost or CatBoost models.
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
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("xgboost not installed. XGBoost models will not be available.")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("catboost not installed. CatBoost models will not be available.")

try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed. Model training and evaluation will be limited.")


class DirectionClassifier:
    """
    Class for training and using binary classifiers for directional price moves.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the direction classifier with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.model_config = self.config.get('ml', {}).get('models', {}).get('direction_classifier', {})
        self.model_type = self.model_config.get('type', 'xgboost')
        self.target = self.model_config.get('target', 'direction')
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
                        'direction_classifier': {
                            'type': 'xgboost',
                            'target': 'direction',
                            'features': ['iv_rank', 'iv_hv_spread', 'rsi', 'days_to_earnings'],
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
        Train the direction classifier model.
        
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
            if self.model_type == 'xgboost':
                if not HAS_XGBOOST:
                    logger.error("xgboost is required for XGBoost models")
                    return {'success': False, 'error': 'xgboost not installed'}
                
                # Default hyperparameters
                params = {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
                
                # Update with provided hyperparameters
                if hyperparams:
                    params.update(hyperparams)
                
                # Create and train model
                self.model = xgb.XGBClassifier(**params)
                
            elif self.model_type == 'catboost':
                if not HAS_CATBOOST:
                    logger.error("catboost is required for CatBoost models")
                    return {'success': False, 'error': 'catboost not installed'}
                
                # Default hyperparameters
                params = {
                    'iterations': 500,
                    'learning_rate': 0.1,
                    'depth': 6,
                    'loss_function': 'Logloss',
                    'eval_metric': 'AUC',
                    'random_seed': 42
                }
                
                # Update with provided hyperparameters
                if hyperparams:
                    params.update(hyperparams)
                
                # Create and train model
                self.model = cb.CatBoostClassifier(**params)
                
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return {'success': False, 'error': f'Unsupported model type: {self.model_type}'}
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Get feature importance
            if self.model_type == 'xgboost':
                importance = self.model.feature_importances_
            elif self.model_type == 'catboost':
                importance = self.model.get_feature_importance()
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
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
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
            y_pred_proba = self.model.predict_proba(X_scaled)
            
            # Add predictions to DataFrame
            result_df = df.copy()
            result_df['predicted_direction'] = y_pred
            result_df['predicted_up_probability'] = y_pred_proba[:, 1]
            result_df['predicted_down_probability'] = y_pred_proba[:, 0]
            
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
            if self.model_type == 'xgboost':
                if not HAS_XGBOOST:
                    logger.error("xgboost is required for XGBoost models")
                    return {'success': False, 'error': 'xgboost not installed'}
                
                model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42)
                
                param_grid = {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
                
            elif self.model_type == 'catboost':
                if not HAS_CATBOOST:
                    logger.error("catboost is required for CatBoost models")
                    return {'success': False, 'error': 'catboost not installed'}
                
                model = cb.CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', random_seed=42)
                
                param_grid = {
                    'iterations': [100, 300, 500],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [4, 6, 8]
                }
                
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return {'success': False, 'error': f'Unsupported model type: {self.model_type}'}
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=self.cv_folds,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_scaled, y)
            
            # Get best parameters and score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
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
    classifier = DirectionClassifier()
    
    # Example: Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = {
        'iv_rank': np.random.uniform(0, 1, n_samples),
        'iv_hv_spread': np.random.normal(0, 0.1, n_samples),
        'rsi': np.random.uniform(0, 100, n_samples),
        'days_to_earnings': np.random.randint(1, 60, n_samples)
    }
    
    # Generate synthetic target (direction)
    # Higher IV rank and RSI > 70 tend to predict downward movement
    # Higher IV-HV spread and RSI < 30 tend to predict upward movement
    direction = np.zeros(n_samples)
    
    for i in range(n_samples):
        iv_rank = data['iv_rank'][i]
        iv_hv_spread = data['iv_hv_spread'][i]
        rsi = data['rsi'][i]
        
        # Logic for synthetic data generation
        if (iv_rank > 0.7 and rsi > 70) or (iv_hv_spread < -0.05):
            # Downward movement (0)
            direction[i] = 0
        elif (iv_rank < 0.3 and rsi < 30) or (iv_hv_spread > 0.05):
            # Upward movement (1)
            direction[i] = 1
        else:
            # Random direction with slight upward bias
            direction[i] = np.random.choice([0, 1], p=[0.45, 0.55])
    
    # Create DataFrame
    data['direction'] = direction
    df = pd.DataFrame(data)
    
    # Train model
    print("Training model...")
    result = classifier.train(df)
    
    if result['success']:
        print(f"Training successful. Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"Feature importance: {result['feature_importance']}")
        
        # Save model
        classifier.save_model("../models/direction_classifier.joblib")
        
        # Make predictions on new data
        new_data = df.drop('direction', axis=1).head(5)
        predictions = classifier.predict(new_data)
        
        print("\nPredictions:")
        print(predictions[['iv_rank', 'rsi', 'predicted_direction', 'predicted_up_probability']])
    else:
        print(f"Training failed: {result['error']}")
