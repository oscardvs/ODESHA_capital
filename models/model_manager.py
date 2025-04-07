"""
Model Management Module

This module handles model versioning, tracking, and lifecycle management
for machine learning models in the trading system.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
import json
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

# Try to import MLflow for experiment tracking
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning("mlflow not installed. Advanced model tracking will be limited.")


class ModelManager:
    """
    Class for managing machine learning models, including versioning and tracking.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the model manager with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.ml_config = self.config.get('ml', {})
        self.model_registry_dir = self.ml_config.get('training', {}).get('save_dir', '../models/saved')
        self.use_mlflow = self.ml_config.get('tracking', {}).get('use_mlflow', False) and HAS_MLFLOW
        self.mlflow_tracking_uri = self.ml_config.get('tracking', {}).get('mlflow_uri', 'file:../mlruns')
        
        # Initialize model registry
        self._initialize_registry()
        
        # Initialize MLflow if enabled
        if self.use_mlflow:
            self._initialize_mlflow()
    
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
                    'training': {
                        'schedule': 'weekly',
                        'save_dir': '../models/saved'
                    },
                    'tracking': {
                        'use_mlflow': False,
                        'mlflow_uri': 'file:../mlruns'
                    }
                }
            }
    
    def _initialize_registry(self):
        """Initialize the model registry directory."""
        try:
            # Create registry directory if it doesn't exist
            os.makedirs(self.model_registry_dir, exist_ok=True)
            
            # Create registry metadata file if it doesn't exist
            registry_file = os.path.join(self.model_registry_dir, 'registry.json')
            
            if not os.path.exists(registry_file):
                registry_data = {
                    'models': {},
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                
                with open(registry_file, 'w') as f:
                    json.dump(registry_data, f, indent=2)
            
            logger.info(f"Model registry initialized at {self.model_registry_dir}")
            
        except Exception as e:
            logger.error(f"Error initializing model registry: {e}")
    
    def _initialize_mlflow(self):
        """Initialize MLflow for experiment tracking."""
        if not HAS_MLFLOW:
            logger.error("MLflow is required for experiment tracking")
            return
            
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            # Create experiment if it doesn't exist
            experiment_name = "quant_ml_trader"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            
            logger.info(f"MLflow initialized with tracking URI: {self.mlflow_tracking_uri}")
            
        except Exception as e:
            logger.error(f"Error initializing MLflow: {e}")
    
    def register_model(self, 
                      model_path: str, 
                      model_type: str, 
                      model_name: str, 
                      version: str = None, 
                      metadata: Dict = None) -> Dict[str, Any]:
        """
        Register a model in the model registry.
        
        Args:
            model_path: Path to the saved model file
            model_type: Type of model (e.g., 'direction_classifier', 'iv_predictor')
            model_name: Name of the model
            version: Version of the model (default: timestamp)
            metadata: Additional metadata for the model
            
        Returns:
            Dictionary with registration results
        """
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return {'success': False, 'error': 'Model file not found'}
            
            # Generate version if not provided
            if version is None:
                version = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Load registry metadata
            registry_file = os.path.join(self.model_registry_dir, 'registry.json')
            
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Create model entry if it doesn't exist
            if model_name not in registry_data['models']:
                registry_data['models'][model_name] = {
                    'versions': {},
                    'current_version': None,
                    'created_at': datetime.now().isoformat()
                }
            
            # Create version entry
            model_info = {
                'model_type': model_type,
                'version': version,
                'path': model_path,
                'created_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            # Add version to registry
            registry_data['models'][model_name]['versions'][version] = model_info
            
            # Set as current version if first version
            if registry_data['models'][model_name]['current_version'] is None:
                registry_data['models'][model_name]['current_version'] = version
            
            # Update registry metadata
            registry_data['updated_at'] = datetime.now().isoformat()
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info(f"Model {model_name} version {version} registered successfully")
            
            return {
                'success': True,
                'model_name': model_name,
                'model_type': model_type,
                'version': version,
                'path': model_path
            }
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_model_info(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """
        Get information about a registered model.
        
        Args:
            model_name: Name of the model
            version: Version of the model (default: current version)
            
        Returns:
            Dictionary with model information
        """
        try:
            # Load registry metadata
            registry_file = os.path.join(self.model_registry_dir, 'registry.json')
            
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Check if model exists
            if model_name not in registry_data['models']:
                logger.error(f"Model {model_name} not found in registry")
                return {'success': False, 'error': 'Model not found'}
            
            model_data = registry_data['models'][model_name]
            
            # Get version
            if version is None:
                version = model_data['current_version']
            
            # Check if version exists
            if version not in model_data['versions']:
                logger.error(f"Version {version} of model {model_name} not found")
                return {'success': False, 'error': 'Version not found'}
            
            # Get model info
            model_info = model_data['versions'][version]
            
            return {
                'success': True,
                'model_name': model_name,
                'model_type': model_info['model_type'],
                'version': version,
                'path': model_info['path'],
                'created_at': model_info['created_at'],
                'metadata': model_info['metadata'],
                'is_current': version == model_data['current_version']
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'success': False, 'error': str(e)}
    
    def list_models(self) -> Dict[str, Any]:
        """
        List all registered models.
        
        Returns:
            Dictionary with list of models
        """
        try:
            # Load registry metadata
            registry_file = os.path.join(self.model_registry_dir, 'registry.json')
            
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Get list of models
            models = []
            
            for model_name, model_data in registry_data['models'].items():
                current_version = model_data['current_version']
                
                if current_version is not None and current_version in model_data['versions']:
                    current_version_info = model_data['versions'][current_version]
                    
                    models.append({
                        'model_name': model_name,
                        'model_type': current_version_info['model_type'],
                        'current_version': current_version,
                        'created_at': model_data['created_at'],
                        'version_count': len(model_data['versions'])
                    })
            
            return {
                'success': True,
                'models': models,
                'count': len(models)
            }
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {'success': False, 'error': str(e)}
    
    def list_versions(self, model_name: str) -> Dict[str, Any]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with list of versions
        """
        try:
            # Load registry metadata
            registry_file = os.path.join(self.model_registry_dir, 'registry.json')
            
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Check if model exists
            if model_name not in registry_data['models']:
                logger.error(f"Model {model_name} not found in registry")
                return {'success': False, 'error': 'Model not found'}
            
            model_data = registry_data['models'][model_name]
            current_version = model_data['current_version']
            
            # Get list of versions
            versions = []
            
            for version, version_info in model_data['versions'].items():
                versions.append({
                    'version': version,
                    'model_type': version_info['model_type'],
                    'created_at': version_info['created_at'],
                    'is_current': version == current_version
                })
            
            # Sort versions by creation date (newest first)
            versions.sort(key=lambda x: x['created_at'], reverse=True)
            
            return {
                'success': True,
                'model_name': model_name,
                'versions': versions,
                'count': len(versions),
                'current_version': current_version
            }
            
        except Exception as e:
            logger.error(f"Error listing versions: {e}")
            return {'success': False, 'error': str(e)}
    
    def set_current_version(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Set the current version of a model.
        
        Args:
            model_name: Name of the model
            version: Version to set as current
            
        Returns:
            Dictionary with result
        """
        try:
            # Load registry metadata
            registry_file = os.path.join(self.model_registry_dir, 'registry.json')
            
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Check if model exists
            if model_name not in registry_data['models']:
                logger.error(f"Model {model_name} not found in registry")
                return {'success': False, 'error': 'Model not found'}
            
            model_data = registry_data['models'][model_name]
            
            # Check if version exists
            if version not in model_data['versions']:
                logger.error(f"Version {version} of model {model_name} not found")
                return {'success': False, 'error': 'Version not found'}
            
            # Set current version
            registry_data['models'][model_name]['current_version'] = version
            registry_data['updated_at'] = datetime.now().isoformat()
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info(f"Current version of model {model_name} set to {version}")
            
            return {
                'success': True,
                'model_name': model_name,
                'version': version
            }
            
        except Exception as e:
            logger.error(f"Error setting current version: {e}")
            return {'success': False, 'error': str(e)}
    
    def load_model(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Version of the model (default: current version)
            
        Returns:
            Dictionary with loaded model
        """
        try:
            # Get model info
            model_info = self.get_model_info(model_name, version)
            
            if not model_info['success']:
                return model_info
            
            # Load model
            model_path = model_info['path']
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return {'success': False, 'error': 'Model file not found'}
            
            model_data = joblib.load(model_path)
            
            return {
                'success': True,
                'model_name': model_name,
                'model_type': model_info['model_type'],
                'version': model_info['version'],
                'model_data': model_data
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {'success': False, 'error': str(e)}
    
    def delete_version(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Delete a version of a model.
        
        Args:
            model_name: Name of the model
            version: Version to delete
            
        Returns:
            Dictionary with result
        """
        try:
            # Load registry metadata
            registry_file = os.path.join(self.model_registry_dir, 'registry.json')
            
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Check if model exists
            if model_name not in registry_data['models']:
                logger.error(f"Model {model_name} not found in registry")
                return {'success': False, 'error': 'Model not found'}
            
            model_data = registry_data['models'][model_name]
            
            # Check if version exists
            if version not in model_data['versions']:
                logger.error(f"Version {version} of model {model_name} not found")
                return {'success': False, 'error': 'Version not found'}
            
            # Check if version is current
            if version == model_data['current_version']:
                logger.error(f"Cannot delete current version of model {model_name}")
                return {'success': False, 'error': 'Cannot delete current version'}
            
            # Get model path
            model_path = model_data['versions'][version]['path']
            
            # Delete model file if it exists
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # Remove version from registry
            del registry_data['models'][model_name]['versions'][version]
            registry_data['updated_at'] = datetime.now().isoformat()
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info(f"Version {version} of model {model_name} deleted")
            
            return {
                'success': True,
                'model_name': model_name,
                'version': version
            }
            
        except Exception as e:
            logger.error(f"Error deleting version: {e}")
            return {'success': False, 'error': str(e)}
    
    def log_training_run(self, 
                        model_name: str, 
                        model_type: str, 
                        params: Dict, 
                        metrics: Dict, 
                        artifacts: Dict = None) -> Dict[str, Any]:
        """
        Log a training run using MLflow.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            params: Model parameters
            metrics: Training metrics
            artifacts: Artifacts to log (e.g., plots, feature importance)
            
        Returns:
            Dictionary with result
        """
        if not self.use_mlflow:
            logger.warning("MLflow tracking is disabled or not available")
            return {'success': False, 'error': 'MLflow tracking is disabled'}
            
        try:
            # Start MLflow run
            experiment_name = "quant_ml_trader"
            
            with mlflow.start_run(experiment_name=experiment_name):
                # Log model info
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("model_type", model_type)
                
                # Log parameters
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log artifacts
                if artifacts:
                    for artifact_name, artifact_path in artifacts.items():
                        if os.path.exists(artifact_path):
                            mlflow.log_artifact(artifact_path, artifact_name)
                
                # Get run ID
                run_id = mlflow.active_run().info.run_id
            
            logger.info(f"Training run logged with MLflow, run_id: {run_id}")
            
            return {
                'success': True,
                'model_name': model_name,
                'model_type': model_type,
                'run_id': run_id
            }
            
        except Exception as e:
            logger.error(f"Error logging training run: {e}")
            return {'success': False, 'error': str(e)}
    
    def compare_versions(self, model_name: str, versions: List[str]) -> Dict[str, Any]:
        """
        Compare multiple versions of a model.
        
        Args:
            model_name: Name of the model
            versions: List of versions to compare
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Load registry metadata
            registry_file = os.path.join(self.model_registry_dir, 'registry.json')
            
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Check if model exists
            if model_name not in registry_data['models']:
                logger.error(f"Model {model_name} not found in registry")
                return {'success': False, 'error': 'Model not found'}
            
            model_data = registry_data['models'][model_name]
            
            # Check if versions exist
            for version in versions:
                if version not in model_data['versions']:
                    logger.error(f"Version {version} of model {model_name} not found")
                    return {'success': False, 'error': f'Version {version} not found'}
            
            # Get version info
            version_info = []
            
            for version in versions:
                info = model_data['versions'][version]
                
                # Load model to get metrics
                model_path = info['path']
                
                if os.path.exists(model_path):
                    model_data = joblib.load(model_path)
                    metrics = model_data.get('metrics', {})
                else:
                    metrics = {}
                
                version_info.append({
                    'version': version,
                    'created_at': info['created_at'],
                    'is_current': version == model_data['current_version'],
                    'metrics': metrics,
                    'metadata': info['metadata']
                })
            
            return {
                'success': True,
                'model_name': model_name,
                'versions': version_info
            }
            
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return {'success': False, 'error': str(e)}


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    manager = ModelManager()
    
    # Example: Register a model
    model_path = "../models/direction_classifier.joblib"
    
    if os.path.exists(model_path):
        result = manager.register_model(
            model_path=model_path,
            model_type="direction_classifier",
            model_name="stock_direction",
            metadata={
                'description': 'Binary classifier for stock direction prediction',
                'features': ['iv_rank', 'iv_hv_spread', 'rsi', 'days_to_earnings']
            }
        )
        
        print(f"Model registration result: {result['success']}")
        
        # List models
        models = manager.list_models()
        print(f"Registered models: {models['count']}")
        
        if models['success'] and models['count'] > 0:
            for model in models['models']:
                print(f"- {model['model_name']} (type: {model['model_type']}, version: {model['current_version']})")
    else:
        print(f"Model file not found: {model_path}")
