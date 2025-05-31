
"""
Production Model Prediction Script
Generated on: 2025-05-31T11:08:55.737313
Model Type: Pipeline
"""

import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Union, Any
import warnings
warnings.filterwarnings("ignore")

class ProductionPredictor:
    """Production-ready prediction class with comprehensive validation."""
    
    def __init__(self, model_path: str = "trained_model.joblib", 
                 metadata_path: str = "model_metadata.json"):
        """Initialize the predictor with trained model and metadata."""
        self.model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata['model_info']['feature_names']
        self.target_name = self.metadata['model_info']['target_name']
        self.model_version = self.metadata['model_info']['model_version']
        self.model_hash = self.metadata['model_info']['model_hash']
        
        print(f"Model loaded: {self.metadata['model_info']['model_type']}")
        print(f"Version: {self.model_version}")
        print(f"Features: {len(self.feature_names)} features")
        print(f"Target: {self.target_name}")
    
    def predict(self, input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Make predictions with comprehensive validation and monitoring."""
        
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame(input_data)
        else:
            input_df = input_data.copy()
        
        # Validate input
        validation_result = self._validate_input(input_df)
        if not validation_result['is_valid']:
            return {
                'success': False,
                'error': 'Validation failed',
                'validation_errors': validation_result['errors'],
                'predictions': None
            }
        
        try:
            # Ensure correct feature order and handle missing features
            input_df = self._prepare_features(input_df)
            
            # Make predictions
            predictions = self.model.predict(input_df)
            
            # Calculate prediction confidence (if possible)
            confidence_scores = self._calculate_confidence(input_df, predictions)
            
            return {
                'success': True,
                'predictions': predictions.tolist(),
                'confidence_scores': confidence_scores,
                'model_version': self.model_version,
                'prediction_timestamp': pd.Timestamp.now().isoformat(),
                'input_shape': input_df.shape
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'predictions': None
            }
    
    def _validate_input(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive input validation."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for required features
        missing_features = set(self.feature_names) - set(input_data.columns)
        if missing_features:
            validation_result['errors'].append(f"Missing required features: {missing_features}")
            validation_result['is_valid'] = False
        
        # Check for extra features
        extra_features = set(input_data.columns) - set(self.feature_names)
        if extra_features:
            validation_result['warnings'].append(f"Extra features will be ignored: {extra_features}")
        
        # Check data types and ranges
        for feature in self.feature_names:
            if feature in input_data.columns:
                # Check for all null values
                if input_data[feature].isnull().all():
                    validation_result['errors'].append(f"Feature '{feature}' is entirely null")
                    validation_result['is_valid'] = False
        
        return validation_result
    
    def _prepare_features(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features in the correct order and format."""
        # Select only required features in correct order
        prepared_data = input_data[self.feature_names].copy()
        
        # Handle any missing values (basic imputation)
        for col in prepared_data.columns:
            if prepared_data[col].dtype in ['object', 'category']:
                prepared_data[col] = prepared_data[col].fillna('Unknown')
            else:
                prepared_data[col] = prepared_data[col].fillna(prepared_data[col].median())
        
        return prepared_data
    
    def _calculate_confidence(self, input_data: pd.DataFrame, predictions: np.ndarray) -> List[float]:
        """Calculate prediction confidence scores."""
        # This is a simplified confidence calculation
        # In practice, you might use prediction intervals, ensemble variance, etc.
        
        if hasattr(self.model, 'predict_proba'):
            # For models with probability estimates
            probas = self.model.predict_proba(input_data)
            return np.max(probas, axis=1).tolist()
        else:
            # Simple confidence based on prediction magnitude
            # This is very basic and should be replaced with domain-specific logic
            confidence = np.ones(len(predictions)) * 0.8  # Default confidence
            return confidence.tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': self.metadata['model_info']['model_type'],
            'target_name': self.target_name,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'model_version': self.model_version,
            'model_hash': self.model_hash,
            'created_date': self.metadata['model_info']['created_date']
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform model health check."""
        try:
            # Create dummy data for testing
            dummy_data = {feature: 0.0 for feature in self.feature_names}
            test_result = self.predict(dummy_data)
            
            return {
                'status': 'healthy' if test_result['success'] else 'unhealthy',
                'model_loaded': True,
                'prediction_test': test_result['success'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }

# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = ProductionPredictor()
    
    # Health check
    health = predictor.health_check()
    print(f"Health check: {health}")
    
    # Example prediction (replace with actual feature values)
    sample_data = {feature: 0.0 for feature in predictor.feature_names}
    
    try:
        result = predictor.predict(sample_data)
        print(f"Prediction result: {result}")
    except Exception as e:
        print(f"Prediction error: {e}")
