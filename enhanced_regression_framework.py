# Enhanced Production Regression Analysis Framework
# Modular, scalable, and production-ready ML pipeline

import warnings
warnings.filterwarnings("ignore")

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import joblib
import json
import yaml
from pathlib import Path
import sys
import io
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import hashlib

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats

# Interactive components
from IPython.display import display, HTML, clear_output
try:
    import ipywidgets as widgets
    from ipywidgets import interact, interactive, fixed, interact_manual
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# 1. CONFIGURATION MANAGEMENT
# =============================================================================

@dataclass
class RegressionConfig:
    """Centralized configuration for regression analysis."""
    
    # Data settings
    test_size: float = 0.2
    random_state: int = 42
    max_rows_for_analysis: int = 50000
    
    # Model settings
    cv_folds: int = 5
    models_to_include: List[str] = None
    scoring_metric: str = 'r2'
    
    # Preprocessing settings
    handle_missing: str = 'auto'  # 'auto', 'drop', 'impute'
    encode_categorical: str = 'onehot'  # 'onehot', 'label', 'target'
    scale_features: bool = True
    
    # Optimization settings
    hyperparameter_tuning: bool = True
    tuning_method: str = 'random'  # 'grid', 'random', 'bayesian'
    tuning_iterations: int = 50
    
    # Output settings
    save_plots: bool = True
    create_deployment_package: bool = True
    output_directory: str = "regression_output"
    
    def __post_init__(self):
        if self.models_to_include is None:
            self.models_to_include = [
                'linear', 'ridge', 'lasso', 'elastic_net', 
                'random_forest', 'gradient_boosting', 'svr'
            ]
    
    @classmethod
    def from_file(cls, config_path: str) -> 'RegressionConfig':
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
        
        return cls(**config_dict)
    
    def save_to_file(self, config_path: str):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_dict = asdict(self)
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")

# =============================================================================
# 2. DATA VALIDATION AND UTILITIES
# =============================================================================

class DataValidator:
    """Comprehensive data validation for regression analysis."""
    
    @staticmethod
    def validate_regression_data(data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Validate data for regression analysis."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Basic validation
        if data.empty:
            validation_results['errors'].append("Dataset is empty")
            validation_results['is_valid'] = False
            return validation_results
        
        if target_column not in data.columns:
            validation_results['errors'].append(f"Target column '{target_column}' not found")
            validation_results['is_valid'] = False
            return validation_results
        
        # Target validation
        target_series = data[target_column]
        
        if target_series.isnull().all():
            validation_results['errors'].append("Target column is entirely null")
            validation_results['is_valid'] = False
        
        valid_target_ratio = target_series.count() / len(data)
        if valid_target_ratio < 0.5:
            validation_results['warnings'].append(
                f"Target has only {valid_target_ratio:.1%} valid values"
            )
        
        if not pd.api.types.is_numeric_dtype(target_series):
            try:
                pd.to_numeric(target_series, errors='raise')
            except (ValueError, TypeError):
                validation_results['errors'].append(
                    f"Target column '{target_column}' is not numeric and cannot be converted"
                )
                validation_results['is_valid'] = False
        
        # Feature validation
        feature_columns = [col for col in data.columns if col != target_column]
        
        if len(feature_columns) == 0:
            validation_results['errors'].append("No feature columns found")
            validation_results['is_valid'] = False
        
        # Check for sufficient data
        if len(data) < 50:
            validation_results['warnings'].append(
                f"Dataset is very small ({len(data)} rows). Results may be unreliable."
            )
        
        # Check for high cardinality categorical columns
        for col in feature_columns:
            if data[col].dtype == 'object':
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio > 0.8:
                    validation_results['warnings'].append(
                        f"Column '{col}' has very high cardinality ({unique_ratio:.1%} unique values)"
                    )
        
        return validation_results
    
    @staticmethod
    def get_data_summary(data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary."""
        # Convert dtypes to strings to avoid JSON serialization issues
        dtypes_dict = {}
        for dtype, count in data.dtypes.value_counts().items():
            dtypes_dict[str(dtype)] = int(count)
        
        summary = {
            'shape': data.shape,
            'memory_usage_mb': float(data.memory_usage(deep=True).sum() / 1024**2),
            'dtypes': dtypes_dict,
            'missing_values': {k: int(v) for k, v in data.isnull().sum().to_dict().items()},
            'duplicate_rows': int(data.duplicated().sum()),
            'numerical_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': data.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        # Add statistical summary for numerical columns
        if summary['numerical_columns']:
            # Convert numpy types to Python native types
            stats_dict = data[summary['numerical_columns']].describe().to_dict()
            summary['numerical_stats'] = {
                col: {stat: float(val) if isinstance(val, (np.floating, np.integer)) else val 
                      for stat, val in col_stats.items()}
                for col, col_stats in stats_dict.items()
            }
        
        return summary

class DataProcessor:
    """Handle data preprocessing with various strategies."""
    
    def __init__(self, config: RegressionConfig):
        self.config = config
        self.preprocessing_steps = []
        
    def optimize_for_large_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle large datasets efficiently."""
        if len(data) > self.config.max_rows_for_analysis:
            logger.warning(
                f"Dataset large ({len(data):,} rows). Sampling {self.config.max_rows_for_analysis:,} for analysis."
            )
            sampled_data = data.sample(n=self.config.max_rows_for_analysis, random_state=self.config.random_state)
            self.preprocessing_steps.append(f"Sampled {self.config.max_rows_for_analysis:,} rows from {len(data):,}")
            return sampled_data
        return data
    
    def handle_missing_values(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Handle missing values based on configuration."""
        data = data.copy()
        
        if self.config.handle_missing == 'drop':
            data = data.dropna()
            self.preprocessing_steps.append("Dropped rows with missing values")
        
        elif self.config.handle_missing == 'impute' or self.config.handle_missing == 'auto':
            for col in data.columns:
                if col == target_column:
                    continue
                    
                if data[col].isnull().sum() > 0:
                    if data[col].dtype in ['object', 'category']:
                        mode_value = data[col].mode()
                        fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                        data[col] = data[col].fillna(fill_value)
                        self.preprocessing_steps.append(f"Filled missing values in '{col}' with mode")
                    else:
                        data[col] = data[col].fillna(data[col].median())
                        self.preprocessing_steps.append(f"Filled missing values in '{col}' with median")
        
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Encode categorical features based on configuration."""
        data = data.copy()
        feature_columns = [col for col in data.columns if col != target_column]
        categorical_cols = data[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            return data
        
        if self.config.encode_categorical == 'onehot' or self.config.encode_categorical == 'one_hot_encoding':
            data = pd.get_dummies(data, columns=categorical_cols, prefix_sep='_', drop_first=True)
            self.preprocessing_steps.append(f"One-hot encoded {len(categorical_cols)} categorical columns")
        
        elif self.config.encode_categorical == 'label' or self.config.encode_categorical == 'label_encoding':
            for col in categorical_cols:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.preprocessing_steps.append(f"Label encoded '{col}'")
        
        return data
    
    def get_preprocessing_summary(self) -> List[str]:
        """Get summary of preprocessing steps performed."""
        return self.preprocessing_steps.copy()
    
    def transform_features(self, data: pd.DataFrame, target_column: Optional[str] = None, 
                          is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Apply the complete preprocessing pipeline to transform features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data to transform
        target_column : str, optional
            Name of the target column (None for prediction data)
        is_training : bool
            Whether this is training data or prediction data
        
        Returns:
        --------
        Tuple[pd.DataFrame, Optional[pd.Series]]
            Processed features and target (if provided)
        """
        data = data.copy()
        
        # Handle missing values
        if target_column:
            data = self.handle_missing_values(data, target_column)
        else:
            # For prediction data, apply the same missing value strategy
            data = self.handle_missing_values(data, '__no_target__')
        
        # Encode categorical features  
        if target_column:
            data = self.encode_categorical_features(data, target_column)
        else:
            # For prediction data, apply the same encoding strategy
            data = self.encode_categorical_features(data, '__no_target__')
        
        # Separate features and target
        if target_column and target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            return X, y
        else:
            # Return all columns as features when no target specified
            return data, None

# =============================================================================
# 3. MODEL TRAINING AND EVALUATION
# =============================================================================

class ModelTrainer:
    """Handle model training with various algorithms."""
    
    def __init__(self, config: RegressionConfig):
        self.config = config
        self.models = {}
        self.trained_models = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize regression models based on configuration."""
        available_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(random_state=self.config.random_state),
            'lasso': Lasso(random_state=self.config.random_state),
            'elastic_net': ElasticNet(random_state=self.config.random_state),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.config.random_state
            ),
            'svr': SVR()
        }
        
        self.models = {
            name: model for name, model in available_models.items()
            if name in self.config.models_to_include
        }
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        return self.models
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train a single model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available")
        
        model = self.models[model_name]
        
        # Handle scaling for models that need it
        if model_name == 'svr' or self.config.scale_features:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            pipeline.fit(X_train, y_train)
            self.trained_models[model_name] = pipeline
            return pipeline
        else:
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model
            return model

class ModelEvaluator:
    """Evaluate model performance with comprehensive metrics."""
    
    def __init__(self, config: RegressionConfig):
        self.config = config
    
    def evaluate_model(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, y_test: pd.Series, model_name: str) -> Dict[str, float]:
        """Evaluate a single model with cross-validation and test metrics."""
        
        # Cross-validation on training set
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self.config.cv_folds,
            scoring=self.config.scoring_metric,
            n_jobs=-1
        )
        
        # Test set predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_r2': r2_score(y_test, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        logger.info(f"{model_name}: R² = {metrics['test_r2']:.3f}, RMSE = {metrics['test_rmse']:.3f}")
        
        return metrics, cv_scores
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare model results and return sorted DataFrame."""
        results_list = []
        
        for model_name, metrics in results.items():
            result_row = {'Model': model_name}
            result_row.update(metrics)
            results_list.append(result_row)
        
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('test_r2', ascending=False)
        
        return results_df

# =============================================================================
# 4. HYPERPARAMETER OPTIMIZATION
# =============================================================================

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization."""
    
    def __init__(self, config: RegressionConfig):
        self.config = config
        self.param_grids = self._get_parameter_grids()
    
    def _get_parameter_grids(self) -> Dict[str, Dict[str, List]]:
        """Define parameter grids for each model."""
        return {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
            },
            'lasso': {
                'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
            },
            'elastic_net': {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'svr': {
                'model__C': [0.1, 1, 10, 100],
                'model__epsilon': [0.01, 0.1, 0.2],
                'model__kernel': ['linear', 'rbf']
            }
        }
    
    def optimize_model(self, model: Any, model_name: str, X_train: pd.DataFrame, 
                      y_train: pd.Series) -> Tuple[Any, Dict[str, Any]]:
        """Optimize hyperparameters for a specific model."""
        
        if model_name not in self.param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return model, {}
        
        param_grid = self.param_grids[model_name]
        
        if self.config.tuning_method == 'grid':
            search = GridSearchCV(
                model, param_grid,
                cv=self.config.cv_folds,
                scoring=self.config.scoring_metric,
                n_jobs=-1
            )
        else:  # random search
            search = RandomizedSearchCV(
                model, param_grid,
                cv=self.config.cv_folds,
                scoring=self.config.scoring_metric,
                n_iter=self.config.tuning_iterations,
                n_jobs=-1,
                random_state=self.config.random_state
            )
        
        search.fit(X_train, y_train)
        
        optimization_results = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': search.cv_results_
        }
        
        logger.info(f"Optimized {model_name}: Best CV score = {search.best_score_:.3f}")
        
        return search.best_estimator_, optimization_results

# =============================================================================
# 5. MODEL INTERPRETATION AND MONITORING
# =============================================================================

class ModelInterpreter:
    """Advanced model interpretation and analysis."""
    
    def __init__(self, config: RegressionConfig):
        self.config = config
    
    def analyze_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Analyze feature importance from the model."""
        importance_data = None
        method = None
        
        # Extract base model from pipeline if necessary
        base_model = model
        if hasattr(model, 'named_steps'):
            base_model = model.named_steps.get('model', model)
        
        # Get feature importance based on model type
        if hasattr(base_model, 'feature_importances_'):
            importance_data = base_model.feature_importances_
            method = "Tree-based Feature Importance"
        elif hasattr(base_model, 'coef_'):
            importance_data = np.abs(base_model.coef_)
            method = "Coefficient Magnitude"
        
        if importance_data is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_data,
                'method': method
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return pd.DataFrame()
    
    def analyze_residuals(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Comprehensive residual analysis."""
        residuals = y_true - y_pred
        
        analysis = {
            'residual_stats': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'skewness': residuals.skew(),
                'kurtosis': residuals.kurtosis()
            },
            'normality_test': None,
            'heteroscedasticity_detected': False
        }
        
        # Normality test (if sample size appropriate)
        if len(residuals) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            analysis['normality_test'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        
        # Simple heteroscedasticity check
        # Divide residuals into groups and check variance differences
        sorted_indices = np.argsort(y_pred)
        n_groups = 3
        group_size = len(residuals) // n_groups
        
        group_vars = []
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < n_groups - 1 else len(residuals)
            group_residuals = residuals.iloc[sorted_indices[start_idx:end_idx]]
            group_vars.append(group_residuals.var())
        
        # Check if variance increases significantly
        if max(group_vars) / min(group_vars) > 3:
            analysis['heteroscedasticity_detected'] = True
        
        return analysis
    
    def performance_breakdown(self, y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
        """Performance breakdown by prediction ranges."""
        
        # Create prediction ranges using quantiles
        pred_ranges = pd.qcut(y_pred, q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        
        breakdown_results = []
        
        for range_label in pred_ranges.categories:
            mask = pred_ranges == range_label
            if mask.sum() > 0:
                range_y_true = y_true[mask]
                range_y_pred = y_pred[mask]
                
                r2 = r2_score(range_y_true, range_y_pred)
                rmse = np.sqrt(mean_squared_error(range_y_true, range_y_pred))
                mae = mean_absolute_error(range_y_true, range_y_pred)
                
                breakdown_results.append({
                    'range': range_label,
                    'count': mask.sum(),
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                })
        
        return pd.DataFrame(breakdown_results)

class ModelMonitor:
    """Monitor model performance and detect data drift."""
    
    def __init__(self, config: RegressionConfig):
        self.config = config
    
    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         new_data: pd.DataFrame, threshold: float = 0.05) -> Dict[str, Any]:
        """Detect data drift using statistical tests."""
        
        drift_results = {
            'drift_detected': False,
            'drift_features': [],
            'test_results': {}
        }
        
        common_features = set(reference_data.columns) & set(new_data.columns)
        
        for feature in common_features:
            ref_values = reference_data[feature].dropna()
            new_values = new_data[feature].dropna()
            
            if ref_values.dtype in ['object', 'category']:
                # Chi-square test for categorical variables
                ref_counts = ref_values.value_counts()
                new_counts = new_values.value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(new_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                new_aligned = [new_counts.get(cat, 0) for cat in all_categories]
                
                try:
                    chi2_stat, p_value = stats.chisquare(new_aligned, ref_aligned)
                    drift_results['test_results'][feature] = {
                        'test': 'chi_square',
                        'statistic': chi2_stat,
                        'p_value': p_value
                    }
                    
                    if p_value < threshold:
                        drift_results['drift_features'].append(feature)
                except ValueError:
                    continue
            
            else:
                # Kolmogorov-Smirnov test for numerical variables
                ks_stat, p_value = stats.ks_2samp(ref_values, new_values)
                drift_results['test_results'][feature] = {
                    'test': 'kolmogorov_smirnov',
                    'statistic': ks_stat,
                    'p_value': p_value
                }
                
                if p_value < threshold:
                    drift_results['drift_features'].append(feature)
        
        drift_results['drift_detected'] = len(drift_results['drift_features']) > 0
        
        return drift_results
    
    def performance_degradation_alert(self, current_metrics: Dict[str, float],
                                    baseline_metrics: Dict[str, float],
                                    threshold: float = 0.1) -> Dict[str, Any]:
        """Alert when performance drops significantly."""
        
        alert_results = {
            'degradation_detected': False,
            'degraded_metrics': [],
            'severity': 'none'
        }
        
        for metric_name in ['test_r2', 'test_rmse', 'test_mae']:
            if metric_name in current_metrics and metric_name in baseline_metrics:
                current_val = current_metrics[metric_name]
                baseline_val = baseline_metrics[metric_name]
                
                if metric_name == 'test_r2':
                    # For R², lower is worse
                    degradation = (baseline_val - current_val) / baseline_val
                else:
                    # For RMSE and MAE, higher is worse
                    degradation = (current_val - baseline_val) / baseline_val
                
                if degradation > threshold:
                    alert_results['degraded_metrics'].append({
                        'metric': metric_name,
                        'current': current_val,
                        'baseline': baseline_val,
                        'degradation_pct': degradation * 100
                    })
        
        if alert_results['degraded_metrics']:
            alert_results['degradation_detected'] = True
            max_degradation = max(m['degradation_pct'] for m in alert_results['degraded_metrics'])
            
            if max_degradation > 25:
                alert_results['severity'] = 'high'
            elif max_degradation > 15:
                alert_results['severity'] = 'medium'
            else:
                alert_results['severity'] = 'low'
        
        return alert_results

# =============================================================================
# 6. ENHANCED VISUALIZATION
# =============================================================================

class EnhancedVisualizer:
    """Create interactive and comprehensive visualizations."""
    
    def __init__(self, config: RegressionConfig):
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_interactive_model_comparison(self, results_df: pd.DataFrame) -> go.Figure:
        """Create interactive model comparison plot."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² Score Comparison', 'RMSE Comparison',
                          'Cross-Validation Performance', 'Performance vs Complexity'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        models = results_df['Model']
        
        # R² Score Comparison
        fig.add_trace(
            go.Bar(x=results_df['test_r2'], y=models, orientation='h',
                  name='R² Score', marker_color='lightblue'),
            row=1, col=1
        )
        
        # RMSE Comparison
        fig.add_trace(
            go.Bar(x=results_df['test_rmse'], y=models, orientation='h',
                  name='RMSE', marker_color='lightcoral'),
            row=1, col=2
        )
        
        # Cross-validation performance
        fig.add_trace(
            go.Scatter(x=models, y=results_df['cv_mean'],
                      error_y=dict(type='data', array=results_df['cv_std']),
                      mode='markers+lines', name='CV Performance',
                      marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Performance vs Complexity (simplified)
        complexity_map = {
            'Linear Regression': 1, 'Ridge Regression': 2, 'Lasso Regression': 2,
            'Elastic Net': 3, 'Support Vector Regression': 5,
            'Random Forest': 8, 'Gradient Boosting': 9
        }
        
        complexity_scores = [complexity_map.get(model, 5) for model in models]
        
        fig.add_trace(
            go.Scatter(x=complexity_scores, y=results_df['test_r2'],
                      mode='markers+text', text=models,
                      textposition="top center", name='Performance vs Complexity',
                      marker=dict(size=10, color=results_df['test_rmse'],
                                colorscale='Viridis', showscale=True)),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="Interactive Model Comparison Dashboard")
        
        if self.config.save_plots:
            fig.write_html(self.output_dir / "model_comparison_interactive.html")
        
        return fig
    
    def create_interactive_feature_importance(self, importance_df: pd.DataFrame) -> go.Figure:
        """Create interactive feature importance plot."""
        
        # Take top 20 features for readability
        top_features = importance_df.head(20)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                y=top_features['feature'],
                x=top_features['importance'],
                orientation='h',
                marker=dict(
                    color=top_features['importance'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=top_features['importance'].round(4),
                textposition='inside'
            )
        )
        
        fig.update_layout(
            title=f"Interactive Feature Importance ({importance_df['method'].iloc[0]})",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        if self.config.save_plots:
            fig.write_html(self.output_dir / "feature_importance_interactive.html")
        
        return fig
    
    def create_residual_analysis_plots(self, y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
        """Create comprehensive residual analysis plots."""
        
        residuals = y_true - y_pred
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals vs Predicted', 'Residual Distribution',
                          'Q-Q Plot', 'Residuals vs Fitted (Scaled)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers',
                      name='Residuals', marker=dict(opacity=0.6)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[y_pred.min(), y_pred.max()], y=[0, 0],
                      mode='lines', name='Zero Line', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Residual Distribution
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=30, name='Residual Distribution',
                        marker_color='lightblue', opacity=0.7),
            row=1, col=2
        )
        
        # Q-Q Plot data
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_residuals,
                      mode='markers', name='Q-Q Plot'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                      mode='lines', name='Perfect Normal', line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # Standardized residuals
        std_residuals = residuals / residuals.std()
        fig.add_trace(
            go.Scatter(x=y_pred, y=std_residuals, mode='markers',
                      name='Standardized Residuals', marker=dict(opacity=0.6)),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="Comprehensive Residual Analysis")
        
        if self.config.save_plots:
            fig.write_html(self.output_dir / "residual_analysis.html")
        
        return fig

# =============================================================================
# 7. DEPLOYMENT AND PACKAGING
# =============================================================================

class DeploymentManager:
    """Enhanced deployment preparation and packaging."""
    
    def __init__(self, config: RegressionConfig):
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_model_package(self, model: Any, feature_names: List[str],
                           target_name: str, model_metadata: Dict[str, Any],
                           preprocessing_steps: List[str] = None) -> Path:
        """Create comprehensive model deployment package."""
        
        # Create package directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_dir = self.output_dir / f"model_package_{timestamp}"
        package_dir.mkdir(exist_ok=True)
        
        # Save the trained model
        model_path = package_dir / "trained_model.joblib"
        joblib.dump(model, model_path)
        
        # Create comprehensive metadata
        metadata = {
            'model_info': {
                'model_type': type(model).__name__,
                'target_name': target_name,
                'feature_names': feature_names,
                'n_features': len(feature_names),
                'created_date': datetime.now().isoformat(),
                'model_version': '1.0.0'
            },
            'training_info': model_metadata,
            'preprocessing_steps': preprocessing_steps or [],
            'configuration': asdict(self.config),
            'model_hash': self._calculate_model_hash(model)
        }
        
        # Save metadata
        metadata_path = package_dir / "model_metadata.json"
        
        # Custom JSON encoder to handle pandas types
        def json_serializable(obj):
            """Convert pandas and numpy types to JSON serializable formats."""
            # Handle numpy dtypes (like Float64DType)
            if hasattr(obj, '__class__') and 'dtype' in obj.__class__.__name__.lower():
                return str(obj)
            # Handle values with dtype attribute
            elif hasattr(obj, 'dtype'):
                if 'int' in str(obj.dtype):
                    return int(obj)
                elif 'float' in str(obj.dtype):
                    return float(obj)
                else:
                    return str(obj)
            # Handle numpy scalars
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            # Handle numpy arrays
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle pandas specific types
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, 'to_list'):
                return obj.to_list()
            # Handle other objects
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return str(obj)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=json_serializable)
        
        # Create prediction script
        self._create_production_prediction_script(package_dir, metadata)
        
        # Create requirements file
        self._create_requirements_file(package_dir)
        
        # Create deployment guide
        self._create_deployment_guide(package_dir, metadata)
        
        # Create API example
        self._create_api_example(package_dir, metadata)
        
        # Create monitoring script
        self._create_monitoring_script(package_dir, metadata)
        
        logger.info(f"Model package created: {package_dir}")
        return package_dir
    
    def _calculate_model_hash(self, model: Any) -> str:
        """Calculate hash of the model for versioning."""
        import pickle
        model_bytes = pickle.dumps(model)
        return hashlib.md5(model_bytes).hexdigest()
    
    def _create_production_prediction_script(self, package_dir: Path, metadata: Dict[str, Any]):
        """Create production-ready prediction script with validation."""
        
        script_content = f'''
"""
Production Model Prediction Script
Generated on: {datetime.now().isoformat()}
Model Type: {metadata['model_info']['model_type']}
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
        
        print(f"Model loaded: {{self.metadata['model_info']['model_type']}}")
        print(f"Version: {{self.model_version}}")
        print(f"Features: {{len(self.feature_names)}} features")
        print(f"Target: {{self.target_name}}")
    
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
            return {{
                'success': False,
                'error': 'Validation failed',
                'validation_errors': validation_result['errors'],
                'predictions': None
            }}
        
        try:
            # Ensure correct feature order and handle missing features
            input_df = self._prepare_features(input_df)
            
            # Make predictions
            predictions = self.model.predict(input_df)
            
            # Calculate prediction confidence (if possible)
            confidence_scores = self._calculate_confidence(input_df, predictions)
            
            return {{
                'success': True,
                'predictions': predictions.tolist(),
                'confidence_scores': confidence_scores,
                'model_version': self.model_version,
                'prediction_timestamp': pd.Timestamp.now().isoformat(),
                'input_shape': input_df.shape
            }}
            
        except Exception as e:
            return {{
                'success': False,
                'error': str(e),
                'predictions': None
            }}
    
    def _validate_input(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive input validation."""
        validation_result = {{
            'is_valid': True,
            'errors': [],
            'warnings': []
        }}
        
        # Check for required features
        missing_features = set(self.feature_names) - set(input_data.columns)
        if missing_features:
            validation_result['errors'].append(f"Missing required features: {{missing_features}}")
            validation_result['is_valid'] = False
        
        # Check for extra features
        extra_features = set(input_data.columns) - set(self.feature_names)
        if extra_features:
            validation_result['warnings'].append(f"Extra features will be ignored: {{extra_features}}")
        
        # Check data types and ranges
        for feature in self.feature_names:
            if feature in input_data.columns:
                # Check for all null values
                if input_data[feature].isnull().all():
                    validation_result['errors'].append(f"Feature '{{feature}}' is entirely null")
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
        return {{
            'model_type': self.metadata['model_info']['model_type'],
            'target_name': self.target_name,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'model_version': self.model_version,
            'model_hash': self.model_hash,
            'created_date': self.metadata['model_info']['created_date']
        }}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform model health check."""
        try:
            # Create dummy data for testing
            dummy_data = {{feature: 0.0 for feature in self.feature_names}}
            test_result = self.predict(dummy_data)
            
            return {{
                'status': 'healthy' if test_result['success'] else 'unhealthy',
                'model_loaded': True,
                'prediction_test': test_result['success'],
                'timestamp': pd.Timestamp.now().isoformat()
            }}
        except Exception as e:
            return {{
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }}

# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = ProductionPredictor()
    
    # Health check
    health = predictor.health_check()
    print(f"Health check: {{health}}")
    
    # Example prediction (replace with actual feature values)
    sample_data = {{feature: 0.0 for feature in predictor.feature_names}}
    
    try:
        result = predictor.predict(sample_data)
        print(f"Prediction result: {{result}}")
    except Exception as e:
        print(f"Prediction error: {{e}}")
'''
        
        script_path = package_dir / "predict.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
    
    def _create_api_example(self, package_dir: Path, metadata: Dict[str, Any]):
        """Create FastAPI example for model serving."""
        
        api_content = f'''
"""
FastAPI Model Serving Example
Generated on: {datetime.now().isoformat()}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
from predict import ProductionPredictor
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Regression Model API",
    description="Production API for regression model predictions",
    version="{metadata['model_info']['model_version']}"
)

# Initialize model
predictor = ProductionPredictor()

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")
    
class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    predictions: Optional[List[float]] = None
    confidence_scores: Optional[List[float]] = None
    error: Optional[str] = None
    model_version: str
    prediction_timestamp: str

class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    model_loaded: bool
    prediction_test: bool
    timestamp: str

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {{
        "message": "Regression Model API",
        "model_type": "{metadata['model_info']['model_type']}",
        "version": "{metadata['model_info']['model_version']}",
        "target": "{metadata['model_info']['target_name']}"
    }}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    health_result = predictor.health_check()
    return HealthResponse(**health_result)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions on input data."""
    try:
        result = predictor.predict(request.data)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_model=Dict[str, Any])
async def get_model_info():
    """Get model information."""
    return predictor.get_model_info()

@app.get("/features", response_model=List[str])
async def get_features():
    """Get required feature names."""
    return predictor.feature_names

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        api_path = package_dir / "api_server.py"
        with open(api_path, 'w') as f:
            f.write(api_content)
    
    def _create_monitoring_script(self, package_dir: Path, metadata: Dict[str, Any]):
        """Create monitoring script for production deployment."""
        
        monitoring_content = f'''
"""
Model Monitoring Script
Generated on: {datetime.now().isoformat()}
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from predict import ProductionPredictor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model performance and data quality in production."""
    
    def __init__(self, predictor: ProductionPredictor, baseline_metrics: Dict[str, float] = None):
        self.predictor = predictor
        self.baseline_metrics = baseline_metrics or {{}}
        self.alerts = []
    
    def monitor_prediction_batch(self, predictions: List[Dict[str, Any]], 
                               actual_values: List[float] = None) -> Dict[str, Any]:
        """Monitor a batch of predictions."""
        
        monitoring_results = {{
            'timestamp': datetime.now().isoformat(),
            'batch_size': len(predictions),
            'successful_predictions': 0,
            'failed_predictions': 0,
            'data_quality_issues': [],
            'performance_metrics': {{}},
            'alerts': []
        }}
        
        # Count successful vs failed predictions
        for pred in predictions:
            if pred.get('success', False):
                monitoring_results['successful_predictions'] += 1
            else:
                monitoring_results['failed_predictions'] += 1
        
        # Calculate performance metrics if actual values provided
        if actual_values:
            pred_values = [p.get('predictions', [0])[0] for p in predictions if p.get('success')]
            if len(pred_values) == len(actual_values):
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                
                monitoring_results['performance_metrics'] = {{
                    'r2_score': r2_score(actual_values, pred_values),
                    'rmse': np.sqrt(mean_squared_error(actual_values, pred_values)),
                    'mae': mean_absolute_error(actual_values, pred_values)
                }}
                
                # Check for performance degradation
                if self.baseline_metrics:
                    self._check_performance_degradation(
                        monitoring_results['performance_metrics'], 
                        monitoring_results
                    )
        
        # Log results
        logger.info(f"Batch monitoring complete: {{monitoring_results['successful_predictions']}}/{{len(predictions)}} successful")
        
        return monitoring_results
    
    def _check_performance_degradation(self, current_metrics: Dict[str, float], 
                                     monitoring_results: Dict[str, Any]):
        """Check for performance degradation."""
        
        for metric, current_value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                
                if metric == 'r2_score':
                    # For R², lower is worse
                    degradation = (baseline_value - current_value) / baseline_value
                else:
                    # For RMSE and MAE, higher is worse  
                    degradation = (current_value - baseline_value) / baseline_value
                
                if degradation > 0.1:  # 10% degradation threshold
                    alert = {{
                        'type': 'performance_degradation',
                        'metric': metric,
                        'current_value': current_value,
                        'baseline_value': baseline_value,
                        'degradation_pct': degradation * 100,
                        'severity': 'high' if degradation > 0.25 else 'medium'
                    }}
                    monitoring_results['alerts'].append(alert)
                    self.alerts.append(alert)
    
    def check_data_drift(self, recent_data: pd.DataFrame, 
                        reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data drift in recent predictions."""
        
        # This is a simplified drift detection
        # In practice, you'd want more sophisticated statistical tests
        
        drift_results = {{
            'drift_detected': False,
            'drift_features': [],
            'timestamp': datetime.now().isoformat()
        }}
        
        common_features = set(recent_data.columns) & set(reference_data.columns)
        
        for feature in common_features:
            recent_mean = recent_data[feature].mean()
            reference_mean = reference_data[feature].mean()
            reference_std = reference_data[feature].std()
            
            # Simple drift detection: mean shift > 2 standard deviations
            if abs(recent_mean - reference_mean) > 2 * reference_std:
                drift_results['drift_features'].append({{
                    'feature': feature,
                    'recent_mean': recent_mean,
                    'reference_mean': reference_mean,
                    'drift_magnitude': abs(recent_mean - reference_mean) / reference_std
                }})
        
        if drift_results['drift_features']:
            drift_results['drift_detected'] = True
            logger.warning(f"Data drift detected in {{len(drift_results['drift_features'])}} features")
        
        return drift_results
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        
        report = {{
            'report_timestamp': datetime.now().isoformat(),
            'model_info': self.predictor.get_model_info(),
            'health_status': self.predictor.health_check(),
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'alert_summary': {{
                'total_alerts': len(self.alerts),
                'high_severity': len([a for a in self.alerts if a.get('severity') == 'high']),
                'medium_severity': len([a for a in self.alerts if a.get('severity') == 'medium'])
            }}
        }}
        
        return report

# Example usage
if __name__ == "__main__":
    # Initialize predictor and monitor
    predictor = ProductionPredictor()
    
    # Example baseline metrics (replace with actual values)
    baseline_metrics = {{
        'r2_score': 0.85,
        'rmse': 1000.0,
        'mae': 750.0
    }}
    
    monitor = ModelMonitor(predictor, baseline_metrics)
    
    # Generate monitoring report
    report = monitor.generate_monitoring_report()
    print(json.dumps(report, indent=2, default=str))
'''
        
        monitoring_path = package_dir / "monitor.py"
        with open(monitoring_path, 'w') as f:
            f.write(monitoring_content)
    
    def _create_requirements_file(self, package_dir: Path):
        """Create comprehensive requirements file."""
        
        requirements = [
            "pandas>=1.5.0",
            "numpy>=1.21.0", 
            "scikit-learn>=1.2.0",
            "joblib>=1.2.0",
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8.0",
            "pyyaml>=6.0",
            "plotly>=5.0.0",
            "scipy>=1.9.0"
        ]
        
        req_path = package_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
    
    def _create_deployment_guide(self, package_dir: Path, metadata: Dict[str, Any]):
        """Create comprehensive deployment guide."""
        
        guide_content = f'''
# Model Deployment Guide

## Package Information
- **Model Type**: {metadata['model_info']['model_type']}
- **Target Variable**: {metadata['model_info']['target_name']}
- **Features**: {len(metadata['model_info']['feature_names'])} features
- **Created**: {metadata['model_info']['created_date']}
- **Version**: {metadata['model_info']['model_version']}

## Package Contents
- `trained_model.joblib`: Serialized trained model
- `model_metadata.json`: Comprehensive model metadata
- `predict.py`: Production prediction script with validation
- `api_server.py`: FastAPI server example
- `monitor.py`: Production monitoring script
- `requirements.txt`: Python dependencies
- `deployment_guide.md`: This guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Basic Usage
```python
from predict import ProductionPredictor

# Load the model
predictor = ProductionPredictor()

# Make a prediction
result = predictor.predict({{
{', '.join([f"    '{feature}': 0.0" for feature in metadata['model_info']['feature_names'][:5]])}
    # ... add all required features
}})

print(result)
```

### 3. API Server
```bash
# Start the API server
python api_server.py

# Test the API
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"data": [{{"feature1": 1.0, "feature2": 2.0}}]}}'
```

### 4. Monitoring
```python
from monitor import ModelMonitor
from predict import ProductionPredictor

predictor = ProductionPredictor()
monitor = ModelMonitor(predictor)

# Monitor predictions
report = monitor.generate_monitoring_report()
print(report)
```

## Production Deployment Options

### Option 1: Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "api_server.py"]
```

Build and run:
```bash
docker build -t regression-model .
docker run -p 8000:8000 regression-model
```

### Option 2: Cloud Function/Lambda
The `predict.py` script can be easily adapted for serverless deployment:
- AWS Lambda
- Google Cloud Functions  
- Azure Functions

### Option 3: Kubernetes Deployment
Create deployment manifests for scalable production deployment.

## Monitoring and Maintenance

### Performance Monitoring
- Track prediction accuracy on new data
- Monitor for data drift in input features
- Set up alerts for performance degradation

### Model Updates
- Retrain periodically with new data
- Version control your models
- A/B test new models before full deployment

### Data Quality Monitoring
- Validate input data format and ranges
- Check for missing or anomalous values
- Monitor feature distributions

## Security Considerations

### Input Validation
- Validate all input data
- Sanitize inputs to prevent injection attacks
- Implement rate limiting

### API Security
- Use HTTPS in production
- Implement authentication/authorization
- Log all prediction requests

### Model Security
- Protect model files from unauthorized access
- Monitor for adversarial inputs
- Regular security audits

## Troubleshooting

### Common Issues
1. **Missing Features**: Ensure all required features are present in input data
2. **Data Type Mismatches**: Check feature data types match training data
3. **Performance Degradation**: Monitor model metrics and retrain if needed

### Logs and Debugging
- Check application logs for detailed error messages
- Use the health check endpoint to verify model status
- Monitor prediction confidence scores

## Support and Maintenance

### Model Information
- Model Hash: `{metadata['model_info']['model_hash']}`
- Training Configuration: See `model_metadata.json`
- Feature Requirements: See `get_features()` endpoint

### Contact Information
For questions or issues, refer to the model training documentation or contact the ML team.
'''
        
        guide_path = package_dir / "deployment_guide.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)

# =============================================================================
# 8. MAIN REGRESSION WORKFLOW ORCHESTRATOR  
# =============================================================================

class EnhancedRegressionWorkflow:
    """Main orchestrator for the enhanced regression analysis workflow."""
    
    def __init__(self, config: RegressionConfig = None):
        self.config = config or RegressionConfig()
        self.data = None
        self.target_column = None
        self.feature_columns = None
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.model_evaluator = ModelEvaluator(self.config)
        self.hyperparameter_optimizer = HyperparameterOptimizer(self.config)
        self.model_interpreter = ModelInterpreter(self.config)
        self.model_monitor = ModelMonitor(self.config)
        self.visualizer = EnhancedVisualizer(self.config)
        self.deployment_manager = DeploymentManager(self.config)
        
        # Results storage
        self.results = {}
        self.best_model = None
        self.feature_importance = None
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up comprehensive logging."""
        log_dir = Path(self.config.output_directory) / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"regression_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    def load_and_validate_data(self, data: pd.DataFrame, target_column: str) -> bool:
        """Load and validate data for regression analysis."""
        
        logger.info("Starting data loading and validation")
        
        # Validate data
        validation_results = DataValidator.validate_regression_data(data, target_column)
        
        if not validation_results['is_valid']:
            logger.error(f"Data validation failed: {validation_results['errors']}")
            for error in validation_results['errors']:
                print(f"❌ {error}")
            return False
        
        # Display warnings
        for warning in validation_results['warnings']:
            logger.warning(warning)
            print(f"⚠️  {warning}")
        
        # Store data and metadata
        self.data = data.copy()
        self.target_column = target_column
        self.feature_columns = [col for col in data.columns if col != target_column]
        
        # Generate data summary
        self.results['data_summary'] = DataValidator.get_data_summary(data)
        self.results['validation_results'] = validation_results
        
        logger.info(f"Data loaded successfully: {data.shape}")
        print(f"✅ Data loaded: {data.shape[0]:,} rows × {data.shape[1]} columns")
        print(f"📊 Target: {target_column}")
        print(f"📈 Features: {len(self.feature_columns)}")
        
        return True
    
    def preprocess_data(self) -> bool:
        """Preprocess data according to configuration."""
        
        if self.data is None:
            logger.error("No data loaded. Call load_and_validate_data first.")
            return False
        
        logger.info("Starting data preprocessing")
        
        try:
            # Optimize for large datasets
            processed_data = self.data_processor.optimize_for_large_data(self.data)
            
            # Handle missing values
            processed_data = self.data_processor.handle_missing_values(processed_data, self.target_column)
            
            # Log categorical columns before encoding
            feature_cols = [col for col in processed_data.columns if col != self.target_column]
            categorical_cols = processed_data[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                logger.info(f"Found categorical columns: {categorical_cols}")
                print(f"   • Found categorical columns: {categorical_cols}")
            
            # Encode categorical features
            processed_data = self.data_processor.encode_categorical_features(processed_data, self.target_column)
            
            # Update feature columns after encoding
            self.feature_columns = [col for col in processed_data.columns if col != self.target_column]
            
            # Store processed data
            self.data = processed_data
            self.results['preprocessing_steps'] = self.data_processor.get_preprocessing_summary()
            self.results['final_feature_count'] = len(self.feature_columns)
            
            logger.info(f"Preprocessing complete. Final features: {len(self.feature_columns)}")
            print(f"✅ Preprocessing complete")
            print(f"📊 Final features: {len(self.feature_columns)}")
            
            for step in self.results['preprocessing_steps']:
                print(f"   • {step}")
            
            return True
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            print(f"❌ Preprocessing failed: {str(e)}")
            return False
    
    def train_and_evaluate_models(self) -> bool:
        """Train and evaluate multiple regression models."""
        
        if self.data is None:
            logger.error("No data loaded. Call load_and_validate_data first.")
            return False
        
        logger.info("Starting model training and evaluation")
        
        try:
            # Prepare data
            X = self.data[self.feature_columns]
            y = self.data[self.target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            # Store split data
            self.results['X_train'] = X_train
            self.results['X_test'] = X_test
            self.results['y_train'] = y_train
            self.results['y_test'] = y_test
            
            # Initialize models
            models = self.model_trainer.initialize_models()
            
            # Train and evaluate each model
            model_results = {}
            cross_validation_scores = {}
            
            for model_name in models:
                logger.info(f"Training {model_name}")
                print(f"🔄 Training {model_name}...")
                
                # Train model
                trained_model = self.model_trainer.train_model(model_name, X_train, y_train)
                
                # Evaluate model
                metrics, cv_scores = self.model_evaluator.evaluate_model(
                    trained_model, X_train, X_test, y_train, y_test, model_name
                )
                
                model_results[model_name] = {
                    'model': trained_model,
                    'metrics': metrics
                }
                cross_validation_scores[model_name] = cv_scores
            
            # Compare models
            results_df = self.model_evaluator.compare_models(
                {name: result['metrics'] for name, result in model_results.items()}
            )
            
            # Store results
            self.results['model_results'] = model_results
            self.results['comparison_df'] = results_df
            self.results['cross_validation'] = cross_validation_scores
            
            # Identify best model
            best_model_name = results_df.iloc[0]['Model']
            self.best_model = model_results[best_model_name]['model']
            self.results['best_model_name'] = best_model_name
            self.results['best_model'] = self.best_model
            
            logger.info(f"Model evaluation complete. Best model: {best_model_name}")
            print(f"✅ Model evaluation complete")
            print(f"🏆 Best model: {best_model_name}")
            print(f"📊 R² Score: {results_df.iloc[0]['test_r2']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            print(f"❌ Model training failed: {str(e)}")
            return False
    
    def optimize_hyperparameters(self) -> bool:
        """Optimize hyperparameters for the best model."""
        
        if not self.config.hyperparameter_tuning or self.best_model is None:
            logger.info("Skipping hyperparameter optimization")
            return True
        
        logger.info("Starting hyperparameter optimization")
        
        try:
            # Get base model name
            best_model_name = self.results['best_model_name']
            base_model = self.model_trainer.models[best_model_name]
            
            # Optimize hyperparameters
            optimized_model, optimization_results = self.hyperparameter_optimizer.optimize_model(
                base_model, best_model_name,
                self.results['X_train'], self.results['y_train']
            )
            
            # Evaluate optimized model
            optimized_metrics, _ = self.model_evaluator.evaluate_model(
                optimized_model,
                self.results['X_train'], self.results['X_test'],
                self.results['y_train'], self.results['y_test'],
                f"{best_model_name}_optimized"
            )
            
            # Compare with original
            original_r2 = self.results['model_results'][best_model_name]['metrics']['test_r2']
            optimized_r2 = optimized_metrics['test_r2']
            improvement = optimized_r2 - original_r2
            
            # Update best model if improvement is significant
            if improvement > 0.001:  # Small threshold for improvement
                self.best_model = optimized_model
                self.results['best_model'] = optimized_model
                self.results['optimization_results'] = optimization_results
                self.results['optimization_improvement'] = improvement
                
                logger.info(f"Hyperparameter optimization improved R² by {improvement:.4f}")
                print(f"✅ Hyperparameter optimization complete")
                print(f"📈 Improvement: +{improvement:.4f} R²")
            else:
                logger.info("Hyperparameter optimization did not improve performance")
                print(f"✅ Hyperparameter optimization complete (no improvement)")
            
            return True
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            print(f"❌ Hyperparameter optimization failed: {str(e)}")
            return False
    
    def interpret_model(self) -> bool:
        """Generate comprehensive model interpretation."""
        
        if self.best_model is None:
            logger.error("No trained model available for interpretation")
            return False
        
        logger.info("Starting model interpretation")
        
        try:
            # Feature importance analysis
            self.feature_importance = self.model_interpreter.analyze_feature_importance(
                self.best_model, self.feature_columns
            )
            
            # Residual analysis
            y_pred = self.best_model.predict(self.results['X_test'])
            residual_analysis = self.model_interpreter.analyze_residuals(
                self.results['y_test'], y_pred
            )
            
            # Performance breakdown
            performance_breakdown = self.model_interpreter.performance_breakdown(
                self.results['y_test'], y_pred
            )
            
            # Store interpretation results
            self.results['feature_importance'] = self.feature_importance
            self.results['residual_analysis'] = residual_analysis
            self.results['performance_breakdown'] = performance_breakdown
            self.results['predictions'] = y_pred
            
            logger.info("Model interpretation complete")
            print(f"✅ Model interpretation complete")
            
            if not self.feature_importance.empty:
                print(f"📊 Top 5 important features:")
                for i, (_, row) in enumerate(self.feature_importance.head(5).iterrows(), 1):
                    print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model interpretation failed: {str(e)}")
            print(f"❌ Model interpretation failed: {str(e)}")
            return False
    
    def create_visualizations(self) -> bool:
        """Create comprehensive visualizations."""
        
        if 'comparison_df' not in self.results:
            logger.error("No model results available for visualization")
            return False
        
        logger.info("Creating visualizations")
        
        try:
            # Model comparison visualization
            comparison_fig = self.visualizer.create_interactive_model_comparison(
                self.results['comparison_df']
            )
            
            # Feature importance visualization
            if not self.feature_importance.empty:
                importance_fig = self.visualizer.create_interactive_feature_importance(
                    self.feature_importance
                )
            
            # Residual analysis visualization
            if 'predictions' in self.results:
                residual_fig = self.visualizer.create_residual_analysis_plots(
                    self.results['y_test'], self.results['predictions']
                )
            
            logger.info("Visualizations created successfully")
            print(f"✅ Interactive visualizations created")
            
            if self.config.save_plots:
                print(f"📁 Plots saved to: {self.config.output_directory}")
            
            return True
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            print(f"❌ Visualization creation failed: {str(e)}")
            return False
    
    def create_deployment_package(self) -> bool:
        """Create production deployment package."""
        
        if self.best_model is None:
            logger.error("No trained model available for deployment")
            return False
        
        if not self.config.create_deployment_package:
            logger.info("Skipping deployment package creation")
            return True
        
        logger.info("Creating deployment package")
        
        try:
            # Prepare metadata
            model_metadata = {
                'training_metrics': self.results['model_results'][self.results['best_model_name']]['metrics'],
                'data_summary': self.results['data_summary'],
                'preprocessing_steps': self.results.get('preprocessing_steps', []),
                'optimization_results': self.results.get('optimization_results', {}),
                'feature_importance': self.feature_importance.to_dict('records') if not self.feature_importance.empty else []
            }
            
            # Create deployment package
            package_path = self.deployment_manager.create_model_package(
                model=self.best_model,
                feature_names=self.feature_columns,
                target_name=self.target_column,
                model_metadata=model_metadata,
                preprocessing_steps=self.results.get('preprocessing_steps', [])
            )
            
            self.results['deployment_package_path'] = package_path
            
            logger.info(f"Deployment package created: {package_path}")
            print(f"✅ Deployment package created")
            print(f"📦 Package location: {package_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment package creation failed: {str(e)}")
            print(f"❌ Deployment package creation failed: {str(e)}")
            return False
    
    def run_complete_workflow(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Run the complete regression analysis workflow."""
        
        logger.info("Starting complete regression workflow")
        print("🎯 Starting Enhanced Regression Analysis Workflow")
        print("=" * 60)
        
        workflow_success = True
        
        # Step 1: Load and validate data
        print("\n📁 Step 1: Data Loading & Validation")
        if not self.load_and_validate_data(data, target_column):
            return {'success': False, 'error': 'Data validation failed'}
        
        # Step 2: Preprocess data
        print("\n🔧 Step 2: Data Preprocessing")
        if not self.preprocess_data():
            workflow_success = False
        
        # Step 3: Train and evaluate models
        print("\n🤖 Step 3: Model Training & Evaluation")
        if not self.train_and_evaluate_models():
            return {'success': False, 'error': 'Model training failed'}
        
        # Step 4: Optimize hyperparameters
        print("\n🎛  Step 4: Hyperparameter Optimization")
        if not self.optimize_hyperparameters():
            workflow_success = False
        
        # Step 5: Interpret model
        print("\n🔍 Step 5: Model Interpretation")
        if not self.interpret_model():
            workflow_success = False
        
        # Step 6: Create visualizations
        print("\n📊 Step 6: Visualization Creation")
        if not self.create_visualizations():
            workflow_success = False
        
        # Step 7: Create deployment package
        print("\n📦 Step 7: Deployment Package Creation")
        if not self.create_deployment_package():
            workflow_success = False
        
        # Generate final summary
        print("\n" + "=" * 60)
        if workflow_success:
            print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        else:
            print("⚠️  WORKFLOW COMPLETED WITH WARNINGS")
        
        self._print_summary()
        
        # Return comprehensive results
        return {
            'success': workflow_success,
            'best_model': self.best_model,
            'results': self.results,
            'config': self.config
        }
    
    def _print_summary(self):
        """Print workflow summary."""
        
        print("\n📊 WORKFLOW SUMMARY")
        print("-" * 40)
        
        if 'data_summary' in self.results:
            data_summary = self.results['data_summary']
            print(f"Dataset: {data_summary['shape'][0]:,} rows × {data_summary['shape'][1]} columns")
            print(f"Memory Usage: {data_summary['memory_usage_mb']:.1f} MB")
        
        if 'best_model_name' in self.results:
            print(f"Best Model: {self.results['best_model_name']}")
            
        if 'comparison_df' in self.results and not self.results['comparison_df'].empty:
            best_metrics = self.results['comparison_df'].iloc[0]
            print(f"Performance:")
            print(f"  • R² Score: {best_metrics['test_r2']:.4f}")
            print(f"  • RMSE: {best_metrics['test_rmse']:.4f}")
            print(f"  • MAE: {best_metrics['test_mae']:.4f}")
        
        if 'optimization_improvement' in self.results:
            improvement = self.results['optimization_improvement']
            print(f"Optimization Improvement: +{improvement:.4f} R²")
        
        if 'deployment_package_path' in self.results:
            print(f"Deployment Package: {self.results['deployment_package_path']}")
        
        print(f"Output Directory: {self.config.output_directory}")

# =============================================================================
# 9. UTILITIES AND TESTING
# =============================================================================

class RegressionTester:
    """Testing framework for regression workflow."""
    
    @staticmethod
    def create_test_dataset(n_samples: int = 1000, n_features: int = 10, 
                           noise: float = 0.1, random_state: int = 42) -> pd.DataFrame:
        """Create synthetic test dataset for regression."""
        
        np.random.seed(random_state)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target with known relationship
        coefficients = np.random.randn(n_features)
        y = X @ coefficients + noise * np.random.randn(n_samples)
        
        # Create DataFrame
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y
        
        # Add some categorical features
        data['category_A'] = np.random.choice(['A1', 'A2', 'A3'], n_samples)
        data['category_B'] = np.random.choice(['B1', 'B2'], n_samples)
        
        return data
    
    @staticmethod
    def test_workflow_with_sample_data(config: RegressionConfig = None) -> Dict[str, Any]:
        """Test complete workflow with sample data."""
        
        print("🧪 Testing Enhanced Regression Framework")
        print("=" * 50)
        
        # Create test configuration
        test_config = config or RegressionConfig(
            test_size=0.2,
            models_to_include=['linear', 'ridge', 'random_forest'],
            hyperparameter_tuning=True,
            create_deployment_package=True,
            output_directory="test_output"
        )
        
        # Create test dataset
        test_data = RegressionTester.create_test_dataset(n_samples=500)
        
        # Initialize workflow
        workflow = EnhancedRegressionWorkflow(test_config)
        
        # Run complete workflow
        results = workflow.run_complete_workflow(test_data, 'target')
        
        # Validate results
        assert results['success'], "Workflow should complete successfully"
        assert results['best_model'] is not None, "Best model should be available"
        assert 'comparison_df' in results['results'], "Model comparison should be available"
        
        print("\n✅ All tests passed!")
        
        return results

def load_sample_datasets() -> Dict[str, pd.DataFrame]:
    """Load sample datasets for testing and demonstration."""
    
    datasets = {}
    np.random.seed(42)
    
    # 1. House prices dataset
    n_houses = 1000
    datasets['house_prices'] = pd.DataFrame({
        'size_sqft': np.random.normal(2000, 500, n_houses),
        'bedrooms': np.random.randint(1, 6, n_houses),
        'bathrooms': np.random.randint(1, 4, n_houses),
        'age_years': np.random.randint(0, 50, n_houses),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_houses),
        'garage': np.random.choice(['Yes', 'No'], n_houses),
        'price': None  # Will be generated based on features
    })
    
    # Generate realistic price based on features
    price = (
        datasets['house_prices']['size_sqft'] * 100 +
        datasets['house_prices']['bedrooms'] * 10000 +
        datasets['house_prices']['bathrooms'] * 5000 +
        (50 - datasets['house_prices']['age_years']) * 1000 +
        np.where(datasets['house_prices']['location'] == 'Urban', 50000,
                np.where(datasets['house_prices']['location'] == 'Suburban', 30000, 10000)) +
        np.where(datasets['house_prices']['garage'] == 'Yes', 15000, 0) +
        np.random.normal(0, 20000, n_houses)
    )
    datasets['house_prices']['price'] = np.maximum(price, 50000)  # Minimum price
    
    # 2. Marketing ROI dataset
    n_campaigns = 800
    datasets['marketing_roi'] = pd.DataFrame({
        'ad_spend': np.random.exponential(5000, n_campaigns),
        'impressions': np.random.poisson(10000, n_campaigns),
        'clicks': np.random.poisson(500, n_campaigns),
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_campaigns),
        'channel': np.random.choice(['Social', 'Search', 'Display', 'Email'], n_campaigns),
        'target_audience': np.random.choice(['Young', 'Middle', 'Senior'], n_campaigns),
        'roi': None
    })
    
    # Generate ROI based on realistic relationships
    base_roi = (
        datasets['marketing_roi']['clicks'] * 0.05 +
        datasets['marketing_roi']['impressions'] * 0.001 +
        np.where(datasets['marketing_roi']['channel'] == 'Search', 1000,
                np.where(datasets['marketing_roi']['channel'] == 'Social', 800, 500)) +
        np.random.normal(0, 200, n_campaigns)
    )
    datasets['marketing_roi']['roi'] = np.maximum(base_roi, 0)
    
    # 3. Employee performance dataset
    n_employees = 600
    datasets['employee_performance'] = pd.DataFrame({
        'experience_years': np.random.exponential(5, n_employees),
        'education_level': np.random.choice(['Bachelor', 'Master', 'PhD'], n_employees),
        'training_hours': np.random.exponential(40, n_employees),
        'team_size': np.random.randint(3, 15, n_employees),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'Support'], n_employees),
        'remote_work_pct': np.random.uniform(0, 100, n_employees),
        'performance_score': None
    })
    
    # Generate performance score
    performance = (
        datasets['employee_performance']['experience_years'] * 2 +
        np.where(datasets['employee_performance']['education_level'] == 'PhD', 15,
                np.where(datasets['employee_performance']['education_level'] == 'Master', 10, 5)) +
        datasets['employee_performance']['training_hours'] * 0.1 +
        np.where(datasets['employee_performance']['department'] == 'Engineering', 5, 0) +
        np.random.normal(75, 10, n_employees)
    )
    datasets['employee_performance']['performance_score'] = np.clip(performance, 0, 100)
    
    return datasets

# =============================================================================
# 10. MAIN INTERFACE FUNCTIONS
# =============================================================================

def quick_regression_analysis(data: pd.DataFrame = None, target_column: str = None,
                            config: RegressionConfig = None) -> Dict[str, Any]:
    """Quick regression analysis with minimal configuration."""
    
    # Use sample data if none provided
    if data is None:
        print("🎲 Using sample house price data for demonstration...")
        sample_datasets = load_sample_datasets()
        data = sample_datasets['house_prices']
        target_column = 'price'
    
    # Use default config if none provided
    if config is None:
        config = RegressionConfig(
            models_to_include=['linear', 'ridge', 'random_forest', 'gradient_boosting'],
            hyperparameter_tuning=False,  # Skip for quick analysis
            save_plots=True,
            create_deployment_package=True
        )
    
    # Run workflow
    workflow = EnhancedRegressionWorkflow(config)
    results = workflow.run_complete_workflow(data, target_column)
    
    return results

def create_regression_analysis_interface():
    """Create comprehensive regression analysis interface."""
    
    interface_html = """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 25px; border-radius: 15px; margin: 20px 0;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
        <h1 style="margin: 0; font-size: 32px; font-weight: 300;">
            🚀 Enhanced Regression Analysis Framework
        </h1>
        <p style="margin: 15px 0 0 0; font-size: 18px; opacity: 0.9;">
            Production-ready ML pipeline with advanced features
        </p>
    </div>
    """
    
    usage_html = """
    <div style="background: #f8f9fa; padding: 25px; border-radius: 10px; margin: 20px 0;">
        <h3 style="color: #495057; margin-top: 0;">🎯 Quick Start Options</h3>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
            
            <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745;">
                <h4 style="color: #28a745; margin-top: 0;">📊 Quick Demo</h4>
                <code style="background: #f1f3f4; padding: 10px; display: block; border-radius: 4px;">
                    results = quick_regression_analysis()
                </code>
                <p style="margin: 10px 0 0 0; font-size: 14px; color: #6c757d;">
                    Run complete analysis with sample data
                </p>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff;">
                <h4 style="color: #007bff; margin-top: 0;">📁 Your Data</h4>
                <code style="background: #f1f3f4; padding: 10px; display: block; border-radius: 4px;">
                    data = pd.read_csv('your_file.csv')<br>
                    results = quick_regression_analysis(data, 'target_col')
                </code>
                <p style="margin: 10px 0 0 0; font-size: 14px; color: #6c757d;">
                    Analyze your own dataset
                </p>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #6f42c1;">
                <h4 style="color: #6f42c1; margin-top: 0;">⚙️ Custom Config</h4>
                <code style="background: #f1f3f4; padding: 10px; display: block; border-radius: 4px;">
                    config = RegressionConfig.from_file('config.yaml')<br>
                    workflow = EnhancedRegressionWorkflow(config)
                </code>
                <p style="margin: 10px 0 0 0; font-size: 14px; color: #6c757d;">
                    Advanced configuration options
                </p>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #dc3545;">
                <h4 style="color: #dc3545; margin-top: 0;">🧪 Testing</h4>
                <code style="background: #f1f3f4; padding: 10px; display: block; border-radius: 4px;">
                    test_results = RegressionTester.test_workflow_with_sample_data()
                </code>
                <p style="margin: 10px 0 0 0; font-size: 14px; color: #6c757d;">
                    Test framework functionality
                </p>
            </div>
        </div>
    </div>
    """
    
    features_html = """
    <div style="background: #e3f2fd; padding: 25px; border-radius: 10px; margin: 20px 0;">
        <h3 style="color: #1976d2; margin-top: 0;">✨ Enhanced Features</h3>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
            <div>
                <h4 style="color: #1976d2; margin: 0 0 10px 0;">🔧 Configuration Management</h4>
                <ul style="margin: 0; padding-left: 20px; color: #424242;">
                    <li>YAML/JSON configuration files</li>
                    <li>Environment-specific settings</li>
                    <li>Parameter validation</li>
                </ul>
            </div>
            
            <div>
                <h4 style="color: #1976d2; margin: 0 0 10px 0;">📊 Advanced Visualization</h4>
                <ul style="margin: 0; padding-left: 20px; color: #424242;">
                    <li>Interactive Plotly charts</li>
                    <li>Comprehensive dashboards</li>
                    <li>Exportable reports</li>
                </ul>
            </div>
            
            <div>
                <h4 style="color: #1976d2; margin: 0 0 10px 0;">🛡️ Production Monitoring</h4>
                <ul style="margin: 0; padding-left: 20px; color: #424242;">
                    <li>Data drift detection</li>
                    <li>Performance monitoring</li>
                    <li>Automated alerts</li>
                </ul>
            </div>
            
            <div>
                <h4 style="color: #1976d2; margin: 0 0 10px 0;">🚀 Deployment Ready</h4>
                <ul style="margin: 0; padding-left: 20px; color: #424242;">
                    <li>FastAPI server templates</li>
                    <li>Docker configuration</li>
                    <li>Comprehensive documentation</li>
                </ul>
            </div>
        </div>
    </div>
    """
    
    display(HTML(interface_html))
    display(HTML(usage_html))
    display(HTML(features_html))
    
    print("🎯 Enhanced Regression Framework Ready!")
    print("=" * 50)
    print("💡 Start with: quick_regression_analysis() for instant results")
    print("📚 Or explore the sample datasets: load_sample_datasets()")
    
    return {
        'quick_analysis': quick_regression_analysis,
        'workflow_class': EnhancedRegressionWorkflow,
        'config_class': RegressionConfig,
        'sample_datasets': load_sample_datasets(),
        'tester': RegressionTester
    }

# =============================================================================
# 11. INITIALIZATION AND SETUP
# =============================================================================

def setup_enhanced_environment():
    """Setup the enhanced regression analysis environment."""
    
    print("🔧 SETTING UP ENHANCED REGRESSION ENVIRONMENT")
    print("=" * 55)
    
    # Check required libraries
    required_libs = {
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'sklearn': __import__('sklearn').__version__,
        'matplotlib': __import__('matplotlib').__version__,
        'seaborn': sns.__version__,
        'plotly': px.__version__ if hasattr(px, '__version__') else 'available'
    }
    
    print("📦 Checking required libraries:")
    for lib, version in required_libs.items():
        print(f"   ✅ {lib}: {version}")
    
    # Check optional libraries
    optional_libs = [
        ('ipywidgets', WIDGETS_AVAILABLE),
        ('yaml', True),  # Usually available
        ('scipy', True)  # Usually available
    ]
    
    for lib, available in optional_libs:
        status = "✅" if available else "⚠️"
        print(f"   {status} {lib}: {'available' if available else 'not installed (optional)'}")
    
    print(f"\n🎯 Enhanced Environment Ready!")
    print(f"🚀 New Features:")
    print(f"   • Configuration management with YAML/JSON")
    print(f"   • Interactive Plotly visualizations")
    print(f"   • Production monitoring and drift detection")
    print(f"   • Comprehensive deployment packages")
    print(f"   • Advanced testing framework")
    
    return True

# Initialize when imported
if __name__ == "__main__":
    setup_enhanced_environment()
    
    print("\n" + "🎯 " * 20)
    print("ENHANCED REGRESSION FRAMEWORK LOADED")
    print("🎯 " * 20)
    print("Ready for production-grade regression analysis!")
    print("Run create_regression_analysis_interface() to begin!")

# Auto-setup when imported
setup_enhanced_environment()