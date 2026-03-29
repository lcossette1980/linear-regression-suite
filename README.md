# Linear Regression Analysis Suite - Streamlit App

A powerful and user-friendly Streamlit application for comprehensive linear regression analysis with multiple models, interactive visualizations, and production-ready deployment features.

## 🚀 New: Enhanced UI/UX Version Available!

We now offer an enhanced version with modern UI/UX design featuring:
- 🎨 **Beautiful Gradient UI**: Modern color schemes with smooth animations
- 📊 **Enhanced Visualizations**: More interactive and informative charts
- 🎯 **Smart Recommendations**: AI-powered insights and suggestions
- 📈 **Visual Progress Tracking**: See your workflow progress at a glance
- 💡 **Intelligent Metric Cards**: Real-time updates with contextual information
- 🔥 **Professional Design**: Enterprise-grade interface with improved usability

### Run Enhanced Version
```bash
streamlit run enhanced_streamlit_app.py
```

## Features

- **📤 Easy Data Upload**: Support for CSV files with automatic data profiling
- **🔧 Smart Preprocessing**: Handle missing values, encode categorical variables, and scale features
- **🤖 Multiple Models**: Train and compare 7 different regression models
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Elastic Net
  - Random Forest
  - Gradient Boosting
  - Support Vector Regression
- **📊 Rich Visualizations**: Interactive plots using Plotly
- **🎯 Model Insights**: Feature importance, residual analysis, and performance metrics
- **💾 Export & Deploy**: Save trained models and generate deployment code
- **🔮 Real-time Predictions**: Make single or batch predictions

## Installation

1. Clone this repository:
```bash
git clone https://github.com/lcossette1980/linear-regression-suite.git
cd linear-regression-suite
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Standard Version
Run the standard Streamlit app:
```bash
streamlit run enhanced_streamlit_app.py
```

### Enhanced UI/UX Version (Recommended)
Run the enhanced version with modern UI:
```bash
streamlit run enhanced_streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Quick Start

1. **Home Page**: Get an overview and optionally load sample datasets
2. **Data Upload**: Upload your CSV file and select the target variable
3. **Preprocessing**: Configure data cleaning and feature engineering options
4. **Model Training**: Select models and train with automatic hyperparameter tuning
5. **Results & Insights**: Explore comprehensive visualizations and model performance
6. **Export & Deploy**: Save your best model and get deployment code
7. **Make Predictions**: Use your trained model for new predictions

## App Structure

- `enhanced_streamlit_app.py`: Streamlit application with modern UI/UX design
- `enhanced_regression_framework.py`: Core regression analysis framework
- `requirements.txt`: Python dependencies
- `model_exports/`: Directory for saved models (created automatically)

## Key Features Explained

### Data Preprocessing
- Automatic handling of missing values
- Smart encoding of categorical variables
- Feature scaling and normalization
- Outlier detection and removal

### Model Training
- Automatic cross-validation
- Hyperparameter tuning
- Parallel model training
- Performance comparison

### Visualizations
- Model performance comparison charts
- Predictions vs actual scatter plots
- Feature importance rankings
- Residual analysis plots
- Correlation heatmaps

### Deployment
- Export trained models as pickle files
- Generate deployment scripts
- Comprehensive performance reports
- Batch prediction capabilities

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependencies

## Tips for Best Results

1. **Data Quality**: Ensure your CSV has clean column names and consistent data types
2. **Target Variable**: Select a numerical column as your regression target
3. **Feature Selection**: The app automatically handles feature selection, but you can review feature importance
4. **Model Selection**: Start with all models to see which performs best on your data
5. **Hyperparameter Tuning**: Enable this for better performance (takes longer)

## Support

For issues or questions, please open an issue in the repository.