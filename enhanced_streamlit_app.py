import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import json
import joblib
from datetime import datetime
from dataclasses import asdict
import scipy.stats as stats
import base64
from PIL import Image

# Import the regression framework
from enhanced_regression_framework import (
    RegressionConfig,
    DataProcessor,
    ModelTrainer,
    ModelEvaluator,
    EnhancedVisualizer,
    DeploymentManager,
    EnhancedRegressionWorkflow
)

# Page configuration - Sets browser tab and layout
st.set_page_config(
    page_title="Linear Regression Analysis Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for modern, professional appearance
st.markdown("""
<style>
    /* Import brand fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lato:wght@400;500&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Lato', sans-serif;
        font-weight: 400;
    }
    
    /* Main header styling */
    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #A44A3F 0%, #2A2A2A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
    }
    
    /* Sub-header styling */
    .sub-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #2A2A2A;
        margin-bottom: 1rem;
        border-bottom: 2px solid #A44A3F;
        padding-bottom: 0.3rem;
    }
    
    /* Metric cards with modern design */
    .metric-card {
        background: linear-gradient(135deg, #F5F2EA 0%, #D7CEB2 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(42, 42, 42, 0.1);
        border: 1px solid rgba(164, 74, 63, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(42, 42, 42, 0.15);
    }
    
    /* Success metric card */
    .metric-card-success {
        background: linear-gradient(135deg, #D7CEB2 0%, #A59E8C 100%);
        border-left: 4px solid #A59E8C;
    }
    
    /* Warning metric card */
    .metric-card-warning {
        background: linear-gradient(135deg, #F5F2EA 0%, #D7CEB2 100%);
        border-left: 4px solid #A44A3F;
    }
    
    /* Error metric card */
    .metric-card-error {
        background: linear-gradient(135deg, #D7CEB2 0%, #A44A3F 100%);
        border-left: 4px solid #A44A3F;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        font-family: 'Lato', sans-serif;
        font-weight: 500;
        width: 100%;
        border-radius: 8px;
        height: 2.8rem;
        font-size: 1rem;
        background: linear-gradient(135deg, #A44A3F 0%, #2A2A2A 100%);
        color: #F5F2EA;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(164, 74, 63, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(164, 74, 63, 0.4);
        background: linear-gradient(135deg, #2A2A2A 0%, #A44A3F 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2A2A2A 0%, #A44A3F 100%);
    }
    
    /* File uploader enhancement */
    .stFileUploader > div {
        border: 2px dashed #A44A3F;
        border-radius: 10px;
        padding: 1.5rem;
        background: linear-gradient(135deg, #F5F2EA 0%, #D7CEB2 100%);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #A44A3F 0%, #2A2A2A 100%);
        border-radius: 10px;
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(42, 42, 42, 0.1);
        font-family: 'Lato', sans-serif;
    }
    
    /* Alert boxes */
    .alert-info {
        background: linear-gradient(135deg, #F5F2EA 0%, #D7CEB2 100%);
        border-left: 4px solid #A59E8C;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-family: 'Lato', sans-serif;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #D7CEB2 0%, #A59E8C 100%);
        border-left: 4px solid #A59E8C;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-family: 'Lato', sans-serif;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #F5F2EA 0%, #D7CEB2 100%);
        border-left: 4px solid #A44A3F;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-family: 'Lato', sans-serif;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background: linear-gradient(135deg, #F5F2EA 0%, #D7CEB2 100%);
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
        font-family: 'Lato', sans-serif;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #A44A3F 0%, #2A2A2A 100%);
        color: #F5F2EA;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #F5F2EA 0%, #D7CEB2 100%);
        border-radius: 10px;
        font-family: 'Lato', sans-serif;
        font-weight: 500;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #A59E8C;
        font-family: 'Lato', sans-serif;
    }
    
    /* Number input styling */
    .stNumberInput > div > div {
        border-radius: 10px;
        border: 2px solid #A59E8C;
        font-family: 'Lato', sans-serif;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #A44A3F 0%, #2A2A2A 100%);
    }
    
    /* Custom spacing */
    .space-small {
        margin: 0.5rem 0;
    }
    
    .space-medium {
        margin: 1rem 0;
    }
    
    .space-large {
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced UI Helper Functions
def create_metric_card(title, value, delta=None, card_type="default"):
    """Create a styled metric card with optional delta."""
    
    card_class = {
        "success": "metric-card metric-card-success",
        "warning": "metric-card metric-card-warning", 
        "error": "metric-card metric-card-error",
        "default": "metric-card"
    }
    
    delta_html = ""
    if delta is not None:
        delta_color = "#A59E8C" if delta >= 0 else "#A44A3F"
        delta_symbol = "▲" if delta >= 0 else "▼"
        delta_html = f'<p style="color: {delta_color}; margin: 0.5rem 0 0 0; font-family: "Lato", sans-serif; font-weight: 500;">{delta_symbol} {delta}</p>'
    
    card_html = f"""
    <div class="{card_class.get(card_type, 'metric-card')}">
        <h3 style="margin: 0; color: #2A2A2A; font-size: 1rem; font-family: 'Lato', sans-serif; font-weight: 500;">{title}</h3>
        <h2 style="margin: 0.3rem 0 0 0; color: #2A2A2A; font-size: 1.8rem; font-family: 'Playfair Display', serif; font-weight: 700;">{value}</h2>
        {delta_html}
    </div>
    """
    
    return st.markdown(card_html, unsafe_allow_html=True)

def create_status_alert(message, alert_type="info"):
    """Create styled alert boxes."""
    
    icons = {
        "info": "ℹ️",
        "success": "✅", 
        "warning": "⚠️",
        "error": "❌"
    }
    
    alert_html = f"""
    <div class="alert-{alert_type}">
        <strong>{icons.get(alert_type, "ℹ️")} {message}</strong>
    </div>
    """
    
    return st.markdown(alert_html, unsafe_allow_html=True)

def create_progress_indicator(current_step, total_steps, step_names):
    """Create a visual progress indicator."""
    
    progress_html = """
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
    """
    
    for i, step_name in enumerate(step_names):
        if i < current_step:
            # Completed step
            circle_color = "#A59E8C"
            text_color = "#A59E8C"
            icon = "✓"
        elif i == current_step:
            # Current step
            circle_color = "#A44A3F"
            text_color = "#A44A3F"
            icon = str(i + 1)
        else:
            # Future step
            circle_color = "#D7CEB2"
            text_color = "#A59E8C"
            icon = str(i + 1)
        
        progress_html += f"""
            <div style="text-align: center; flex: 1;">
                <div style="
                    width: 40px; 
                    height: 40px; 
                    border-radius: 50%; 
                    background: {circle_color};
                    color: white;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 0.5rem auto;
                    font-weight: bold;
                ">{icon}</div>
                <p style="color: {text_color}; font-size: 0.9rem; margin: 0; font-family: 'Lato', sans-serif; font-weight: 500;">{step_name}</p>
            </div>
        """
    
    progress_html += """
        </div>
        <div style="background: #D7CEB2; height: 4px; border-radius: 2px; margin-top: 1rem;">
            <div style="
                background: linear-gradient(90deg, #A44A3F 0%, #2A2A2A 100%); 
                height: 100%; 
                border-radius: 2px;
                width: """ + f"{(current_step / total_steps) * 100}%" + """
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    """
    
    return st.markdown(progress_html, unsafe_allow_html=True)

def create_feature_importance_chart(importance_df):
    """Create an enhanced feature importance chart."""
    
    # Take top 15 features for better readability
    top_features = importance_df.head(15)
    
    fig = go.Figure()
    
    # Create color scale based on importance values
    colors = px.colors.sample_colorscale(
        "viridis", 
        [i/(len(top_features)-1) for i in range(len(top_features))]
    )
    
    fig.add_trace(
        go.Bar(
            y=top_features['feature'],
            x=top_features['importance'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.8)', width=1)
            ),
            text=[f"{val:.4f}" for val in top_features['importance']],
            textposition='inside',
            textfont=dict(color='#F5F2EA', size=12, family='Lato')
        )
    )
    
    fig.update_layout(
        title={
            'text': "🎯 Feature Importance Analysis",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Playfair Display', 'color': '#2A2A2A'}
        },
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white',
        font=dict(family='Lato', size=12),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_model_comparison_chart(results_df):
    """Create an enhanced model comparison chart."""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('R² Score Comparison', 'RMSE Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # R² Score chart
    fig.add_trace(
        go.Bar(
            x=results_df['test_r2'],
            y=results_df['Model'],
            orientation='h',
            name='R² Score',
            marker=dict(
                color=results_df['test_r2'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="R² Score", x=0.45)
            ),
            text=[f"{val:.3f}" for val in results_df['test_r2']],
            textposition='inside'
        ),
        row=1, col=1
    )
    
    # RMSE chart
    fig.add_trace(
        go.Bar(
            x=results_df['test_rmse'],
            y=results_df['Model'],
            orientation='h',
            name='RMSE',
            marker=dict(
                color=results_df['test_rmse'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="RMSE", x=1.02)
            ),
            text=[f"{val:.1f}" for val in results_df['test_rmse']],
            textposition='inside'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white',
        font=dict(family='Lato', size=12),
        title={
            'text': "🏆 Model Performance Comparison",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Playfair Display', 'color': '#2A2A2A'}
        }
    )
    
    return fig

def create_residual_analysis_plot(y_true, y_pred):
    """Create enhanced residual analysis plots."""
    
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Residuals vs Predicted Values',
            'Residual Distribution', 
            'Q-Q Plot',
            'Residuals vs Fitted (Standardized)'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                color='rgba(164, 74, 63, 0.6)',
                size=6,
                line=dict(width=1, color='#F5F2EA')
            )
        ),
        row=1, col=1
    )
    
    # Add zero line
    fig.add_trace(
        go.Scatter(
            x=[y_pred.min(), y_pred.max()],
            y=[0, 0],
            mode='lines',
            name='Zero Line',
            line=dict(color='red', dash='dash', width=2)
        ),
        row=1, col=1
    )
    
    # Residual histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Distribution',
            marker=dict(
                color='rgba(165, 158, 140, 0.7)',
                line=dict(width=1, color='#F5F2EA')
            )
        ),
        row=1, col=2
    )
    
    # Q-Q Plot
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode='markers',
            name='Q-Q Plot',
            marker=dict(
                color='rgba(164, 74, 63, 0.6)',
                size=6
            )
        ),
        row=2, col=1
    )
    
    # Perfect normal line
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=theoretical_quantiles,
            mode='lines',
            name='Perfect Normal',
            line=dict(color='red', dash='dash', width=2)
        ),
        row=2, col=1
    )
    
    # Standardized residuals
    std_residuals = residuals / residuals.std()
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=std_residuals,
            mode='markers',
            name='Standardized',
            marker=dict(
                color='rgba(164, 74, 63, 0.6)',
                size=6
            )
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white',
        font=dict(family='Lato', size=12),
        title={
            'text': "🔍 Comprehensive Residual Analysis",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Playfair Display', 'color': '#2A2A2A'}
        }
    )
    
    return fig

# Enhanced Session State Management
def initialize_session_state():
    """Initialize all session state variables with default values."""
    
    default_states = {
        'workflow': None,
        'data': None,
        'processed_data': None,
        'models': None,
        'best_model': None,
        'results': None,
        'config': RegressionConfig(encode_categorical='onehot'),
        'current_step': 0,
        'analysis_complete': False,
        'model_trained': False,
        'data_uploaded': False
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Enhanced Sidebar with Modern Navigation
def create_enhanced_sidebar():
    """Create an enhanced sidebar with better navigation."""
    
    # Try to load and display logo
    logo_path = "logo1.png"  # Change this to your actual logo filename
    
    if Path(logo_path).exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        
        st.sidebar.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #A44A3F 0%, #2A2A2A 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
        ">
            <div style="
                width: 120px;
                height: 120px;
                margin: 0 auto 1rem auto;
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: #F5F2EA;
                border-radius: 50%;
                overflow: hidden;
            ">
                <img src="data:image/png;base64,{logo_data}" style="max-width: 100%; max-height: 100%; object-fit: contain;">
            </div>
            <h2 style="color: #F5F2EA; margin: 0; font-family: 'Playfair Display', serif; font-weight: 700;">
                ML Suite
            </h2>
            <p style="color: #D7CEB2; margin: 0.5rem 0 0 0; font-size: 0.9rem; font-family: 'Lato', sans-serif;">
                Production-Ready Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback to text if logo not found
        st.sidebar.markdown("""
        <div style="
            background: linear-gradient(135deg, #A44A3F 0%, #2A2A2A 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
        ">
            <div style="
                width: 120px;
                height: 120px;
                background-color: #F5F2EA;
                border-radius: 50%;
                margin: 0 auto 1rem auto;
                display: flex;
                align-items: center;
                justify-content: center;
                font-family: 'Playfair Display', serif;
                font-size: 2rem;
                color: #2A2A2A;
                font-weight: 700;
            ">
                LOGO
            </div>
            <h2 style="color: #F5F2EA; margin: 0; font-family: 'Playfair Display', serif; font-weight: 700;">
                ML Suite
            </h2>
            <p style="color: #D7CEB2; margin: 0.5rem 0 0 0; font-size: 0.9rem; font-family: 'Lato', sans-serif;">
                Production-Ready Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation with icons and descriptions
    pages = {
        "🏠 Home": "Overview & Getting Started",
        "📤 Data Upload": "Load & Validate Your Data", 
        "🔧 Preprocessing": "Clean & Prepare Data",
        "🤖 Model Training": "Train & Compare Models",
        "📊 Results & Insights": "Analyze Performance",
        "💾 Export & Deploy": "Package for Production",
        "🔮 Make Predictions": "Test Your Model"
    }
    
    st.sidebar.markdown("### 🧭 Navigation")
    
    # Create radio buttons with enhanced styling
    page = st.sidebar.radio(
        "Choose your workflow step:",
        list(pages.keys()),
        format_func=lambda x: x,
        help="Navigate through the ML pipeline"
    )
    
    # Display page description
    if page in pages:
        st.sidebar.markdown(f"*{pages[page]}*")
    
    # Update current step based on page
    step_mapping = {
        "🏠 Home": 0,
        "📤 Data Upload": 1,
        "🔧 Preprocessing": 2,
        "🤖 Model Training": 3,
        "📊 Results & Insights": 4,
        "💾 Export & Deploy": 5,
        "🔮 Make Predictions": 6
    }
    st.session_state.current_step = step_mapping.get(page, 0)
    
    # Progress indicator in sidebar
    step_names = ["Upload", "Process", "Train", "Analyze", "Deploy"]
    current_step = min(st.session_state.current_step - 1, len(step_names) - 1)
    current_step = max(0, current_step)  # Ensure non-negative
    
    st.sidebar.markdown("### 📈 Progress")
    progress_percentage = (current_step / len(step_names)) * 100 if current_step > 0 else 0
    st.sidebar.progress(progress_percentage / 100)
    st.sidebar.markdown(f"**Step {current_step + 1} of {len(step_names)}: {step_names[current_step]}**")
    
    # Quick stats in sidebar
    if st.session_state.get('data') is not None:
        st.sidebar.markdown("### 📊 Data Stats")
        data = st.session_state.data
        st.sidebar.metric("Rows", f"{len(data):,}")
        st.sidebar.metric("Columns", len(data.columns))
        
        if st.session_state.get('models') is not None:
            st.sidebar.metric("Models Trained", len(st.session_state.models))
    
    return page

# Initialize session state
initialize_session_state()

# Create enhanced sidebar and get current page
page = create_enhanced_sidebar()

# Main app logic with enhanced UI
if page == "🏠 Home":
    # Try to load and display logo for home page
    logo_path = "logo.png"  # Change this to your actual logo filename
    
    if Path(logo_path).exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        
        st.markdown(f"""
       <div style="text-align: center; margin-bottom: 2rem;">
            <div style="width: 100%; max-width: 600px; margin: 0 auto 1rem auto; border: 3px solid #A44A3F; border-radius: 12px; box-shadow: 0 4px 15px rgba(164, 74, 63, 0.2); background-color: #F5F2EA; overflow: hidden;">
                <div style="width: 100%; height: 80px; display: flex; align-items: center; justify-content: center; background-color: #F5F2EA; padding: 0 2rem; gap: 1.5rem;">
                    <img src="data:image/png;base64,{logo_data}" style="height: 50px; width: auto; object-fit: contain;">
                    <h1 class="main-header" style="margin: 0; font-size: 1.8rem; color: #A44A3F;">🚀 Linear Regression Analysis Suite</h1>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback to text if logo not found
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="
                width: 150px;
                height: 150px;
                background-color: #F5F2EA;
                border: 3px solid #A44A3F;
                border-radius: 50%;
                margin: 0 auto 1rem auto;
                display: flex;
                align-items: center;
                justify-content: center;
                font-family: 'Playfair Display', serif;
                font-size: 2rem;
                color: #2A2A2A;
                font-weight: 700;
                box-shadow: 0 4px 15px rgba(164, 74, 63, 0.2);
            ">
                LOGO
            </div>
            <h1 class="main-header">🚀 Linear Regression Analysis Suite</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Hero section with metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Models Available", "7+", None, "success")
    
    with col2:
        create_metric_card("Auto-Tuning", "✅", None, "success")
    
    with col3:
        create_metric_card("Visualizations", "15+", None, "warning")
    
    with col4:
        create_metric_card("Deploy Ready", "100%", None, "default")
    
    st.markdown("<hr style='margin: 1rem 0; border: 1px solid #D7CEB2;'/>", unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="font-family: 'Playfair Display', serif; color: #2A2A2A;">📊 Comprehensive Analysis</h3>
            <p style="font-family: 'Lato', sans-serif; color: #2A2A2A;">Multiple models with automatic tuning and cross-validation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="font-family: 'Playfair Display', serif; color: #2A2A2A;">🎨 Interactive Visualizations</h3>
            <p style="font-family: 'Lato', sans-serif; color: #2A2A2A;">Beautiful, interactive plots powered by Plotly for deep insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="font-family: 'Playfair Display', serif; color: #2A2A2A;">🚀 Production Ready Deployment</h3>
            <p style="font-family: 'Lato', sans-serif; color: #2A2A2A;">Export models and generate deployment code with one click</p>
        </div>
        """, unsafe_allow_html=True)
    
    
    # Quick start guide with enhanced styling
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        ### How to use this app:
        
        1. **📤 Upload Data**: Start by uploading your CSV file in the Data Upload page
        2. **🔧 Preprocess**: Configure data preprocessing options (handle missing values, encode categories, etc.)
        3. **🤖 Train Models**: Select and train multiple regression models with automatic optimization
        4. **📊 Analyze Results**: Explore comprehensive visualizations and model insights
        5. **💾 Export & Deploy**: Save your best model and create deployment packages
        6. **🔮 Make Predictions**: Use your trained model to make predictions on new data
        
        ### Supported Models:
        - Linear Regression
        - Ridge Regression (L2 regularization)
        - Lasso Regression (L1 regularization)
        - Elastic Net (L1 + L2 regularization)
        - Random Forest Regressor
        - Gradient Boosting Regressor
        - Support Vector Regression (SVR)
        """)
    
    # Sample data section with enhanced buttons
    st.markdown("### 🎯 Try with Sample Data")
    create_status_alert("Load sample datasets to explore the app's capabilities", "info")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏘️ Load Housing Data"):
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing(as_frame=True)
            df = data.frame
            st.session_state.data = df
            st.session_state.data_uploaded = True
            create_status_alert("California Housing data loaded! Go to Data Upload page.", "success")
    
    with col2:
        if st.button("💼 Load Sales Data"):
            # Create synthetic sales data
            np.random.seed(42)
            n_samples = 1000
            df = pd.DataFrame({
                'advertising_spend': np.random.uniform(1000, 50000, n_samples),
                'num_salespeople': np.random.randint(5, 50, n_samples),
                'market_size': np.random.choice(['small', 'medium', 'large'], n_samples),
                'season': np.random.choice(['spring', 'summer', 'fall', 'winter'], n_samples),
                'competitor_price': np.random.uniform(50, 200, n_samples)
            })
            # Create target with some noise
            df['sales'] = (
                0.5 * df['advertising_spend'] + 
                1000 * df['num_salespeople'] + 
                df['market_size'].map({'small': 10000, 'medium': 30000, 'large': 50000}) +
                df['season'].map({'spring': 5000, 'summer': 15000, 'fall': 10000, 'winter': -5000}) +
                -100 * df['competitor_price'] +
                np.random.normal(0, 5000, n_samples)
            )
            st.session_state.data = df
            st.session_state.data_uploaded = True
            create_status_alert("Sample sales data loaded! Go to Data Upload page.", "success")
    
    with col3:
        if st.button("🎓 Load Student Data"):
            # Create synthetic student performance data
            np.random.seed(42)
            n_samples = 800
            df = pd.DataFrame({
                'study_hours': np.random.uniform(0, 10, n_samples),
                'attendance_rate': np.random.uniform(0.5, 1.0, n_samples),
                'previous_grade': np.random.uniform(50, 100, n_samples),
                'assignments_completed': np.random.uniform(0, 1, n_samples),
                'extra_activities': np.random.randint(0, 5, n_samples)
            })
            # Create target
            df['final_grade'] = (
                5 * df['study_hours'] + 
                40 * df['attendance_rate'] + 
                0.3 * df['previous_grade'] +
                20 * df['assignments_completed'] +
                2 * df['extra_activities'] +
                np.random.normal(0, 5, n_samples)
            ).clip(0, 100)
            st.session_state.data = df
            st.session_state.data_uploaded = True
            create_status_alert("Sample student data loaded! Go to Data Upload page.", "success")

elif page == "📤 Data Upload":
    st.markdown('<h1 class="main-header">📤 Data Upload & Exploration</h1>', unsafe_allow_html=True)
    
    # File upload section with enhanced UI
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your dataset in CSV format. The file should contain both features and target variable."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.session_state.data_uploaded = True
            create_status_alert(f"Successfully loaded {uploaded_file.name} with {len(df):,} rows and {len(df.columns)} columns", "success")

            # Display both column blocks here
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### 📋 Dataset Overview")
                col11, col12, col13, col14 = st.columns(4)
                with col11:
                    create_metric_card("Rows", f"{df.shape[0]:,}")
                with col12:
                    create_metric_card("Columns", str(df.shape[1]))
                with col13:
                    create_metric_card("Memory", f"{df.memory_usage().sum() / 1024**2:.2f} MB")
                with col14:
                    missing_count = df.isnull().sum().sum()
                    card_type = "success" if missing_count == 0 else "warning"
                    create_metric_card("Missing Values", str(missing_count), None, card_type)

            with col2:
                st.markdown("### 🎯 Select Target Variable")
                target_column = st.selectbox(
                    "Choose the target column for regression:",
                    df.columns,
                    help="This is the variable you want to predict"
                )

                if st.button("Confirm Target Selection", type="primary"):
                    st.session_state.target_column = target_column
                    create_status_alert(f"Target variable set to: {target_column}", "success")

        except Exception as e:
            create_status_alert(f"Error loading file: {str(e)}", "error")
    
    # Display data preview with enhanced tabs
    if st.session_state.data is not None:
        st.markdown("### 👀 Data Exploration")
        
        # Enhanced tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Sample", "🏷️ Data Types", "📈 Statistics", "❓ Missing Values", "🔍 Correlations"])
        
        with tab1:
            st.dataframe(st.session_state.data.head(100), use_container_width=True)
        
        with tab2:
            dtype_df = pd.DataFrame({
                'Column': st.session_state.data.columns,
                'Type': st.session_state.data.dtypes.astype(str),
                'Non-Null Count': st.session_state.data.count(),
                'Null Count': st.session_state.data.isnull().sum(),
                'Unique Values': st.session_state.data.nunique()
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with tab3:
            st.dataframe(st.session_state.data.describe(), use_container_width=True)
        
        with tab4:
            missing_df = pd.DataFrame({
                'Column': st.session_state.data.columns,
                'Missing Count': st.session_state.data.isnull().sum(),
                'Missing Percentage': (st.session_state.data.isnull().sum() / len(st.session_state.data) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
                
                # Enhanced missing values visualization
                fig = px.bar(
                    missing_df, 
                    x='Column', 
                    y='Missing Percentage',
                    title='Missing Values by Column',
                    labels={'Missing Percentage': 'Missing (%)'},
                    color='Missing Percentage',
                    color_continuous_scale='reds'
                )
                fig.update_layout(
                    font=dict(family='Inter'),
                    title_font_size=20
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                create_status_alert("No missing values found in the dataset!", "success")
        
        with tab5:
            # Correlation heatmap for numeric columns
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = st.session_state.data[numeric_cols].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                fig.update_layout(
                    title="Feature Correlation Matrix",
                    height=600,
                    font=dict(family='Inter'),
                    title_font_size=20
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis")

elif page == "🔧 Preprocessing":
    st.markdown('<h1 class="main-header">🔧 Data Preprocessing</h1>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        create_status_alert("Please upload data first!", "warning")
        st.stop()
    
    if not hasattr(st.session_state, 'target_column'):
        create_status_alert("Please select a target variable in the Data Upload page!", "warning")
        st.stop()
    
    # Preprocessing options with enhanced UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="sub-header">🧹 Data Cleaning</h3>', unsafe_allow_html=True)
        
        # Missing values handling
        missing_strategy = st.selectbox(
            "Missing Values Strategy",
            ["Drop rows with missing values", "Fill with mean", "Fill with median", 
             "Fill with mode", "Forward fill", "Backward fill"],
            help="Choose how to handle missing values in your dataset"
        )
        
        # Outlier detection
        remove_outliers = st.checkbox("Remove outliers (>3 standard deviations)", help="Remove extreme values that might affect model performance")
        
        # Duplicate handling
        remove_duplicates = st.checkbox("Remove duplicate rows", help="Remove identical rows from the dataset")
    
    with col2:
        st.markdown('<h3 class="sub-header">🔄 Feature Engineering</h3>', unsafe_allow_html=True)
        
        # Feature scaling
        scale_features = st.checkbox("Scale numerical features", value=True, 
                                   help="Standardize features to have zero mean and unit variance")
        
        # Categorical encoding
        encoding_method = st.selectbox(
            "Categorical Encoding Method",
            ["One-Hot Encoding", "Label Encoding", "Target Encoding"],
            help="Choose how to convert categorical variables to numeric format"
        )
        
        # Feature selection
        feature_selection = st.checkbox("Automatic feature selection", 
                                      help="Use correlation and importance metrics to select best features")
    
    st.markdown("<hr style='margin: 1rem 0; border: 1px solid #D7CEB2;'/>", unsafe_allow_html=True)
    
    # Apply preprocessing button
    if st.button("🚀 Apply Preprocessing", type="primary"):
        with st.spinner("Processing data..."):
            try:
                # Create a copy of the data
                processed_data = st.session_state.data.copy()
                
                # Handle missing values
                if missing_strategy == "Drop rows with missing values":
                    processed_data = processed_data.dropna()
                elif missing_strategy == "Fill with mean":
                    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                    processed_data[numeric_cols] = processed_data[numeric_cols].fillna(processed_data[numeric_cols].mean())
                elif missing_strategy == "Fill with median":
                    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                    processed_data[numeric_cols] = processed_data[numeric_cols].fillna(processed_data[numeric_cols].median())
                elif missing_strategy == "Fill with mode":
                    for col in processed_data.columns:
                        if processed_data[col].isnull().any():
                            mode_val = processed_data[col].mode()
                            if len(mode_val) > 0:
                                processed_data[col] = processed_data[col].fillna(mode_val[0])
                elif missing_strategy == "Forward fill":
                    processed_data = processed_data.fillna(method='ffill')
                elif missing_strategy == "Backward fill":
                    processed_data = processed_data.fillna(method='bfill')
                
                # Remove duplicates
                if remove_duplicates:
                    processed_data = processed_data.drop_duplicates()
                
                # Store processed data
                st.session_state.processed_data = processed_data
                st.session_state.config.scale_features = scale_features
                st.session_state.config.encode_categorical = encoding_method.lower().replace(" ", "_").replace("-", "_")
                
                create_status_alert("Preprocessing completed successfully!", "success")
                
                # Show before/after comparison with enhanced metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    create_metric_card("Original Rows", f"{len(st.session_state.data):,}")
                with col2:
                    create_metric_card("Processed Rows", f"{len(processed_data):,}")
                with col3:
                    rows_removed = len(st.session_state.data) - len(processed_data)
                    card_type = "warning" if rows_removed > 0 else "success"
                    create_metric_card("Rows Removed", str(rows_removed), None, card_type)
                
            except Exception as e:
                create_status_alert(f"Error during preprocessing: {str(e)}", "error")
    
    # Display processed data info
    if st.session_state.processed_data is not None:
        st.markdown("### 📊 Processed Data Overview")
        
        # Show feature types with enhanced visualization
        features = [col for col in st.session_state.processed_data.columns if col != st.session_state.target_column]
        numeric_features = st.session_state.processed_data[features].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = st.session_state.processed_data[features].select_dtypes(include=['object']).columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            create_metric_card("Total Features", str(len(features)))
        with col2:
            create_metric_card("Numeric Features", str(len(numeric_features)), None, "success")
        with col3:
            create_metric_card("Categorical Features", str(len(categorical_features)), None, "warning")
        
        # Enhanced feature correlation heatmap
        if len(numeric_features) > 1:
            st.markdown("### 🔥 Feature Correlation Heatmap")
            corr_matrix = st.session_state.processed_data[numeric_features + [st.session_state.target_column]].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig.update_layout(
                height=600,
                font=dict(family='Inter'),
                title={
                    'text': "Feature Correlations with Target Variable",
                    'x': 0.5,
                    'font': {'size': 20, 'color': '#2c3e50'}
                }
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Training":
    st.markdown('<h1 class="main-header">🤖 Model Training & Optimization</h1>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        create_status_alert("Please preprocess your data first!", "warning")
        st.stop()
    
    # Model selection with enhanced UI
    st.markdown("### 🎯 Select Models to Train")
    
    col1, col2 = st.columns(2)
    
    with col1:
        models_to_train = st.multiselect(
            "Choose regression models:",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", 
             "Elastic Net", "Random Forest", "Gradient Boosting", "SVR"],
            default=["Linear Regression", "Ridge Regression", "Random Forest"],
            help="Select multiple models to compare their performance"
        )
        
        # Convert to internal names
        model_mapping = {
            "Linear Regression": "linear",
            "Ridge Regression": "ridge",
            "Lasso Regression": "lasso",
            "Elastic Net": "elastic_net",
            "Random Forest": "random_forest",
            "Gradient Boosting": "gradient_boosting",
            "SVR": "svr"
        }
        st.session_state.config.models_to_include = [model_mapping[m] for m in models_to_train]
    
    with col2:
        st.markdown("### ⚙️ Training Configuration")
        
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05, help="Fraction of data to use for testing")
        st.session_state.config.test_size = test_size
        
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5, help="Number of folds for cross-validation")
        st.session_state.config.cv_folds = cv_folds
        
        hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning", value=True, 
                                          help="Automatically find the best parameters for each model")
        st.session_state.config.hyperparameter_tuning = hyperparameter_tuning
    
    # Display expected training time
    if len(models_to_train) > 0:
        estimated_time = len(models_to_train) * (2 if hyperparameter_tuning else 1)
        create_status_alert(f"Estimated training time: ~{estimated_time} minutes", "info")
    
    st.markdown("<hr style='margin: 1rem 0; border: 1px solid #D7CEB2;'/>", unsafe_allow_html=True)
    
    # Train models button
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Initialize workflow
                workflow = EnhancedRegressionWorkflow(config=st.session_state.config)
                st.session_state.workflow = workflow
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run the workflow with stdout suppression to prevent random output
                data_to_use = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
                
                # Capture stdout to prevent random print statements from appearing
                import sys
                from io import StringIO
                
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()
                
                try:
                    workflow_results = workflow.run_complete_workflow(
                        data=data_to_use,
                        target_column=st.session_state.target_column
                    )
                finally:
                    # Restore stdout
                    sys.stdout = old_stdout
                
                # Extract results
                results = workflow_results['results']
                st.session_state.results = results
                st.session_state.workflow_results = workflow_results
                st.session_state.models = {name: res['model'] for name, res in results['model_results'].items()}
                st.session_state.best_model = workflow_results['best_model']
                st.session_state.model_trained = True
                
                progress_bar.progress(100)
                status_text.text("Training completed!")
                
                create_status_alert("Model training completed successfully!", "success")
                
                # Display results summary with enhanced visualization
                st.markdown("### 🏆 Model Performance Summary")
                
                if 'comparison_df' in results and not results['comparison_df'].empty:
                    # Use the enhanced model comparison chart
                    fig = create_model_comparison_chart(results['comparison_df'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Best model info
                    best_model_name = results['best_model_name']
                    best_metrics = results['model_results'][best_model_name]['metrics']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        create_metric_card("Best Model", best_model_name, None, "success")
                    with col2:
                        create_metric_card("R² Score", f"{best_metrics['test_r2']:.4f}", None, "success")
                    with col3:
                        create_metric_card("RMSE", f"{best_metrics['test_rmse']:.4f}", None, "success")
                
            except Exception as e:
                create_status_alert(f"Error during model training: {str(e)}", "error")
                st.exception(e)

elif page == "📊 Results & Insights":
    st.markdown('<h1 class="main-header">📊 Results & Insights</h1>', unsafe_allow_html=True)
    
    if st.session_state.results is None:
        create_status_alert("Please train models first!", "warning")
        st.stop()
    
    # Results tabs with enhanced styling
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Performance", "🎯 Predictions", "🔍 Features", "📊 Residuals", "🔮 Insights"]
    )
    
    with tab1:
        st.markdown("### 📊 Detailed Performance Metrics")
        
        # Enhanced model comparison visualization
        model_names = list(st.session_state.results['model_results'].keys())
        metrics_data = []
        
        for model_name in model_names:
            metrics = st.session_state.results['model_results'][model_name]['metrics']
            metrics_data.append({
                'Model': model_name,
                'R² Score': metrics['test_r2'],
                'RMSE': metrics['test_rmse'],
                'MAE': metrics['test_mae']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create interactive comparison chart
        fig = go.Figure()
        
        # Add R² scores
        fig.add_trace(go.Bar(
            name='R² Score',
            x=metrics_df['Model'],
            y=metrics_df['R² Score'],
            text=[f"{val:.3f}" for val in metrics_df['R² Score']],
            textposition='auto',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            font=dict(family='Inter', size=12),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cross-validation scores
        if 'cross_validation' in st.session_state.results:
            st.markdown("### 📊 Cross-Validation Results")
            cv_data = []
            for model_name, scores in st.session_state.results['cross_validation'].items():
                cv_data.append({
                    'Model': model_name,
                    'Mean CV Score': np.mean(scores),
                    'Std Dev': np.std(scores),
                    'Min Score': np.min(scores),
                    'Max Score': np.max(scores)
                })
            
            cv_df = pd.DataFrame(cv_data).sort_values('Mean CV Score', ascending=False)
            
            # Display with color coding
            st.dataframe(
                cv_df.style.background_gradient(subset=['Mean CV Score'], cmap='RdYlGn'),
                use_container_width=True
            )
    
    with tab2:
        st.markdown("### 🎯 Predictions vs Actual Values")
        
        # Select model to visualize
        selected_model = st.selectbox(
            "Select model to visualize:",
            list(st.session_state.results['model_results'].keys()),
            index=0
        )
        
        if selected_model in st.session_state.results['model_results']:
            model = st.session_state.results['model_results'][selected_model]['model']
            X_test = st.session_state.results.get('X_test')
            y_test = st.session_state.results.get('y_test')
            
            if model is not None and X_test is not None and y_test is not None:
                predictions = model.predict(X_test)
                
                # Enhanced scatter plot
                fig = go.Figure()
                
                # Add scatter plot with color gradient
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=predictions,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        color=np.abs(y_test - predictions),
                        colorscale='earth',
                        size=8,
                        opacity=0.6,
                        colorbar=dict(title="Absolute Error")
                    ),
                    text=[f"Actual: {a:.2f}<br>Predicted: {p:.2f}<br>Error: {abs(a-p):.2f}" 
                          for a, p in zip(y_test, predictions)],
                    hovertemplate='%{text}<extra></extra>'
                ))
                
                # Add perfect prediction line
                min_val = min(y_test.min(), predictions.min())
                max_val = max(y_test.max(), predictions.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                fig.update_layout(
                    title=f'{selected_model} - Predictions vs Actual',
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values',
                    height=500,
                    font=dict(family='Lato')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction error distribution
                errors = y_test - predictions
                fig_hist = px.histogram(
                    x=errors, 
                    nbins=30,
                    title='Prediction Error Distribution',
                    labels={'x': 'Prediction Error', 'y': 'Frequency'},
                    color_discrete_sequence=['#A44A3F']
                )
                fig_hist.update_layout(font=dict(family='Lato'))
                st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔍 Feature Importance Analysis")
        
        if 'feature_importance' in st.session_state.results:
            importance_df = st.session_state.results['feature_importance']
            
            if not importance_df.empty:
                # Use enhanced feature importance chart
                fig = create_feature_importance_chart(importance_df.sort_values('importance', ascending=False))
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance table
                st.markdown("#### Feature Importance Details")
                st.dataframe(
                    importance_df.sort_values('importance', ascending=False).style.background_gradient(subset=['importance'], cmap='Greens'),
                    use_container_width=True
                )
        else:
            create_status_alert("Feature importance is available for tree-based models (Random Forest, Gradient Boosting)", "info")
    
    with tab4:
        st.markdown("### 📊 Residual Analysis")
        
        # Select model for residual analysis
        selected_model_residual = st.selectbox(
            "Select model for residual analysis:",
            list(st.session_state.results['model_results'].keys()),
            key="residual_model"
        )
        
        if selected_model_residual in st.session_state.results['model_results']:
            model = st.session_state.results['model_results'][selected_model_residual]['model']
            X_test = st.session_state.results.get('X_test')
            y_test = st.session_state.results.get('y_test')
            
            if model is not None and X_test is not None and y_test is not None:
                predictions = model.predict(X_test)
                
                # Use enhanced residual analysis plot
                fig = create_residual_analysis_plot(y_test, predictions)
                st.plotly_chart(fig, use_container_width=True)
                
                # Residual statistics
                residuals = y_test - predictions
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    create_metric_card("Mean Residual", f"{np.mean(residuals):.4f}")
                with col2:
                    create_metric_card("Std Residual", f"{np.std(residuals):.4f}")
                with col3:
                    create_metric_card("Min Residual", f"{np.min(residuals):.4f}")
                with col4:
                    create_metric_card("Max Residual", f"{np.max(residuals):.4f}")
    
    with tab5:
        st.markdown("### 🔮 Model Insights & Recommendations")
        
        # Best model summary with enhanced cards
        best_model_name = st.session_state.results['best_model_name']
        best_metrics = st.session_state.results['model_results'][best_model_name]['metrics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Best Model Performance")
            
            # Performance metrics cards
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                create_metric_card("Model", best_model_name, None, "success")
                create_metric_card("R² Score", f"{best_metrics['test_r2']:.4f}", None, "success")
            with subcol2:
                create_metric_card("RMSE", f"{best_metrics['test_rmse']:.4f}", None, "success")
                create_metric_card("MAE", f"{best_metrics['test_mae']:.4f}", None, "success")
            
            # Interpretation
            st.info(f"""
            **Model Interpretation:**
            - The model explains **{best_metrics['test_r2']*100:.1f}%** of the variance in {st.session_state.target_column}
            - On average, predictions are off by **{best_metrics['test_mae']:.2f}** units
            - The root mean squared error is **{best_metrics['test_rmse']:.2f}**
            """)
        
        with col2:
            st.markdown("#### 💡 Recommendations")
            
            # Generate intelligent recommendations
            recommendations = []
            
            # Performance-based recommendations
            if best_metrics['test_r2'] < 0.5:
                recommendations.append(("⚠️ Low R² Score", "Consider adding more features or trying non-linear models", "warning"))
            elif best_metrics['test_r2'] < 0.7:
                recommendations.append(("📊 Moderate R² Score", "Try feature engineering or ensemble methods", "info"))
            else:
                recommendations.append(("✅ Good R² Score", "Model is performing well", "success"))
            
            # Model-specific recommendations
            if best_model_name in ['random_forest', 'gradient_boosting']:
                recommendations.append(("🌳 Tree-Based Model", "Good for capturing non-linear relationships", "info"))
            elif best_model_name in ['ridge', 'lasso', 'elastic_net']:
                recommendations.append(("📏 Regularized Model", "Good for preventing overfitting", "info"))
            
            # Display recommendations
            for title, desc, alert_type in recommendations:
                create_status_alert(f"{title}: {desc}", alert_type)

elif page == "💾 Export & Deploy":
    st.markdown('<h1 class="main-header">💾 Export & Deploy</h1>', unsafe_allow_html=True)
    
    if st.session_state.best_model is None:
        create_status_alert("Please train models first!", "warning")
        st.stop()
    
    st.markdown("### 📦 Model Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💾 Save Trained Model")
        
        model_name = st.text_input("Model name:", value="regression_model", help="Name for your exported model")
        
        if st.button("Export Model Package", type="primary"):
            try:
                # Create export directory
                export_dir = Path("model_exports")
                export_dir.mkdir(exist_ok=True)
                
                # Save model
                model_path = export_dir / f"{model_name}.pkl"
                joblib.dump(st.session_state.best_model, model_path)
                
                # Save metadata
                metadata = {
                    "model_type": st.session_state.results['best_model_name'],
                    "performance_metrics": st.session_state.results['model_results'][st.session_state.results['best_model_name']]['metrics'],
                    "features": list(st.session_state.workflow.feature_columns),
                    "target_column": st.session_state.target_column,
                    "training_date": datetime.now().isoformat(),
                    "config": asdict(st.session_state.config)
                }
                
                # Custom JSON encoder for metadata
                def json_serializable(obj):
                    """Convert numpy/pandas types to JSON serializable formats."""
                    if isinstance(obj, (np.integer, np.int64, np.int32)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64, np.float32)):
                        return float(obj)
                    elif isinstance(obj, (np.bool_, np.bool8)):
                        return bool(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif hasattr(obj, '__class__') and 'dtype' in obj.__class__.__name__.lower():
                        return str(obj)
                    elif hasattr(obj, 'to_dict'):
                        return obj.to_dict()
                    elif hasattr(obj, 'to_list'):
                        return obj.to_list()
                    else:
                        return str(obj)
                
                metadata_path = export_dir / f"{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=json_serializable)
                
                create_status_alert(f"Model exported successfully to {model_path}", "success")
                
                # Offer download
                with open(model_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Model",
                        data=f.read(),
                        file_name=f"{model_name}.pkl",
                        mime="application/octet-stream"
                    )
                
            except Exception as e:
                create_status_alert(f"Error exporting model: {str(e)}", "error")
    
    with col2:
        st.markdown("#### 📊 Export Results Report")
        
        if st.button("Generate Comprehensive Report", type="primary"):
            try:
                # Create comprehensive report
                report = {
                    "Executive Summary": {
                        "Best Model": st.session_state.results['best_model_name'],
                        "R² Score": st.session_state.results['model_results'][st.session_state.results['best_model_name']]['metrics']['test_r2'],
                        "RMSE": st.session_state.results['model_results'][st.session_state.results['best_model_name']]['metrics']['test_rmse'],
                        "MAE": st.session_state.results['model_results'][st.session_state.results['best_model_name']]['metrics']['test_mae']
                    },
                    "All Models Performance": {name: res['metrics'] for name, res in st.session_state.results['model_results'].items()},
                    "Cross Validation Results": st.session_state.results.get('cross_validation', {}),
                    "Dataset Information": {
                        "Total Samples": len(st.session_state.processed_data),
                        "Features": len(st.session_state.workflow.feature_columns),
                        "Target Variable": st.session_state.target_column
                    },
                    "Configuration": asdict(st.session_state.config)
                }
                
                # Custom JSON encoder to handle numpy/pandas types
                def json_serializable(obj):
                    """Convert numpy/pandas types to JSON serializable formats."""
                    if isinstance(obj, (np.integer, np.int64, np.int32)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64, np.float32)):
                        return float(obj)
                    elif isinstance(obj, (np.bool_, np.bool8)):
                        return bool(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif hasattr(obj, '__class__') and 'dtype' in obj.__class__.__name__.lower():
                        return str(obj)
                    elif hasattr(obj, 'to_dict'):
                        return obj.to_dict()
                    elif hasattr(obj, 'to_list'):
                        return obj.to_list()
                    else:
                        return str(obj)
                
                # Convert to JSON
                report_json = json.dumps(report, indent=2, default=json_serializable)
                
                st.download_button(
                    label="📥 Download Report (JSON)",
                    data=report_json,
                    file_name=f"regression_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                create_status_alert("Report generated successfully!", "success")
                
            except Exception as e:
                create_status_alert(f"Error generating report: {str(e)}", "error")
    
    st.markdown("<hr style='margin: 1rem 0; border: 1px solid #D7CEB2;'/>", unsafe_allow_html=True)
    
    # Deployment code section
    st.markdown("### 🚀 Deployment Code")
    
    with st.expander("View Deployment Script", expanded=True):
        deployment_code = f"""
# Deployment script for {st.session_state.results['best_model_name']}
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('{model_name}.pkl')

# Define feature names
feature_names = {st.session_state.workflow.feature_columns}

def predict(data):
    '''
    Make predictions using the trained model.
    
    Parameters:
    -----------
    data : dict or DataFrame
        Input data with features matching the training set
    
    Returns:
    --------
    prediction : float or array
        Model predictions
    '''
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data
    
    # Ensure all features are present
    df = df[feature_names]
    
    # Make prediction
    prediction = model.predict(df)
    
    return prediction

# Example usage
if __name__ == "__main__":
    sample_data = {{
        {', '.join([f'"{feat}": 0' for feat in st.session_state.workflow.feature_columns[:3]])}
        # Add more features as needed
    }}
    
    result = predict(sample_data)
    print(f"Prediction: {{result}}")
"""
        
        st.code(deployment_code, language='python')
        
        st.download_button(
            label="📥 Download Deployment Script",
            data=deployment_code,
            file_name=f"{model_name}_deploy.py",
            mime="text/plain"
        )

elif page == "🔮 Make Predictions":
    st.markdown('<h1 class="main-header">🔮 Make Predictions</h1>', unsafe_allow_html=True)
    
    if st.session_state.best_model is None:
        create_status_alert("Please train models first!", "warning")
        st.stop()
    
    st.markdown("### 🎯 Enter Values for Prediction")
    
    # Create input form with enhanced UI
    prediction_data = {}
    
    # Use original feature names (exclude target column)
    original_features = [col for col in st.session_state.data.columns if col != st.session_state.target_column]
    num_cols = 3
    cols = st.columns(num_cols)
    
    for idx, feature in enumerate(original_features):
        col_idx = idx % num_cols
        with cols[col_idx]:
            # Get feature statistics from original data
            try:
                if pd.api.types.is_numeric_dtype(st.session_state.data[feature]):
                    min_val = float(st.session_state.data[feature].min())
                    max_val = float(st.session_state.data[feature].max())
                    mean_val = float(st.session_state.data[feature].mean())
                    
                    prediction_data[feature] = st.number_input(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        help=f"Range: {min_val:.2f} - {max_val:.2f}",
                        key=f"pred_{feature}"
                    )
                else:
                    # For categorical features, show selectbox with unique values
                    unique_values = st.session_state.data[feature].unique()
                    prediction_data[feature] = st.selectbox(
                        feature,
                        options=unique_values,
                        key=f"pred_{feature}"
                    )
            except:
                prediction_data[feature] = st.number_input(
                    feature,
                    value=0.0,
                    key=f"pred_{feature}"
                )
    
    st.markdown("<hr style='margin: 1rem 0; border: 1px solid #D7CEB2;'/>", unsafe_allow_html=True)
    
    # Make prediction button
    if st.button("🔮 Make Prediction", type="primary"):
        try:
            # Create DataFrame with original feature structure
            pred_df = pd.DataFrame([prediction_data])
            
            # Apply the same preprocessing as during training
            if hasattr(st.session_state.workflow, 'data_processor'):
                # Process data through the same pipeline as training
                processed_pred_df = st.session_state.workflow.data_processor.transform_features(
                    pred_df, 
                    target_column=None,  # No target for predictions
                    is_training=False
                )[0]  # Get features only
                
                # Ensure we have all the columns that the model expects
                expected_features = st.session_state.workflow.feature_columns
                
                # Add missing columns with zeros
                for col in expected_features:
                    if col not in processed_pred_df.columns:
                        processed_pred_df[col] = 0
                
                # Select only the expected features in the correct order
                processed_pred_df = processed_pred_df[expected_features]
            else:
                # Fallback: assume data is already processed correctly
                processed_pred_df = pred_df
            
            # Make prediction
            prediction = st.session_state.best_model.predict(processed_pred_df)[0]
            
            # Display result with enhanced UI
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                create_metric_card(
                    f"Predicted {st.session_state.target_column}", 
                    f"{prediction:.4f}", 
                    None, 
                    "success"
                )
            
            # Compare with training data statistics
            target_mean = st.session_state.data[st.session_state.target_column].mean()
            target_std = st.session_state.data[st.session_state.target_column].std()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                create_metric_card("Prediction", f"{prediction:.2f}")
            with col2:
                create_metric_card("Training Mean", f"{target_mean:.2f}")
            with col3:
                deviation = (prediction - target_mean) / target_std
                card_type = "success" if abs(deviation) < 1 else "warning"
                create_metric_card("Std Deviations", f"{deviation:.2f}", None, card_type)
            
        except Exception as e:
            create_status_alert(f"Error making prediction: {str(e)}", "error")
    
    st.markdown("<hr style='margin: 1rem 0; border: 1px solid #D7CEB2;'/>", unsafe_allow_html=True)
    
    # Batch predictions section
    st.markdown("### 📊 Batch Predictions")
    
    uploaded_prediction_file = st.file_uploader(
        "Upload CSV for batch predictions",
        type=['csv'],
        key="batch_pred",
        help="Upload a CSV file with the same features as the training data"
    )
    
    if uploaded_prediction_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_prediction_file)
            create_status_alert(f"Loaded {len(batch_df)} rows for prediction", "info")
            
            if st.button("Make Batch Predictions", key="batch_predict"):
                # Check features
                missing_features = set(original_features) - set(batch_df.columns)
                if missing_features:
                    create_status_alert(f"Missing features: {missing_features}", "error")
                else:
                    # Apply preprocessing
                    if hasattr(st.session_state.workflow, 'data_processor'):
                        # Process data through the same pipeline as training
                        batch_processed = st.session_state.workflow.data_processor.transform_features(
                            batch_df[original_features], 
                            target_column=None,  # No target for predictions
                            is_training=False
                        )[0]  # Get features only
                        
                        # Ensure we have all the columns that the model expects
                        expected_features = st.session_state.workflow.feature_columns
                        
                        # Add missing columns with zeros
                        for col in expected_features:
                            if col not in batch_processed.columns:
                                batch_processed[col] = 0
                        
                        # Select only the expected features in the correct order
                        batch_processed = batch_processed[expected_features]
                    else:
                        # Fallback: assume data is already processed correctly
                        batch_processed = batch_df[original_features]
                    
                    # Make predictions
                    predictions = st.session_state.best_model.predict(batch_processed)
                    
                    # Add predictions to dataframe
                    result_df = batch_df.copy()
                    result_df[f'predicted_{st.session_state.target_column}'] = predictions
                    
                    # Display results
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        create_metric_card("Predictions Made", str(len(predictions)))
                    with col2:
                        create_metric_card("Mean Prediction", f"{np.mean(predictions):.2f}")
                    with col3:
                        create_metric_card("Std Prediction", f"{np.std(predictions):.2f}")
                    
                    # Download results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            create_status_alert(f"Error processing batch predictions: {str(e)}", "error")

# Footer with enhanced styling
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 2rem 0;'>
        <p style='color: #A44A3F; font-family: "Lato", sans-serif; font-weight: 500;'>Built with ❤️ using Streamlit | Linear Regression Analysis Suite v2.0</p>
        <p style='color: #A59E8C; font-size: 0.9rem; font-family: "Lato", sans-serif;'>Enhanced UI/UX Edition</p>
    </div>
    """,
    unsafe_allow_html=True
)