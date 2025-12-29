"""
CREDIT CARD FRAUD DETECTION PIPELINE - PROFESSIONAL DASHBOARD
=============================================================
Professional dashboard for fraud detection analysis

Author: Dylan Fernandez
Date: December 15, 2025
Course: Advanced Programming 2025

How to run:
    streamlit run dashboard_clean.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional

# ==================== CONFIGURATION & CONSTANTS ====================

class Colors:
    """Professional color palette"""
    FRAUD = "#e74c3c"
    LEGITIMATE = "#2ecc71"
    SUPERVISED = "#2C3E50"
    UNSUPERVISED = "#3498db"
    SEMI_SUPERVISED = "#9b59b6"
    ENSEMBLE = "#34495e"
    TURQUOISE = "#45D9E8"  # Added turquoise
    TEAL = "#2E8B8B"       # Added teal
    BACKGROUND = "#ffffff"
    CARD_BG = "#f8f9fa"
    TEXT_PRIMARY = "#2c3e50"
    TEXT_SECONDARY = "#7f8c8d"
    ACCENT_PRIMARY = "#3498db"

class DatasetInfo:
    """Dataset statistics"""
    TOTAL_TRANSACTIONS = 283_726
    TOTAL_FRAUD = 492
    TOTAL_LEGITIMATE = 283_234
    FRAUD_RATE = 0.17
    TRAIN_SIZE = 181_584
    VAL_SIZE = 45_396
    TEST_SIZE = 56_746
    TOTAL_FEATURES = 54
    ORIGINAL_FEATURES = 30
    ENGINEERED_FEATURES = 24


# Feature definitions
ORIGINAL_FEATURES_LIST = [
    'Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'
]

ENGINEERED_FEATURES_LIST = [
    # Temporal features (7)
    'Hour', 'Minute', 'Second', 'Hour_sin', 'Hour_cos', 'Minute_sin', 'Minute_cos',
    # Amount features (6)
    'Amount_log', 'Amount_bin', 'Amount_bin_freq', 'Amount_zscore', 'Amount_hour_mean', 'Amount_hour_std',
    # Time-based features (5)
    'Time_of_day', 'Time_freq', 'Time_hour_mean', 'Time_hour_std', 'Time_zscore',
    # Interactions (6)
    'Amount_Time_interaction', 'V1_Amount', 'V2_Amount', 'V3_Amount', 'V4_Amount', 'Hour_Amount_interaction'
]

KEY_ENGINEERED_FEATURES = [
    {'name': 'Hour_sin / Hour_cos', 'type': 'Temporal', 'description': 'Cyclical encoding of transaction hour'},
    {'name': 'Amount_log', 'type': 'Amount', 'description': 'Log transformation to handle skewness'},
    {'name': 'Amount_zscore', 'type': 'Amount', 'description': 'Standardized amount within time periods'},
    {'name': 'Time_of_day', 'type': 'Temporal', 'description': 'Categorical: Morning/Afternoon/Evening/Night'},
    {'name': 'Amount_bin_freq', 'type': 'Amount', 'description': 'Frequency of transactions in amount bins'},
    {'name': 'Hour_Amount_interaction', 'type': 'Interaction', 'description': 'Product of hour and amount features'}
]

ENSEMBLE_COMPOSITION = {
    'Random Forest': 42.25,
    'Logistic Regression': 38.94,
    'LOF Semi-Supervised': 18.81
}

METRIC_DEFINITIONS = {
    'F1-Score': 'Harmonic mean of Precision and Recall. Balances both metrics equally.',
    'Precision': 'Proportion of predicted frauds that are actually frauds (TP / (TP + FP)).',
    'Recall': 'Proportion of actual frauds that are detected (TP / (TP + FN)).',
    'ROC-AUC': 'Area Under the ROC Curve. Measures model ability to distinguish classes.',
    'Validation Calibration': 'Using validation set for all decisions to prevent test leakage.',
    'Test Leakage': 'Using test data for model decisions (threshold, weights) causing overfitting.',
    'SMOTE': 'Synthetic Minority Over-sampling Technique. Generates synthetic fraud samples.',
    'Class Weighting': 'Assigns higher loss weights to minority class during training.',
}

PERFORMANCE_DATA = pd.DataFrame({
    'Model': [
        'Logistic Regression', 'Random Forest',
        'Isolation Forest', 'Local Outlier Factor', 'Gaussian Mixture',
        'IF Semi-Supervised', 'LOF Semi-Supervised', 'GMM Semi-Supervised',
        'Ensemble (Top 3)'
    ],
    'Paradigm': [
        'Supervised', 'Supervised',
        'Unsupervised', 'Unsupervised', 'Unsupervised',
        'Semi-Supervised', 'Semi-Supervised', 'Semi-Supervised',
        'Ensemble'
    ],
    'Fraud_Precision': [0.78, 0.85, 0.02, 0.07, 0.01, 0.13, 0.12, 0.26, 0.45],
    'Fraud_Recall': [0.64, 0.72, 0.43, 0.18, 0.30, 0.47, 0.23, 0.36, 0.64],
    'Fraud_F1': [0.70, 0.78, 0.04, 0.10, 0.02, 0.20, 0.16, 0.31, 0.53],
    'Macro_F1': [0.85, 0.89, 0.52, 0.55, 0.50, 0.60, 0.58, 0.65, 0.76],
    'ROC_AUC': [0.980, 0.962, 0.870, 0.890, 0.764, 0.939, 0.884, 0.958, 0.933]
})

BEST_MODEL = {
    'name': 'Random Forest',
    'paradigm': 'Supervised',
    'fraud_f1': 0.78,
    'fraud_precision': 0.85,
    'fraud_recall': 0.72,
    'macro_f1': 0.89,
    'roc_auc': 0.962,
    'why': 'Highest fraud F1-score with strong precision-recall balance'
}

KEY_FINDINGS = [
    "Supervised models outperform unsupervised/semi-supervised significantly",
    "Random Forest achieves best fraud detection (F1=0.78, Precision=0.85)",
    "Semi-supervised GMM improves 15x over unsupervised version (0.02 → 0.31)",
    "Ensemble provides robustness but trades peak performance for stability",
    "ROC-AUC misleading for extreme imbalance (high AUC, poor fraud detection)"
]

LIMITATIONS = [
    "SMOTE uses labels → not truly unsupervised",
    "Single dataset limits generalizability",
    "No temporal drift evaluation",
    "Computational cost for IF/LOF requires subsampling (30%)"
]

# ==================== HELPER FUNCTIONS ====================

def format_number(value: int) -> str:
    return f"{value:,}"

def format_percentage(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"

def get_model_by_paradigm(paradigm: str) -> pd.DataFrame:
    if paradigm == "All":
        return PERFORMANCE_DATA
    return PERFORMANCE_DATA[PERFORMANCE_DATA['Paradigm'] == paradigm]

def get_top_n_models(n: int = 3, metric: str = 'Fraud_F1') -> pd.DataFrame:
    return PERFORMANCE_DATA.nlargest(n, metric)

def calculate_expected_cost(precision: float, recall: float, 
                           fraud_rate: float = 0.0017, 
                           total_transactions: int = 56746) -> dict:
    FP_COST = 1
    FN_COST = 10
    
    total_fraud = int(total_transactions * fraud_rate)
    total_legit = total_transactions - total_fraud
    
    TP = int(recall * total_fraud)
    FN = total_fraud - TP
    
    if precision > 0:
        FP = int(TP * (1 - precision) / precision)
    else:
        FP = total_legit
    
    total_cost = FP * FP_COST + FN * FN_COST
    
    return {'TP': TP, 'FP': FP, 'TN': total_legit - FP, 'FN': FN, 'total_cost': total_cost}

# ==================== UI COMPONENTS ====================

def kpi_card(label: str, value: str, delta: Optional[str] = None, 
             delta_color: str = "normal", card_type: str = "default"):
    card_class = {
        'default': 'kpi-card',
        'fraud': 'kpi-card kpi-card-fraud',
        'legitimate': 'kpi-card kpi-card-legitimate',
        'warning': 'kpi-card kpi-card-warning'
    }.get(card_type, 'kpi-card')
    
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)
    st.markdown('</div>', unsafe_allow_html=True)

def section_header(title: str, subtitle: Optional[str] = None, divider: bool = True):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<p class="text-muted mb-2">{subtitle}</p>', unsafe_allow_html=True)
    if divider:
        st.markdown("---")

def info_card(title: str, content: str, card_type: str = "info"):
    if card_type == "info":
        st.info(f"**{title}**\n\n{content}")
    elif card_type == "success":
        st.success(f"**{title}**\n\n{content}")
    elif card_type == "warning":
        st.warning(f"**{title}**\n\n{content}")
    elif card_type == "error":
        st.error(f"**{title}**\n\n{content}")

def model_card(model_name: str, paradigm: str, f1_score: float, 
               precision: float, recall: float, auc: float, color: str = "#3498db"):
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {color};
            padding: 1rem;
            background-color: white;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        ">
            <h4 style="margin-top: 0; color: #2c3e50;">{model_name}</h4>
            <p style="color: #7f8c8d; font-size: 0.875rem; margin-bottom: 0.5rem;">
                <em>{paradigm}</em>
            </p>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem;">
                <div>
                    <p style="margin: 0; font-size: 0.75rem; color: #95a5a6;">F1</p>
                    <p style="margin: 0; font-weight: 600; color: #2c3e50;">{f1_score:.2f}</p>
                </div>
                <div>
                    <p style="margin: 0; font-size: 0.75rem; color: #95a5a6;">Precision</p>
                    <p style="margin: 0; font-weight: 600; color: #2c3e50;">{precision:.2f}</p>
                </div>
                <div>
                    <p style="margin: 0; font-size: 0.75rem; color: #95a5a6;">Recall</p>
                    <p style="margin: 0; font-weight: 600; color: #2c3e50;">{recall:.2f}</p>
                </div>
                <div>
                    <p style="margin: 0; font-size: 0.75rem; color: #95a5a6;">AUC</p>
                    <p style="margin: 0; font-weight: 600; color: #2c3e50;">{auc:.3f}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================== CHART FUNCTIONS ====================

def create_f1_comparison_chart(df: pd.DataFrame, metric: str = 'Fraud_F1') -> go.Figure:
    df_sorted = df.sort_values(metric, ascending=True)
    
    color_map = {
        'Supervised': Colors.SUPERVISED,
        'Unsupervised': Colors.UNSUPERVISED,
        'Semi-Supervised': Colors.SEMI_SUPERVISED,
        'Ensemble': Colors.ENSEMBLE
    }
    
    colors = [color_map.get(p, '#95a5a6') for p in df_sorted['Paradigm']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted['Model'],
        x=df_sorted[metric],
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=df_sorted[metric].apply(lambda x: f'{x:.2f}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>F1: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Model Comparison: {metric.replace("_", " ")}',
        xaxis=dict(title=metric.replace('_', ' '), gridcolor='#ecf0f1', showgrid=True),
        yaxis=dict(title='', tickfont=dict(size=11)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_precision_recall_scatter(df: pd.DataFrame) -> go.Figure:
    color_map = {
        'Supervised': Colors.SUPERVISED,
        'Unsupervised': Colors.UNSUPERVISED,
        'Semi-Supervised': Colors.SEMI_SUPERVISED,
        'Ensemble': Colors.ENSEMBLE
    }
    
    fig = go.Figure()
    
    for paradigm in df['Paradigm'].unique():
        subset = df[df['Paradigm'] == paradigm]
        fig.add_trace(go.Scatter(
            x=subset['Fraud_Recall'],
            y=subset['Fraud_Precision'],
            mode='markers+text',
            name=paradigm,
            text=subset['Model'].str[:10],
            textposition='top center',
            textfont=dict(size=9),
            marker=dict(size=12, color=color_map.get(paradigm, '#95a5a6'), 
                       line=dict(color='white', width=2)),
            hovertemplate='<b>%{text}</b><br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Precision vs Recall Trade-off (Fraud Detection)',
        xaxis=dict(title='Recall', range=[0, 1], gridcolor='#ecf0f1', showgrid=True),
        yaxis=dict(title='Precision', range=[0, 1], gridcolor='#ecf0f1', showgrid=True),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

def create_class_distribution_chart(fraud_count: int, legitimate_count: int, 
                                    use_log_scale: bool = False) -> go.Figure:
    """Create class distribution bar chart"""
    total = fraud_count + legitimate_count
    fraud_pct = (fraud_count / total) * 100
    legit_pct = (legitimate_count / total) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Legitimate', 'Fraud'],
        y=[legitimate_count, fraud_count],
        marker=dict(color=[Colors.LEGITIMATE, Colors.FRAUD], 
                   line=dict(color='white', width=2)),
        text=[
            f'{legitimate_count:,}<br>({legit_pct:.2f}%)',
            f'{fraud_count:,}<br>({fraud_pct:.3f}%)'
        ],
        textposition=['inside', 'outside'],  # inside pour grande barre, outside pour petite
        insidetextanchor='middle',
        textfont=dict(color=['white', '#2c3e50'], size=12),
        hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Class Distribution: Extreme Imbalance',
        xaxis=dict(title='Class', tickfont=dict(size=12)),
        yaxis=dict(title='Number of Transactions', 
                  type='log' if use_log_scale else 'linear',
                  gridcolor='#ecf0f1', showgrid=True),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        showlegend=False,
        annotations=[
            dict(
                text=f'Imbalance Ratio: 1:{legitimate_count/fraud_count:.0f}',
                xref='paper', yref='paper', x=0.5, y=-0.12,
                showarrow=False, font=dict(size=12, color='#7f8c8d')
            )
        ],
        margin=dict(l=60, r=60, t=80, b=140)
    )
    
    return fig

def create_ensemble_composition_chart(ensemble_dict: Dict[str, float]) -> go.Figure:
    """Create PIE chart with exact colors from screenshot"""
    # Exact colors: Dark blue, Turquoise, Teal
    colors_pie = ['#2C3E50', '#45D9E8', '#2E8B8B']
    
    labels = list(ensemble_dict.keys())
    values = list(ensemble_dict.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors_pie, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textposition='inside',
        textfont=dict(size=11, color='white'),
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title='F1-Weighted Ensemble Composition',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02
        )
    )
    
    return fig

def create_cost_comparison_chart(models: List[str], costs: List[float], 
                                 paradigms: List[str]) -> go.Figure:
    color_map = {
        'Supervised': Colors.SUPERVISED,
        'Unsupervised': Colors.UNSUPERVISED,
        'Semi-Supervised': Colors.SEMI_SUPERVISED,
        'Ensemble': Colors.ENSEMBLE
    }
    
    colors = [color_map.get(p, '#95a5a6') for p in paradigms]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=costs,
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=[f'${c:.0f}' for c in costs],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Expected Cost: $%{y:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Expected Cost Comparison (FN cost = 10x FP cost)',
        xaxis=dict(title='Model', tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(title='Expected Cost ($)', gridcolor='#ecf0f1', showgrid=True),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        showlegend=False
    )
    
    return fig

# ==================== CSS STYLING ====================

def get_custom_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #2c3e50;
        }
        
        .main { background-color: #f8f9fa; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        
        h1 {
            color: #2c3e50;
            font-weight: 700;
            font-size: 2.5rem;
            letter-spacing: -0.02em;
            margin-bottom: 1rem;
            border-bottom: 3px solid #45D9E8;
            padding-bottom: 0.5rem;
        }
        
        h2 {
            color: #34495e;
            font-weight: 600;
            font-size: 1.75rem;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        h3 {
            color: #34495e;
            font-weight: 600;
            font-size: 1.25rem;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }
        
        [data-testid="stSidebar"] {
            background-color: #2c3e50;
            padding-top: 2rem;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #ecf0f1;
        }
        
        [data-testid="stSidebar"] .row-widget.stRadio > div label {
            color: #ecf0f1 !important;
            font-weight: 500;
            font-size: 0.95rem;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: all 0.2s ease;
        }
        
        [data-testid="stSidebar"] .row-widget.stRadio > div label:hover {
            background-color: rgba(69, 217, 232, 0.2);
            color: #45D9E8 !important;
        }
        
        [data-testid="stSidebar"] .row-widget.stRadio > div[data-baseweb="radio"] > div:first-child {
            background-color: #45D9E8;
        }
        
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {
            color: #bdc3c7;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #2c3e50;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.875rem;
            font-weight: 500;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .kpi-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 1.25rem;
            border-radius: 8px;
            border-left: 4px solid #45D9E8;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .kpi-card-fraud { border-left-color: #e74c3c; }
        .kpi-card-legitimate { border-left-color: #2ecc71; }
        .kpi-card-warning { border-left-color: #f39c12; }
        
        .stAlert {
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #45D9E8;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .dataframe {
            font-size: 0.9rem;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .dataframe thead th {
            background-color: #34495e;
            color: white;
            font-weight: 600;
            padding: 0.75rem;
        }
        
        .dataframe tbody tr:nth-child(even) { background-color: #f8f9fa; }
        .dataframe tbody tr:hover { background-color: #e9ecef; }
        
        .stButton > button {
            background: linear-gradient(135deg, #45D9E8 0%, #2E8B8B 100%);
            color: white;
            font-weight: 600;
            border-radius: 6px;
            border: none;
            padding: 0.5rem 2rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(69, 217, 232, 0.3);
        }
        
        .text-muted { color: #7f8c8d; }
        .mb-2 { margin-bottom: 1rem; }
        
        hr {
            margin: 2rem 0;
            border: none;
            border-top: 2px solid #dee2e6;
        }
    </style>
    """

st.set_page_config(
    page_title="Fraud Detection Pipeline",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(get_custom_css(), unsafe_allow_html=True)

# ==================== SIDEBAR ====================

st.sidebar.title("Fraud Detection")
st.sidebar.markdown("**Credit Card Fraud Detection Pipeline**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "Home",
        "Executive Summary",
        "Dataset Overview",
        "Methodology",
        "Models Evaluated",
        "Performance Results",
        "Technical Details"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
st.sidebar.metric("Total Transactions", format_number(DatasetInfo.TOTAL_TRANSACTIONS))
st.sidebar.metric("Fraud Rate", f"{DatasetInfo.FRAUD_RATE}%")
st.sidebar.metric("Features", "54 (30 original + 24 engineered)")
st.sidebar.metric("Best F1-Score", f"{BEST_MODEL['fraud_f1']:.2f}")

st.sidebar.markdown("---")
st.sidebar.info("**Advanced Programming 2025**\n\nDylan Fernandez")

# Default filter values (filters removed from UI but variables needed in code)
paradigm_filter = "All"
show_details = False

# ==================== PAGES ====================

if page == "Home":
    # Simple elegant home page
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Title
    st.markdown("""
    <h1 style='
        text-align: center;
        color: #2c3e50;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    '>
        Credit Card Fraud Detection Pipeline
    </h1>
    """, unsafe_allow_html=True)
    
    # Spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Full subtitle with two parts
    st.markdown("""
    <div style='text-align: center; margin: 0 auto 3rem auto; max-width: 900px;'>
        <p style='
            color: #2c3e50;
            font-size: 1.3rem;
            font-weight: 500;
            margin: 0 0 0.5rem 0;
            padding: 0 2rem;
            line-height: 1.5;
        '>
            Credit Card Fraud Detection Using Machine Learning:
        </p>
        <p style='
            color: #5dade2;
            font-size: 1.2rem;
            font-weight: 400;
            font-style: italic;
            letter-spacing: 0.3px;
            line-height: 1.6;
            margin: 0;
            padding: 0 2rem;
        '>
            A Comparative Study of Supervised, Unsupervised,<br>and Semi-Supervised Approaches
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

elif page == "Executive Summary":
    st.title("Executive Summary")
    st.markdown("**Credit Card Fraud Detection: A Comparative Analysis of Learning Paradigms**")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Best Model", BEST_MODEL['name'], card_type="default")
    with col2:
        kpi_card("Fraud F1-Score", f"{BEST_MODEL['fraud_f1']:.2f}", 
                delta="+11% vs Ensemble", card_type="default")
    with col3:
        kpi_card("Precision", f"{BEST_MODEL['fraud_precision']:.2f}", 
                card_type="legitimate")
    with col4:
        kpi_card("Recall", f"{BEST_MODEL['fraud_recall']:.2f}", card_type="fraud")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        section_header("Project Objective", divider=False)
        st.markdown(f"""
        Detect fraudulent credit card transactions in a highly imbalanced dataset 
        ({format_percentage(DatasetInfo.FRAUD_RATE/100, 2)} fraud rate) by comparing **supervised**, 
        **unsupervised**, and **semi-supervised** learning paradigms.
        
        **Primary Challenge:** Extreme class imbalance (1:{DatasetInfo.TOTAL_LEGITIMATE // DatasetInfo.TOTAL_FRAUD} ratio)
        
        **Key Constraint:** False Negatives (missed fraud) cost significantly more than False Positives
        """)
        
        section_header("Key Findings", divider=False)
        for i, finding in enumerate(KEY_FINDINGS, 1):
            st.markdown(f"{i}. {finding}")
    
    with col2:
        section_header("Best Model", divider=False)
        model_card(
            BEST_MODEL['name'], BEST_MODEL['paradigm'],
            BEST_MODEL['fraud_f1'], BEST_MODEL['fraud_precision'],
            BEST_MODEL['fraud_recall'], BEST_MODEL['roc_auc'],
            color=Colors.SUPERVISED
        )
        st.markdown(f"**Why this model?**\n\n{BEST_MODEL['why']}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Recommendations", divider=False)
        info_card("Production Deployment", 
                 "Random Forest (supervised) with class weighting", 
                 card_type="success")
        info_card("Limited Labels Scenario", 
                 "GMM Semi-Supervised or IF Semi-Supervised", 
                 card_type="info")
        info_card("Metric Priority", 
                 "Prioritize Recall > Precision (missing fraud costs more)", 
                 card_type="warning")
    
    with col2:
        section_header("Limitations", divider=False)
        for limitation in LIMITATIONS:
            st.markdown(f"- {limitation}")
        st.markdown("")
        info_card("Important Note",
                 "SMOTE uses labels for resampling. Therefore, 'unsupervised' models "
                 "are not strictly unsupervised. This is documented transparently.",
                 card_type="warning")
    
    if show_details:
        st.markdown("---")
        section_header("Metric Definitions", divider=False)
        col1, col2 = st.columns(2)
        with col1:
            for metric in ['F1-Score', 'Precision', 'Recall', 'ROC-AUC']:
                with st.expander(f"{metric}"):
                    st.markdown(METRIC_DEFINITIONS[metric])
        with col2:
            for metric in ['Validation Calibration', 'Test Leakage', 'SMOTE', 'Class Weighting']:
                with st.expander(f"{metric}"):
                    st.markdown(METRIC_DEFINITIONS[metric])

elif page == "Dataset Overview":
    st.title("Dataset Overview")
    st.markdown("**Kaggle Credit Card Fraud Detection Dataset**")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Transactions", format_number(DatasetInfo.TOTAL_TRANSACTIONS))
    with col2:
        kpi_card("Fraudulent", format_number(DatasetInfo.TOTAL_FRAUD), card_type="fraud")
    with col3:
        kpi_card("Legitimate", format_number(DatasetInfo.TOTAL_LEGITIMATE), card_type="legitimate")
    with col4:
        kpi_card("Fraud Rate", f"{DatasetInfo.FRAUD_RATE}%", card_type="warning")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        section_header("Class Distribution", "Extreme imbalance visualization")
        use_log = st.checkbox("Use logarithmic scale", value=False)
        fig_dist = create_class_distribution_chart(
            DatasetInfo.TOTAL_FRAUD, DatasetInfo.TOTAL_LEGITIMATE, use_log_scale=use_log
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        section_header("Challenge", divider=False)
        info_card("Extreme Imbalance",
                 f"""
                 **Imbalance Ratio:** 1:{DatasetInfo.TOTAL_LEGITIMATE // DatasetInfo.TOTAL_FRAUD}
                 
                 - Naive "predict all legitimate" = 99.83% accuracy
                 - Standard metrics (accuracy) meaningless
                 - Requires specialized handling
                 - Precision-Recall > ROC-AUC
                 """,
                 card_type="warning")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Source & Privacy", divider=False)
        st.markdown(f"""
        - **Source:** [Kaggle European Credit Card Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
        - **Original Size:** 284,807 transactions
        - **Cleaned Size:** {format_number(DatasetInfo.TOTAL_TRANSACTIONS)} (-1,081 duplicates)
        - **Time Period:** September 2013 (48 hours)
        - **Privacy:** PCA-transformed features (V1-V28)
        - **Non-PCA:** Time, Amount, Class (target)
        """)
    
    with col2:
        section_header("Feature Engineering", divider=False)
        st.markdown(f"""
        **Total Features:** {DatasetInfo.TOTAL_FEATURES}
        - **Original:** {DatasetInfo.ORIGINAL_FEATURES} (28 PCA + Time + Amount)
        - **Engineered:** +{DatasetInfo.ENGINEERED_FEATURES} new features
        """)
        
        st.markdown("**Key Engineered Features:**")
        
        # Create DataFrame for key features
        key_features_df = pd.DataFrame(KEY_ENGINEERED_FEATURES)
        st.dataframe(
            key_features_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("Feature", width="medium"),
                "type": st.column_config.TextColumn("Type", width="small"),
                "description": st.column_config.TextColumn("Description", width="large")
            }
        )
        
        # Expander for all features
        with st.expander("📋 View All 54 Features"):
            st.markdown("**Original Features (30):**")
            cols_orig = st.columns(3)
            for i, feature in enumerate(ORIGINAL_FEATURES_LIST):
                with cols_orig[i % 3]:
                    st.markdown(f"• {feature}")
            
            st.markdown("")
            st.markdown("**Engineered Features (24):**")
            
            # Group by type
            temporal = [f for f in ENGINEERED_FEATURES_LIST if any(x in f for x in ['Hour', 'Minute', 'Second', 'Time', 'sin', 'cos'])]
            amount = [f for f in ENGINEERED_FEATURES_LIST if 'Amount' in f]
            interaction = [f for f in ENGINEERED_FEATURES_LIST if 'interaction' in f or any(f.startswith(f'V{i}_') for i in range(1,30))]
            
            st.markdown(f"*Temporal ({len(temporal)}):*")
            cols_temp = st.columns(3)
            for i, f in enumerate(temporal):
                with cols_temp[i % 3]:
                    st.markdown(f"• {f}")
            
            st.markdown(f"*Amount-based ({len(amount)}):*")
            cols_amt = st.columns(3)
            for i, f in enumerate(amount):
                with cols_amt[i % 3]:
                    st.markdown(f"• {f}")
            
            st.markdown(f"*Interactions ({len(interaction)}):*")
            cols_int = st.columns(3)
            for i, f in enumerate(interaction):
                with cols_int[i % 3]:
                    st.markdown(f"• {f}")
    
    st.markdown("---")
    
    section_header("Chronological Split", "Train-Validation-Test preserving temporal order")
    
    split_df = pd.DataFrame({
        'Split': ['Training', 'Validation', 'Test'],
        'Percentage': [64, 16, 20],
        'Samples': [DatasetInfo.TRAIN_SIZE, DatasetInfo.VAL_SIZE, DatasetInfo.TEST_SIZE]
    })
    
    fig_split = go.Figure()
    colors_split = [Colors.SUPERVISED, Colors.TURQUOISE, Colors.TEAL]
    
    for i, row in split_df.iterrows():
        fig_split.add_trace(go.Bar(
            name=row['Split'],
            x=[row['Split']],
            y=[row['Percentage']],
            text=f"{format_number(row['Samples'])} samples<br>({row['Percentage']}%)",
            textposition='auto',
            marker=dict(color=colors_split[i])
        ))
    
    fig_split.update_layout(
        title="Data Split Strategy",
        xaxis_title="", yaxis_title="Percentage (%)",
        barmode='group', showlegend=False,
        plot_bgcolor='white', paper_bgcolor='white', height=400
    )
    
    st.plotly_chart(fig_split, use_container_width=True)
    
    info_card("Why Chronological Split?",
             """
             - **Preserves temporal order:** Models tested on future transactions
             - **Realistic evaluation:** Mimics production deployment
             - **Prevents leakage:** No future info in training
             - **Validation set:** Used for threshold & ensemble selection
             - **Test set:** Strictly held out for final evaluation
             """,
             card_type="info")

elif page == "Methodology":
    st.title("Methodology")
    st.markdown("**Experimental framework and validation protocol**")
    st.markdown("---")
    
    section_header("Research Questions", divider=False)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        info_card("Question 1", 
                 "**Performance vs Cost**\n\nDo unsupervised/semi-supervised methods "
                 "justify labeling cost savings with competitive performance?",
                 card_type="info")
    
    with col2:
        info_card("Question 2",
                 "**Imbalance Handling**\n\nWhich strategy works best per paradigm: "
                 "class weighting, SMOTE, or both?",
                 card_type="info")
    
    with col3:
        info_card("Question 3",
                 "**Validation Protocol**\n\nDoes validation-based calibration prevent "
                 "test-set leakage while maintaining generalization?",
                 card_type="info")
    
    st.markdown("---")
    
    section_header("Training Strategies by Paradigm")
    
    strategies_df = pd.DataFrame({
        'Paradigm': ['Supervised', 'Unsupervised', 'Semi-Supervised'],
        'Imbalance': ['Class Weighting', 'SMOTE (both classes)', 'SMOTE → Normal only'],
        'Training Data': ['Original (imbalanced)', 'Resampled (balanced)', 'Resampled (normal only)'],
        'Output': ['Probabilities', 'Anomaly scores', 'Anomaly scores'],
        'Calibration': ['Direct', 'Validation threshold', 'Validation threshold']
    })
    
    st.dataframe(strategies_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    section_header("Validation-Based Calibration Protocol")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Critical Methodological Contribution:**
        
        All model selection decisions made exclusively on **validation set** to prevent 
        test leakage and ensure unbiased performance.
        
        **Validation Set Used For:**
        1. Hyperparameter optimization (Optuna, 20 trials)
        2. Threshold calibration (maximize F1-score)
        3. Ensemble model selection (top 3 by validation F1)
        4. Ensemble weight calculation (proportional to validation F1)
        
        **Test Set:**
        - Strictly held out
        - Used only for final reporting
        - Never touched during training/calibration
        - Provides unbiased generalization estimate
        """)
    
    with col2:
        info_card("Why This Matters",
                 "Many studies optimize thresholds on test data, leading to inflated metrics. "
                 "Our protocol ensures all reported metrics are **true generalization estimates**.",
                 card_type="success")
    
    if show_details:
        st.code("""
def get_optimal_threshold_f1(anomaly_scores, y_true):
    '''VALIDATION data only - prevents test leakage'''
    precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    return thresholds[optimal_idx]

# Apply to VALIDATION
threshold = get_optimal_threshold_f1(val_scores, y_val)
# Use on TEST (no leakage)
y_test_pred = (test_scores >= threshold).astype(int)
        """, language='python')
    
    st.markdown("---")
    
    section_header("Hyperparameter Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Optuna Framework:**
        - Bayesian optimization (TPE sampler)
        - 20 trials per model
        - Cross-validation on training set
        - Consistent across paradigms
        
        **Objective:**
        - Maximize mean F1-score
        - TimeSeriesSplit (3 folds)
        - Preserves temporal ordering
        """)
    
    with col2:
        info_card("Efficiency Gains",
                 "Optuna provides 3-4× speedup over GridSearchCV while exploring "
                 "parameter spaces more intelligently.\n\n"
                 "Full pipeline training: **15-30 minutes** on MacBook Air M4",
                 card_type="info")

elif page == "Models Evaluated":
    st.title("Models Evaluated")
    st.markdown("**9 models across 3 learning paradigms + 1 ensemble**")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Supervised", "2", card_type="default")
    with col2:
        kpi_card("Unsupervised", "3", card_type="default")
    with col3:
        kpi_card("Semi-Supervised", "3", card_type="default")
    with col4:
        kpi_card("Ensemble", "1", card_type="default")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        section_header("Supervised", divider=False)
        st.markdown("""
        **Models:**
        - Logistic Regression
        - Random Forest
        
        **Training:**
        - Class weighting
        - Original imbalanced data
        - Direct probability outputs
        """)
    
    with col2:
        section_header("Unsupervised", divider=False)
        st.markdown("""
        **Models:**
        - Isolation Forest
        - Local Outlier Factor
        - Gaussian Mixture Model
        
        **Training:**
        - SMOTE resampling
        - Anomaly score outputs
        - Validation threshold
        """)
    
    with col3:
        section_header("Semi-Supervised", divider=False)
        st.markdown("""
        **Models:**
        - IF Semi-Supervised
        - LOF Semi-Supervised
        - GMM Semi-Supervised
        
        **Training:**
        - SMOTE → normal only
        - One-class learning
        - Validation threshold
        """)
    
    st.markdown("---")
    
    section_header("Ensemble Composition", "F1-weighted combination of top 3 models")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_ensemble = create_ensemble_composition_chart(ENSEMBLE_COMPOSITION)
        st.plotly_chart(fig_ensemble, use_container_width=True)
    
    with col2:
        st.markdown("### Details")
        st.markdown("""
        **Selection:**
        - Top 3 by validation F1-score
        
        **Weighting:**
        - Proportional to validation F1
        
        **Components:**
        """)
        
        for model, weight in sorted(ENSEMBLE_COMPOSITION.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"- {model}: **{weight:.2f}%**")
        
        st.markdown("")
        info_card("Trade-off",
                 "Ensemble sacrifices peak performance (0.78 → 0.53) for "
                 "improved robustness through model diversity.",
                 card_type="info")

elif page == "Performance Results":
    st.title("Performance Results")
    st.markdown("**Test set performance with validation-based calibration**")
    st.markdown("---")
    
    filtered_data = get_model_by_paradigm(paradigm_filter)
    
    section_header("Top Performers", "Best models by fraud F1-score")
    
    top_3 = get_top_n_models(3, 'Fraud_F1')
    cols = st.columns(3)
    
    for i, (_, row) in enumerate(top_3.iterrows()):
        with cols[i]:
            model_card(
                row['Model'], row['Paradigm'],
                row['Fraud_F1'], row['Fraud_Precision'],
                row['Fraud_Recall'], row['ROC_AUC'],
                color=[Colors.SUPERVISED, Colors.SUPERVISED, Colors.ENSEMBLE][i]
            )
    
    st.markdown("---")
    
    section_header("Model Performance Comparison")
    
    tab1, tab2, tab3 = st.tabs(["F1-Score Comparison", "Precision-Recall Trade-off", "Performance Table"])
    
    with tab1:
        fig_f1 = create_f1_comparison_chart(filtered_data, 'Fraud_F1')
        st.plotly_chart(fig_f1, use_container_width=True)
        info_card("Key Insight",
                 "Supervised models significantly outperform unsupervised/semi-supervised. "
                 "Random Forest achieves highest fraud F1-score (0.78).",
                 card_type="success")
    
    with tab2:
        fig_pr = create_precision_recall_scatter(filtered_data)
        st.plotly_chart(fig_pr, use_container_width=True)
        info_card("Precision-Recall Trade-off",
                 "- **High Precision (0.85):** Random Forest minimizes false alarms\n"
                 "- **High Recall (0.72):** Detects 72% of actual fraud\n"
                 "- **Unsupervised:** Very low precision, moderate recall",
                 card_type="info")
    
    with tab3:
        display_df = filtered_data[['Model', 'Paradigm', 'Fraud_Precision', 'Fraud_Recall', 
                                     'Fraud_F1', 'Macro_F1', 'ROC_AUC']].copy()
        for col in ['Fraud_Precision', 'Fraud_Recall', 'Fraud_F1', 'Macro_F1', 'ROC_AUC']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    if show_details:
        st.markdown("---")
        section_header("Cost Analysis", "Expected cost assuming FN = 10× FP")
        
        costs = []
        for _, row in filtered_data.iterrows():
            cost_info = calculate_expected_cost(row['Fraud_Precision'], row['Fraud_Recall'])
            costs.append(cost_info['total_cost'])
        
        fig_cost = create_cost_comparison_chart(
            filtered_data['Model'].tolist(), costs, filtered_data['Paradigm'].tolist()
        )
        st.plotly_chart(fig_cost, use_container_width=True)
        
        info_card("Cost Interpretation",
                 "Assuming FP cost = $1, FN cost = $10:\n\n"
                 "**Random Forest has lowest expected cost** due to high precision and recall. "
                 "Unsupervised models incur high costs from many false positives.",
                 card_type="info")
    
    st.markdown("---")
    
    section_header("Why ROC-AUC is Misleading")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **The Problem:**
        
        With 99.83% legitimate transactions, correctly classifying the majority class 
        dominates ROC-AUC. Models achieve high AUC (0.87-0.98) while performing 
        poorly on fraud (F1 = 0.02-0.10).
        
        **Example:**
        - GMM Unsupervised: AUC = 0.764, Fraud F1 = 0.02
        - Only 1% precision despite moderate AUC
        
        **Better Metrics:**
        - Precision-Recall curves
        - F1-score (fraud class)
        - Cost-sensitive metrics
        """)
    
    with col2:
        info_card("Recommendation",
                 "For fraud detection with extreme imbalance:\n\n"
                 "✅ **Use:** F1-score, Precision-Recall AUC\n\n"
                 "❌ **Avoid:** ROC-AUC as primary metric",
                 card_type="warning")

elif page == "Technical Details":
    st.title("Technical Details")
    st.markdown("**Implementation, code, and reproducibility**")
    st.markdown("---")
    
    section_header("Code Implementations")
    
    with st.expander("Validation-Based Threshold Optimization"):
        st.markdown("**File:** `main.py`")
        st.code("""
def get_optimal_threshold_f1(anomaly_scores, y_true):
    '''
    IMPORTANT: Call on VALIDATION data only to prevent test leakage.
    '''
    precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    return thresholds[optimal_idx]
        """, language='python')
        info_card("Key Point", 
                 "Threshold optimized on VALIDATION set only. Test set untouched.",
                 card_type="success")
    
    with st.expander("Paradigm-Specific Training"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Supervised**")
            st.code("""
lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
lr.fit(X_train, y_train)
            """, language='python')
        
        with col2:
            st.markdown("**Unsupervised**")
            st.code("""
X_res, y_res = SMOTE().fit_resample(
    X_train, y_train
)
if_model = IsolationForest()
if_model.fit(X_res)
            """, language='python')
        
        with col3:
            st.markdown("**Semi-Supervised**")
            st.code("""
X_res, y_res = SMOTE().fit_resample(
    X_train, y_train
)
X_normal = X_res[y_res == 0]
if_semi = IsolationForest()
if_semi.fit(X_normal)
            """, language='python')
    
    with st.expander("F1-Weighted Ensemble"):
        st.markdown("**File:** `models_application.py`")
        st.code("""
# Select top 3 by validation F1
sorted_models = sorted(val_f1_scores.items(), key=lambda x: x[1], reverse=True)
top_3 = sorted_models[:3]

# F1-proportional weights
total_f1 = sum(f1 for _, f1 in top_3)
weights = [f1 / total_f1 for _, f1 in top_3]

# Ensemble on VALIDATION (no test leakage)
ensemble_proba_val = sum(w * pred for w, pred in zip(weights, predictions_val))
threshold = get_optimal_threshold_f1(ensemble_proba_val, y_val)
        """, language='python')
    
    st.markdown("---")
    
    section_header("Project Structure")
    
    st.code("""
Project_data_science_source/
├── READ/                                     # Documentation
│   ├── README.md                                   # Project overview, setup & usage instructions
│   └── Proposal.md                   (70 lines)    # Project proposal
│
├── src/                                      # Source code
│   ├── main.py                       (290 lines)   # Data loading & preprocessing
│   ├── models_calibration.py         (702 lines)   # Hyperparameter optimization
│   ├── models_application.py         (429 lines)   # Model evaluation
│   ├── performance_visualization.py  (589 lines)   # Results & data visualization
│   └── menu.py                       (670 lines)   # Interactive menu system
│
├── data/                                    # Dataset storage
│   └── creditcard.csv                              # Kaggle dataset (auto-downloaded)
│
├── saved_models/                            # Trained model storage
│   └── trained_models.pkl                          # All 8 models + ensemble
│
├── output/                                  # Generated visualizations
│   ├── 0_class_distribution.png
│   ├── 0_amount_distribution.png
│   ├── 1_confusion_matrices.png
│   ├── 2_performance_metrics.png
│   ├── 3_f1_score_comparison.png
│   ├── 4_roc_curves_all_models.png
│   ├── 5_roc_curves_top_performers.png
│   ├── 6_precision_recall_curves.png
│   ├── 7_feature_importance_rf.png
│   ├── 8_feature_importance_lr.png
│   └── 9_lr_coefficients_signed.png
│
├── environment.yml                          # Conda environment specification
├── AI-USAGE.md                              # AI tools transparency disclosure
└── .gitignore                               # Git ignore rules
    """, language='text')
    
    st.markdown("---")
    
    section_header("Quick Start Guide")
    
    with st.expander("1. Environment Setup"):
        st.code("""
# Create conda environment
conda env create -f environment.yml
conda activate data_science_project

# Verify installation
python menu.py
        """, language='bash')
    
    with st.expander("2. Data Preparation"):
        st.code(f"""
# Run pipeline
python menu.py → Option 1

# Output:
# - {format_number(DatasetInfo.TOTAL_TRANSACTIONS)} transactions
# - {DatasetInfo.TOTAL_FEATURES} features
# - 64-16-20 split
        """, language='bash')
    
    with st.expander("3. Model Training"):
        st.code("""
# Train all models (10-30 min)
python menu.py → Option 2

# Output:
# - saved_models/trained_models.pkl
        """, language='bash')
    
    st.markdown("---")
    
    section_header("Video Presentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Structure")
        st.markdown("""
        1. Introduction & Problem
        2. Data & Methodology
        3. Code Demonstrations
        4. CLI Menu Demo
        5. Results & Conclusion
        """)
    
    with col2:
        st.markdown("### Tools")
        st.markdown("""
        - **Slides:** PowerPoint
        - **Recording:** Loom (macOS)
        - **Platform:** macOS
        """)
    
    st.markdown("---")
    
    section_header("Key References")
    
    st.markdown("""
    - **Dataset:** [Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
    - **SMOTE:** Chawla et al. (2002)
    - **Isolation Forest:** Liu et al. (2008)
    - **Optuna:** Akiba et al. (2019)
    - **Dal Pozzolo et al. (2014):** Practitioner perspective on fraud detection
    """)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 8px;'>
    <h3 style='color: #2c3e50; margin-bottom: 0.5rem;'>Credit Card Fraud Detection Pipeline</h3>
    <p style='color: #7f8c8d; margin: 0;'>
        <strong>Dylan Fernandez</strong> | Advanced Programming 2025 | December 15, 2025
    </p>
    <p style='color: #95a5a6; font-size: 0.875rem; margin-top: 0.5rem;'>
        University of Lausanne
    </p>
</div>
""", unsafe_allow_html=True)
