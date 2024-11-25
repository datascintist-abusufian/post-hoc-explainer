import streamlit as st
import pandas as pd
import numpy as np
import shap
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_curve, auc, precision_recall_curve)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import survival_analysis

# Configure Streamlit page
st.set_page_config(
    page_title="Heart Disease Analysis Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main { padding: 2rem 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff; }
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    div.stButton > button:hover { background-color: #45a049; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess data with caching"""
    try:
        dtypes = {
            'age': 'int32', 'gender': 'int8', 'height': 'int32',
            'weight': 'float32', 'ap_hi': 'int32', 'ap_lo': 'int32',
            'cholesterol': 'int8', 'gluc': 'int8', 'smoke': 'int8',
            'alco': 'int8', 'active': 'int8', 'cardio': 'int8'
        }
        url = "https://raw.githubusercontent.com/datascintist-abusufian/post-hoc-explainer/main/data/cardio_train.csv"
        data = pd.read_csv(url, sep=';', dtype=dtypes)
        
        # Preprocess data
        data['age'] = data['age'] / 365.25
        data['bmi'] = data['weight'] / ((data['height']/100) ** 2)
        data['pulse_pressure'] = data['ap_hi'] - data['ap_lo']
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

class ModelAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.explainer = None
        self.shap_values = None
        
    @st.cache_data
    def prepare_data(self, data):
        """Prepare data for modeling with caching"""
        features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                   'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                   'bmi', 'pulse_pressure']
        
        X = data[features]
        y = data['cardio']
        self.feature_names = features
        X_scaled = self.scaler.fit_transform(X)
        self.X_train = pd.DataFrame(X_scaled, columns=self.feature_names)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    @st.cache_data
    def train_model(self, X_train, y_train, n_estimators, max_depth):
        """Train model with caching"""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        return self.model

    @st.cache_data
    def compute_shap_values(self):
        """Compute SHAP values with caching"""
        self.explainer = shap.TreeExplainer(self.model)
        return self.explainer.shap_values(self.X_train)

    @st.cache_data
    def generate_metrics(self, X_test, y_test):
        """Generate model metrics with caching"""
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_prob,
            'cv_scores': cross_val_score(self.model, X_test, y_test, cv=5),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': dict(zip(self.feature_names, 
                                        self.model.feature_importances_))
        }

class SurvivalAnalyzer:
    @st.cache_data
    def analyze(self, data, group_col=None):
        """Run survival analysis using Kaplan-Meier estimate"""
        fig = plt.figure(figsize=(10, 6))
        
        if group_col:
            for value in sorted(data[group_col].unique()):
                mask = data[group_col] == value
                group_data = data[mask]
                
                # Calculate survival curve
                time_points = np.sort(group_data['age'].unique())
                survival_prob = 1 - group_data.groupby('age')['cardio'].mean().cumsum() / len(time_points)
                
                plt.plot(time_points, survival_prob, 
                        label=f'{group_col}={value}')
        else:
            time_points = np.sort(data['age'].unique())
            survival_prob = 1 - data.groupby('age')['cardio'].mean().cumsum() / len(time_points)
            plt.plot(time_points, survival_prob, label='Overall')
            
        plt.title('Survival Analysis')
        plt.xlabel('Age (years)')
        plt.ylabel('Survival probability')
        plt.legend()
        plt.grid(True)
        return fig

@st.cache_data
def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix with caching"""
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Disease', 'Disease'],
               yticklabels=['No Disease', 'Disease'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return fig

@st.cache_data
def plot_roc_curve(y_true, y_prob):
    """Plot ROC curve with caching"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                           name=f'ROC curve (AUC = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                           line=dict(dash='dash'), name='Random'))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800, height=500
    )
    return fig

def main():
    # Add custom CSS for GIF container
    st.markdown("""
        <style>
        .gif-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
        .gif-container img {
            width: 100%;
            max-width: 800px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .gif-container img:hover {
            transform: scale(1.02);
        }
        </style>
    """, unsafe_allow_html=True)

    # Add GIF with enhanced styling
    st.markdown("""
        <div class="gif-container">
            <img src="https://raw.githubusercontent.com/datascintist-abusufian/post-hoc-explainer/main/Transformer%20ExplainableAI.gif" 
                 alt="Explainable AI Visualization">
        </div>
    """, unsafe_allow_html=True)

    st.title("🫀 Advanced Heart Disease Analysis Dashboard")

    # Initialize session state
    if 'model_analyzer' not in st.session_state:
        st.session_state.model_analyzer = ModelAnalyzer()
    if 'survival_analyzer' not in st.session_state:
        st.session_state.survival_analyzer = SurvivalAnalyzer()

    # Sidebar
    with st.sidebar:
        st.header("📊 Analysis Controls")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Data Overview", "Model Performance", "SHAP Analysis", 
             "Feature Engineering", "Survival Analysis"]
        )
        
        n_estimators = st.slider("Number of trees", 50, 200, 100)
        max_depth = st.slider("Maximum tree depth", 5, 20, 10)

    # Load data once
    data = load_data()
    if data is None:
        return

    # Display selected analysis
    if analysis_type == "Data Overview":
        st.header("📋 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{len(data):,}")
        col2.metric("Features", len(data.columns) - 1)
        col3.metric("Disease Prevalence", f"{(data['cardio'].mean() * 100):.1f}%")
        
        st.dataframe(data.head())
        
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select feature:", data.columns)
        fig = px.histogram(data, x=feature, color='cardio',
                         title=f"Distribution of {feature}")
        st.plotly_chart(fig)

    elif analysis_type == "Model Performance":
        st.header("📊 Model Performance")
        
        if st.button("Run Analysis"):
            progress = st.progress(0)
            
            # Prepare data
            progress.progress(25)
            X_train, X_test, y_train, y_test = st.session_state.model_analyzer.prepare_data(data)
            
            # Train model
            progress.progress(50)
            model = st.session_state.model_analyzer.train_model(X_train, y_train, n_estimators, max_depth)
            
            # Generate metrics
            progress.progress(75)
            metrics = st.session_state.model_analyzer.generate_metrics(X_test, y_test)
            
            progress.progress(100)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_roc_curve(y_test, metrics['probabilities']))
            with col2:
                st.pyplot(plot_confusion_matrix(y_test, metrics['predictions']))
            
            st.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")

    elif analysis_type == "SHAP Analysis":
        st.header("🎯 SHAP Analysis")
        
        if st.button("Generate SHAP Analysis"):
            with st.spinner("Computing SHAP values..."):
                X_train, X_test, y_train, y_test = st.session_state.model_analyzer.prepare_data(data)
                model = st.session_state.model_analyzer.train_model(X_train, y_train, n_estimators, max_depth)
                shap_values = st.session_state.model_analyzer.compute_shap_values()
                
                fig = plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values[1], st.session_state.model_analyzer.X_train, show=False)
                st.pyplot(fig)

    elif analysis_type == "Feature Engineering":
        st.header("🔧 Feature Engineering")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(data, x='bmi', y='age', color='cardio',
                           title='BMI vs Age')
            st.plotly_chart(fig)
        with col2:
            fig = px.scatter(data, x='ap_hi', y='ap_lo', color='cardio',
                           title='Blood Pressure Analysis')
            st.plotly_chart(fig)

    else:  # Survival Analysis
        st.header("⏳ Survival Analysis")
        
        group_col = st.selectbox(
            "Select grouping variable:", 
            ['None', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        )
        
        if st.button("Generate Survival Curves"):
            with st.spinner("Calculating survival curves..."):
                fig = st.session_state.survival_analyzer.analyze(
                    data,
                    None if group_col == 'None' else group_col
                )
                st.pyplot(fig)

    st.markdown("---")
    st.caption("Heart Disease Analysis Dashboard | Version 2.0")

if __name__ == "__main__":
    main()
