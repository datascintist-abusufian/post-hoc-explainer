import streamlit as st
import pandas as pd
import numpy as np
import shap
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_curve, auc, precision_recall_curve)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Heart Disease Analysis Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    .plot-container {
        background-color: white;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

class DataLoader:
    @staticmethod
    @st.cache_data
    def load_data():
        try:
            dtypes = {
                'age': 'int32', 'gender': 'int8', 'height': 'int32',
                'weight': 'float32', 'ap_hi': 'int32', 'ap_lo': 'int32',
                'cholesterol': 'int8', 'gluc': 'int8', 'smoke': 'int8',
                'alco': 'int8', 'active': 'int8', 'cardio': 'int8'
            }
            url = "https://raw.githubusercontent.com/datascintist-abusufian/Post-hoc-explanation-cardio-phenotype-interpretability/main/cardio_train.csv"
            data = pd.read_csv(url, sep=';', dtype=dtypes)
            
            # Preprocess age from days to years
            data['age'] = data['age'] / 365.25
            
            return data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

class HeartDiseaseAnalyzer:
    def __init__(self, n_estimators=100, max_depth=10):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.explainer = None

    def preprocess_data(self, data):
        X = data.drop('cardio', axis=1)
        y = data['cardio']
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.X_train = pd.DataFrame(X_scaled, columns=self.feature_names)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_prob,
            'cv_scores': cross_val_score(self.model, X_test, y_test, cv=5),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }

    def plot_shap_summary(self):
        shap_values = self.explainer.shap_values(self.X_train)
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[1], self.X_train, show=False)
        plt.title("SHAP Feature Importance Summary")
        return fig

    def plot_shap_dependence(self, feature_name):
        shap_values = self.explainer.shap_values(self.X_train)
        fig = plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name, 
            shap_values[1], 
            self.X_train,
            show=False
        )
        plt.title(f"SHAP Dependence Plot for {feature_name}")
        return fig

    # ... [Previous plotting methods remain the same] ...

def plot_survival_curves(data):
    kmf = KaplanMeierFitter()
    
    # Create survival data
    duration = data['age']
    event_observed = data['cardio']
    
    fig = plt.figure(figsize=(10, 6))
    
    # Plot survival curves for different groups
    for cholesterol in [1, 2, 3]:
        mask = data['cholesterol'] == cholesterol
        kmf.fit(
            duration[mask], 
            event_observed[mask], 
            label=f'Cholesterol Level {cholesterol}'
        )
        kmf.plot()
    
    plt.title('Survival Curves by Cholesterol Level')
    plt.xlabel('Age (years)')
    plt.ylabel('Survival probability')
    return fig

def main():
    st.title("🫀 Advanced Heart Disease Analysis Dashboard")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Analysis Controls")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Model Performance", "SHAP Analysis", "Survival Analysis"]
        )
        
        n_estimators = st.slider("Number of trees", 50, 200, 100)
        max_depth = st.slider("Maximum tree depth", 5, 20, 10)

    # Load data
    data = DataLoader.load_data()
    if data is None:
        return

    # Main content
    if analysis_type == "Model Performance":
        st.header("📊 Model Performance Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(data.head())
        with col2:
            st.metric("Total Records", len(data))
            st.metric("Features", len(data.columns) - 1)
            st.metric("Disease Prevalence", f"{(data['cardio'].mean() * 100):.1f}%")

        if st.button("🔍 Run Model Analysis"):
            analyzer = HeartDiseaseAnalyzer(n_estimators, max_depth)
            
            with st.spinner("Processing data and training model..."):
                X_train, X_test, y_train, y_test = analyzer.preprocess_data(data)
                metrics = analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)

            col1, col2, col3 = st.columns(3)
            col1.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("CV Mean", f"{metrics['cv_scores'].mean():.2%}")
            col3.metric("CV Std", f"{metrics['cv_scores'].std():.2%}")

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(analyzer.plot_roc_curve(y_test, metrics['probabilities']))
            with col2:
                st.pyplot(analyzer.plot_confusion_matrix(y_test, metrics['predictions']))

    elif analysis_type == "SHAP Analysis":
        st.header("🎯 SHAP Feature Analysis")
        
        if st.button("🔍 Generate SHAP Analysis"):
            analyzer = HeartDiseaseAnalyzer(n_estimators, max_depth)
            
            with st.spinner("Calculating SHAP values..."):
                X_train, X_test, y_train, y_test = analyzer.preprocess_data(data)
                analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)
                
                st.pyplot(analyzer.plot_shap_summary())
                
                feature = st.selectbox("Select feature for dependence plot:", analyzer.feature_names)
                st.pyplot(analyzer.plot_shap_dependence(feature))

    else:  # Survival Analysis
        st.header("📈 Survival Analysis")
        
        if st.button("🔍 Generate Survival Analysis"):
            with st.spinner("Calculating survival curves..."):
                st.pyplot(plot_survival_curves(data))
                
                # Cox Proportional Hazards Model
                cph = CoxPHFitter()
                survival_data = data.copy()
                survival_data['duration'] = survival_data['age']
                survival_data['event'] = survival_data['cardio']
                
                cph.fit(survival_data, 'duration', 'event')
                st.write("Cox Proportional Hazards Model Summary:")
                st.write(cph.print_summary())

    st.markdown("---")
    st.markdown(
        """<div style='text-align: center; color: #666;'>
            <p>Advanced Heart Disease Analysis Dashboard | Version 2.0</p>
            <p>Featuring SHAP Explanations and Survival Analysis</p>
        </div>""",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
