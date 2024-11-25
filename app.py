import streamlit as st
import pandas as pd
import numpy as np
import shap
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
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #45a049;
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
            # Using your GitHub repository URL
            url = "https://raw.githubusercontent.com/datascintist-abusufian/post-hoc-explainer/main/data/cardio_train.csv"
            data = pd.read_csv(url, sep=';', dtype=dtypes)
            
            # Preprocess age from days to years
            data['age'] = data['age'] / 365.25
            
            # Calculate BMI
            data['bmi'] = data['weight'] / ((data['height']/100) ** 2)
            
            # Calculate Pulse Pressure
            data['pulse_pressure'] = data['ap_hi'] - data['ap_lo']
            
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
        # Select features including engineered ones
        features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                   'cholesterol', 'gluc', 'smoke', 'alco', 'active', 
                   'bmi', 'pulse_pressure']
        
        X = data[features]
        y = data['cardio']
        self.feature_names = features
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
            'report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': dict(zip(self.feature_names, 
                                        self.model.feature_importances_))
        }

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return fig

    def plot_roc_curve(self, y_true, y_prob):
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
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=800, height=600
        )
        return fig

    def plot_feature_importance(self):
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='Feature',
            y='Importance',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(width=800, height=600)
        return fig

    def plot_shap_summary(self):
        shap_values = self.explainer.shap_values(self.X_train)
        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[1], self.X_train, show=False)
        plt.title("SHAP Feature Importance Summary")
        return fig

    def plot_shap_dependence(self, feature_name):
        shap_values = self.explainer.shap_values(self.X_train)
        fig = plt.figure(figsize=(12, 8))
        shap.dependence_plot(
            feature_name, 
            shap_values[1], 
            self.X_train,
            show=False
        )
        plt.title(f"SHAP Dependence Plot for {feature_name}")
        return fig

def main():
    # Add GIF at the top
    st.markdown("""
        <div style="display: flex; justify-content: center;">
            <img src="https://raw.githubusercontent.com/datascintist-abusufian/post-hoc-explainer/main/Transformer%20ExplainableAI.gif" 
                 alt="Explainable AI Visualization"
                 style="width: 100%; max-width: 800px; margin-bottom: 20px;">
        </div>
    """, unsafe_allow_html=True)

    st.title("🫀 Advanced Heart Disease Analysis Dashboard")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Analysis Controls")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Data Overview", "Model Performance", "SHAP Analysis", "Feature Engineering"]
        )
        
        n_estimators = st.slider("Number of trees", 50, 200, 100)
        max_depth = st.slider("Maximum tree depth", 5, 20, 10)
        
        st.markdown("---")
        st.markdown("""
        ### About
        Post hoc explanation dashboard provides interactive insights into heart disease prediction using:
        advanced indepth scientific analysis for trust worthy and transparent to the clinician.
        It helps to clinical expert to take a decision on the basis what input has been influenced on prediction!!
        """)
    # Load data
    data = DataLoader.load_data()
    if data is None:
        return

    # Main content based on analysis type
    if analysis_type == "Data Overview":
        st.header("📋 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(data):,}")
        col2.metric("Features", len(data.columns) - 1)
        col3.metric("Disease Prevalence", f"{(data['cardio'].mean() * 100):.1f}%")
        col4.metric("Age Range", f"{data['age'].min():.1f}-{data['age'].max():.1f} years")
        
        st.subheader("Sample Data")
        st.dataframe(data.head(10))
        
        st.subheader("Statistical Summary")
        st.dataframe(data.describe())
        
        st.subheader("Feature Distributions")
        feature_to_plot = st.selectbox("Select feature:", data.columns)
        fig = px.histogram(data, x=feature_to_plot, color='cardio',
                         title=f"Distribution of {feature_to_plot}")
        st.plotly_chart(fig)

    elif analysis_type == "Model Performance":
        st.header("📊 Model Performance Analysis")
        
        if st.button("🔍 Run Model Analysis", key='run_model'):
            analyzer = HeartDiseaseAnalyzer(n_estimators, max_depth)
            
            with st.spinner("Processing data and training model..."):
                X_train, X_test, y_train, y_test = analyzer.preprocess_data(data)
                metrics = analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)

            col1, col2, col3 = st.columns(3)
            col1.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("CV Mean", f"{metrics['cv_scores'].mean():.2%}")
            col3.metric("CV Std", f"{metrics['cv_scores'].std():.2%}")

            st.subheader("ROC Curve & Confusion Matrix")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(analyzer.plot_roc_curve(y_test, metrics['probabilities']))
            with col2:
                st.pyplot(analyzer.plot_confusion_matrix(y_test, metrics['predictions']))
            
            st.subheader("Feature Importance")
            st.plotly_chart(analyzer.plot_feature_importance())

    elif analysis_type == "SHAP Analysis":
        st.header("🎯 SHAP Feature Analysis")
        
        if st.button("🔍 Generate SHAP Analysis", key='run_shap'):
            analyzer = HeartDiseaseAnalyzer(n_estimators, max_depth)
            
            with st.spinner("Calculating SHAP values..."):
                X_train, X_test, y_train, y_test = analyzer.preprocess_data(data)
                analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)
                
                st.subheader("SHAP Summary Plot")
                st.pyplot(analyzer.plot_shap_summary())
                
                st.subheader("Feature Dependence Analysis")
                feature = st.selectbox("Select feature for dependence plot:", 
                                     analyzer.feature_names)
                st.pyplot(analyzer.plot_shap_dependence(feature))

    else:  # Feature Engineering
        st.header("🔧 Feature Engineering Analysis")
        
        st.subheader("Body Mass Index (BMI) Analysis")
        fig = px.scatter(data, x='bmi', y='age', color='cardio',
                        title='BMI vs Age by Disease Status')
        st.plotly_chart(fig)
        
        st.subheader("Blood Pressure Analysis")
        fig = px.scatter(data, x='ap_hi', y='ap_lo', color='cardio',
                        title='Systolic vs Diastolic Blood Pressure')
        st.plotly_chart(fig)
        
        st.subheader("Risk Factor Combinations")
        risk_factors = data[['cholesterol', 'gluc', 'smoke', 'alco']].mean()
        fig = px.bar(risk_factors, title='Average Risk Factor Prevalence')
        st.plotly_chart(fig)

    st.markdown("---")
    st.markdown(
        """<div style='text-align: center; color: #666;'>
            <p>Advanced Heart Disease Analysis Dashboard | Version 2.0</p>
            <p>Featuring SHAP Explanations and Advanced Analytics</p>
        </div>""",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
