import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_curve, auc)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Heart Disease Analysis", page_icon="🫀", layout="wide")

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
            # Update path to use data from your repository
            data = pd.read_csv('data/cardio_train.csv', sep=';', dtype=dtypes)
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

    def preprocess_data(self, data):
        X = data.drop('cardio', axis=1)
        y = data['cardio']
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_prob,
            'cv_scores': cv_scores,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
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
            width=700, height=500
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
        
        return fig

def main():
    st.title("🫀 Heart Disease Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Analysis Controls")
        
        # Model parameters
        n_estimators = st.slider("Number of trees", 50, 200, 100)
        max_depth = st.slider("Maximum tree depth", 5, 20, 10)

    # Load data
    data = DataLoader.load_data()
    if data is None:
        return

    # Data Overview
    st.header("📋 Data Overview")
    
    tab1, tab2 = st.tabs(["Data Preview", "Feature Descriptions"])
    with tab1:
        st.dataframe(data.head())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(data))
        col2.metric("Features", len(data.columns) - 1)
        col3.metric("Disease Prevalence", f"{(data['cardio'].mean() * 100):.1f}%")
        
    with tab2:
        # Feature descriptions
        st.markdown("### Feature Descriptions")
        feature_descriptions = {
            'age': 'Age in days',
            'gender': '0: Female, 1: Male',
            'height': 'Height in centimeters',
            'weight': 'Weight in kilograms',
            'ap_hi': 'Systolic blood pressure',
            'ap_lo': 'Diastolic blood pressure',
            'cholesterol': '1: Normal, 2: Above Normal, 3: Well Above Normal',
            'gluc': '1: Normal, 2: Above Normal, 3: Well Above Normal',
            'smoke': '0: Non-smoker, 1: Smoker',
            'alco': '0: No alcohol, 1: Alcohol consumption',
            'active': '0: Inactive, 1: Physically active',
            'cardio': '0: No disease, 1: Disease present'
        }
        for feature, description in feature_descriptions.items():
            st.markdown(f"**{feature}**: {description}")

    # Analysis
    if st.button("🔍 Run Analysis"):
        analyzer = HeartDiseaseAnalyzer(n_estimators, max_depth)
        
        with st.spinner("Processing data..."):
            X_train, X_test, y_train, y_test = analyzer.preprocess_data(data)
        
        with st.spinner("Training model..."):
            metrics = analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)

        # Results tabs
        tabs = st.tabs([
            "📊 Model Performance",
            "🎯 Feature Analysis",
            "📈 Model Metrics"
        ])

        with tabs[0]:
            col1, col2, col3 = st.columns(3)
            col1.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("Cross-validation Mean", f"{metrics['cv_scores'].mean():.2%}")
            col3.metric("Cross-validation Std", f"{metrics['cv_scores'].std():.2%}")
            
            st.subheader("Confusion Matrix")
            st.pyplot(analyzer.plot_confusion_matrix(y_test, metrics['predictions']))
            
            st.subheader("ROC Curve")
            st.plotly_chart(analyzer.plot_roc_curve(y_test, metrics['probabilities']))

        with tabs[1]:
            st.plotly_chart(analyzer.plot_feature_importance())

        with tabs[2]:
            st.subheader("Classification Report")
            report_df = pd.DataFrame(metrics['report']).transpose()
            st.dataframe(report_df)

    st.markdown("---")
    st.markdown(
        """<div style='text-align: center; color: #666;'>
            <p>Heart Disease Analysis Tool | Version 1.0</p>
        </div>""",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
