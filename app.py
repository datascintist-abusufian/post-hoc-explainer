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
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
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
        data['map'] = data['ap_lo'] + (data['pulse_pressure'] / 3)  # Mean Arterial Pressure
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Remove the cached decorator from compute_shap_values function
def compute_shap_values(X_train, model):
    """Compute SHAP values without caching"""
    try:
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values
        X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        shap_values = explainer.shap_values(X_train_array)
        
        # Store explainer in session state for force plots
        st.session_state['explainer'] = explainer
        
        return shap_values, X_train.columns if isinstance(X_train, pd.DataFrame) else None
    except Exception as e:
        st.error(f"Error computing SHAP values: {str(e)}")
        return None, None

class ModelAnalyzer:
    def compute_shap_values(self, model, X_train):
        """Direct computation of SHAP values without caching"""
        return compute_shap_values(X_train, model)

# Modified SHAP Analysis section in main():
elif analysis_type == "SHAP Analysis":
    st.header("🎯 SHAP Analysis")
    
    if st.button("Generate SHAP Analysis"):
        if 'current_model' not in st.session_state:
            st.warning("Please run the Model Performance analysis first!")
            return
            
        with st.spinner("Computing SHAP values..."):
            shap_values, feature_names = compute_shap_values(
                st.session_state.model_analyzer.X_train,
                st.session_state['current_model']
            )
            
            if shap_values is not None:
                # Store values in session state for reuse
                st.session_state['shap_values'] = shap_values
                
                # Summary plot
                st.subheader("SHAP Summary Plot")
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(
                    shap_values[1],
                    st.session_state.model_analyzer.X_train,
                    feature_names=feature_names,
                    show=False
                )
                st.pyplot(fig)
                plt.close()

                # Individual prediction explanation
                if st.checkbox("Show Individual Prediction Explanations"):
                    st.subheader("Individual Prediction Explanation")
                    sample_idx = st.slider(
                        "Select sample index",
                        0,
                        len(st.session_state.model_analyzer.X_train)-1,
                        0
                    )
                    
                    fig, ax = plt.subplots(figsize=(10, 3))
                    shap.force_plot(
                        st.session_state['explainer'].expected_value[1],
                        shap_values[1][sample_idx],
                        st.session_state.model_analyzer.X_train.iloc[sample_idx],
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                  
    def prepare_data(self, data):
        """Prepare data for modeling"""
        features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                   'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                   'bmi', 'pulse_pressure', 'map']
        
        X = data[features].copy()
        y = data['cardio'].copy()
        self.feature_names = features
        X_scaled = self.scaler.fit_transform(X)
        self.X_train = pd.DataFrame(X_scaled, columns=self.feature_names)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    @st.cache_data
    def compute_metrics(_self, X_train, X_test, y_train, y_test, n_estimators, max_depth):
        """Train model and compute metrics with caching"""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'predictions': y_pred.tolist(),
            'probabilities': y_prob.tolist(),
            'cv_scores': cross_val_score(model, X_test, y_test, cv=5).tolist(),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': dict(zip(_self.feature_names, 
                                         model.feature_importances_.tolist()))
        }
        
        return model, metrics

    def compute_shap_values(self, model, X_train):
        """Wrapper method for SHAP computation"""
        return compute_shap_values(self, X_train, model)

class SurvivalAnalyzer:
    @st.cache_data
    def analyze(self, data, group_col=None):
        """Run survival analysis using custom implementation"""
        fig = plt.figure(figsize=(10, 6))
        
        if group_col:
            for value in sorted(data[group_col].unique()):
                mask = data[group_col] == value
                group_data = data[mask]
                
                age_groups = pd.qcut(group_data['age'], q=20)
                survival_prob = 1 - group_data.groupby(age_groups)['cardio'].mean()
                
                plt.plot(
                    [group.right for group in survival_prob.index], 
                    survival_prob.values,
                    label=f'{group_col}={value}',
                    marker='o',
                    markersize=4
                )
        else:
            age_groups = pd.qcut(data['age'], q=20)
            survival_prob = 1 - data.groupby(age_groups)['cardio'].mean()
            
            plt.plot(
                [group.right for group in survival_prob.index], 
                survival_prob.values,
                label='Overall',
                marker='o',
                markersize=4
            )
            
        plt.title('Survival Analysis')
        plt.xlabel('Age (years)')
        plt.ylabel('Survival probability')
        plt.grid(True)
        plt.legend()
        return fig

    @st.cache_data
    def analyze_risk_factors(self, data):
        """Analyze risk factors impact on survival"""
        risk_factors = ['cholesterol', 'gluc', 'smoke', 'alco', 'active']
        results = {}
        
        for factor in risk_factors:
            risk_high = data[data[factor] > 1]['cardio'].mean()
            risk_low = data[data[factor] == 1]['cardio'].mean()
            risk_ratio = risk_high / risk_low if risk_low > 0 else 1
            results[factor] = risk_ratio
            
        return pd.Series(results).sort_values(ascending=False)

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
    st.markdown("""
        <div class="gif-container">
            <img src="https://raw.githubusercontent.com/datascintist-abusufian/post-hoc-explainer/main/Transformer%20ExplainableAI.gif" 
                 alt="Explainable AI Visualization">
        </div>
    """, unsafe_allow_html=True)

    st.title("🫀 Advanced Heart Disease Analysis Dashboard")

    if 'model_analyzer' not in st.session_state:
        st.session_state.model_analyzer = ModelAnalyzer()
    if 'survival_analyzer' not in st.session_state:
        st.session_state.survival_analyzer = SurvivalAnalyzer()

    with st.sidebar:
        st.header("📊 Analysis Controls")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Data Overview", "Model Performance", "SHAP Analysis", 
             "Feature Engineering", "Survival Analysis"]
        )
        
        n_estimators = st.slider("Number of trees", 50, 200, 100)
        max_depth = st.slider("Maximum tree depth", 5, 20, 10)

    data = load_data()
    if data is None:
        return

    if analysis_type == "Data Overview":
        st.header("📋 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", f"{len(data):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Features", len(data.columns) - 1)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Disease Prevalence", f"{(data['cardio'].mean() * 100):.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("Sample Data")
        st.dataframe(data.head())
        
        st.subheader("Feature Distributions")
        col1, col2 = st.columns(2)
        with col1:
            feature = st.selectbox("Select feature:", data.columns)
        with col2:
            bin_count = st.slider("Number of bins", 10, 100, 30)
        
        fig = px.histogram(data, x=feature, color='cardio',
                         nbins=bin_count,
                         title=f"Distribution of {feature}",
                         labels={'cardio': 'Heart Disease'})
        st.plotly_chart(fig)

        if st.checkbox("Show Statistical Summary"):
            st.subheader("Statistical Summary")
            st.dataframe(data.describe())

    elif analysis_type == "Model Performance":
        st.header("📊 Model Performance")
        
        if st.button("Run Analysis"):
            progress = st.progress(0)
            
            X_train, X_test, y_train, y_test = st.session_state.model_analyzer.prepare_data(data)
            progress.progress(25)
            
            model, metrics = st.session_state.model_analyzer.compute_metrics(
                X_train, X_test, y_train, y_test, n_estimators, max_depth
            )
            progress.progress(50)
            
            st.session_state['current_model'] = model
            st.session_state['current_X_train'] = X_train
            
            progress.progress(100)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("CV Mean", f"{np.mean(metrics['cv_scores']):.2%}")
            col3.metric("CV Std", f"{np.std(metrics['cv_scores']):.2%}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_roc_curve(y_test, metrics['probabilities']))
            with col2:
                st.pyplot(plot_confusion_matrix(y_test, metrics['predictions']))
            
            importance_df = pd.DataFrame({
                'Feature': st.session_state.model_analyzer.feature_names,
                'Importance': list(metrics['feature_importance'].values())
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Feature', y='Importance',
                        title='Feature Importance',
                        color='Importance',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig)

    elif analysis_type == "SHAP Analysis":
        st.header("🎯 SHAP Analysis")
        
        if st.button("Generate SHAP Analysis"):
            if 'current_model' not in st.session_state:
                st.warning("Please run the Model Performance analysis first!")
                return
                
            with st.spinner("Computing SHAP values..."):
                shap_values, feature_names = st.session_state.model_analyzer.compute_shap_values(
                    st.session_state['current_model'],
                    st.session_state.model_analyzer.X_train
                )
                
                if shap_values is not None:
                    st.subheader("SHAP Summary Plot")
                    fig = plt.figure(figsize=(10, 8))
                    shap.summary_plot(
                        shap_values[1], 
                        st.session_state.model_analyzer.X_train,
                        feature_names=feature_names,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()

                    # Add SHAP force plot for individual predictions
                    st.subheader("Individual Prediction Explanation")
                    sample_idx = st.slider("Select sample index", 0, len(st.session_state.model_analyzer.X_train)-1, 0)
                    
                    fig = plt.figure(figsize=(10, 3))
                    shap.force_plot(
                        explainer.expected_value[1],
                        shap_values[1][sample_idx],
                        st.session_state.model_analyzer.X_train.iloc[sample_idx],
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()

    elif analysis_type == "Feature Engineering":
        st.header("🔧 Feature Engineering")
        
        st.write("""
        ### Engineered Features:
        - **BMI (Body Mass Index)**: Calculated from height and weight
        - **Pulse Pressure**: Difference between systolic and diastolic pressure
        - **MAP (Mean Arterial Pressure)**: Average blood pressure during cardiac cycle
        """)
        
        st.subheader("Interactive Feature Analysis")
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Select X-axis feature:", data.columns, index=data.columns.get_loc('age'))
        with col2:
            y_feature = st.selectbox("Select Y-axis feature:", data.columns, index=data.columns.get_loc('bmi'))
        
        fig = px.scatter(data, x=x_feature, y=y_feature, 
                        color='cardio',
                        color_discrete_map={0: 'blue', 1: 'red'},
                        title=f'Relationship between {x_feature} and {y_feature}',
                        labels={'cardio': 'Heart Disease'})
        st.plotly_chart(fig)
        
        st.subheader("Feature Correlations")
        corr_matrix = data.corr()
        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title("Feature Correlation Matrix")
        st.pyplot(fig)
        plt.close()

    else:  # Survival Analysis
        st.header("⏳ Survival Analysis")
        
        st.info("""
        This analysis examines the relationship between various risk factors and the 
        probability of heart disease over time. The survival probability represents 
        the likelihood of not developing heart disease at different ages.
        """)
        
        tab1, tab2 = st.tabs(["Survival Curves", "Risk Factor Analysis"])
        
        with tab1:
            group_col = st.selectbox(
                "Select grouping variable:", 
                ['None', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
            )
            
            if st.button("Generate Survival Curves"):
                with st.spinner("Calculating survival probabilities..."):
                    fig = st.session_state.survival_analyzer.analyze(
                        data,
                        None if group_col == 'None' else group_col
                    )
                    st.pyplot(fig)
                    plt.close()
                    
                    if group_col != 'None':
                        st.subheader(f"Risk Analysis for {group_col}")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            counts = data[group_col].value_counts().sort_index()
                            fig = px.bar(
                                x=counts.index,
                                y=counts.values,
                                title=f"Distribution of {group_col}",
                                labels={'x': group_col, 'y': 'Count'}
                            )
                            st.plotly_chart(fig)
                        
                        with col2:
                            prev = data.groupby(group_col)['cardio'].mean() * 100
                            fig = px.bar(
                                x=prev.index,
                                y=prev.values,
                                title=f"Disease Prevalence by {group_col}",
                                labels={'x': group_col, 'y': 'Prevalence (%)'}
                            )
                            st.plotly_chart(fig)
        
        with tab2:
            if st.button("Analyze Risk Factors"):
                with st.spinner("Analyzing risk factors..."):
                    risk_scores = st.session_state.survival_analyzer.analyze_risk_factors(data)
                    
                    fig = px.bar(
                        x=risk_scores.index,
                        y=risk_scores.values,
                        title="Risk Factor Impact Analysis",
                        labels={'x': 'Risk Factor', 'y': 'Risk Ratio'},
                        color=risk_scores.values,
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig)
                    
                    st.write("""
                    ### Interpretation:
                    - Risk Ratio > 1: Increased risk of heart disease
                    - Risk Ratio = 1: No effect
                    - Risk Ratio < 1: Decreased risk of heart disease
                    """)

    st.markdown("---")
    st.caption("Heart Disease Analysis Dashboard | Version 2.0")

if __name__ == "__main__":
    main()

      
