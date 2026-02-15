"""Streamlit ML Classification App"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, roc_curve, auc
from ml_models import MultiClassificationModels, prepare_dataset
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Classification", page_icon="🤖", layout="wide")


@st.cache_data
def load_dataset():
    """Load Breast Cancer dataset (569 samples, 30 features)"""
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    return df


@st.cache_data
def load_user_data(uploaded_file):
    """Load user uploaded CSV file"""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return None


def plot_confusion_matrix(cm, model_name):
    """Create confusion matrix heatmap"""
    fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[f"Class {i}" for i in range(cm.shape[1])],
                    y=[f"Class {i}" for i in range(cm.shape[0])],
                    color_continuous_scale="Blues", text_auto=True,
                    title=f"Confusion Matrix - {model_name}")
    fig.update_layout(width=500, height=400)
    return fig


def plot_roc_curves(results_dict, y_test):
    """Plot ROC curves for binary classification"""
    fig = go.Figure()
    for name, results in results_dict.items():
        if results['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, results['probabilities'][:, 1])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={roc_auc:.3f})', mode='lines'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', mode='lines', line=dict(dash='dash')))
    fig.update_layout(title='ROC Curves', xaxis_title='FPR', yaxis_title='TPR')
    return fig


def plot_metrics_comparison(comparison_df):
    """Create metrics comparison bar chart"""
    fig = go.Figure()
    for metric in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
        fig.add_trace(go.Bar(name=metric, x=comparison_df['Model'], y=comparison_df[metric],
                            text=comparison_df[metric].round(3), textposition='outside'))
    fig.update_layout(title='Model Performance Comparison', barmode='group', height=500)
    return fig


def main():
    st.title("🤖 ML Classification Models")
    st.markdown("### Breast Cancer Classification")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Dataset selection
    st.sidebar.subheader("1. Dataset")
    use_custom = st.sidebar.checkbox("Upload custom CSV")
    
    df = None
    target_column = None
    
    if use_custom:
        uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=['csv'])
        if uploaded_file:
            df = load_user_data(uploaded_file)
            if df is not None:
                st.sidebar.success("✓ File uploaded")
                target_column = st.sidebar.selectbox("Target column:", df.columns.tolist())
    else:
        df = load_dataset()
        target_column = 'target'
        st.sidebar.info("Using Breast Cancer Dataset\n569 samples, 30 features")
    
    # Model configuration
    st.sidebar.subheader("2. Configuration")
    test_size = st.sidebar.slider("Test size:", 0.1, 0.4, 0.2, 0.05)
    scale_features = st.sidebar.checkbox("Scale features", value=True)
    random_state = st.sidebar.number_input("Random state:", value=42, min_value=0)
    
    # Model selection dropdown
    st.sidebar.subheader("3. Model Selection")
    available_models = ['All Models', 'Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors',
                       'Naive Bayes', 'Random Forest', 'XGBoost', 'SVM', 'Gradient Boosting']
    selected_model_option = st.sidebar.selectbox("Select model to train:", available_models)
    
    train_button = st.sidebar.button("🚀 Train Models", type="primary", use_container_width=True)
    
    # Main content
    if df is not None and target_column is not None:
        st.header("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Samples", df.shape[0])
        col2.metric("Features", df.shape[1] - 1)
        col3.metric("Classes", df[target_column].nunique())
        col4.metric("Missing", df.isnull().sum().sum())
        
        with st.expander("🔍 View Data"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Training
        if train_button:
            with st.spinner("Training models..."):
                X_train, X_test, y_train, y_test, scaler, features = prepare_dataset(
                    df, target_column, test_size, scale_features, random_state)
                
                ml_models = MultiClassificationModels(X_train, X_test, y_train, y_test, random_state)
                ml_models.initialize_models()
                ml_models.train_all_models()
                ml_models.evaluate_all_models()
                
                st.session_state['ml_models'] = ml_models
                st.session_state['y_test'] = y_test
                st.success("✅ Training complete!")
        
        # Results display
        if 'ml_models' in st.session_state:
            ml_models = st.session_state['ml_models']
            y_test = st.session_state['y_test']
            
            st.markdown("---")
            st.header("📈 Model Results")
            
            comparison_df = ml_models.get_comparison_dataframe()
            
            # Display metrics table
            st.subheader("📋 Evaluation Metrics")
            st.dataframe(comparison_df.style.highlight_max(
                subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'], color='lightgreen'
            ).format({col: '{:.4f}' for col in comparison_df.columns if col != 'Model'}),
            use_container_width=True)
            
            # Download buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("📥 Download Metrics", comparison_df.to_csv(index=False),
                                 "metrics.csv", "text/csv", key="dl_metrics")
            with col2:
                test_df = pd.DataFrame({'True': y_test, 
                    'Predicted': ml_models.results[ml_models.get_best_model('accuracy')[0]]['predictions']})
                st.download_button("📥 Download Predictions", test_df.to_csv(index=False),
                                 "predictions.csv", "text/csv", key="dl_pred")
            with col3:
                import json
                results_dict = {name: {k: float(v) for k, v in res.items() 
                    if k in ['accuracy', 'precision', 'recall', 'f1_score', 'mcc']}
                    for name, res in ml_models.results.items()}
                st.download_button("📥 Download Results", json.dumps(results_dict, indent=2),
                                 "results.json", "application/json", key="dl_json")
            
            best_name, _, best_res = ml_models.get_best_model('accuracy')
            st.info(f"🏆 Best Model: {best_name} | Accuracy: {best_res['accuracy']:.4f}")
            
            # Visualizations
            st.subheader("📊 Performance Comparison")
            st.plotly_chart(plot_metrics_comparison(comparison_df), key="viz_comp")
            
            # ROC Curves
            st.subheader("📉 ROC Curves")
            st.plotly_chart(plot_roc_curves(ml_models.results, y_test), key="viz_roc")
            
            # Confusion Matrix
            st.subheader("🔲 Confusion Matrix")
            selected = st.selectbox("Select model for confusion matrix:", list(ml_models.results.keys()))
            if selected:
                st.plotly_chart(plot_confusion_matrix(ml_models.results[selected]['confusion_matrix'], selected),
                              key=f"cm_{selected}")
            
            with st.expander("View All Confusion Matrices"):
                models_list = list(ml_models.results.keys())
                for i in range(0, len(models_list), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(models_list):
                            name = models_list[i + j]
                            with col:
                                st.plotly_chart(plot_confusion_matrix(
                                    ml_models.results[name]['confusion_matrix'], name),
                                    key=f"cm_grid_{i}_{j}")
    else:
        st.info("Configure settings in sidebar and click Train Models")


if __name__ == "__main__":
    main()
