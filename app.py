"""Streamlit ML Classification App"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.metrics import confusion_matrix, roc_curve, auc
from ml_models import MultiClassificationModels, prepare_dataset
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Classification", page_icon="🤖", layout="wide")


@st.cache_data
def load_sample_datasets():
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    
    cancer = load_breast_cancer()
    cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    cancer_df['target'] = cancer.target
    
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['target'] = wine.target
    
    return {
        'Iris': {'frame': iris_df},
        'Breast Cancer': {'frame': cancer_df},
        'Wine': {'frame': wine_df}
    }


@st.cache_data
def load_user_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return None


def plot_confusion_matrix(cm, model_name):
    fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[f"Class {i}" for i in range(cm.shape[1])],
                    y=[f"Class {i}" for i in range(cm.shape[0])],
                    color_continuous_scale="Blues", text_auto=True,
                    title=f"Confusion Matrix - {model_name}")
    fig.update_layout(width=500, height=400)
    return fig


def plot_roc_curves(results_dict, y_test):
    fig = go.Figure()
    for name, results in results_dict.items():
        if results['probabilities'] is not None and len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, results['probabilities'][:, 1])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={roc_auc:.3f})', mode='lines'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', mode='lines', line=dict(dash='dash', color='gray')))
    fig.update_layout(title='ROC Curves', xaxis_title='FPR', yaxis_title='TPR', width=700, height=500)
    return fig


def plot_metrics_comparison(comparison_df):
    fig = go.Figure()
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        fig.add_trace(go.Bar(name=metric, x=comparison_df['Model'], y=comparison_df[metric],
                            text=comparison_df[metric].round(3), textposition='outside'))
    fig.update_layout(title='Model Performance', barmode='group', height=500)
    return fig


def main():
    st.title("🤖 ML Classification Models")
    st.markdown("### Interactive Model Training Platform")
    st.markdown("---")
    
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.subheader("1. Dataset")
    data_source = st.sidebar.radio("Source:", ["Sample Datasets", "Upload Custom"])
    
    df = None
    target_column = None
    
    if data_source == "Sample Datasets":
        datasets = load_sample_datasets()
        dataset_name = st.sidebar.selectbox("Select:", list(datasets.keys()))
        df = datasets[dataset_name]['frame']
        target_column = 'target'
        st.sidebar.success(f"✓ Loaded {dataset_name}")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx', 'xls'])
        if uploaded_file:
            df = load_user_data(uploaded_file)
            if df is not None:
                st.sidebar.success("✓ Uploaded")
                target_column = st.sidebar.selectbox("Target column:", df.columns.tolist())
    
    st.sidebar.subheader("2. Configuration")
    test_size = st.sidebar.slider("Test size:", 0.1, 0.4, 0.2, 0.05)
    scale_features = st.sidebar.checkbox("Scale features", value=True)
    random_state = st.sidebar.number_input("Random state:", value=42, min_value=0)
    train_button = st.sidebar.button("🚀 Train All Models", type="primary", use_container_width=True)
    
    if df is not None and target_column is not None:
        st.header("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Samples", df.shape[0])
        col2.metric("Features", df.shape[1] - 1)
        col3.metric("Classes", df[target_column].nunique())
        col4.metric("Missing", df.isnull().sum().sum())
        
        with st.expander("🔍 View Data"):
            st.dataframe(df.head(10), use_container_width=True)
        
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
        
        if 'ml_models' in st.session_state:
            ml_models = st.session_state['ml_models']
            y_test = st.session_state['y_test']
            
            st.markdown("---")
            st.header("📈 Results")
            comparison_df = ml_models.get_comparison_dataframe()
            
            st.subheader("📋 Metrics")
            st.dataframe(comparison_df.style.highlight_max(
                subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='lightgreen'
            ).format({col: '{:.4f}' for col in comparison_df.columns if col != 'Model'}),
            use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("📥 Metrics CSV", comparison_df.to_csv(index=False),
                                 "metrics.csv", "text/csv", key="dl_metrics")
            with col2:
                test_df = pd.DataFrame({'True': y_test, 
                    'Predicted': ml_models.results[ml_models.get_best_model('accuracy')[0]]['predictions']})
                st.download_button("📥 Predictions", test_df.to_csv(index=False),
                                 "predictions.csv", "text/csv", key="dl_pred")
            with col3:
                import json
                results_dict = {name: {k: float(v) for k, v in res.items() 
                    if k in ['accuracy', 'precision', 'recall', 'f1_score']}
                    for name, res in ml_models.results.items()}
                st.download_button("📥 Results JSON", json.dumps(results_dict, indent=2),
                                 "results.json", "application/json", key="dl_json")
            
            best_name, _, best_res = ml_models.get_best_model('accuracy')
            st.info(f"🏆 Best: {best_name} (Accuracy: {best_res['accuracy']:.4f})")
            
            st.subheader("📊 Visualizations")
            st.plotly_chart(plot_metrics_comparison(comparison_df), use_container_width=True, key="viz_comp")
            
            if len(np.unique(y_test)) == 2:
                st.subheader("📉 ROC Curves")
                st.plotly_chart(plot_roc_curves(ml_models.results, y_test), use_container_width=True, key="viz_roc")
            
            st.subheader("🔲 Confusion Matrices")
            selected = st.selectbox("Select model:", list(ml_models.results.keys()))
            if selected:
                st.plotly_chart(plot_confusion_matrix(ml_models.results[selected]['confusion_matrix'], selected),
                              use_container_width=True, key=f"cm_{selected}")
            
            with st.expander("All Confusion Matrices"):
                models_list = list(ml_models.results.keys())
                for i in range(0, len(models_list), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(models_list):
                            name = models_list[i + j]
                            with col:
                                st.plotly_chart(plot_confusion_matrix(
                                    ml_models.results[name]['confusion_matrix'], name),
                                    use_container_width=True, key=f"cm_grid_{i}_{j}")
    else:
        st.info("👈 Select or upload a dataset to begin")


if __name__ == "__main__":
    main()
