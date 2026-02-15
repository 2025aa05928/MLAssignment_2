"""
Streamlit Web Application for ML Classification Models
Interactive UI for model demonstration and evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Import custom models
from ml_models import MultiClassificationModels, prepare_dataset

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_sample_datasets():
    """Load sample datasets for demonstration."""
    import pandas as pd
    
    # Load Iris
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    
    # Load Breast Cancer
    cancer = load_breast_cancer()
    cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    cancer_df['target'] = cancer.target
    
    # Load Wine
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['target'] = wine.target
    
    datasets = {
        'Iris': {'frame': iris_df, 'target_names': iris.target_names, 'target': iris.target},
        'Breast Cancer': {'frame': cancer_df, 'target_names': cancer.target_names, 'target': cancer.target},
        'Wine': {'frame': wine_df, 'target_names': wine.target_names, 'target': wine.target}
    }
    return datasets


@st.cache_data
def load_user_data(uploaded_file):
    """Load user uploaded dataset."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def plot_confusion_matrix(cm, model_name):
    """Create a heatmap for confusion matrix."""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[f"Class {i}" for i in range(cm.shape[1])],
        y=[f"Class {i}" for i in range(cm.shape[0])],
        color_continuous_scale="Blues",
        text_auto=True,
        title=f"Confusion Matrix - {model_name}"
    )
    fig.update_layout(width=500, height=400)
    return fig


def plot_roc_curves(results_dict, y_test):
    """Plot ROC curves for all models."""
    fig = go.Figure()
    
    for name, results in results_dict.items():
        if results['probabilities'] is not None and len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, results['probabilities'][:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'{name} (AUC = {roc_auc:.3f})',
                mode='lines'
            ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves - All Models',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_metrics_comparison(comparison_df):
    """Create bar chart comparing model metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df['Model'],
            y=comparison_df[metric],
            text=comparison_df[metric].round(3),
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🤖 Machine Learning Classification Models")
    st.markdown("### Interactive Model Training and Evaluation Platform")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Dataset selection
    st.sidebar.subheader("1. Select Dataset")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Sample Datasets", "Upload Custom Dataset"]
    )
    
    df = None
    target_column = None
    
    if data_source == "Sample Datasets":
        datasets = load_sample_datasets()
        dataset_name = st.sidebar.selectbox(
            "Select a dataset:",
            list(datasets.keys())
        )
        
        dataset = datasets[dataset_name]
        df = dataset['frame']
        target_column = 'target'
        
        st.sidebar.success(f"✓ Loaded {dataset_name} dataset")
        
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload your dataset (CSV or Excel)",
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            df = load_user_data(uploaded_file)
            if df is not None:
                st.sidebar.success("✓ File uploaded successfully")
                target_column = st.sidebar.selectbox(
                    "Select target column:",
                    df.columns.tolist()
                )
    
    # Model configuration
    st.sidebar.subheader("2. Model Configuration")
    test_size = st.sidebar.slider("Test set size:", 0.1, 0.4, 0.2, 0.05)
    scale_features = st.sidebar.checkbox("Scale features", value=True)
    random_state = st.sidebar.number_input("Random state:", value=42, min_value=0)
    
    # Train button
    train_button = st.sidebar.button("🚀 Train All Models", type="primary", use_container_width=True)
    
    # Main content
    if df is not None and target_column is not None:
        
        # Dataset overview
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1] - 1)
        with col3:
            st.metric("Classes", df[target_column].nunique())
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Show data preview
        with st.expander("🔍 View Dataset", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Statistical Summary")
                st.dataframe(df.describe(), use_container_width=True)
            with col2:
                st.subheader("Target Distribution")
                target_counts = df[target_column].value_counts()
                fig = px.pie(
                    values=target_counts.values,
                    names=target_counts.index,
                    title=f"Distribution of {target_column}"
                )
                st.plotly_chart(fig, use_container_width=True, key="target_distribution_pie")
        
        # Model training and evaluation
        if train_button:
            with st.spinner("Training models... This may take a moment."):
                
                # Prepare dataset
                X_train, X_test, y_train, y_test, scaler, feature_names = prepare_dataset(
                    df, target_column, test_size, scale_features, random_state
                )
                
                # Initialize and train models
                ml_models = MultiClassificationModels(X_train, X_test, y_train, y_test, random_state)
                ml_models.initialize_models()
                ml_models.train_all_models()
                ml_models.evaluate_all_models()
                
                # Store in session state
                st.session_state['ml_models'] = ml_models
                st.session_state['y_test'] = y_test
                
                st.success("✅ All models trained successfully!")
        
        # Display results if models are trained
        if 'ml_models' in st.session_state:
            ml_models = st.session_state['ml_models']
            y_test = st.session_state['y_test']
            
            st.markdown("---")
            st.header("📈 Model Performance Results")
            
            # Get comparison dataframe
            comparison_df = ml_models.get_comparison_dataframe()
            
            # Display metrics table
            st.subheader("📋 Performance Metrics Comparison")
            st.dataframe(
                comparison_df.style.highlight_max(
                    subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    color='lightgreen'
                ).format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1-Score': '{:.4f}',
                    'ROC AUC': '{:.4f}',
                    'CV Mean': '{:.4f}',
                    'CV Std': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # Download buttons for results
            col1, col2, col3 = st.columns(3)
            with col1:
                # Download metrics as CSV
                metrics_csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Metrics CSV",
                    data=metrics_csv,
                    file_name="model_metrics.csv",
                    mime="text/csv",
                    key="download_metrics"
                )
            with col2:
                # Download test data predictions
                test_df = pd.DataFrame({
                    'True_Label': y_test,
                    'Best_Model_Prediction': ml_models.results[ml_models.get_best_model('accuracy')[0]]['predictions']
                })
                test_csv = test_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Test Predictions",
                    data=test_csv,
                    file_name="test_predictions.csv",
                    mime="text/csv",
                    key="download_predictions"
                )
            with col3:
                # Download full results as JSON
                import json
                results_dict = {name: {
                    'accuracy': float(res['accuracy']),
                    'precision': float(res['precision']),
                    'recall': float(res['recall']),
                    'f1_score': float(res['f1_score'])
                } for name, res in ml_models.results.items()}
                results_json = json.dumps(results_dict, indent=2)
                st.download_button(
                    label="📥 Download Full Results",
                    data=results_json,
                    file_name="results.json",
                    mime="application/json",
                    key="download_json"
                )
            
            # Best model highlight
            best_model_name, best_model, best_results = ml_models.get_best_model('accuracy')
            st.info(f"🏆 **Best Model:** {best_model_name} with Accuracy: {best_results['accuracy']:.4f}")
            
            # Visualizations
            st.subheader("📊 Performance Visualizations")
            
            # Metrics comparison chart
            st.plotly_chart(plot_metrics_comparison(comparison_df), use_container_width=True, key="metrics_comparison_chart")
            
            # ROC curves (for binary classification)
            if len(np.unique(y_test)) == 2:
                st.subheader("📉 ROC Curves")
                roc_fig = plot_roc_curves(ml_models.results, y_test)
                st.plotly_chart(roc_fig, use_container_width=True, key="roc_curves_chart")
            
            # Confusion matrices
            st.subheader("🔲 Confusion Matrices")
            
            # Model selector for detailed view
            selected_model = st.selectbox(
                "Select model for detailed confusion matrix:",
                list(ml_models.results.keys())
            )
            
            if selected_model:
                results = ml_models.results[selected_model]
                cm_fig = plot_confusion_matrix(results['confusion_matrix'], selected_model)
                st.plotly_chart(cm_fig, use_container_width=True, key=f"cm_selected_{selected_model}")
            
            # Display all confusion matrices in grid
            with st.expander("View All Confusion Matrices"):
                models_list = list(ml_models.results.keys())
                n_models = len(models_list)
                n_cols = 3
                n_rows = (n_models + n_cols - 1) // n_cols
                
                for i in range(0, n_models, n_cols):
                    cols = st.columns(n_cols)
                    for j, col in enumerate(cols):
                        if i + j < n_models:
                            model_name = models_list[i + j]
                            results = ml_models.results[model_name]
                            with col:
                                cm_fig = plot_confusion_matrix(
                                    results['confusion_matrix'],
                                    model_name
                                )
                                st.plotly_chart(cm_fig, use_container_width=True, key=f"cm_grid_{i}_{j}_{model_name}")
            
            # Model details
            st.markdown("---")
            st.subheader("🔍 Detailed Model Analysis")
            
            detailed_model = st.selectbox(
                "Select model for detailed analysis:",
                list(ml_models.results.keys()),
                key="detailed_model"
            )
            
            if detailed_model:
                results = ml_models.results[detailed_model]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{results['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{results['recall']:.4f}")
                with col4:
                    st.metric("F1-Score", f"{results['f1_score']:.4f}")
                
                if results['roc_auc']:
                    st.metric("ROC AUC Score", f"{results['roc_auc']:.4f}")
                
                if results['cv_mean']:
                    st.info(f"**Cross-Validation Score:** {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
    
    else:
        st.info("👈 Please select or upload a dataset from the sidebar to begin.")
        
        # Show instructions
        st.markdown("""
        ### 📖 How to Use This Application
        
        1. **Select Dataset**: Choose from sample datasets or upload your own CSV/Excel file
        2. **Configure Models**: Adjust test set size, feature scaling, and random state
        3. **Train Models**: Click the "Train All Models" button to train 8 different classifiers
        4. **Analyze Results**: View performance metrics, visualizations, and detailed analysis
        
        ### 🎯 Available Models
        
        This application trains and compares the following classification models:
        
        - Logistic Regression
        - Random Forest Classifier
        - Support Vector Machine (SVM)
        - K-Nearest Neighbors (KNN)
        - Decision Tree Classifier
        - Gradient Boosting Classifier
        - XGBoost Classifier
        - Naive Bayes Classifier
        
        ### 📊 Evaluation Metrics
        
        - **Accuracy**: Overall correctness of predictions
        - **Precision**: Ratio of correct positive predictions
        - **Recall**: Ratio of actual positives identified
        - **F1-Score**: Harmonic mean of precision and recall
        - **ROC AUC**: Area under the ROC curve (binary classification)
        - **Cross-Validation**: 5-fold CV for robust evaluation
        
        ### 💡 Tips
        
        - Use feature scaling for distance-based algorithms (KNN, SVM)
        - Larger test sets provide more reliable evaluation
        - Compare multiple metrics to choose the best model
        - Check confusion matrices for detailed error analysis
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>ML Assignment 2 - Classification Models | Built with Streamlit</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
