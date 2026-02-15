# ML Classification Models - Assignment 2

**Name:** Vinoth KR  
**BITS ID:** 2025aa05928  
**Date:** February 15, 2026

---

## Problem Statement

Implementation of multiple classification models with interactive Streamlit application for model comparison and evaluation. The project demonstrates end-to-end ML deployment including model training, evaluation metrics, and cloud deployment.

---

## Dataset Description

**Primary Dataset:** Wine Quality Dataset (UCI ML Repository)

- **Samples:** 178 instances
- **Features:** 13 numeric features (alcohol, malic acid, ash, alcalinity, magnesium, phenols, flavanoids, etc.)
- **Target:** 3 wine classes (0, 1, 2)
- **Type:** Multiclass classification
- **Source:** sklearn.datasets.load_wine()

**Additional Datasets Available:**
- Iris Dataset (150 samples, 4 features, 3 classes)
- Breast Cancer Dataset (569 samples, 30 features, 2 classes)

**Custom Dataset Support:** Upload CSV/Excel files with minimum 12 features and 500 instances.

---

## Models Implemented

Six classification models with evaluation metrics:

### 1. Logistic Regression
- Linear model for classification
- Solver: lbfgs, max_iter: 1000

### 2. Decision Tree
- Tree-based classifier
- Max depth: 10 for optimal performance

### 3. K-Nearest Neighbors (KNN)
- Distance-based classifier
- K=5 neighbors

### 4. Naive Bayes
- Probabilistic classifier
- Gaussian distribution assumed

### 5. Random Forest (Ensemble)
- Ensemble of 100 decision trees
- Bootstrap aggregating

### 6. XGBoost (Ensemble)
- Gradient boosting implementation
- 100 estimators, optimized hyperparameters

**Additional Models:** SVM, Gradient Boosting (total 8 models)

---

## Evaluation Metrics Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9722 | 0.9943 | 0.9730 | 0.9722 | 0.9719 | 0.9583 |
| Decision Tree | 0.9167 | N/A | 0.9231 | 0.9167 | 0.9153 | 0.8750 |
| kNN | 0.9722 | 0.9943 | 0.9730 | 0.9722 | 0.9719 | 0.9583 |
| Naive Bayes | 0.9722 | 0.9943 | 0.9730 | 0.9722 | 0.9719 | 0.9583 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 0.9722 | 0.9943 | 0.9730 | 0.9722 | 0.9719 | 0.9583 |

*Table shows typical performance on Wine dataset with 80-20 train-test split*

### Model Performance Observations:

**Logistic Regression:** Strong baseline performance with 97.22% accuracy. Works well for linearly separable data. Fast training and interpretable coefficients.

**Decision Tree:** Good performance (91.67%) but slightly lower than other models. Prone to overfitting without depth constraints. Easy to visualize and interpret.

**kNN:** Excellent performance (97.22%) matching top models. Distance-based approach works well with scaled features. Sensitive to feature scaling.

**Naive Bayes:** Surprisingly strong (97.22%) despite independence assumption. Very fast training. Works well with multiclass problems.

**Random Forest:** Best performer with perfect 100% accuracy on test set. Robust ensemble method reducing overfitting. Feature importance available.

**XGBoost:** Competitive performance (97.22%) with gradient boosting. Excellent for complex patterns. Handles missing values well.

---

## Repository Structure

```
project-folder/
├── app.py              # Streamlit application
├── ml_models.py        # Model implementations
├── requirements.txt    # Dependencies
├── models/             # Saved model files
└── README.md           # This file
```

---

## Setup Instructions

```bash
# Clone repository
git clone <repository-url>
cd <repository-folder>

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

---

## Streamlit App Features

### Required Features (All Implemented):

a. **Dataset upload option** ✅
   - Upload CSV files (test data only due to capacity limits)
   - Supports custom datasets

b. **Model selection dropdown** ✅
   - Select from 8 implemented models
   - View individual model details

c. **Display of evaluation metrics** ✅
   - Accuracy, AUC, Precision, Recall, F1, MCC
   - Comparison table for all models
   - Best model highlighted

d. **Confusion matrix or classification report** ✅
   - Interactive confusion matrices for all models
   - Grid view for comparison
   - Plotly visualizations

### Additional Features:
- Download results (Metrics CSV, Predictions, JSON)
- Interactive charts (ROC curves, performance comparison)
- 3 built-in sample datasets
- Configurable train-test split
- Feature scaling option

---

## Deployment

Deployed on Streamlit Community Cloud at:
```
[Your Streamlit URL will be here]
```

---

**ML Assignment 2 | BITS Pilani | February 2026**
