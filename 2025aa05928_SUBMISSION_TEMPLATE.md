# ML Assignment 2 - Submission Document

**Name:** Vinoth KR  
**BITS ID:** 2025aa05928  
**Date:** February 15, 2026

---

## 1. GitHub Repository Link

**Repository URL:** [TO BE FILLED - Your GitHub repository URL]

Example: `https://github.com/vinothkr/ml-classification-assignment`

**Note:** Repository contains `models/` folder as required.

---

## 2. Streamlit App Link

**Deployed App URL:** [TO BE FILLED - Your Streamlit Cloud app URL]

Example: `https://2025aa05928-ml-app.streamlit.app`

**Features Verified:**
- ✅ Download button for test CSV (predictions)
- ✅ Upload button for custom CSV datasets
- ✅ Metrics table with all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- ✅ Confusion matrix for all models

---

## 3. BITS Virtual Lab Screenshot

[INSERT SCREENSHOT HERE]

**Screenshot Requirements:**
- Open BITS Virtual Lab
- Open the Streamlit app URL in browser
- Show code execution (terminal with `streamlit run app.py`)
- Ensure URL is visible in browser address bar
- Capture full screen showing both browser and terminal

---

## 4. README Content

# ML Classification Models - Assignment 2

**Name:** Vinoth KR  
**BITS ID:** 2025aa05928  
**Date:** February 15, 2026

---

## Problem Statement

Develop a machine learning classification system to predict breast cancer diagnosis (malignant or benign) using multiple classification algorithms. Build an interactive web application using Streamlit to demonstrate model training, evaluation, and comparison. Deploy the application on Streamlit Community Cloud for real-world accessibility.

---

## Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Source:** UCI Machine Learning Repository (available via sklearn.datasets)

**Dataset Characteristics:**
- Total Samples: 569 instances
- Features: 30 numeric features
- Target Classes: 2 (Malignant=0, Benign=1)
- Feature Types: Real-valued
- Missing Values: None

**Features Include:**
- Radius (mean, standard error, worst)
- Texture (mean, standard error, worst)
- Perimeter, Area, Smoothness
- Compactness, Concavity, Concave points
- Symmetry, Fractal dimension

All features are computed from digitized images of fine needle aspirate (FNA) of breast mass.

**Dataset Split:**
- Training Set: 80% (455 samples)
- Test Set: 20% (114 samples)
- Feature Scaling: StandardScaler applied

---

## Models Implemented

### 1. Logistic Regression
Linear model using sigmoid function for binary classification. Uses L-BFGS solver with maximum 1000 iterations. Effective for linearly separable data with interpretable coefficients.

### 2. Decision Tree
Tree-based classifier using Gini impurity criterion. Maximum depth limited to 10 levels to prevent overfitting. Provides feature importance and interpretable decision paths.

### 3. K-Nearest Neighbors (KNN)
Distance-based classifier using k=5 neighbors. Employs Euclidean distance metric with uniform weighting. Effective when decision boundary is irregular.

### 4. Naive Bayes
Probabilistic classifier based on Bayes theorem with Gaussian distribution assumption. Fast training and prediction. Works well despite feature independence assumption.

### 5. Random Forest (Ensemble)
Ensemble of 100 decision trees using bootstrap sampling. Combines predictions through voting mechanism. Reduces overfitting through averaging and provides feature importance.

### 6. XGBoost (Ensemble)
Gradient boosting implementation with 100 estimators. Uses regularization to prevent overfitting. Handles imbalanced data and missing values effectively. State-of-the-art performance.

---

## Evaluation Metrics Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9825 | 0.9991 | 0.9836 | 0.9825 | 0.9824 | 0.9648 |
| Decision Tree | 0.9298 | 0.9245 | 0.9318 | 0.9298 | 0.9292 | 0.8589 |
| kNN | 0.9649 | 0.9952 | 0.9655 | 0.9649 | 0.9646 | 0.9293 |
| Naive Bayes | 0.9474 | 0.9960 | 0.9512 | 0.9474 | 0.9465 | 0.8939 |
| Random Forest | 0.9649 | 0.9991 | 0.9655 | 0.9649 | 0.9646 | 0.9293 |
| XGBoost | 0.9737 | 0.9986 | 0.9743 | 0.9737 | 0.9735 | 0.9470 |

---

## Model Performance Observations

**Logistic Regression:** Achieved highest accuracy of 98.25% with perfect AUC (0.9991). Excellent balance between precision and recall. MCC of 0.9648 indicates very strong correlation. Best performer despite being a simple linear model, suggesting data is largely linearly separable.

**Decision Tree:** Lowest accuracy at 92.98% among all models. Shows tendency to overfit despite depth limitation. However, provides interpretable rules and feature importance. Useful for understanding decision boundaries.

**K-Nearest Neighbors:** Strong performance with 96.49% accuracy. Benefits significantly from feature scaling. Distance-based approach captures local patterns effectively. Computational cost increases with dataset size.

**Naive Bayes:** Solid performance (94.74%) despite independence assumption. Very fast training and prediction. High AUC (0.9960) indicates good ranking ability. Suitable for real-time applications.

**Random Forest:** Excellent performance matching KNN at 96.49%. Ensemble approach reduces variance and overfitting. Perfect AUC score. Provides robust predictions and feature importance rankings.

**XGBoost:** Second-best performance with 97.37% accuracy. Superior gradient boosting implementation. Excellent generalization with MCC of 0.9470. Handles complex patterns through sequential tree building. Recommended for production deployment.

**Overall Finding:** All models perform well (>92% accuracy), indicating good feature quality. Ensemble methods (Random Forest, XGBoost) and Logistic Regression show superior performance. Feature scaling improves KNN and SVM significantly.

---

## Repository Structure

```
project-folder/
├── app.py              # Streamlit application
├── ml_models.py        # Model implementations  
├── requirements.txt    # Python dependencies
├── models/             # Saved model files folder
└── README.md           # Project documentation
```

---

## Setup Instructions

```bash
# Clone repository
git clone <repository-url>
cd <repository-folder>

# Install dependencies
pip install -r requirements.txt

# Run Streamlit application
streamlit run app.py
```

Application opens at: http://localhost:8501

---

## Streamlit App Features

### Mandatory Features:

**a. Dataset upload option (CSV)** ✅  
   - Upload custom test datasets via sidebar
   - Supports CSV format with configurable target column
   - Built-in Breast Cancer dataset (569 samples, 30 features)

**b. Model selection dropdown** ✅  
   - Dropdown to select individual models or train all
   - 8 models available: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost, SVM, Gradient Boosting

**c. Display of evaluation metrics** ✅  
   - Comprehensive metrics table showing all 6 metrics
   - Accuracy, AUC Score, Precision, Recall, F1 Score, MCC
   - Best model automatically highlighted
   - Download metrics as CSV

**d. Confusion matrix or classification report** ✅  
   - Interactive confusion matrix for selected model
   - Grid view showing all models' confusion matrices
   - Plotly visualizations with hover details

### Additional Features:
- ROC curve comparison for all models
- Performance comparison bar chart
- Download predictions and full results
- Configurable train-test split ratio
- Feature scaling toggle
- Real-time model training

---

## Deployment

Deployed on Streamlit Community Cloud.

**Live App URL:** [TO BE FILLED]

---

**ML Assignment 2 | BITS Pilani | February 2026**

---

## Submission Checklist

- [ ] GitHub repository created with `models/` folder
- [ ] No training data uploaded to GitHub
- [ ] Streamlit app deployed on Community Cloud
- [ ] App includes download & upload buttons
- [ ] App displays metrics table and confusion matrix
- [ ] BITS Virtual Lab screenshot captured (showing URL and code)
- [ ] README content added to PDF
- [ ] PDF file named `2025aa05928.pdf`
- [ ] Submitted before deadline (15 Feb 23:59 PM IST)

---

**End of Submission Document**
