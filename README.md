# 🤖 Machine Learning Classification Models - Streamlit Application

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 📋 Project Overview

This project implements **8 different classification machine learning models** with an interactive Streamlit web application for model training, evaluation, and comparison. The application provides comprehensive performance metrics, visualizations, and supports both built-in sample datasets and custom dataset uploads.

## 🎯 Features

### Multiple Classification Models
- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**
- **Gradient Boosting Classifier**
- **XGBoost Classifier**
- **Naive Bayes Classifier**

### Interactive Streamlit Dashboard
- ✅ Dataset selection (sample datasets or custom upload)
- ✅ Configurable hyperparameters
- ✅ Real-time model training
- ✅ Comprehensive performance metrics
- ✅ Interactive visualizations
- ✅ Confusion matrix analysis
- ✅ ROC curve comparison
- ✅ Cross-validation scores

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC Score
- Confusion Matrix
- 5-Fold Cross-Validation

## 🚀 Live Demo

**Streamlit App Link:** [Your Deployed App URL Here]

> The application is deployed on Streamlit Community Cloud (FREE)

## 📁 Project Structure

```
ML/
│
├── app.py                  # Main Streamlit application
├── ml_models.py           # ML models implementation
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore           # Git ignore file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone the repository**
```bash
git clone <your-github-repo-url>
cd ML
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit application**
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📊 How to Use

### Step 1: Select Dataset
- Choose from **sample datasets** (Iris, Breast Cancer, Wine)
- Or **upload your own** CSV/Excel file

### Step 2: Configure Models
- Adjust test set size (10% - 40%)
- Enable/disable feature scaling
- Set random state for reproducibility

### Step 3: Train Models
- Click **"Train All Models"** button
- Wait for training to complete (usually < 1 minute)

### Step 4: Analyze Results
- View **performance metrics table**
- Explore **interactive visualizations**
- Compare **confusion matrices**
- Analyze **ROC curves** (for binary classification)
- Check **detailed model statistics**

## 📈 Sample Datasets Included

1. **Iris Dataset**
   - 150 samples, 4 features
   - 3 classes (flower species)
   
2. **Breast Cancer Dataset**
   - 569 samples, 30 features
   - 2 classes (malignant/benign)
   
3. **Wine Dataset**
   - 178 samples, 13 features
   - 3 classes (wine types)

## 🔧 Custom Dataset Format

Your custom dataset should be in CSV or Excel format with:
- Features in columns
- One target/label column
- No missing values (or minimal preprocessing required)
- Numeric features (categorical encoding supported)

Example format:
```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
...
```

## 📦 Dependencies

```
streamlit==1.31.0
pandas==2.2.0
numpy==1.26.3
scikit-learn==1.4.0
xgboost==2.0.3
matplotlib==3.8.2
seaborn==0.13.2
plotly==5.18.0
imbalanced-learn==0.12.0
```

## 🌐 Deployment on Streamlit Community Cloud

### Step 1: Prepare Repository
1. Push your code to GitHub
2. Ensure `requirements.txt` is up to date
3. Make repository public

### Step 2: Deploy on Streamlit
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Set main file path: `app.py`
6. Click **"Deploy"**

### Step 3: Share Your App
Once deployed, you'll receive a public URL like:
```
https://your-app-name.streamlit.app
```

## 📝 Assignment Submission Checklist

- [ ] ✅ Implemented multiple classification models (8 models)
- [ ] ✅ Created interactive Streamlit application
- [ ] ✅ Uploaded code to GitHub repository
- [ ] ✅ Deployed app on Streamlit Community Cloud
- [ ] ✅ Tested app functionality
- [ ] ✅ Shared clickable links:
  - GitHub Repository: `<your-repo-url>`
  - Streamlit App: `<your-app-url>`
- [ ] ✅ Took screenshot of BITS Lab execution (if applicable)

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline implementation
- Multiple classification algorithms
- Model evaluation and comparison
- Interactive web application development
- Cloud deployment on Streamlit
- Version control with Git/GitHub

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Module not found error**
```bash
# Solution: Ensure all dependencies are installed
pip install -r requirements.txt
```

**Issue 2: Streamlit not opening**
```bash
# Solution: Check if port 8501 is available or specify different port
streamlit run app.py --server.port 8502
```

**Issue 3: Memory error with large datasets**
```
# Solution: Reduce dataset size or use sampling
# Add this in app.py if needed:
df = df.sample(n=10000, random_state=42)
```

## 📧 Contact & Support

For any issues or questions:
- Email: neha.vinayak@pilani.bits-pilani.ac.in
- Subject: "ML Assignment 2: [Your Query]"

## 📄 License

This project is created for educational purposes as part of ML Assignment 2.

## 🙏 Acknowledgments

- BITS Pilani Virtual Lab
- Streamlit Community
- scikit-learn Documentation
- XGBoost Team

---

### 📸 Screenshots

*Add screenshots of your application here after deployment*

**Dashboard View:**
![Dashboard](screenshots/dashboard.png)

**Model Comparison:**
![Comparison](screenshots/comparison.png)

**Confusion Matrix:**
![Confusion Matrix](screenshots/confusion_matrix.png)

---

**Created for ML Assignment 2 | BITS Pilani | February 2026**
