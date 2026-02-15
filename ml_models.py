"""
Machine Learning Classification Models Implementation
Author: ML Assignment 2
Date: February 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class MultiClassificationModels:
    """
    A comprehensive class to train and evaluate multiple classification models.
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, random_state=42):
        """
        Initialize the classification models framework.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        X_test : array-like
            Testing features
        y_train : array-like
            Training labels
        y_test : array-like
            Testing labels
        random_state : int
            Random state for reproducibility
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.trained_models = {}
        
    def initialize_models(self):
        """Initialize all classification models."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Support Vector Machine': SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'Naive Bayes': GaussianNB()
        }
        
    def train_all_models(self):
        """Train all initialized models."""
        print("=" * 80)
        print("TRAINING MULTIPLE CLASSIFICATION MODELS")
        print("=" * 80)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                self.trained_models[name] = model
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"✗ Error training {name}: {str(e)}")
                
    def evaluate_model(self, name, model):
        """
        Evaluate a single model and store results.
        
        Parameters:
        -----------
        name : str
            Name of the model
        model : sklearn estimator
            Trained model object
            
        Returns:
        --------
        dict : Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = None
        
        # Get probability predictions if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Handle binary and multiclass scenarios
        n_classes = len(np.unique(self.y_test))
        average_method = 'binary' if n_classes == 2 else 'weighted'
        
        precision = precision_score(self.y_test, y_pred, average=average_method, zero_division=0)
        recall = recall_score(self.y_test, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average=average_method, zero_division=0)
        
        # Calculate ROC AUC if probability predictions are available
        roc_auc = None
        if y_pred_proba is not None:
            try:
                if n_classes == 2:
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except:
            cv_mean = None
            cv_std = None
        
        results = {
            'model_name': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return results
    
    def evaluate_all_models(self):
        """Evaluate all trained models."""
        print("\n" + "=" * 80)
        print("EVALUATING ALL MODELS")
        print("=" * 80)
        
        for name, model in self.trained_models.items():
            print(f"\nEvaluating {name}...")
            results = self.evaluate_model(name, model)
            self.results[name] = results
            
            print(f"  Accuracy:  {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall:    {results['recall']:.4f}")
            print(f"  F1-Score:  {results['f1_score']:.4f}")
            if results['roc_auc']:
                print(f"  ROC AUC:   {results['roc_auc']:.4f}")
            if results['cv_mean']:
                print(f"  CV Score:  {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
    
    def get_comparison_dataframe(self):
        """
        Create a comparison dataframe of all models.
        
        Returns:
        --------
        pd.DataFrame : Comparison of all model metrics
        """
        comparison_data = []
        
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC AUC': results['roc_auc'] if results['roc_auc'] else np.nan,
                'CV Mean': results['cv_mean'] if results['cv_mean'] else np.nan,
                'CV Std': results['cv_std'] if results['cv_std'] else np.nan
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_best_model(self, metric='accuracy'):
        """
        Get the best performing model based on specified metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison (accuracy, f1_score, roc_auc)
            
        Returns:
        --------
        tuple : (model_name, model_object, results)
        """
        best_score = -1
        best_model_name = None
        
        for name, results in self.results.items():
            score = results.get(metric, 0)
            if score and score > best_score:
                best_score = score
                best_model_name = name
        
        if best_model_name:
            return (
                best_model_name,
                self.trained_models[best_model_name],
                self.results[best_model_name]
            )
        return None
    
    def print_summary(self):
        """Print a comprehensive summary of all models."""
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        
        df = self.get_comparison_dataframe()
        print("\n" + df.to_string(index=False))
        
        print("\n" + "=" * 80)
        best = self.get_best_model('accuracy')
        if best:
            print(f"BEST MODEL (by Accuracy): {best[0]}")
            print(f"Accuracy: {best[2]['accuracy']:.4f}")
            print("=" * 80)


def prepare_dataset(data, target_column, test_size=0.2, scale_features=True, random_state=42):
    """
    Prepare dataset for training: split and optionally scale.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target_column : str
        Name of the target column
    test_size : float
        Proportion of test set
    scale_features : bool
        Whether to scale features
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Encode target if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features if requested
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, feature_names


def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to tune
    param_grid : dict
        Parameter grid for tuning
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    sklearn estimator : Best model after tuning
    """
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_
