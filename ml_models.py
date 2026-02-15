"""ML Classification Models Implementation"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')


class MultiClassificationModels:
    
    def __init__(self, X_train, X_test, y_train, y_test, random_state=42):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.trained_models = {}
        
    def initialize_models(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state, max_depth=10),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=self.random_state, eval_metric='logloss')
        }
        
    def train_all_models(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
                
    def evaluate_model(self, name, model):
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(self.y_test, y_pred)
        n_classes = len(np.unique(self.y_test))
        average_method = 'binary' if n_classes == 2 else 'weighted'
        
        precision = precision_score(self.y_test, y_pred, average=average_method, zero_division=0)
        recall = recall_score(self.y_test, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average=average_method, zero_division=0)
        
        roc_auc = None
        if y_pred_proba is not None:
            try:
                if n_classes == 2:
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                pass
        
        cm = confusion_matrix(self.y_test, y_pred)
        mcc = matthews_corrcoef(self.y_test, y_pred)
        
        try:
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except:
            cv_mean = None
            cv_std = None
        
        return {
            'model_name': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'mcc': mcc,
            'confusion_matrix': cm,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def evaluate_all_models(self):
        for name, model in self.trained_models.items():
            results = self.evaluate_model(name, model)
            self.results[name] = results
    
    def get_comparison_dataframe(self):
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'AUC': results['roc_auc'] if results['roc_auc'] else np.nan,
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1': results['f1_score'],
                'MCC': results['mcc']
            })
        return pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    def get_best_model(self, metric='accuracy'):
        best_score = -1
        best_model_name = None
        for name, results in self.results.items():
            score = results.get(metric, 0)
            if score and score > best_score:
                best_score = score
                best_model_name = name
        if best_model_name:
            return (best_model_name, self.trained_models[best_model_name], self.results[best_model_name])
        return None


def prepare_dataset(data, target_column, test_size=0.2, scale_features=True, random_state=42):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, feature_names
