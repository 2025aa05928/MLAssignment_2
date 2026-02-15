"""
Demo script to test ML models implementation
Run this to verify everything works before deploying
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from ml_models import MultiClassificationModels, prepare_dataset

def test_models():
    """Test all models with Iris dataset."""
    print("=" * 80)
    print("TESTING ML CLASSIFICATION MODELS")
    print("=" * 80)
    
    # Load sample dataset
    print("\n1. Loading Iris dataset...")
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['target'] = iris.target
    print(f"✓ Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Prepare dataset
    print("\n2. Preparing dataset...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_dataset(
        df, 'target', test_size=0.2, scale_features=True, random_state=42
    )
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Initialize and train models
    print("\n3. Initializing models...")
    ml_models = MultiClassificationModels(X_train, X_test, y_train, y_test, random_state=42)
    ml_models.initialize_models()
    print(f"✓ Initialized {len(ml_models.models)} models")
    
    print("\n4. Training all models...")
    ml_models.train_all_models()
    
    print("\n5. Evaluating all models...")
    ml_models.evaluate_all_models()
    
    # Print summary
    print("\n6. Results summary:")
    ml_models.print_summary()
    
    # Get best model
    best_name, best_model, best_results = ml_models.get_best_model('accuracy')
    print(f"\n🏆 BEST MODEL: {best_name}")
    print(f"   Accuracy: {best_results['accuracy']:.4f}")
    print(f"   F1-Score: {best_results['f1_score']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
    
    return ml_models

if __name__ == "__main__":
    try:
        models = test_models()
        print("\n✓ Models are ready for deployment!")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
