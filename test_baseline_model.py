#!/usr/bin/env python3
"""
Test script for baseline model training module.
Tests the XGBoost model training with GroupKFold cross-validation.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_baseline_model_training():
    """Test the baseline model training with actual features."""
    print("Testing Baseline Model Training")
    print("="*50)
    
    try:
        # Import the model training module
        from model_train import train_baseline_model
        
        # Check if features file exists
        features_path = "data/features_sample.parquet"
        if not os.path.exists(features_path):
            print(f"Features file not found: {features_path}")
            print("Please run the feature extraction test first to create features.")
            return None
        
        print(f"Using features from: {features_path}")
        
        # Test model training
        print("\nStarting baseline model training...")
        model, cv_results = train_baseline_model(
            features_path=features_path,
            turbine_id_col="WTG",
            rul_col="rul_hours",
            n_splits=3,  # Use fewer folds for testing
            model_path="models/rul_xgb_test.json",
            results_path="models/training_results_test.json"
        )
        
        print(f"\n✅ Baseline model training completed successfully!")
        print(f"Model saved to: models/rul_xgb_test.json")
        print(f"Results saved to: models/training_results_test.json")
        
        # Display results
        print(f"\nCross-validation Results:")
        print(f"  Number of folds: {cv_results['n_splits']}")
        print(f"  Total samples: {cv_results['total_samples']}")
        print(f"  Total turbines: {cv_results['total_turbines']}")
        print(f"  Average MAE: {cv_results['avg_mae']:.2f} ± {cv_results['mae_std']:.2f} hours")
        print(f"  Average RMSE: {cv_results['avg_rmse']:.2f} ± {cv_results['rmse_std']:.2f} hours")
        print(f"  Average R²: {cv_results['avg_r2']:.4f} ± {cv_results['r2_std']:.4f}")
        
        # Check if model files were created
        if os.path.exists("models/rul_xgb_test.json"):
            file_size_mb = os.path.getsize("models/rul_xgb_test.json") / (1024 * 1024)
            print(f"\nModel file size: {file_size_mb:.2f} MB")
        
        if os.path.exists("models/training_results_test.json"):
            file_size_mb = os.path.getsize("models/training_results_test.json") / (1024 * 1024)
            print(f"Results file size: {file_size_mb:.2f} MB")
        
        return model, cv_results
        
    except Exception as e:
        print(f"❌ Baseline model training test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_loading():
    """Test loading a trained model."""
    print("\n" + "="*50)
    print("Testing Model Loading")
    print("="*50)
    
    try:
        from model_train import load_trained_model
        
        model_path = "models/rul_xgb_test.json"
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
        
        # Load the trained model
        loaded_model = load_trained_model(model_path)
        
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(loaded_model)}")
        print(f"Number of features: {loaded_model.n_features_in_}")
        
        return loaded_model
        
    except Exception as e:
        print(f"❌ Model loading test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_prediction():
    """Test making predictions with the trained model."""
    print("\n" + "="*50)
    print("Testing Model Prediction")
    print("="*50)
    
    try:
        # Load a small sample of features for prediction
        features_path = "data/features_sample.parquet"
        if not os.path.exists(features_path):
            print(f"Features file not found: {features_path}")
            return None
        
        # Load features
        df = pd.read_parquet(features_path)
        
        # Prepare a small sample for prediction
        sample_df = df.head(100)  # Use first 100 rows
        
        # Get feature columns (exclude non-feature columns)
        exclude_cols = ['WTG', 'rul_hours', 'Unnamed: 0', 'TurbineStatus', 'WindDirection']
        feature_cols = [col for col in sample_df.columns if col not in exclude_cols]
        
        X_sample = sample_df[feature_cols].values
        
        # Handle missing values
        X_sample = np.nan_to_num(X_sample, nan=0.0)
        
        # Load model
        from model_train import load_trained_model
        model = load_trained_model("models/rul_xgb_test.json")
        
        # Make predictions
        predictions = model.predict(X_sample)
        
        print("✅ Model prediction test completed successfully!")
        print(f"Input shape: {X_sample.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Prediction range: {predictions.min():.2f} to {predictions.max():.2f} hours")
        print(f"Sample predictions: {predictions[:5]}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Model prediction test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all tests."""
    print("Wind Turbine Baseline Model Training Tests")
    print("="*50)
    
    # Test 1: Model training
    result = test_baseline_model_training()
    
    if result is not None:
        model, cv_results = result
        print("\n✅ Baseline model training test PASSED!")
        
        # Test 2: Model loading
        loaded_model = test_model_loading()
        
        if loaded_model is not None:
            print("\n✅ Model loading test PASSED!")
            
            # Test 3: Model prediction
            predictions = test_model_prediction()
            
            if predictions is not None:
                print("\n✅ Model prediction test PASSED!")
                
                print("\n" + "="*50)
                print("All Tests Completed Successfully! 🎉")
                print("="*50)
                print("Your baseline XGBoost model is working correctly!")
                print("You can now proceed with sequence model training and API development.")
            else:
                print("\n⚠️ Model prediction test failed, but training and loading work.")
        else:
            print("\n⚠️ Model loading test failed, but training works.")
    else:
        print("\n❌ Baseline model training test failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
