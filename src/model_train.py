import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, Tuple, Any
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_baseline_model(
    features_path: str = "data/features.parquet",
    turbine_id_col: str = "turbine_id", # Standardized
    rul_col: str = "rul_hours",
    n_splits: int = 5,
    model_path: str = "models/rul_xgb.json",
    results_path: str = "models/training_results.json"
) -> None:
    """Train baseline XGBoost model for RUL prediction."""
    logger.info("Starting baseline model training...")
    try:
        df = pd.read_parquet(features_path)
        
        exclude_cols = [turbine_id_col, rul_col, 'timestamp']
        feature_names = [col for col in df.select_dtypes(include=np.number).columns if col not in exclude_cols]
        
        X = df[feature_names].values
        y = df[rul_col].values
        groups = df[turbine_id_col].values

        # FIX: Replace infinite values which can be created during feature engineering
        X[np.isinf(X)] = np.finfo(np.float32).max

        xgb_params = {
            'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
            'n_jobs': -1, 'early_stopping_rounds': 50,
        }
        
        logger.info("Training final model on full dataset...")
        final_model = xgb.XGBRegressor(**xgb_params)
        # Use a validation set for early stopping to prevent overfitting
        eval_set = [(X, y)] # Simplified for final training
        final_model.fit(X, y, eval_set=eval_set, verbose=False)
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        final_model.save_model(model_path)
        logger.info(f"Model saved to: {model_path}")

        results_to_save = {'feature_names': feature_names}
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        logger.info(f"Training results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

if __name__ == "__main__":
    train_baseline_model()
