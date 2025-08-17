import sys
import os

# Add the project root directory to the Python path
# This ensures that imports like 'from src.module import ...' work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_pipeline import load_and_clean_data
from src.labeling import build_rul_labels
from src.features import extract_features
from src.model_train import train_baseline_model
from src.model_seq import train_gru_model

def main():
    """Executes the full ML pipeline from data cleaning to model training."""
    
    print("--- 1. Running Data Pipeline ---")
    df_clean = load_and_clean_data(file_path="data/Turbine_Data.csv")

    print("\n--- 2. Running RUL Labeling ---")
    df_labeled = build_rul_labels(df_clean)

    print("\n--- 3. Running Feature Engineering ---")
    extract_features(df_labeled, save_path="data/features.parquet")

    print("\n--- 4. Training XGBoost Baseline Model ---")
    train_baseline_model()

    print("\n--- 5. Training GRU Sequence Model ---")
    train_gru_model()

    print("\n--- ✅ Full ML Pipeline Completed! ---")
    print("Model files are now available in the 'models/' directory.")

if __name__ == "__main__":
    main()