import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_rolling_features(df: pd.DataFrame, sensor_columns: List[str], turbine_id_col: str, window_sizes: List[int]) -> pd.DataFrame:
    """Create rolling window features for numeric sensors."""
    logger.info(f"Creating rolling features for {len(sensor_columns)} sensors...")
    
    feature_df = df.copy()
    new_features_list = []

    for sensor in sensor_columns:
        if sensor not in df.columns:
            continue
        for window in window_sizes:
            grouped_sensor = df.groupby(turbine_id_col)[sensor].rolling(window=window, min_periods=1)
            rolling_mean = grouped_sensor.mean().reset_index(0, drop=True).rename(f'{sensor}_rolling_mean_{window}')
            rolling_std = grouped_sensor.std().reset_index(0, drop=True).rename(f'{sensor}_rolling_std_{window}')
            new_features_list.extend([rolling_mean, rolling_std])

    if new_features_list:
        feature_df = pd.concat([feature_df] + new_features_list, axis=1)

    return feature_df.copy()

def create_temporal_features(df: pd.DataFrame, sensor_columns: List[str], turbine_id_col: str) -> pd.DataFrame:
    """Create temporal features like first differences and rate of change."""
    logger.info("Creating temporal features...")
    feature_df = df.copy()
    for sensor in sensor_columns:
        if sensor in df.columns:
            feature_df[f'{sensor}_diff'] = df.groupby(turbine_id_col)[sensor].diff()
            feature_df[f'{sensor}_pct_change'] = df.groupby(turbine_id_col)[sensor].pct_change(fill_method='ffill')
    return feature_df

def add_power_curve_residuals(df: pd.DataFrame, power_col: str, wind_speed_col: str) -> pd.DataFrame:
    """Fit power curve and add residuals."""
    logger.info("Adding power curve residuals...")
    feature_df = df.copy()
    fit_data = df[[power_col, wind_speed_col]].dropna()
    if len(fit_data) > 10:
        X = fit_data[wind_speed_col].values.reshape(-1, 1)
        y = fit_data[power_col].values
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        
        predict_data = df[[wind_speed_col]].dropna()
        X_pred_poly = poly.transform(predict_data[wind_speed_col].values.reshape(-1, 1))
        predicted_power = model.predict(X_pred_poly)
        
        residuals = pd.Series(df[power_col] - predicted_power, index=predict_data.index)
        feature_df['power_curve_residual'] = residuals
    return feature_df

def extract_features(
    df: pd.DataFrame,
    turbine_id_col: str = "turbine_id", # Standardized
    save_path: Optional[str] = "data/features.parquet"
) -> pd.DataFrame:
    """Master function to extract all features."""
    logger.info("Starting feature extraction process...")
    
    sensor_columns = [col for col in df.select_dtypes(include=np.number).columns if col != turbine_id_col]
    
    df_rolling = create_rolling_features(df, sensor_columns, turbine_id_col=turbine_id_col, window_sizes=[10, 30, 60])
    df_temporal = create_temporal_features(df_rolling, sensor_columns, turbine_id_col=turbine_id_col)
    df_final = add_power_curve_residuals(df_temporal, power_col="power_output", wind_speed_col="wind_speed")
    
    initial_rows = len(df_final)
    df_final.dropna(inplace=True)
    logger.info(f"Dropped {initial_rows - len(df_final)} rows with NaN features.")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_final.to_parquet(save_path, index=False)
        logger.info(f"Features saved to: {save_path}")
        
    return df_final
