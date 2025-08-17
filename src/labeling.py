import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('labeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def compute_rul_from_failures(df: pd.DataFrame, turbine_id_col: str, datetime_col: str, failure_flag_col: str) -> pd.Series:
    """Computes RUL by backward counting from failure flags for each turbine."""
    df_copy = df.sort_values([turbine_id_col, datetime_col])
    rul_series = pd.Series(np.nan, index=df_copy.index)

    for turbine_id in df_copy[turbine_id_col].unique():
        turbine_mask = df_copy[turbine_id_col] == turbine_id
        turbine_data = df_copy.loc[turbine_mask]
        failure_indices = turbine_data[turbine_data[failure_flag_col] == 1].index

        if not failure_indices.empty:
            cycle_starts = failure_indices[failure_indices.to_series().diff().ne(1)]
            for start_idx in cycle_starts:
                end_idx = failure_indices[failure_indices >= start_idx].max()
                cycle_data = turbine_data.loc[turbine_data.index <= end_idx]
                
                time_to_failure = (turbine_data.loc[end_idx, datetime_col] - cycle_data[datetime_col])
                rul_in_hours = time_to_failure.dt.total_seconds() / 3600
                rul_series.update(rul_in_hours)
    
    return rul_series

def build_health_index(df: pd.DataFrame, sensor_columns: List[str]) -> pd.Series:
    """Builds a Health Index using PCA on sensor data."""
    logger.info("Building Health Index using PCA...")
    valid_sensors = [col for col in sensor_columns if col in df.columns]
    if not valid_sensors:
        raise ValueError("None of the specified sensor columns were found.")
        
    feature_matrix = df[valid_sensors].copy()
    feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    feature_matrix.ffill(inplace=True)
    feature_matrix.bfill(inplace=True)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)
    
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(features_scaled)
    
    health_index = pd.Series(principal_component.flatten(), index=df.index)
    health_index = (health_index - health_index.min()) / (health_index.max() - health_index.min())
    
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.4f}")
    return health_index

def detect_pseudo_failures(
    df_with_hi: pd.DataFrame, 
    turbine_id_col: str,
    health_index_col: str = 'health_index',
    threshold: float = 0.3, 
    consecutive_windows: int = 5
) -> pd.Series:
    """Detects pseudo-failures when Health Index drops below a threshold."""
    logger.info(f"Detecting pseudo-failures with threshold {threshold}...")
    failure_flags = pd.Series(0, index=df_with_hi.index, name='pseudo_failure')
    
    for _, group in df_with_hi.groupby(turbine_id_col):
        below_threshold = group[health_index_col] < threshold
        failure_periods = below_threshold.rolling(window=consecutive_windows, min_periods=consecutive_windows).sum()
        failures = (failure_periods >= consecutive_windows)
        failure_flags.loc[group.index] = failures.astype(int)
        
    logger.info(f"Detected {failure_flags.sum()} pseudo-failures.")
    return failure_flags

def build_rul_labels(
    df: pd.DataFrame,
    turbine_id_col: str = "turbine_id",
    datetime_col: str = "timestamp",
    failure_flag_col: Optional[str] = None
) -> pd.DataFrame:
    """Builds RUL labels, either from a flag or by generating a Health Index."""
    df_out = df.copy()
    
    if failure_flag_col and failure_flag_col in df_out.columns:
        logger.info("Computing RUL from existing failure flags.")
        rul_series = compute_rul_from_failures(df_out, turbine_id_col, datetime_col, failure_flag_col)
    else:
        logger.info("No failure flags found. Generating RUL from Health Index.")
        sensor_cols = [col for col in df_out.select_dtypes(include=np.number).columns if col != turbine_id_col]
        
        df_out['health_index'] = build_health_index(df_out, sensor_cols)
        
        pseudo_failure_flags = detect_pseudo_failures(df_out, turbine_id_col)
        
        df_out['pseudo_failure'] = pseudo_failure_flags
        
        rul_series = compute_rul_from_failures(df_out, turbine_id_col, datetime_col, 'pseudo_failure')

    df_out['rul_hours'] = rul_series
    logger.info("RUL labeling completed.")
    return df_out
