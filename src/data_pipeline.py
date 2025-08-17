import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the datetime column in the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Name of the datetime column or None if not found
    """
    # Common datetime column names
    datetime_patterns = ['timestamp', 'time', 'date', 'datetime', 'ts']
    
    for col in df.columns:
        col_lower = col.lower()
        # Check if column name matches datetime patterns
        if any(pattern in col_lower for pattern in datetime_patterns):
            logger.info(f"Detected potential datetime column: {col}")
            return col
        
        # Check if column contains datetime-like data
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].iloc[:100])  # Test first 100 rows
                logger.info(f"Detected datetime column by content: {col}")
                return col
            except:
                continue
    
    logger.warning("No datetime column detected automatically")
    return None

def load_and_clean_data(
    file_path: str = "data/Turbine_Data.csv",
    turbine_id_col: str = "turbine_id",
    datetime_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Load and clean wind turbine sensor data.
    
    Args:
        file_path: Path to the CSV file
        turbine_id_col: Name of the turbine ID column
        datetime_col: Name of the datetime column (auto-detected if None)
        
    Returns:
        Cleaned dataframe with proper data types and sorted order
    """
    logger.info(f"Starting data loading and cleaning from: {file_path}")
    
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data
    logger.info("Loading CSV data...")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise
    
    # Log initial data info
    logger.info(f"Initial data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Data types:\n{df.dtypes}")
    
    # Detect datetime column if not provided
    if datetime_col is None:
        datetime_col = detect_datetime_column(df)
        if datetime_col is None:
            logger.error("Could not detect datetime column. Please specify manually.")
            raise ValueError("Datetime column not found")
    
    # Parse datetime column
    logger.info(f"Parsing datetime column: {datetime_col}")
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        invalid_dates = df[datetime_col].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Found {invalid_dates} invalid datetime values")
    except Exception as e:
        logger.error(f"Error parsing datetime column: {e}")
        raise
    
    # Check for turbine_id column
    if turbine_id_col not in df.columns:
        logger.error(f"Turbine ID column '{turbine_id_col}' not found")
        raise ValueError(f"Turbine ID column '{turbine_id_col}' not found")
    
    # Sort by turbine_id and timestamp
    logger.info("Sorting data by turbine_id and timestamp...")
    df = df.sort_values([turbine_id_col, datetime_col]).reset_index(drop=True)
    
    # Log missing values before cleaning
    missing_before = df.isnull().sum()
    logger.info(f"Missing values before cleaning:\n{missing_before}")
    
    # Handle missing values for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    logger.info(f"Processing {len(numeric_columns)} numeric columns for missing values")
    
    for col in numeric_columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            logger.info(f"Processing column '{col}' with {missing_count} missing values")
            
            # Forward fill within each turbine group
            df[col] = df.groupby(turbine_id_col)[col].ffill()
            
            # Interpolate remaining missing values
            remaining_missing = df[col].isnull().sum()
            if remaining_missing > 0:
                logger.info(f"Interpolating {remaining_missing} remaining missing values in '{col}'")
                df[col] = df.groupby(turbine_id_col)[col].interpolate(method='linear')
                
                # If still missing, use global mean
                final_missing = df[col].isnull().sum()
                if final_missing > 0:
                    logger.info(f"Filling {final_missing} remaining missing values with global mean")
                    df[col] = df[col].fillna(df[col].mean())
    
    # Handle missing values in non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_columns:
        if col != datetime_col:  # Skip datetime column
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                logger.info(f"Forward filling {missing_count} missing values in non-numeric column '{col}'")
                df[col] = df.groupby(turbine_id_col)[col].ffill()
    
    # Log missing values after cleaning
    missing_after = df.isnull().sum()
    logger.info(f"Missing values after cleaning:\n{missing_after}")
    
    # Final data validation
    logger.info("Performing final data validation...")
    logger.info(f"Final data shape: {df.shape}")
    logger.info(f"Data types after cleaning:\n{df.dtypes}")
    
    # Check for any remaining issues
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        logger.warning(f"Warning: {total_missing} missing values still remain")
    else:
        logger.info("All missing values have been successfully handled")
    
    # Log summary statistics
    logger.info("Data cleaning completed successfully!")
    logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Turbine IDs: {df[turbine_id_col].nunique()}")
    logger.info(f"Date range: {df[datetime_col].min()} to {df[datetime_col].max()}")
    
    return df

def validate_cleaned_data(df: pd.DataFrame, turbine_id_col: str = "turbine_id") -> bool:
    """
    Validate that the cleaned data meets quality standards.
    
    Args:
        df: Cleaned dataframe
        turbine_id_col: Name of the turbine ID column
        
    Returns:
        True if validation passes, False otherwise
    """
    logger.info("Validating cleaned data quality...")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.error(f"Validation failed: {missing_count} missing values found")
        return False
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows")
    
    # Check data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        logger.error("Validation failed: No numeric columns found")
        return False
    
    # Check for infinite values
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} infinite values in numeric columns")
    
    logger.info("Data validation completed successfully")
    return True

if __name__ == "__main__":
    # Example usage
    try:
        df = load_and_clean_data()
        if validate_cleaned_data(df):
            logger.info("Data pipeline completed successfully!")
        else:
            logger.error("Data validation failed!")
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
