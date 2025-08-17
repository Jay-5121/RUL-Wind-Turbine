import pandas as pd
import numpy as np
import pytest
from src.data_pipeline import load_and_clean_data
from src.labeling import build_rul_labels

@pytest.fixture
def create_test_csv(tmp_path):
    """Creates a temporary CSV file for testing the data pipeline."""
    # Using 'h' for hourly frequency as recommended by the warning
    timestamps = pd.to_datetime(pd.date_range('2023-01-01', periods=100, freq='h'))
    data = {
        'Unnamed: 0': timestamps,
        'WTG': ['G01'] * 100, # <-- Turbine ID column is named 'WTG'
        'ActivePower': np.random.rand(100) * 2000,
        'WindSpeed': np.random.rand(100) * 20,
    }
    # Add a missing value to test cleaning
    data['ActivePower'][10] = np.nan
    df = pd.DataFrame(data)
    
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_data_pipeline(create_test_csv):
    """Tests the load_and_clean_data function."""
    # FIX: Explicitly tell the function to use the 'WTG' column
    df_clean = load_and_clean_data(
        file_path=str(create_test_csv), 
        turbine_id_col='WTG',
        datetime_col='Unnamed: 0'
    )
    
    # Assertions to check if the function works correctly
    assert not df_clean.isnull().values.any()
    assert 'ActivePower' in df_clean.columns

def test_labeling(create_test_csv):
    """Tests the build_rul_labels function."""
    df = pd.read_csv(create_test_csv)
    df['failure'] = 0
    df.loc[90, 'failure'] = 1 # Create a failure event
    
    # FIX: Explicitly tell the function to use the 'WTG' column
    df_labeled = build_rul_labels(
        df, 
        turbine_id_col='WTG', 
        datetime_col='Unnamed: 0', 
        failure_flag_col='failure'
    )
    
    # Assertions to check if the function works correctly
    assert 'rul_hours' in df_labeled.columns
    assert df_labeled.loc[90, 'rul_hours'] == 0
    assert df_labeled.loc[89, 'rul_hours'] == 1