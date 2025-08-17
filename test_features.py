import pandas as pd
import numpy as np
import pytest
from src.features import extract_features, create_rolling_features, create_temporal_features

@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    data = {
        'WTG': ['G01'] * 100,
        'ActivePower': np.random.rand(100) * 2000,
        'WindSpeed': np.random.rand(100) * 20,
        'AmbientTemperatue': np.random.rand(100) * 40,
        'rul_hours': np.arange(100, 0, -1)
    }
    df = pd.DataFrame(data)
    df['Unnamed: 0'] = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='10min'))
    return df

def test_rolling_features(sample_dataframe):
    """Test the creation of rolling features."""
    sensor_cols = ['ActivePower', 'WindSpeed']
    feature_df = create_rolling_features(sample_dataframe, sensor_cols)
    
    # Assertions to check if the function works correctly
    assert 'ActivePower_rolling_mean_10' in feature_df.columns
    assert not feature_df['ActivePower_rolling_mean_10'].isnull().all()

def test_temporal_features(sample_dataframe):
    """Test the creation of temporal features."""
    sensor_cols = ['ActivePower', 'WindSpeed']
    feature_df = create_temporal_features(sample_dataframe, sensor_cols)

    # Assertions to check if the function works correctly
    assert 'ActivePower_diff' in feature_df.columns
    # The first value for diff should be NaN
    assert pd.isna(feature_df['ActivePower_diff'].iloc[0])