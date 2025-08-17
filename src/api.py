import sys
import os
import pickle
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

# Add the project root directory to the Python path
# This allows imports like 'from src.features...' to work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now we can import our custom modules
from src.features import extract_features
from src.model_seq import GRUModel

# --- App and Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wind Turbine RUL Prediction API",
    version="1.0.0"
)

# --- Global Model Storage ---
models = {}

# --- Pydantic Models for Data Contracts ---
class SensorReading(BaseModel):
    timestamp: str
    WTG: str
    ActivePower: float | None = None
    WindSpeed: float | None = None
    AmbientTemperatue: float | None = None
    # Add other sensor fields as needed

class PredictionRequest(BaseModel):
    model_type: str = Field("xgboost", enum=["xgboost", "gru"])
    sensor_readings: List[SensorReading]

class PredictionResponse(BaseModel):
    turbine_id: str
    model_type: str
    predicted_rul_hours: float

# --- Model Loading on Startup ---
@app.on_event("startup")
def load_models():
    """Load ML models into memory when the API starts."""
    logger.info("Loading models...")
    
    # Load XGBoost model
    xgb_path = os.path.join(project_root, "models", "rul_xgb.json")
    if os.path.exists(xgb_path):
        models["xgboost"] = xgb.XGBRegressor()
        models["xgboost"].load_model(xgb_path)
        logger.info("XGBoost model loaded.")
    else:
        logger.warning(f"XGBoost model not found at: {xgb_path}")

    # Load GRU model and scaler
    gru_path = os.path.join(project_root, "models", "rul_gru.pth")
    scaler_path = os.path.join(project_root, "models", "gru_scaler.pkl")
    if os.path.exists(gru_path) and os.path.exists(scaler_path):
        checkpoint = torch.load(gru_path, map_location="cpu")
        feature_names = checkpoint.get('feature_names', [])
        input_size = len(feature_names) if feature_names else 99  # Use 99 as fallback based on error message
        
        gru_model = GRUModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        gru_model.load_state_dict(checkpoint['model_state_dict'])
        gru_model.eval()
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Create a config dictionary with the model parameters
        config = {
            'input_size': input_size,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'feature_names': feature_names
        }
        
        models["gru"] = {"model": gru_model, "scaler": scaler, "config": config}
        logger.info("GRU model and scaler loaded.")
    else:
        logger.warning("GRU model or scaler not found.")
        
# --- API Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.sensor_readings:
        raise HTTPException(status_code=400, detail="No sensor readings provided.")
    
    df = pd.DataFrame([reading.dict() for reading in request.sensor_readings])
    turbine_id = df['WTG'].iloc[0]

    # This is a simplified feature extraction for real-time.
    # For production, this should precisely match the training feature pipeline.
    features_df = df[['ActivePower', 'WindSpeed', 'AmbientTemperatue']].ffill().bfill()
    features_df.fillna(0, inplace=True)

    if request.model_type == "xgboost":
        if "xgboost" not in models:
            raise HTTPException(status_code=503, detail="XGBoost model not available.")
        
        # This part needs to be adapted to your full feature set
        # For now, we'll use a placeholder prediction
        prediction = models["xgboost"].predict(features_df.iloc[-1:].values)[0]
    else: # GRU model
        # Placeholder for GRU prediction logic
        prediction = 150.0 # Placeholder value

    return PredictionResponse(
        turbine_id=turbine_id,
        model_type=request.model_type,
        predicted_rul_hours=float(prediction)
    )