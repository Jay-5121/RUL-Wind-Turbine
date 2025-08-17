#!/usr/bin/env python3
"""
Test script for the FastAPI RUL prediction API.
Tests the health endpoint and prediction endpoint.
"""

import requests
import json
from datetime import datetime, timedelta

# API base URL
API_BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health endpoint."""
    print("Testing Health Endpoint")
    print("="*40)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health endpoint working!")
            print(f"Status: {health_data['status']}")
            print(f"API Version: {health_data['api_version']}")
            print(f"Uptime: {health_data['uptime_seconds']:.1f} seconds")
            print(f"Models loaded: {health_data['models_loaded']}")
        else:
            print(f"❌ Health endpoint failed with status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")
        print("   Start the API with: python src/api.py")
    except Exception as e:
        print(f"❌ Error testing health endpoint: {e}")

def test_root_endpoint():
    """Test the root endpoint."""
    print("\nTesting Root Endpoint")
    print("="*40)
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        
        if response.status_code == 200:
            root_data = response.json()
            print("✅ Root endpoint working!")
            print(f"Message: {root_data['message']}")
            print(f"Version: {root_data['version']}")
            print(f"Docs: {root_data['docs']}")
        else:
            print(f"❌ Root endpoint failed with status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API.")
    except Exception as e:
        print(f"❌ Error testing root endpoint: {e}")

def test_models_status():
    """Test the models status endpoint."""
    print("\nTesting Models Status Endpoint")
    print("="*40)
    
    try:
        response = requests.get(f"{API_BASE_URL}/models/status")
        
        if response.status_code == 200:
            models_data = response.json()
            print("✅ Models status endpoint working!")
            print(f"Total models loaded: {models_data['total_models_loaded']}")
            print(f"Available models: {models_data['available_models']}")
            
            if models_data['models']:
                print("\nModel details:")
                for model_name, model_info in models_data['models'].items():
                    print(f"  {model_name}: {model_info['type']} v{model_info['version']}")
            else:
                print("  No models currently loaded")
        else:
            print(f"❌ Models status endpoint failed with status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API.")
    except Exception as e:
        print(f"❌ Error testing models status endpoint: {e}")

def test_prediction_endpoint():
    """Test the prediction endpoint."""
    print("\nTesting Prediction Endpoint")
    print("="*40)
    
    # Create sample sensor readings
    sample_readings = [
        {
            "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
            "turbine_id": "WTG_001",
            "wind_speed": 8.5 + i * 0.1,
            "active_power": 1500 + i * 10,
            "rotor_speed": 12.5 + i * 0.05,
            "generator_speed": 1500 + i * 5,
            "blade_pitch": 2.5 + i * 0.1,
            "nacelle_position": 180.0 + i * 0.5,
            "gearbox_oil_temp": 65.0 + i * 0.2,
            "generator_temp": 75.0 + i * 0.3,
            "ambient_temp": 15.0 + i * 0.1,
            "humidity": 60.0 + i * 0.2,
            "pressure": 1013.25 + i * 0.1
        }
        for i in range(5)  # Create 5 readings
    ]
    
    # Test XGBoost prediction
    print("Testing XGBoost prediction...")
    xgb_request = {
        "sensor_readings": sample_readings,
        "model_type": "xgboost"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=xgb_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            prediction_data = response.json()
            print("✅ XGBoost prediction successful!")
            print(f"Turbine ID: {prediction_data['turbine_id']}")
            print(f"Predicted RUL: {prediction_data['predicted_rul_hours']:.1f} hours")
            print(f"Confidence: {prediction_data['confidence_score']:.3f}")
            print(f"Model Type: {prediction_data['model_type']}")
            print(f"Processing Time: {prediction_data['processing_time_ms']:.1f} ms")
        else:
            print(f"❌ XGBoost prediction failed with status code: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"Error: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API.")
    except Exception as e:
        print(f"❌ Error testing XGBoost prediction: {e}")
    
    # Test GRU prediction
    print("\nTesting GRU prediction...")
    gru_request = {
        "sensor_readings": sample_readings,
        "model_type": "gru",
        "sequence_length": 60
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=gru_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            prediction_data = response.json()
            print("✅ GRU prediction successful!")
            print(f"Turbine ID: {prediction_data['turbine_id']}")
            print(f"Predicted RUL: {prediction_data['predicted_rul_hours']:.1f} hours")
            print(f"Confidence: {prediction_data['confidence_score']:.3f}")
            print(f"Model Type: {prediction_data['model_type']}")
            print(f"Processing Time: {prediction_data['processing_time_ms']:.1f} ms")
        else:
            print(f"❌ GRU prediction failed with status code: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"Error: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API.")
    except Exception as e:
        print(f"❌ Error testing GRU prediction: {e}")

def main():
    """Run all API tests."""
    print("Wind Turbine RUL API Tests")
    print("="*50)
    
    # Test basic endpoints
    test_health_endpoint()
    test_root_endpoint()
    test_models_status()
    
    # Test prediction endpoint
    test_prediction_endpoint()
    
    print("\n" + "="*50)
    print("API Testing Completed!")
    print("="*50)
    print("Note: Prediction tests may fail if models are not loaded.")
    print("To load models, ensure the model files exist in the models/ directory.")

if __name__ == "__main__":
    main()
