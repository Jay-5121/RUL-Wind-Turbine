# Wind Turbine Predictive Maintenance

This project provides a complete machine learning pipeline for predicting the Remaining Useful Life (RUL) of wind turbines. It includes data processing, feature engineering, model training, and an interactive dashboard for visualizing the results.

## Overview

The primary goal of this project is to predict how many hours a wind turbine will continue to operate before it requires maintenance. By predicting the RUL, maintenance can be scheduled proactively, reducing downtime and operational costs.

This repository contains:
- A data pipeline for cleaning and preparing turbine sensor data.
- Feature engineering to create meaningful inputs for the models.
- Training scripts for two types of RUL prediction models: XGBoost and a GRU-based sequence model.
- A FastAPI for serving model predictions.
- An interactive Streamlit dashboard for visualizing turbine health and RUL predictions.

## Features

- **Data Pipeline:** A robust pipeline to process raw sensor data into a clean, labeled dataset.
- **Predictive Models:**
    - **XGBoost Model:** A baseline model for RUL prediction.
    - **GRU Model:** A deep learning (sequence) model to capture temporal dependencies in sensor data.
- **Interactive Dashboard:** A user-friendly web interface built with Streamlit to monitor turbine health, view RUL timelines, and analyze sensor trends.
- **API:** A FastAPI application to serve the trained models and provide RUL predictions programmatically.

## System Architecture

The project is structured into several key components:

1.  **Data Processing:** Raw data from `data/Turbine_Data.csv` is cleaned, labeled, and transformed into features suitable for modeling.
2.  **Model Training:** The `run_pipeline.py` script orchestrates the entire training process, saving the trained models to the `models/` directory.
3.  **Backend API:** The FastAPI application (`src/api.py`) loads the trained models and exposes a `/predict` endpoint.
4.  **Frontend Dashboard:** The Streamlit application (`app/dashboard.py`) provides an interface for users to interact with the data and predictions. It can run in a standalone mode or connect to the FastAPI.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

There are three main ways to interact with this project:

### 1. Run the Full ML Pipeline

To process the data and train the models from scratch, run the main pipeline script. This will generate the cleaned data, features, and trained model files in the `data/` and `models/` directories.

```bash
python run_pipeline.py
```

### 2. Launch the Interactive Dashboard

To explore the turbine data and RUL predictions, launch the Streamlit dashboard.

```bash
streamlit run app/dashboard.py
```

The dashboard will open in your web browser. From the sidebar, click "Load Turbine Data" to get started.

### 3. Run the API Server

To serve the RUL predictions via a REST API, start the FastAPI server.

```bash
uvicorn src.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can view the API documentation at `http://127.0.0.1:8000/docs`.

## Project Structure

```
├── app/
│   └── dashboard.py      # Streamlit dashboard application
├── data/
│   ├── Turbine_Data.csv    # Raw turbine sensor data
│   └── ...                 # Processed data and features
├── models/
│   ├── rul_xgb.json        # Trained XGBoost model
│   ├── rul_gru.pth         # Trained GRU model
│   └── ...                 # Other model artifacts (e.g., scaler)
├── src/
│   ├── api.py              # FastAPI application
│   ├── data_pipeline.py    # Data loading and cleaning
│   ├── features.py         # Feature engineering
│   ├── labeling.py         # RUL labeling logic
│   ├── model_train.py      # XGBoost model training
│   └── model_seq.py        # GRU model training
├── tests/                  # Test scripts
├── run_pipeline.py         # Main script to run the full pipeline
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## License

This project is licensed under the terms of the [MIT License](LICENSE).
