import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import requests
import os

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Wind Turbine RUL Dashboard",
    page_icon="🌪️",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark Theme ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #e13d3d;
    }
    h1, h2, h3 {
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions for Data Loading ---

@st.cache_data
def get_project_root() -> str:
    """Gets the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_data
def load_features_data():
    """Load features data from the data directory."""
    try:
        project_root = get_project_root()
        features_path = os.path.join(project_root, "data", "features.parquet")
        if os.path.exists(features_path):
            df = pd.read_parquet(features_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            st.error(f"Features file not found at: {features_path}")
            return None
    except Exception as e:
        st.error(f"Error loading features data: {e}")
        return None

def load_local_models():
    """Checks for models on the local filesystem."""
    models_found = []
    project_root = get_project_root()

    # Check for XGBoost model
    xgb_path = os.path.join(project_root, "models", "rul_xgb.json")
    if os.path.exists(xgb_path):
        models_found.append("xgboost")

    # Check for GRU model
    gru_path = os.path.join(project_root, "models", "rul_gru.pth")
    if os.path.exists(gru_path):
        models_found.append("gru")

    return models_found

def check_api_status():
    """
    Check the health status of the backend API.
    If the API is not available, it falls back to checking models locally.
    """
    try:
        # Check for an environment variable to see if running on Streamlit Cloud
        # A more robust solution for Streamlit sharing
        is_streamlit_cloud = os.environ.get('HOSTNAME') == 'streamlit' or 'STREAMLIT_SHARING_MODE' in os.environ

        if is_streamlit_cloud:
            # On Streamlit Cloud, don't even try to connect to a local API
            models_loaded = load_local_models()
            return "standalone", models_loaded

        # Try to connect to the local API
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if response.status_code == 200:
            return "healthy", response.json().get("models_loaded", [])
        return "degraded", []

    except requests.exceptions.ConnectionError:
        # If API is not running locally, check for local models
        models_loaded = load_local_models()
        return "standalone", models_loaded

# --- Plotting Functions ---

def create_rul_timeline_plot(df, selected_turbine):
    """Create RUL timeline plot for the selected turbine."""
    if df is None or selected_turbine is None:
        return go.Figure()

    # FIX: Standardized to 'turbine_id'
    turbine_data = df[df['turbine_id'] == selected_turbine].sort_values('timestamp')
    if turbine_data.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=turbine_data['timestamp'],
        y=turbine_data['rul_hours'],
        mode='lines+markers',
        name='RUL (hours)',
        line=dict(color='#ff4b4b', width=3)
    ))
    fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
    fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    fig.update_layout(
        title=f"RUL Timeline - Turbine {selected_turbine}",
        xaxis_title="Time",
        yaxis_title="RUL (hours)",
        template="plotly_dark",
        height=400
    )
    return fig

def create_sensor_trends_plot(df, selected_turbine, selected_sensors):
    """Create sensor trends plot for selected sensors."""
    if df is None or selected_turbine is None or not selected_sensors:
        return go.Figure()

    # FIX: Standardized to 'turbine_id'
    turbine_data = df[df['turbine_id'] == selected_turbine].sort_values('timestamp')
    if turbine_data.empty:
        return go.Figure()

    fig = make_subplots(rows=len(selected_sensors), cols=1, subplot_titles=selected_sensors, vertical_spacing=0.1)
    for i, sensor in enumerate(selected_sensors):
        if sensor in turbine_data.columns:
            fig.add_trace(
                go.Scatter(x=turbine_data['timestamp'], y=turbine_data[sensor], mode='lines', name=sensor),
                row=i + 1, col=1
            )
    fig.update_layout(
        title=f"Sensor Trends - Turbine {selected_turbine}",
        template="plotly_dark",
        height=250 * len(selected_sensors),
        showlegend=False
    )
    return fig

def create_turbine_map(df):
    """Create a geographic map of turbines."""
    if df is None:
        return None
    
    np.random.seed(42)
    # FIX: Standardized to 'turbine_id'
    turbine_locations = df.groupby('turbine_id').agg(
        rul_hours=('rul_hours', 'last')
    ).reset_index()
    turbine_locations['lat'] = 55.6761 + np.random.uniform(-0.5, 0.5, size=len(turbine_locations))
    turbine_locations['lon'] = 12.5683 + np.random.uniform(-0.5, 0.5, size=len(turbine_locations))

    def get_color(rul):
        if rul > 150: return [0, 255, 0, 160]
        if rul > 75: return [255, 255, 0, 160]
        return [255, 0, 0, 160]
    
    turbine_locations['color'] = turbine_locations['rul_hours'].apply(get_color)

    view_state = pdk.ViewState(latitude=55.6761, longitude=12.5683, zoom=7, pitch=50)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=turbine_locations,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius=5000,
        pickable=True,
    )
    # FIX: Standardized to 'turbine_id'
    tooltip = {
        "html": "<b>Turbine ID:</b> {turbine_id}<br/><b>Latest RUL:</b> {rul_hours:.1f} hours",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    return pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10", tooltip=tooltip)

# --- Main Dashboard UI ---

def main():
    """Main function to render the Streamlit dashboard."""
    st.title("🌪️ Wind Turbine RUL Dashboard")
    st.markdown("---")

    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        api_status, models_loaded = check_api_status()

        if api_status == 'healthy':
            status_icon = "✅"
            status_text = "API Connected"
        elif api_status == 'standalone':
            status_icon = "📦"
            status_text = "Standalone Mode"
        else:
            status_icon = "❌"
            status_text = "API Unavailable"

        st.info(f"Mode: **{status_text}** {status_icon}")

        if models_loaded:
            st.write(f"Models Found: `{'`, `'.join(models_loaded)}`")
        else:
            st.warning("No models found.")
        st.markdown("---")

        if 'features_data' not in st.session_state:
            st.session_state.features_data = None

        if st.button("🔄 Load Turbine Data", use_container_width=True):
            with st.spinner("Loading data..."):
                st.session_state.features_data = load_features_data()
        
        if st.session_state.features_data is not None:
            df = st.session_state.features_data
            # FIX: Standardized to 'turbine_id'
            turbines = sorted(df['turbine_id'].unique())
            
            selected_turbine = st.selectbox("🏗️ Select Turbine", options=turbines)
            
            st.subheader("📡 Sensor Selection")
            sensor_cols = sorted([col for col in df.columns if col not in ['turbine_id', 'rul_hours', 'timestamp']])
            default_sensors = [s for s in ['power_output', 'wind_speed', 'gearbox_oil_temp'] if s in sensor_cols]
            selected_sensors = st.multiselect("Select Sensors", options=sensor_cols, default=default_sensors)
        else:
            st.info("Click 'Load Turbine Data' to begin.")
            selected_turbine = None
            selected_sensors = []

    if st.session_state.features_data is None:
        st.warning("Please load the data using the button in the sidebar.")
        return

    df = st.session_state.features_data
    
    st.subheader("📈 Key Metrics")
    col1, col2, col3 = st.columns(3)
    if selected_turbine:
        # FIX: Standardized to 'turbine_id'
        turbine_data = df[df['turbine_id'] == selected_turbine]
        if not turbine_data.empty:
            current_rul = turbine_data['rul_hours'].iloc[-1]
            avg_rul = turbine_data['rul_hours'].mean()
            col1.metric("Current RUL (Selected Turbine)", f"{current_rul:.1f} hrs")
            col2.metric("Average RUL (Selected Turbine)", f"{avg_rul:.1f} hrs")
    
    # FIX: Standardized to 'turbine_id'
    total_turbines = df['turbine_id'].nunique()
    col3.metric("Total Monitored Turbines", total_turbines)
    
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["RUL Timeline", "Sensor Trends", "Turbine Map"])

    with tab1:
        st.plotly_chart(create_rul_timeline_plot(df, selected_turbine), use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_sensor_trends_plot(df, selected_turbine, selected_sensors), use_container_width=True)
        
    with tab3:
        st.pydeck_chart(create_turbine_map(df))

if __name__ == "__main__":
    main()
