import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import os

# --- Configuration (Must match Milestone 3/4) ---
SEED_VALUE = 42
WINDOW_SIZE = 50
RUL_CLIPPING_VALUE = 100
THRESHOLD_WARNING = 30
THRESHOLD_CRITICAL = 10

# Paths for cached data/model (relative to project root)
PROCESSED_DATA_PATH = 'data/processed/train_FD001_processed.csv' 
MODEL_SAVE_PATH = 'models/best_rul_gru_model.h5' 

# Define standard columns (needed for uploaded file consistency)
op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
sensor_cols_full = [f's_{i}' for i in range(1, 27)]
COLUMN_NAMES = ['unit_number', 'time_in_cycles'] + op_cols + sensor_cols_full
TARGET_COLUMN = 'RUL_Clipped'

# --- CRITICAL FIX: Hardcoded list of the 20 features based on the processed data ---
FINAL_FEATURES_LIST = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_11', 
    's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_20', 's_21' 
]
# Total count: 3 Operational + 17 Sensors = 20 features.

# --- Preprocessing and Model Functions ---

def generate_alert(rul_prediction):
    if rul_prediction <= THRESHOLD_CRITICAL:
        return 'CRITICAL'
    elif rul_prediction <= THRESHOLD_WARNING:
        return 'WARNING'
    else:
        return 'HEALTHY'

def build_gru_model(timesteps, features):
    optimizer = Adam(learning_rate=0.0005)
    model = Sequential([
        GRU(128, input_shape=(timesteps, features), return_sequences=True),
        Dropout(0.3),
        GRU(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'mae'])
    return model

def generate_sequences(df, window_size, feature_cols, target_col):
    X, y, groups = [], [], []
    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit]
        features = unit_data[feature_cols].values
        target = unit_data[target_col].values
        unit_groups = unit_data['unit_number'].values
        
        for i in range(len(unit_data) - window_size + 1):
            X.append(features[i : i + window_size])
            y.append(target[i + window_size - 1])
            groups.append(unit_groups[i + window_size - 1])
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(groups)


# --- Cached Assets Loading (Model, Scaler, Demo Data) ---

@st.cache_resource
def load_assets():
    """Load model and fit scaler based on training data."""
    try:
        df_train = pd.read_csv(PROCESSED_DATA_PATH)
        
        features_to_scale = FINAL_FEATURES_LIST
        df_train_features = df_train[features_to_scale]

        scaler = MinMaxScaler()
        scaler.fit(df_train_features)
        
        NUM_FEATURES = len(features_to_scale) 
        custom_objects = {'rmse': tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'mse': tf.keras.losses.MeanSquaredError(), 'relu': tf.keras.activations.relu}
        model = build_gru_model(WINDOW_SIZE, NUM_FEATURES)
        model = load_model(MODEL_SAVE_PATH, custom_objects=custom_objects)
        
        results_df, rmse = run_demo_prediction_logic(df_train, model, scaler, features_to_scale)

        return model, scaler, features_to_scale, results_df, rmse

    except FileNotFoundError as e:
        if 'train_FD001_processed.csv' in str(e):
             st.error(f"FATAL ERROR: Processed data file not found at {PROCESSED_DATA_PATH}. Ensure Milestone 1 was run.")
        elif 'best_rul_gru_model.h5' in str(e):
             st.error(f"FATAL ERROR: Model file not found at {MODEL_SAVE_PATH}. Ensure Milestone 2 was run successfully.")
        else:
             st.error(f"FATAL ERROR: Required project asset not found. Details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading model assets. Details: {e}")
        st.stop()


def run_demo_prediction_logic(df_full, model, scaler, feature_cols):
    """Replicates the Milestone 3 test split logic to get demo data."""
    df_full['RUL_Clipped'] = df_full['RUL'].clip(upper=RUL_CLIPPING_VALUE)
    df_filtered = df_full[~df_full['unit_number'].isna()].copy()
    
    X, y, groups = generate_sequences(df_filtered, WINDOW_SIZE, feature_cols, TARGET_COLUMN)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED_VALUE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_test = X[test_idx]
    y_test = y[test_idx]
    groups_test = groups[test_idx]
    
    del X; del y; del groups # Free memory

    y_pred = model.predict(X_test).flatten()
    rmse = model.evaluate(X_test, y_test, verbose=0)[1]

    results_df = pd.DataFrame({
        'unit_number': groups_test,
        'RUL_Actual': y_test,
        'RUL_Predicted': y_pred,
    })
    results_df['Alert_Status'] = results_df['RUL_Predicted'].apply(generate_alert)
    return results_df, rmse


def predict_uploaded_data(uploaded_file, model, scaler, feature_cols):
    """Processes uploaded file and returns predictions for the last sequence."""
    
    try:
        df_upload = pd.read_csv(uploaded_file, sep='\s+', header=None, names=COLUMN_NAMES, index_col=False)
        df_upload.dropna(axis=1, how='all', inplace=True)
        
        df_upload_features = df_upload[feature_cols]

        scaled_data = scaler.transform(df_upload_features)
        df_scaled_upload = pd.DataFrame(scaled_data, columns=feature_cols)
        
        cycle_count = len(df_upload)

        if cycle_count < WINDOW_SIZE:
            st.error(f"Error: Uploaded data has only {cycle_count} cycles. A minimum of {WINDOW_SIZE} cycles is required for prediction.")
            return None, None
            
        last_sequence = df_scaled_upload.tail(WINDOW_SIZE).values
        X_predict = np.array([last_sequence], dtype=np.float32)

        prediction = model.predict(X_predict).flatten()[0]
        rul_predicted = np.clip(prediction, 0, RUL_CLIPPING_VALUE)
        
        return rul_predicted, cycle_count
    
    except KeyError as e:
        st.error(f"Data Format Error: Column {e} not found. Please ensure your uploaded file contains the correct columns and is space-separated.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        return None, None


# --- Streamlit Dashboard Layout ---

st.set_page_config(layout="wide", page_title="PrognosAI: Predictive Maintenance Dashboard ⚙️")

# Load global assets (Model, Scaler, Demo Data)
model, scaler, feature_cols, results_df, test_rmse = load_assets()

# --- Main Page Content ---

st.title("PrognosAI: Remaining Useful Life (RUL) Dashboard")
st.markdown("---")


## ⚙️ LIVE PREDICTION MODE
st.header("Upload for Live RUL Prediction (New Asset)")
st.markdown("Upload a sensor data file (e.g., TXT or CSV) for a single engine unit. **Must contain at least 50 cycles.**")

# --- FILE UPLOADER ELEMENT ADDED HERE ---
uploaded_file = st.file_uploader("Choose a sensor data file", type=['txt', 'csv'])

if uploaded_file is not None:
    with st.spinner('Processing and Predicting RUL...'):
        rul_pred, cycle_count = predict_uploaded_data(uploaded_file, model, scaler, feature_cols)
    
    if rul_pred is not None:
        last_alert = generate_alert(rul_pred)
        
        st.success("✅ Prediction Complete!")
        
        col_up1, col_up2, col_up3 = st.columns(3)
        col_up1.metric("Total Cycles in File", cycle_count)
        col_up2.metric("Predicted RUL (Cycles)", f"{rul_pred:.2f}")
        col_up3.metric("Maintenance Status", last_alert, delta_color="off")
        
        if last_alert == 'CRITICAL':
            st.warning("⚠️ CRITICAL ALERT: Immediate maintenance is required.")
        elif last_alert == 'WARNING':
            st.info("🟡 WARNING: Plan maintenance within the next 30 cycles.")
        else:
            st.success("🟢 HEALTHY: Asset is operating normally.")
            
    st.markdown("---")
    
    
## 📊 DEMO MODE (CACHED TEST RESULTS)
st.header("Demo Mode: Cached Test Set Evaluation")

col1, col2, col3, col4 = st.columns(4)

alert_counts = results_df['Alert_Status'].value_counts()
critical_count = alert_counts.get('CRITICAL', 0)
warning_count = alert_counts.get('WARNING', 0)
total_engines = results_df['unit_number'].nunique()

col1.metric("Total Test Engines", total_engines)
col2.metric("CRITICAL Alerts", critical_count, delta_color="inverse")
col3.metric("WARNING Alerts", warning_count, delta_color="off")
col4.metric("Test RMSE", f"{test_rmse:.2f} cycles")


st.markdown("---")

## 📈 RUL Prediction Trends (Interactive Plotly Chart)
st.subheader("RUL Prediction Trends and Alert Zones (Test Data)")

engine_options = sorted(results_df['unit_number'].unique().tolist())
default_index = engine_options.index(91) if 91 in engine_options else 0
selected_engine = st.selectbox(
    'Select an Engine Unit to Visualize (from cached test set):',
    options=engine_options,
    index=default_index
)

engine_data = results_df[results_df['unit_number'] == selected_engine].reset_index(drop=True)
last_prediction = engine_data['RUL_Predicted'].iloc[-1]
last_alert = engine_data['Alert_Status'].iloc[-1]

st.info(f"Engine Unit **{selected_engine}** Last Predicted RUL: **{last_prediction:.2f}** cycles. Current Status: **{last_alert}**")

fig = go.Figure()

# Add RUL Actual 
fig.add_trace(go.Scatter(
    x=engine_data.index, y=engine_data['RUL_Actual'], mode='lines', name='Actual RUL (Clipped)',
    line=dict(color='darkblue', width=4), opacity=0.5, visible='legendonly' 
))

# Add RUL Predicted
fig.add_trace(go.Scatter(
    x=engine_data.index, y=engine_data['RUL_Predicted'], mode='lines', name='Predicted RUL',
    line=dict(color='red', width=3, dash='dash')
))

# Add Alert Zones
fig.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=THRESHOLD_CRITICAL, x1=1, y1=THRESHOLD_WARNING,
    fillcolor="orange", opacity=0.2, layer="below", line_width=0,
)
fig.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=0, x1=1, y1=THRESHOLD_CRITICAL,
    fillcolor="red", opacity=0.3, layer="below", line_width=0,
)

# Add Alert Threshold Lines
fig.add_hline(y=THRESHOLD_WARNING, line_dash="dash", line_color="orange", annotation_text="WARNING THRESHOLD")
fig.add_hline(y=THRESHOLD_CRITICAL, line_dash="dash", line_color="red", annotation_text="CRITICAL THRESHOLD")

# Update Layout
fig.update_layout(
    title=f'RUL Prediction for Engine Unit {selected_engine}', xaxis_title='Cycle Index (Relative to Start of Test Sequence)',
    yaxis_title='RUL (Remaining Useful Life in Cycles)', height=550, hovermode="x unified",
    template="plotly_dark", yaxis=dict(range=[0, RUL_CLIPPING_VALUE * 1.1])
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

## 🚨 Alert Status Distribution (Bar Chart)
st.header("Current Alert Status Distribution")
st.markdown("Last Known State of All Test Engines") 

# Bar chart data logic (fixed)
alert_data = results_df.groupby('unit_number')['Alert_Status'].last().value_counts().reset_index()
alert_data.columns = ['Alert_Status', 'Count']

status_order = ['HEALTHY', 'WARNING', 'CRITICAL']
full_status_df = pd.DataFrame({'Alert_Status': status_order})
alert_data = full_status_df.merge(alert_data, on='Alert_Status', how='left').fillna(0)

color_map = {'CRITICAL': 'red', 'WARNING': 'orange', 'HEALTHY': 'green'}
alert_data['Color'] = alert_data['Alert_Status'].map(color_map)

alert_data['Alert_Status'] = pd.Categorical(alert_data['Alert_Status'], categories=status_order, ordered=True)
alert_data = alert_data.sort_values('Alert_Status')


fig_alerts = go.Figure(data=[
    go.Bar(
        x=alert_data['Alert_Status'], y=alert_data['Count'], marker_color=alert_data['Color'],
        text=alert_data['Count'].astype(int), textposition='auto',
    )
])

fig_alerts.update_layout(
    title='', xaxis_title='Alert Level', yaxis_title='Number of Engines',
    template="plotly_dark",
    yaxis=dict(range=[0, total_engines + 1], tickvals=list(range(0, total_engines + 1, 5))), 
    xaxis=dict(categoryorder='array', categoryarray=status_order) 
)
st.plotly_chart(fig_alerts, use_container_width=True)