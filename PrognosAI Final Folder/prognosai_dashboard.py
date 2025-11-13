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
import io

# --- Configuration ---
SEED_VALUE = 42
WINDOW_SIZE = 50
RUL_CLIPPING_VALUE = 100
THRESHOLD_WARNING = 30
THRESHOLD_CRITICAL = 10

# Paths (Relative to where Streamlit is run, assuming 'data' and 'models' are accessible)
PROCESSED_DATA_PATH = 'data/processed/train_FD001_processed.csv' 
MODEL_SAVE_PATH = 'models/best_rul_gru_model.h5' 

# Define standard columns (from 1_data_prep.ipynb)
op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
sensor_cols_full = [f's_{i}' for i in range(1, 27)]
COLUMN_NAMES = ['unit_number', 'time_in_cycles'] + op_cols + sensor_cols_full
TARGET_COLUMN = 'RUL_Clipped'

FINAL_FEATURES_LIST = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_11', 
    's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_20', 's_21' 
]

# --- Preprocessing and Model Functions ---

def generate_alert(rul_prediction):
    """Logic for triggering maintenance alerts based on RUL prediction."""
    if rul_prediction <= THRESHOLD_CRITICAL:
        return 'CRITICAL'
    elif rul_prediction <= THRESHOLD_WARNING:
        return 'WARNING'
    else:
        return 'HEALTHY'

def build_gru_model(timesteps, features):
    """Reconstructs the GRU model architecture for loading weights."""
    # Use Sequential model definition as found in 2_model_training.ipynb
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
    # The metrics definition is crucial for loading the model correctly
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'mae'])
    return model

def create_prediction_sequences_and_cycles(df_scaled, window_size, feature_cols):
    """
    Generates all possible 3D sequences (X) from a single engine's data
    for prediction, along with the corresponding cycle indices.
    """
    features_array = df_scaled[feature_cols].values
    total_cycles = features_array.shape[0]
    
    X_sequences = []
    # Start sequence generation at the first point where a full window is available
    start_index = window_size - 1
    
    for i in range(start_index, total_cycles):
        # The sequence is the window_size cycles *ending* at index i
        sequence = features_array[i - window_size + 1 : i + 1]
        X_sequences.append(sequence)
        
    X = np.array(X_sequences, dtype=np.float32)
    
    # Cycles corresponding to the *end* of each sequence
    # For a window size of 50, cycle 50 is the first one predicted
    cycle_indices = np.arange(window_size, total_cycles + 1)
    
    return X, cycle_indices


def plot_rul_trend(df_plot, title_suffix, has_actual=False):
    """Reusable function to generate an interactive Plotly RUL trend chart."""
    fig = go.Figure()

    if has_actual:
        # Add RUL Actual (only available in demo mode, not for raw upload)
        fig.add_trace(go.Scatter(
            x=df_plot.index + WINDOW_SIZE, 
            y=df_plot['RUL_Actual'], 
            mode='lines', 
            name='Actual RUL (Clipped)',
            line=dict(color='darkblue', width=4), 
            opacity=0.5, 
            visible='legendonly' 
        ))

    # Add RUL Predicted
    fig.add_trace(go.Scatter(
        x=df_plot.index + WINDOW_SIZE, # Offset index to show actual cycle number (e.g., starts at 50)
        y=df_plot['RUL_Predicted'], 
        mode='lines', 
        name='Predicted RUL',
        line=dict(color='red', width=3, dash='dash')
    ))

    # Add Alert Zones
    # Warning Zone (CRITICAL to WARNING Threshold)
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=THRESHOLD_CRITICAL, x1=1, y1=THRESHOLD_WARNING,
        fillcolor="orange", opacity=0.2, layer="below", line_width=0,
    )
    # Critical Zone (0 to CRITICAL Threshold)
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, y0=0, x1=1, y1=THRESHOLD_CRITICAL,
        fillcolor="red", opacity=0.3, layer="below", line_width=0,
    )

    # Add Alert Threshold Lines
    fig.add_hline(y=THRESHOLD_WARNING, line_dash="dash", line_color="orange", annotation_text="WARNING THRESHOLD", annotation_position="top left")
    fig.add_hline(y=THRESHOLD_CRITICAL, line_dash="dash", line_color="red", annotation_text="CRITICAL THRESHOLD", annotation_position="top left")

    # Update Layout
    fig.update_layout(
        title=f'RUL Prediction Trend {title_suffix}', 
        xaxis_title='Cycle Index',
        yaxis_title='RUL (Remaining Useful Life in Cycles)', 
        height=550, 
        hovermode="x unified",
        template="plotly_dark", 
        yaxis=dict(range=[0, RUL_CLIPPING_VALUE * 1.1])
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Cached Assets Loading (Only runs once) ---

@st.cache_resource
def load_assets():
    """Loads the model, scaler, and runs demo predictions once."""
    try:
        # 1. Load Processed Training Data (Used for getting the scaler fit)
        df_train = pd.read_csv(PROCESSED_DATA_PATH)
        
        # 2. Fit Scaler
        features_to_scale = FINAL_FEATURES_LIST
        df_train_features = df_train[features_to_scale]

        scaler = MinMaxScaler()
        scaler.fit(df_train_features)
        
        # 3. Load Model
        NUM_FEATURES = len(features_to_scale) 
        custom_objects = {
            'rmse': tf.keras.metrics.RootMeanSquaredError(name='rmse'), 
            'mse': tf.keras.losses.MeanSquaredError(), 
            'relu': tf.keras.activations.relu
        }
        # Build model structure then load weights
        model = build_gru_model(WINDOW_SIZE, NUM_FEATURES)
        model = load_model(MODEL_SAVE_PATH, custom_objects=custom_objects)
        
        # 4. Run Demo Prediction Logic (for dashboard metrics/charts)
        results_df, test_rmse = run_demo_prediction_logic(df_train, model, scaler, features_to_scale)

        return model, scaler, features_to_scale, results_df, test_rmse

    except FileNotFoundError as e:
        st.error(f"FATAL ERROR: Required project asset not found. Check that the files are present in 'data/processed/' and 'models/' directories. Details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during asset loading: {e}")
        st.stop()


def run_demo_prediction_logic(df_full, model, scaler, feature_cols):
    """
    Performs the original prediction logic on the *test split* of the processed data
    to populate the demo charts and metrics.
    """
    df_full['RUL_Clipped'] = df_full['RUL'].clip(upper=RUL_CLIPPING_VALUE)
    df_filtered = df_full[~df_full['unit_number'].isna()].copy()
    
    # This generates sequences for the *entire* processed set
    # The third return value (groups) is needed for splitting by engine unit
    # (y is needed for RMSE calculation)
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
    
    X, y, groups = generate_sequences(df_filtered, WINDOW_SIZE, feature_cols, TARGET_COLUMN)
    
    # Replicate the split used in the notebook (20% for test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED_VALUE)
    _, test_idx = next(gss.split(X, y, groups=groups))
    
    X_test = X[test_idx]
    y_test = y[test_idx]
    groups_test = groups[test_idx]
    
    del X; del y; del groups # Free up memory 

    # Predict and evaluate
    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse = model.evaluate(X_test, y_test, verbose=0)[1]

    # Create results DataFrame for the test split
    results_df = pd.DataFrame({
        'unit_number': groups_test,
        'RUL_Actual': y_test,
        'RUL_Predicted': y_pred,
    })
    results_df['Alert_Status'] = results_df['RUL_Predicted'].apply(generate_alert)
    return results_df, rmse


def predict_uploaded_data(uploaded_file, model, scaler, feature_cols):
    """
    Processes the uploaded data, scales it using the fitted scaler, 
    and predicts the RUL trend for the entire sequence.
    """
    try:
        # 1. Read Raw Data
        df_upload_raw = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")), 
                                    sep='\s+', 
                                    header=None, 
                                    names=COLUMN_NAMES, 
                                    index_col=False)
        
        # Drop NaN columns (from file parsing) and static columns (from FINAL_FEATURES_LIST definition)
        df_upload_raw.dropna(axis=1, how='all', inplace=True)
        df_upload = df_upload_raw.copy()
        
        # 2. Check Data Sufficiency
        cycle_count = len(df_upload)
        if cycle_count < WINDOW_SIZE:
            st.error(f"Error: Uploaded data has only {cycle_count} cycles. A minimum of {WINDOW_SIZE} cycles is required for generating time sequences.")
            return None, None
            
        # 3. Scale Features
        df_upload_features = df_upload[feature_cols]
        scaled_data = scaler.transform(df_upload_features)
        df_scaled_upload = pd.DataFrame(scaled_data, columns=feature_cols)
        
        # 4. Create Sequences for Prediction Trend
        # The result X contains sequences starting from the WINDOW_SIZE-th cycle
        X_predict, cycle_indices = create_prediction_sequences_and_cycles(df_scaled_upload, WINDOW_SIZE, feature_cols)

        # 5. Predict RUL for all sequences
        predictions = model.predict(X_predict, verbose=0).flatten()
        rul_predicted = np.clip(predictions, 0, RUL_CLIPPING_VALUE)
        
        # 6. Create Results DataFrame
        df_results = pd.DataFrame({
            'Cycle_Index': cycle_indices,
            'RUL_Predicted': rul_predicted
        })
        
        df_results['Alert_Status'] = df_results['RUL_Predicted'].apply(generate_alert)
        
        # Get the final prediction/status from the last cycle
        final_pred = df_results['RUL_Predicted'].iloc[-1]
        final_alert = df_results['Alert_Status'].iloc[-1]
        
        return df_results, final_pred, final_alert, cycle_count
    
    except KeyError as e:
        st.error(f"Data Format Error: Missing required feature column. Please ensure your uploaded file has the correct format (e.g., space-separated with all 26 sensor columns, even if static). Missing: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        return None, None, None, None


# --- Streamlit Dashboard Layout ---

st.set_page_config(layout="wide", page_title="PrognosAI: Predictive Maintenance Dashboard ⚙️", initial_sidebar_state="expanded")

# Load global assets
model, scaler, feature_cols, results_df, test_rmse = load_assets()

# Main Page Content
st.title("PrognosAI: Remaining Useful Life (RUL) Dashboard")
st.markdown("---")


# --- LIVE PREDICTION MODE ---
st.header("1. Live RUL Prediction & Trend (Upload New Asset Data)")
st.markdown("Upload a sensor data file (e.g., TXT or CSV) for a single engine unit. **A minimum of 50 cycles is required** for the initial prediction point.")

# FILE UPLOADER ELEMENT
uploaded_file = st.file_uploader("Choose a sensor data file (NASA CMAPSS FD001 format)", type=['txt', 'csv'])

if uploaded_file is not None:
    with st.spinner(f'Processing {uploaded_file.name} and Predicting RUL Trend...'):
        df_results, final_pred, final_alert, cycle_count = predict_uploaded_data(uploaded_file, model, scaler, feature_cols)
    
    if df_results is not None:
        
        st.subheader(f"Results for Uploaded Asset ({uploaded_file.name})")

        col_up1, col_up2, col_up3 = st.columns(3)
        col_up1.metric("Total Cycles in File", cycle_count)
        col_up2.metric("Last Predicted RUL (Cycles)", f"{final_pred:.2f}")
        col_up3.metric("Current Maintenance Status", final_alert, delta_color="off")
        
        if final_alert == 'CRITICAL':
            st.error("⚠️ CRITICAL ALERT: Immediate maintenance is required.")
        elif final_alert == 'WARNING':
            st.warning("🟡 WARNING: Plan maintenance within the next 30 cycles.")
        else:
            st.success("🟢 HEALTHY: Asset is operating normally.")
            
        
        # PLOT INTERACTIVE RUL TREND FOR UPLOADED FILE
        # Adjusting the DataFrame format slightly for the reusable plot function
        df_results.set_index('Cycle_Index', inplace=True)
        
        plot_rul_trend(
            df_results, 
            f"for {uploaded_file.name} (Predicted Trend)", 
            has_actual=False # No actual RUL for a new prediction file
        )

st.markdown("---")

# --- DEMO MODE (CACHED TEST RESULTS) ---
st.header("2. Demo Mode: Cached Test Set Evaluation")
st.markdown("Explore the performance and alert triggers on a held-out test set from the original training data.")

col1, col2, col3, col4 = st.columns(4)

alert_counts = results_df['Alert_Status'].value_counts()
critical_count = alert_counts.get('CRITICAL', 0)
warning_count = alert_counts.get('WARNING', 0)
total_engines = results_df['unit_number'].nunique()

col1.metric("Total Test Engines", total_engines)
col2.metric("CRITICAL Alerts", critical_count, delta_color="inverse")
col3.metric("WARNING Alerts", warning_count, delta_color="off")
col4.metric("Test RMSE", f"{test_rmse:.2f} cycles")


# --- RUL Prediction Trends (Interactive Plotly Chart) ---
st.subheader("RUL Prediction Trends and Alert Zones (Test Data)")

engine_options = sorted(results_df['unit_number'].unique().tolist())
# Set a default engine that typically shows a degradation curve for a better demo
default_engine = 91
if default_engine not in engine_options:
    # Fallback to the first engine if 91 isn't in the test set
    default_engine = engine_options[0] if engine_options else None
    
default_index = engine_options.index(default_engine) if default_engine else 0

selected_engine = st.selectbox(
    'Select an Engine Unit to Visualize (from cached test set):',
    options=engine_options,
    index=default_index
)

if selected_engine:
    # Filter data for the selected engine and reset index to use as cycle count base
    engine_data = results_df[results_df['unit_number'] == selected_engine].reset_index(drop=True)
    
    # Calculate the last known RUL/Status for the info box
    last_prediction = engine_data['RUL_Predicted'].iloc[-1]
    last_actual = engine_data['RUL_Actual'].iloc[-1]
    last_alert = engine_data['Alert_Status'].iloc[-1]

    st.info(f"Engine Unit **{selected_engine}** | Last Predicted RUL: **{last_prediction:.2f}** cycles | Last Actual RUL: **{last_actual:.2f}** cycles | Current Status: **{last_alert}**")

    # PLOT INTERACTIVE RUL TREND FOR DEMO MODE
    plot_rul_trend(
        engine_data, 
        f"for Engine Unit {selected_engine} (Test Data)", 
        has_actual=True # Actual RUL is available for the test set
    )

st.markdown("---")

# --- Alert Status Distribution (Bar Chart) ---
st.header("3. Current Alert Status Distribution")
st.markdown("Last known state of all test engines based on predicted RUL.") 

# Bar chart data logic (using the last prediction for each engine)
alert_data = results_df.groupby('unit_number')['Alert_Status'].last().value_counts().reset_index()
alert_data.columns = ['Alert_Status', 'Count']

status_order = ['HEALTHY', 'WARNING', 'CRITICAL']
full_status_df = pd.DataFrame({'Alert_Status': status_order})
# Merge with full status list to ensure all statuses are present (even if count is 0)
alert_data = full_status_df.merge(alert_data, on='Alert_Status', how='left').fillna(0)

# Define custom colors for the chart
color_map = {'CRITICAL': 'red', 'WARNING': 'orange', 'HEALTHY': 'green'}
alert_data['Color'] = alert_data['Alert_Status'].map(color_map)

# Ensure correct ordering for the plot
alert_data['Alert_Status'] = pd.Categorical(alert_data['Alert_Status'], categories=status_order, ordered=True)
alert_data = alert_data.sort_values('Alert_Status')


fig_alerts = go.Figure(data=[
    go.Bar(
        x=alert_data['Alert_Status'], 
        y=alert_data['Count'], 
        marker_color=alert_data['Color'],
        text=alert_data['Count'].astype(int), 
        textposition='auto',
        name="Engine Count"
    )
])

fig_alerts.update_layout(
    title='Distribution of Last Known Alert Status for Test Engines', 
    xaxis_title='Alert Level', 
    yaxis_title='Number of Engines',
    template="plotly_dark",
    # Set y-axis to extend slightly past the max count for better visual
    yaxis=dict(range=[0, alert_data['Count'].max() * 1.1 + 1], tickvals=list(range(0, int(alert_data['Count'].max() * 1.1) + 2, max(1, int(total_engines / 10))))), 
    xaxis=dict(categoryorder='array', categoryarray=status_order) 
)
st.plotly_chart(fig_alerts, use_container_width=True)