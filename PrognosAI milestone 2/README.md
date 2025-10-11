# PrognosAI: AI-Driven Predictive Maintenance System

## Project Overview

PrognosAI is a proof-of-concept predictive maintenance (PdM) system designed to estimate the **Remaining Useful Life (RUL)** of industrial machinery (turbofan engines in this case) using multivariate time-series sensor data.The primary objective is to enable timely maintenance decisions, minimize unplanned downtime, and optimize asset utilization.

This system employs deep learning techniques, specifically **Gated Recurrent Units (GRU)**, to recognize degradation patterns and predict when an asset is likely to fail.

## Technical Goals (Milestone Completion)

**Milestone 1: Data Preparation & Feature Engineering:** Successfully loaded, cleaned, and scaled the raw sensor data, calculated the RUL target, and generated rolling window sequences (50 timesteps).
**Milestone 2: Model Development & Training:** Developed and trained a robust GRU-based deep learning model using unit-based splitting and RUL clipping (at 100 cycles) to maximize predictive accuracy.

## Key Results (Current Performance)

The final optimized model achieves state-of-the-art performance on the test set:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Test R² Score** | **> 0.95** | The model explains over 95% of the variance in the RUL target. |
| **Test RMSE** | **< 7.3 cycles** | On average, the RUL prediction is off by fewer than 7.3 cycles. |
| **RUL Clipping** | 100 cycles | Target RUL capped to focus model training on the critical degradation phase. |

## Project Structure

PrognosAI milestone 2/
  -data/
    -processed/
      -train_FD001_processed.csv    # Cleaned, scaled data
    -raw/
      -train_FD001.txt              # NASA CMAPSS Raw Data 
    -sequences/
      -X_train_sequences.npy        # LSTM/GRU Input sequences (50 timesteps)
      -y_train_targets.npy          # Clipped RUL Targets
  -models/
    -best_rul_gru_final.h5            # Saved best model weights (Milestone 2 Deliverable) 
  -notebooks/
    -1_data_prep.ipynb                # Code for Milestone 1
    -2_model_training.ipynb           # Code for Milestone 2 (Training and Evaluation)
  -.gitignore                           # Files/folders to ignore in Git
  -README.md                            # Project documentation

## Usage

1.  Place the raw CMAPSS data (`train_FD001.txt`) into the `data/raw` folder.
2.  Run the notebooks sequentially:
    **`notebooks/1_data_prep.ipynb`**: Executes data cleaning, scaling, and sequence generation.
    **`notebooks/2_model_training.ipynb`**: Loads sequences, trains the optimized GRU model, evaluates performance, and saves the final model to `models/`.