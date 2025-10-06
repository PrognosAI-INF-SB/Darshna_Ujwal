# PrognosAI: AI-Driven Predictive Maintenance System

## 🚀 Project Overview

**PrognosAI** is a deep learning project dedicated to estimating the **Remaining Useful Life (RUL)** of industrial machinery using multivariate time-series sensor data (NASA CMAPSS FD001 subset). The system aims to recognize sequential degradation patterns and enable timely maintenance decisions.

***

## 🎯 Current Status: Milestone 1 Completion

The work for this milestone is contained within the nested **`PrognosAI milestone 1`** subdirectory.

### Milestone 1: Data Preparation & Feature Engineering (Completed)

| Objective | Status |
| :--- | :--- |
| **Objective:** Successfully load, preprocess, and prepare the CMAPSS dataset for model training. | **Achieved** |
| **Deliverable:** Cleaned and preprocessed CMAPSS sensor data. | **Complete** (`data/processed/train_FD001_processed.csv`) |
| **Deliverable:** Python scripts for data loading and preprocessing. | **Complete** (`notebooks/1.0_data_prep.ipynb`) |
| **Deliverable:** Generated rolling window sequences. | **Complete** (`data/sequences/X_train_sequences.npy` and `Y_train_RUL.npy`) |
| **Deliverable:** Computed RUL targets for all engine cycles. | **Complete** |

***

## 📝 Project Workflow Overview

The overall project is structured around five key milestones:

1.  **Data Ingestion & Preprocessing** (Completed)
2.  **Sequence Generation:** Transform data into rolling window sequences for LSTM.
3.  **Model Training:** Design and train the LSTM model.
4.  **Model Evaluation:** Evaluate performance using RMSE.
5.  **Visualization & Dashboard:** Implement alerts and visualization.

***

## 📁 Project Structure

The root of the repository is the **`PrognosAI main folder`**. The milestone content is nested here:
- PrognosAI milestone 1/
  - data
    - raw            # Contains original dataset files (e.g., train_FD001.txt).
    - processed      # Stores cleaned data (e.g., train_FD001_processed.csv).
    - sequenced      # Stores generated sequences (X_train_sequences.npy, Y_train_RUL.npy).
  - notebooks        # Script for loading, preprocessing, and sequence generation.
    - 1_data_prep.ipynb
  - requirements.txt    #List of all Python dependencies.

***

### 2. Installation
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/PrognosAI-INF-SB/Darshna_Ujwal.git](https://github.com/PrognosAI-INF-SB/Darshna_Ujwal.git)
    cd "PrognosAI main folder"
    ```
2.  **Download Large Files (LFS):**
    This step is crucial for downloading the `.npy` sequence files.
    ```bash
    git lfs install
    git lfs pull
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

***

### 3. Running the Pipeline
To re-verify the data preparation and view the final sequence generation logic, open and execute all cells in:
`PrognosAI milestone 1/notebooks/1.0_data_prep.ipynb`