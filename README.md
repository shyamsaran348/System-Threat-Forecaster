# System Threat Forecaster

**System Threat Forecaster** is a machine learning-based system designed to predict the probability of malware threats on devices. It leverages advanced ensemble techniques, combining **LightGBM**, **CatBoost**, and **XGBoost** models to achieve high predictive accuracy. The system also includes a comprehensive explainability module using **SHAP (SHapley Additive exPlanations)** to provide insights into model decisions.

## 🚀 Key Features

*   **Multi-Model Ensemble**: Utilizes a blend of Gradient Boosting Decision Trees (GBDT) for robust performance.
*   **Advanced Preprocessing**: Handles high-cardinality categorical features, missing values, and skewed distributions.
*   **Explainable AI (XAI)**: Integrated SHAP analysis for global feature importance and local instance-level explanations.
*   **Interactive Dashboard**: A Streamlit-based UI for visualizing predictions and explanations.
*   **Automated Pipeline**: End-to-end scripts for training, evaluation, and inference.

---

## 📊 Dataset

The project uses a subset of the **Microsoft Malware Prediction** dataset, focusing on system telemetry to predict infection probability.

*   **Source**: Microsoft / Kaggle
*   **Train Set**: 65,535 samples
*   **Test Set**: 10,000 samples
*   **Features**: 82 columns describing the machine's configuration (e.g., OS version, antivirus state, hardware specs).
*   **Target**: `target` (Binary: 0 = Clean, 1 = Infected)

**Key Features:**
*   `EngineVersion`, `AppVersion`, `AvSigVersion`: Defender state.
*   `RtpStateBitfield`: Real-time protection status.
*   `IsSxsPassiveMode`: Passive mode status.
*   `AVProductStatesIdentifier`: Antivirus product ID.
*   `AVProductsInstalled`: Number of AV products.

---

## 🧠 Machine Learning Architecture

The core of the system is an ensemble of models trained on a large dataset of system telemetry.

### 1. Models
*   **LightGBM**: The primary workhorse, optimized for speed and efficiency on large datasets. We use a Stratified K-Fold cross-validation strategy (k=5) to train multiple instances.
*   **CatBoost**: Handles categorical features natively without extensive preprocessing, providing diversity to the ensemble.
*   **XGBoost**: Adds another layer of boosting diversity.
*   **Ensemble Strategy**: The final prediction is a weighted average of the OOF (Out-Of-Fold) predictions from all models, where weights are optimized based on validation AUC scores.

### 2. Preprocessing Pipeline (`src/ml/preprocess.py`)
*   **Categorical Encoding**: High-cardinality features are hashed or frequency-encoded. Low-cardinality features are label-encoded.
*   **Missing Value Imputation**: Strategic filling of NaNs based on feature type (e.g., -1 for categories, mean/median for numericals).
*   **Feature Engineering**: Creation of interaction terms and aggregated statistics to capture complex relationships.

### 3. Evaluation
*   **Metric**: The primary evaluation metric is **ROC AUC (Area Under the Receiver Operating Characteristic Curve)**, ensuring the model effectively discriminates between infected and non-infected systems.

---

## 🔍 Explainability (SHAP)

Understanding *why* a model predicts a threat is crucial. We use SHAP to interpret the black-box models:

*   **Global Importance**: Identifies which system features (e.g., `AvSigVersion`, `CountryIdentifier`) drive the most risk across the entire population.
*   **Local Force Plots**: For any specific device, we can visualize which features pushed the risk score up (red arrows) or down (blue arrows).

Run the Streamlit app to explore these visualizations interactively.

---

## 📂 Project Structure

```text
.
├── data/                   # Raw and processed data
├── frontend/               # React frontend (if applicable)
├── notebooks/              # Jupyter notebooks for exploration
├── outputs/                # Model artifacts, predictions, and metrics
├── src/
│   ├── api/                # API endpoints
│   ├── ml/                 # ML source code
│   │   ├── train.py        # Single model training script
│   │   ├── train_full.py   # Full pipeline orchestrator
│   │   ├── ensemble_blend.py # Model blending logic
│   │   └── preprocess.py   # Data transformation pipeline
│   └── utils/              # Helper functions
├── app_shap_streamlit.py   # Interactive dashboard
└── requirements.txt        # Python dependencies
```

---

## 🛠️ Getting Started

### Prerequisites
*   Python 3.8+
*   Git

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/shyamsaran348/System-Threat-Forecaster.git
    cd System-Threat-Forecaster
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Training the Models

To run the full training pipeline (LightGBM + CatBoost + Ensemble):

```bash
python src/ml/train_full.py
```

This will:
1.  Train 5 folds of LightGBM.
2.  Train 5 folds of CatBoost.
3.  Compute OOF predictions.
4.  Generate the final ensemble submission file in `outputs/ensemble/`.

### Running the Dashboard

To visualize model explanations:

```bash
streamlit run app_shap_streamlit.py
```

---

## 📈 Results

*   **Validation AUC**: ~0.74 (Ensemble)
*   **Top Features**: `SmartScreen`, `AVProductStatesIdentifier`, `Census_TotalPhysicalRAM`.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.
