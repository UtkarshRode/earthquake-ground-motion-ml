# 🌎 Earthquake Ground-Motion Prediction Using Machine Learning and Deep Learning

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://earthquake-ground-motion-ml-c3ebpb3tww8ybtwgq2vdu.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-9ACD32)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

> An end-to-end geoscience machine-learning system for predicting earthquake ground motion from earthquake source parameters, propagation geometry, geographic information, and site conditions.

**2.87M+ observations · 40 earthquakes · 13 final features · 4 model families · Earthquake-grouped validation · Unseen-earthquake testing · Interactive deployment**

---

## 🚀 Live Application

### [Launch the Interactive Streamlit Dashboard →](https://earthquake-ground-motion-ml-c3ebpb3tww8ybtwgq2vdu.streamlit.app/)

The deployed application provides an interactive interface for:

- 🌎 Exploring completely unseen earthquake events
- 📊 Comparing machine-learning and deep-learning models
- 🔍 Investigating feature importance
- 📈 Examining model response to physical variables
- 🗺️ Visualizing observed, predicted, and residual ground motion
- 🎯 Generating ground-motion predictions from user-defined earthquake/site parameters

---

# 📌 Project Snapshot

| Component | Result |
|---|---:|
| Ground-motion observations | **2.87M+** |
| Earthquake events | **40** |
| Final model features | **13** |
| Modeling approaches | **4** |
| Training earthquakes | **28** |
| Validation earthquakes | **6** |
| Completely unseen test earthquakes | **6** |
| Best grouped-CV R² | **0.8015** |
| Unseen-earthquake R² | **0.6099** |
| Final model | **PyTorch MLP** |
| Deployment | **Streamlit Cloud** |

---

# 🏆 Key Results

The final PyTorch multilayer perceptron achieved the strongest performance during earthquake-grouped cross-validation.

## Grouped 5-Fold Cross-Validation

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| 🥇 **PyTorch MLP** | **0.2128** | **0.1777** | **0.8015** |
| Elastic Net | 0.2474 | 0.2080 | 0.7317 |
| Physical Elastic Net | 0.2999 | 0.2353 | 0.6021 |
| LightGBM | 0.3169 | 0.2385 | 0.5579 |

The MLP was selected as the final model because it achieved the lowest grouped-CV RMSE and MAE and the highest grouped-CV R².

---

# 🎯 Final Unseen-Earthquake Evaluation

The most important evaluation in this project is the final test on **six earthquake events that were completely excluded from model development**.

| Metric | Final Unseen-Earthquake Test |
|---|---:|
| RMSE | **0.3113** |
| MAE | **0.2800** |
| R² | **0.6099** |

The difference between grouped cross-validation and unseen-event performance is intentionally reported rather than hidden.

```text
Grouped CV R²
     0.8015
        ↓
Unseen Test R²
     0.6099
```

This demonstrates that the model performs well on held-out earthquake groups while also revealing the additional difficulty of generalizing to completely unseen seismic events.

---

# 🔬 Why Earthquake-Level Validation?

A conventional random train-test split would be problematic for this dataset.

Each earthquake contains many spatial observations. If observations from the same earthquake appear in both training and validation sets, the model can effectively learn event-specific characteristics.

That can lead to overly optimistic validation performance.

Instead, this project keeps entire earthquakes together during validation.

### Dataset-level split

```text
                    40 Earthquakes
                          │
          ┌───────────────┼───────────────┐
          │               │               │
       Training       Validation        Final Test
       28 events       6 events         6 events
          │               │               │
          └───────────────┴───────────────┘
                          │
             Completely unseen events
```

The final test therefore answers a more meaningful question:

> **Can the model generalize to earthquake events that were not observed during model development?**

---

# 🎯 Project Objective

The objective is to predict **logarithmic Peak Ground Acceleration (log-PGA)** using machine learning and deep learning.

The model uses information describing:

### Earthquake source

- Magnitude
- Event depth

### Propagation geometry

- Epicentral distance
- Hypocentral distance
- Distance transformations
- Magnitude-distance interactions
- Depth-distance interactions

### Site conditions

- Vs30
- Log-transformed Vs30
- Vs30 availability

### Geographic information

- Grid latitude
- Grid longitude

The project combines these variables into a physically informed machine-learning pipeline.

---

# 🌍 Scientific Background

Peak Ground Acceleration (PGA) is a commonly used measure of the intensity of ground shaking during an earthquake.

Ground motion is influenced by several factors:

```text
Earthquake Source
       │
       ├── Magnitude
       └── Depth
       │
       ↓
Propagation
       │
       ├── Epicentral Distance
       └── Hypocentral Distance
       │
       ↓
Site Effects
       │
       └── Vs30
       │
       ↓
Observed Ground Motion
       │
       ↓
PGA
```

The relationship between these variables is nonlinear and can vary across earthquake events and locations.

This makes the problem suitable for comparing:

- Linear regularized models
- Tree-based nonlinear models
- Neural networks

---

# 📊 Dataset

The integrated modeling dataset contains approximately **2.87 million ground-motion observations across 40 earthquakes**.

The data combines earthquake source information, ShakeMap observations, geographic coordinates, propagation distances, and Vs30-based site information.

The integrated dataset contains 26 columns:

```text
grid_lon
grid_lat
PGA
PGV
MMI
PSA03
PSA10
PSA30
event_id
magnitude
event_lat
event_lon
epicentral_distance_km
hypocentral_distance_km
event_depth
Vs30
Vs30_clean
site_class
log_PGA
log_distance
log_epicentral_distance
magnitude_distance_interaction
magnitude_squared
depth_distance_interaction
log_Vs30
Vs30_available
```

---

# 🧹 Data Processing

The project includes a multi-stage data-processing pipeline.

```text
Raw Earthquake Data
        ↓
Quality Control
        ↓
ShakeMap Processing
        ↓
Spatial Grid Processing
        ↓
Earthquake Metadata Integration
        ↓
Vs30 Integration
        ↓
Missing-Value Handling
        ↓
Feature Engineering
        ↓
Final Modeling Dataset
```

The processing pipeline includes:

- Data inspection
- Missing-value analysis
- Physical consistency checks
- Ground-motion transformations
- Distance calculations
- Vs30 integration
- Event-level organization
- Feature generation
- Train/validation/test separation

---

# 🧠 Feature Engineering

The final neural network uses **13 features**.

```text
1.  magnitude
2.  event_depth
3.  epicentral_distance_km
4.  hypocentral_distance_km
5.  log_distance
6.  magnitude_squared
7.  magnitude_distance_interaction
8.  depth_distance_interaction
9.  Vs30_clean
10. log_Vs30
11. Vs30_available
12. grid_lat
13. grid_lon
```

---

## Logarithmic Distance

Ground-motion attenuation is commonly represented using logarithmic distance relationships.

The project therefore uses:

```text
log_distance = log10(hypocentral_distance)
```

This provides a more appropriate representation of the large dynamic range of source-to-site distances.

---

## Magnitude Nonlinearity

Magnitude is transformed using:

```text
magnitude_squared = magnitude²
```

This allows the linear baseline models to represent some nonlinear magnitude behavior.

---

## Magnitude-Distance Interaction

The model includes:

```text
magnitude_distance_interaction
    = magnitude × log_distance
```

This allows the effect of distance to vary with earthquake magnitude.

---

## Depth-Distance Interaction

The project also includes:

```text
depth_distance_interaction
    = event_depth × log_distance
```

This captures an additional interaction between earthquake depth and propagation geometry.

---

## Vs30 Transformation

Vs30 is used as a proxy for local site conditions.

A logarithmic transformation is applied:

```text
log_Vs30 = log10(Vs30)
```

The model also retains:

```text
Vs30_available
```

as an indicator of whether Vs30 information was available.

This allows the model to distinguish between:

- A known site condition
- An imputed site condition

---

# 🤖 Models Evaluated

Four model families were evaluated.

---

## 1. Elastic Net

Elastic Net provides an interpretable regularized linear baseline.

It combines:

- L1 regularization
- L2 regularization

This provides a useful benchmark for determining how much performance can be obtained from approximately linear relationships.

---

## 2. Physical Feature Elastic Net

A second linear model was evaluated using physically motivated engineered features.

This provides a comparison between:

```text
Raw / simpler representation
            vs
Physically engineered representation
```

---

## 3. LightGBM

LightGBM provides a nonlinear tree-based machine-learning benchmark.

It can automatically model:

- Nonlinear relationships
- Feature interactions
- Threshold effects

without requiring all nonlinearities to be manually specified.

---

## 4. PyTorch MLP

The final model is a fully connected multilayer perceptron implemented using PyTorch.

### Architecture

```text
Input
13 features
    │
    ▼
Linear: 13 → 64
    │
    ▼
Batch Normalization
    │
    ▼
Activation
    │
    ▼
Linear: 64 → 32
    │
    ▼
Batch Normalization
    │
    ▼
Activation
    │
    ▼
Linear: 32 → 16
    │
    ▼
Activation
    │
    ▼
Linear: 16 → 1
    │
    ▼
Predicted log-PGA
```

The trained model is stored in:

```text
models/pga_mlp_development.pt
```

---

# 📈 Model Comparison

The grouped cross-validation results are:

| Rank | Model | RMSE | MAE | R² |
|---:|---|---:|---:|---:|
| 🥇 | **PyTorch MLP** | **0.2128** | **0.1777** | **0.8015** |
| 🥈 | Elastic Net | 0.2474 | 0.2080 | 0.7317 |
| 🥉 | Physical Elastic Net | 0.2999 | 0.2353 | 0.6021 |
| 4 | LightGBM | 0.3169 | 0.2385 | 0.5579 |

The MLP provides the best overall grouped-CV performance.

---

# 🔍 Model Interpretability

Predictive performance alone is not sufficient for this project.

The final system therefore includes several interpretability analyses.

---

## Permutation Feature Importance

Each feature is randomly permuted while keeping the trained model fixed.

The resulting increase in RMSE provides an estimate of how strongly the model depends on that feature.

The analysis found:

- **Event depth** to be the strongest feature by permutation importance.
- Depth-distance interaction to provide substantial additional information.
- Distance-related features to be highly influential.
- Vs30-related variables to contribute meaningfully.
- Magnitude to contribute to prediction.
- Geographic coordinates to provide additional predictive information.

A notable observation is that:

```text
Vs30_available
```

has negative permutation RMSE increase in the reported analysis.

This should not be interpreted as proof that Vs30 availability is physically harmful. It indicates that the feature's standalone permutation behavior is affected by feature interactions and redundancy within the trained model.

---

# 📈 Model Response Analysis

The model response is evaluated by changing one physical input variable while holding the other inputs approximately fixed.

Response analyses are available for:

- Event depth
- Hypocentral distance
- Magnitude
- Vs30

These curves help inspect whether the learned model behaves in physically meaningful ways.

---

# 🧪 Residual Analysis

Residuals are calculated as:

```text
Residual = Observed log-PGA − Predicted log-PGA
```

The project evaluates residual behavior against:

- Magnitude
- Distance
- Depth
- Vs30

Additional analyses include:

- Residual correlation matrix
- Event-level residual summaries
- Absolute-error correlations
- Vs30 availability analysis
- Per-event test results

---

# 🗺️ Spatial Analysis

The project compares the observed and predicted spatial distributions of ground motion.

Three major spatial visualizations are generated.

### Observed Ground Motion

```text
Observed log-PGA
```

### Predicted Ground Motion

```text
Predicted log-PGA
```

### Residual Ground Motion

```text
Observed log-PGA − Predicted log-PGA
```

These maps help identify geographic patterns and systematic errors that aggregate metrics may not reveal.

---

# 📊 Key Visualizations

The repository contains the main analysis figures.

### Observed vs Predicted

![Observed vs Predicted](figures/observed_vs_predicted_log_pga.png)

### Permutation Feature Importance

![Feature Importance](figures/mlp_permutation_feature_importance.png)

### Residual vs Depth

![Residual vs Depth](figures/residual_vs_depth.png)

### Residual vs Distance

![Residual vs Distance](figures/residual_vs_distance.png)

### Residual vs Magnitude

![Residual vs Magnitude](figures/residual_vs_magnitude.png)

### Residual vs Vs30

![Residual vs Vs30](figures/residual_vs_vs30.png)

---

# 🗺️ Spatial Ground-Motion Maps

### Observed

![Observed Spatial Ground Motion](figures/spatial_observed_log_pga.png)

### Predicted

![Predicted Spatial Ground Motion](figures/spatial_predicted_log_pga.png)

### Residual

![Spatial Residual Map](figures/spatial_residual_map.png)

---

# 🖥️ Interactive Streamlit Dashboard

The project is deployed as a complete interactive Streamlit application.

## Dashboard Sections

### 🏠 Overview

The Overview page communicates:

- Dataset size
- Number of earthquake events
- Grouped cross-validation performance
- Unseen-earthquake performance
- Project objective
- Validation strategy
- Modeling pipeline

---

### 🌎 Earthquake Explorer

The Earthquake Explorer allows users to select one of the six completely unseen earthquake events.

For each event, the dashboard reports:

- Number of observations
- RMSE
- MAE
- R²
- Observed mean
- Predicted mean
- Mean residual

Users can also inspect individual prediction records.

---

### 🤖 Model Comparison

The Model Comparison page displays:

- Grouped 5-fold CV RMSE
- Grouped 5-fold CV MAE
- Grouped 5-fold CV R²
- Model ranking
- Final unseen-earthquake performance

---

### 🔎 Interpretability

The Interpretability page contains:

- Permutation feature importance
- Response to depth
- Response to hypocentral distance
- Response to magnitude
- Response to Vs30

---

### 🎯 PGA Prediction

The PGA Prediction page provides an interactive inference interface.

Users can provide:

```text
Magnitude
Depth
Epicentral Distance
Hypocentral Distance
Vs30
Latitude
Longitude
```

The application performs the same feature engineering and preprocessing used during model development.

It then returns:

```text
Predicted log-PGA
Predicted PGA
```

---

# 🔄 Production Inference Pipeline

The deployed prediction system follows:

```text
User Inputs
     ↓
Feature Engineering
     ↓
Feature Ordering
     ↓
Missing-Value Imputation
     ↓
Standard Scaling
     ↓
PyTorch MLP
     ↓
Predicted log-PGA
     ↓
PGA Transformation
     ↓
Dashboard Output
```

The final feature order is explicitly maintained to ensure that the deployed model receives inputs in the same order used during training.

---

# 📦 Model Artifacts

The repository contains the trained inference artifacts.

```text
models/
├── pga_imputer.pkl
├── pga_mlp_development.pt
└── pga_scaler.pkl
```

### `pga_mlp_development.pt`

Contains the trained PyTorch MLP state dictionary.

### `pga_scaler.pkl`

Contains the fitted `StandardScaler` used to standardize the 13 model features.

### `pga_imputer.pkl`

Contains the fitted imputation strategy used during preprocessing.

---

# 🏗️ Project Architecture

```text
                         ┌─────────────────────┐
                         │   Raw Earthquake    │
                         │       Data          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Data Acquisition &  │
                         │ Quality Control     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ ShakeMap Processing │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Geological / Vs30   │
                         │ Integration          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
        ┌──────────────────┐                ┌──────────────────┐
        │ Classical ML     │                │ PyTorch MLP      │
        │                  │                │                  │
        │ Elastic Net      │                │ Deep Learning    │
        │ Physical EN      │                │                  │
        │ LightGBM         │                │                  │
        └────────┬─────────┘                └────────┬─────────┘
                 │                                   │
                 └────────────────┬──────────────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ Earthquake-Grouped  │
                       │ Cross Validation    │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Unseen Earthquake   │
                       │ Final Evaluation    │
                       └──────────┬──────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              Residuals   Interpretability   Spatial
                            Analysis          Analysis
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ Streamlit Dashboard │
                       └─────────────────────┘
```

---

# 📁 Repository Structure

```text
earthquake-ground-motion-ml/
│
├── app/
│   ├── app.py
│   └── utils.py
│
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   └── INTERVIEW_GUIDE.md
│
├── figures/
│   ├── mlp_permutation_feature_importance.png
│   ├── observed_vs_predicted_log_pga.png
│   ├── residual_vs_depth.png
│   ├── residual_vs_distance.png
│   ├── residual_vs_magnitude.png
│   ├── residual_vs_vs30.png
│   ├── spatial_observed_log_pga.png
│   ├── spatial_predicted_log_pga.png
│   └── spatial_residual_map.png
│
├── models/
│   ├── pga_imputer.pkl
│   ├── pga_mlp_development.pt
│   └── pga_scaler.pkl
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_shakemap_processing.ipynb
│   ├── 03_eda_and_physical_analysis.ipynb
│   ├── 04_geological_feature_engineering.ipynb
│   ├── 05_feature_engineering_and_dataset_design.ipynb
│   ├── 06_ml_baselines.ipynb
│   └── 07_modeling_evaluation_and_interpretability.ipynb
│
├── reports/
│   ├── figures/
│   ├── master_results.json
│   ├── methodology_summary.txt
│   ├── project_metadata.json
│   ├── results_summary.txt
│   └── scientific_conclusions.txt
│
├── scripts/
│   └── validate_project.py
│
├── src/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── features.py
│   └── models.py
│
├── MODEL_CARD.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 📚 Notebook Pipeline

The analysis is organized into sequential notebooks.

| Notebook | Purpose |
|---|---|
| `01_data_acquisition.ipynb` | Earthquake data acquisition and initial inspection |
| `02_shakemap_processing.ipynb` | ShakeMap processing and integration |
| `03_eda_and_physical_analysis.ipynb` | Exploratory and physical analysis |
| `04_geological_feature_engineering.ipynb` | Geological and Vs30 feature integration |
| `05_feature_engineering_and_dataset_design.ipynb` | Final feature engineering and dataset construction |
| `06_ml_baselines.ipynb` | Classical ML baseline development |
| `07_modeling_evaluation_and_interpretability.ipynb` | MLP development, grouped CV, testing, residuals and interpretability |

---

# 📋 Processed Outputs

The project produces a comprehensive set of processed analytical outputs.

Important outputs include:

```text
final_model_comparison.csv
final_model_results_table.csv
final_test_metrics.csv
final_test_predictions.csv
final_test_residuals.csv
final_test_per_event_results.csv
final_test_event_residual_summary.csv
mlp_grouped_cv_summary.csv
mlp_permutation_feature_importance.csv
mlp_response_depth.csv
mlp_response_hypocentral_distance.csv
mlp_response_magnitude.csv
mlp_response_Vs30.csv
residual_correlation_matrix.csv
residual_by_vs30_availability.csv
event_level_summary.csv
distance_attenuation_summary.csv
```

These outputs make the modeling process reproducible and allow the dashboard to operate without rerunning the full training pipeline.

---

# 🛠️ Technology Stack

## Programming

- Python
- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- LightGBM

## Deep Learning

- PyTorch

## Visualization

- Matplotlib
- Plotly

## Application

- Streamlit

## Development

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

# ⚙️ Running Locally

## 1. Clone the repository

```bash
git clone https://github.com/UtkarshRode/earthquake-ground-motion-ml.git
cd earthquake-ground-motion-ml
```

## 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit application

```bash
streamlit run app/app.py
```

The application should be available at:

```text
http://localhost:8501
```

---

# 🔁 Reproducibility

The project separates:

```text
Data
   ↓
Processing
   ↓
Features
   ↓
Model Development
   ↓
Evaluation
   ↓
Deployment
```

The trained model artifacts are included so that the Streamlit application can perform inference without retraining the neural network.

The feature order is explicitly maintained during inference.

---

# 🧪 Evaluation Philosophy

A major focus of this project is avoiding misleading validation.

The evaluation process considers:

### 1. Baseline performance

How well can regularized linear models perform?

### 2. Nonlinear machine learning

Does a tree-based model capture additional nonlinear structure?

### 3. Deep learning

Does the MLP improve upon classical approaches?

### 4. Grouped validation

Does the model generalize across earthquake events?

### 5. Completely unseen test events

Does the model generalize to earthquake events never observed during development?

### 6. Residual analysis

Where does the model systematically fail?

### 7. Interpretability

Which features drive the predictions?

This creates a more complete evaluation than reporting a single random train-test score.

---

# 🔎 Important Findings

## Finding 1 — The MLP was the strongest model

The PyTorch MLP achieved:

```text
RMSE = 0.2128
MAE  = 0.1777
R²   = 0.8015
```

during grouped cross-validation.

---

## Finding 2 — Unseen-event performance is lower

The final unseen-earthquake evaluation achieved:

```text
RMSE = 0.3113
MAE  = 0.2800
R²   = 0.6099
```

This demonstrates the additional challenge of generalizing across earthquake events.

---

## Finding 3 — Event depth is highly influential

Permutation feature importance identified event depth as the strongest individual feature according to the reported RMSE increase.

This highlights the importance of earthquake source geometry in the learned prediction function.

---

## Finding 4 — Distance-related variables are important

Distance and engineered distance interactions contribute substantially to prediction.

This is consistent with the physical attenuation of ground motion with increasing source-to-site distance.

---

## Finding 5 — Site conditions matter

Vs30-related features contribute meaningfully to the model.

This supports incorporating local site characteristics rather than relying only on earthquake source parameters.

---

# ⚠️ Limitations

This is a research and portfolio-oriented machine-learning system.

It should **not** be interpreted as:

- An earthquake early-warning system
- A real-time emergency response system
- A structural safety assessment
- A replacement for engineering seismic hazard analysis
- A replacement for established ground-motion prediction equations
- A production safety-critical seismic forecasting system

Model performance may change when applied to:

- New geographic regions
- Different earthquake mechanisms
- Magnitudes outside the development range
- Site conditions poorly represented in the dataset
- Earthquakes with substantially different distributions
- Data collected using different processing pipelines

The gap between grouped-CV and unseen-event performance also demonstrates that event-specific variability remains an important source of uncertainty.

---

# 🔮 Future Improvements

Potential future work includes:

### More earthquake events

Increase the number and geographic diversity of earthquake events.

### Additional geological information

Integrate:

- Geological maps
- Fault proximity
- Lithology
- Basin indicators
- Elevation
- Regional tectonic information

### Improved uncertainty estimation

Instead of providing only point predictions, future versions could estimate predictive uncertainty.

Potential approaches include:

- Ensembles
- Quantile regression
- Bayesian neural networks
- Monte Carlo dropout

### Advanced architectures

Potential extensions include:

- Residual neural networks
- Attention-based architectures
- Graph neural networks
- Spatial neural networks

### Stronger seismic baselines

Compare against established empirical ground-motion prediction equations.

### Geographic generalization

Perform explicit region-held-out testing in addition to earthquake-held-out testing.

---

# 🎓 Machine-Learning Concepts Demonstrated

This project demonstrates practical implementation of:

- Regression
- Feature engineering
- Feature scaling
- Missing-value imputation
- Regularization
- Elastic Net
- Gradient boosting
- LightGBM
- Neural networks
- PyTorch
- Batch normalization
- Nonlinear feature transformations
- Feature interactions
- Cross-validation
- Grouped cross-validation
- Event-level splitting
- Unseen-group evaluation
- Residual analysis
- Permutation feature importance
- Model response analysis
- Model serialization
- Production inference
- Streamlit deployment

---

# 🌍 Geoscience Concepts Demonstrated

The project combines machine learning with:

- Earthquake source parameters
- Earthquake magnitude
- Earthquake depth
- Peak Ground Acceleration
- ShakeMap data
- Epicentral distance
- Hypocentral distance
- Ground-motion attenuation
- Vs30
- Site classification
- Spatial seismic analysis
- Ground-motion residual analysis
- Geological/site-condition integration

---

# 📖 Documentation

Additional technical documentation is available in the repository.

### Architecture

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

Describes the system architecture and data/model flow.

### Deployment

[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)

Explains how the Streamlit application is deployed.

### Interview Guide

[`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md)

Contains important project concepts and questions useful for technical interviews.

### Model Card

[`MODEL_CARD.md`](MODEL_CARD.md)

Documents model purpose, intended use, evaluation, limitations and responsible interpretation.

---

# 📊 Project Outputs

The repository contains:

- Processed datasets
- Model evaluation tables
- Cross-validation results
- Final predictions
- Residual datasets
- Feature importance results
- Model response curves
- Scientific analysis figures
- Trained model artifacts
- Streamlit application
- Technical documentation

This allows the project to be inspected at multiple levels:

```text
Scientific Analysis
        ↓
Data Processing
        ↓
Feature Engineering
        ↓
Model Development
        ↓
Model Evaluation
        ↓
Interpretability
        ↓
Deployment
```

---

# 🚀 Deployment

The Streamlit application is deployed using Streamlit Community Cloud.

The deployed application loads:

```text
models/pga_mlp_development.pt
models/pga_scaler.pkl
models/pga_imputer.pkl
```

and performs inference through the same feature pipeline used during model development.

### Live application

**[Open Earthquake Ground-Motion Prediction →](https://earthquake-ground-motion-ml-c3ebpb3tww8ybtwgq2vdu.streamlit.app/)**

---

# 📌 Project Takeaways

This project demonstrates that a strong machine-learning application in geoscience requires more than simply training a model.

The complete workflow combines:

```text
Domain Knowledge
      +
Large-Scale Data Processing
      +
Physical Feature Engineering
      +
Multiple ML Baselines
      +
Deep Learning
      +
Rigorous Event-Level Validation
      +
Unseen-Event Testing
      +
Interpretability
      +
Residual Analysis
      +
Deployment
```

The final result is an end-to-end system that connects **geoscience, machine learning, deep learning, model evaluation, interpretability, and production-style deployment**.

---

# ⭐ Project Highlights

```text
2.87M+ ground-motion observations
40 earthquake events
13 final model features
4 model families
Earthquake-grouped 5-fold cross-validation
6 completely unseen test earthquakes
PyTorch MLP final model
R² = 0.8015 grouped CV
R² = 0.6099 unseen-event test
Permutation feature importance
Physical response analysis
Residual diagnostics
Spatial ground-motion analysis
Interactive Streamlit dashboard
Cloud deployment
Reproducible model artifacts
```

---

# 👤 Author

## Utkarsh Rode

IIT Kharagpur

GitHub:  
https://github.com/UtkarshRode

---

# 📄 Disclaimer

This project is intended for educational, research, and portfolio purposes.

The predictions generated by the application are machine-learning estimates based on the feature distributions and earthquake observations used during model development.

They should not be interpreted as professional seismic hazard assessments, structural-safety evaluations, earthquake early warnings, or emergency-response recommendations.
