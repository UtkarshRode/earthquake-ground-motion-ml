# 🌎 Earthquake Ground-Motion Prediction Using Machine Learning + Deep Learning

> **Large-scale earthquake ground-motion prediction with earthquake-level validation, physical feature engineering, interpretable deep learning, and an interactive Streamlit application.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Portfolio%20Ready-success.svg)]()

---

## 1. Project Overview

This project develops a machine-learning system for predicting **logarithmic Peak Ground Acceleration (log-PGA)** from earthquake source, propagation, geographic, and site-condition variables.

The project combines:

- large-scale ShakeMap observations
- earthquake metadata
- Vs30 site-condition information
- physically motivated feature engineering
- classical machine learning
- a PyTorch multilayer perceptron
- earthquake-grouped cross-validation
- a completely unseen-earthquake test set
- residual and spatial analysis
- permutation feature importance
- model response analysis
- an interactive Streamlit application

The central modeling decision is important:

> **Observations from the same earthquake are kept together.**

This avoids the overly optimistic performance that can occur when millions of correlated observations from one earthquake are randomly split between training and validation.

---

## 2. Executive Results

### Dataset

| Quantity | Result |
|---|---:|
| Total observations | ~2.87 million |
| Total earthquakes | 40 |
| Training earthquakes | 28 |
| Validation earthquakes | 6 |
| Final unseen test earthquakes | 6 |
| Final MLP input features | 13 |

### Grouped 5-Fold Cross-Validation

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| **PyTorch MLP** | **0.2128** | **0.1777** | **0.8015** |
| Elastic Net | 0.2474 | 0.2080 | 0.7317 |
| Physical Elastic Net | 0.2999 | 0.2353 | 0.6021 |
| LightGBM | 0.3169 | 0.2385 | 0.5579 |

### Completely Unseen Earthquake Test

| Metric | Result |
|---|---:|
| RMSE | **0.3113** |
| MAE | **0.2800** |
| R² | **0.6099** |

The gap between grouped CV and the final unseen-earthquake test is deliberately reported rather than hidden. It demonstrates that generalization to new earthquake events is substantially harder than interpolation within the development distribution.

---

## 3. Problem Statement

Peak Ground Acceleration is a fundamental ground-motion quantity used in earthquake engineering and hazard analysis.

The objective here is **not earthquake early warning** and the application should not be interpreted as a structural-safety system.

Instead, the project asks:

> **Given earthquake source parameters, propagation geometry, location, and site-condition information, how accurately can machine-learning models estimate observed ground motion?**

The prediction target is:

```text
log_PGA = log10(PGA)
```

and the deployed application converts the prediction back to PGA units using:

```text
PGA = 10 ^ predicted_log_PGA
```

---

## 4. Why Earthquake-Level Validation?

A random row-wise train/test split would be inappropriate for this problem.

A single earthquake can contain tens of thousands of spatial observations. Those observations share:

- the same event magnitude
- the same event depth
- the same source
- related spatial structure
- related propagation characteristics

Randomly distributing those observations across train and validation sets can therefore leak event-specific information.

### Validation design

```text
40 earthquakes
      │
      ├── Development events: 34
      │       ├── 28 training events
      │       └── 6 validation events
      │
      └── Final test: 6 completely unseen earthquakes
```

Grouped 5-fold CV is performed at the **earthquake-event level**, not the observation level.

This is one of the main methodological strengths of the project.

---

## 5. Data Pipeline

```text
Earthquake Metadata
        │
ShakeMap Ground Motion
        │
Vs30 / Site Conditions
        │
        ▼
Data Cleaning + Quality Control
        │
        ▼
Physical / Exploratory Analysis
        │
        ▼
Feature Engineering
        │
        ├───────────────┐
        ▼               ▼
Classical ML        PyTorch MLP
        │               │
        └───────┬───────┘
                ▼
        Earthquake-Grouped CV
                │
                ▼
      Final Unseen-Event Test
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
     Metrics Residuals Interpretability
                │
                ▼
        Streamlit Dashboard
```

---

## 6. Final Feature Set

The final MLP uses 13 features:

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

### Physical motivation

- **Magnitude** captures source strength.
- **Depth** describes source geometry.
- **Epicentral / hypocentral distance** capture propagation effects.
- **Log distance** provides a more appropriate representation of attenuation.
- **Magnitude²** allows nonlinear source scaling.
- **Magnitude × distance** captures source-distance interaction.
- **Depth × distance** captures coupled source geometry.
- **Vs30** represents near-surface site conditions.
- **log(Vs30)** gives a transformed site-response representation.
- **Vs30 availability** explicitly tells the model whether the site measurement exists.
- **Latitude / longitude** retain geographic/spatial information.

---

## 7. Models Compared

### Elastic Net

A regularized linear baseline providing a strong interpretable reference.

### Physical Elastic Net

An Elastic Net using the physically engineered feature representation.

### LightGBM

A nonlinear tree-based model used to test whether boosted decision trees outperform the engineered linear baseline.

### PyTorch MLP

The final deep-learning model:

```text
13
 ↓
Linear(64)
 ↓
ReLU
 ↓
BatchNorm(64)
 ↓
ReLU
 ↓
Linear(32)
 ↓
ReLU
 ↓
BatchNorm(32)
 ↓
ReLU
 ↓
Linear(16)
 ↓
ReLU
 ↓
Linear(1)
```

The final checkpoint was verified against the expected state-dict architecture.

---

## 8. Interpretation

Permutation importance identifies the features whose disruption causes the largest degradation in model performance.

The strongest contributors include:

1. `event_depth`
2. `depth_distance_interaction`
3. `log_distance`
4. `log_Vs30`
5. `Vs30_clean`
6. spatial coordinates
7. hypocentral distance
8. magnitude

The result is useful because it does not simply rank arbitrary engineered columns: the dominant features correspond to meaningful earthquake-source, propagation, and site-condition effects.

---

## 9. Response Analysis

The application includes one-dimensional model-response analyses for:

- event depth
- hypocentral distance
- magnitude
- Vs30

These response curves provide a sanity check on whether the learned model behaves consistently with broad physical expectations.

They should be interpreted as **model-response diagnostics**, not causal relationships.

---

## 10. Unseen-Earthquake Analysis

The final test contains six earthquakes that were not used during model development.

The application allows each event to be explored interactively.

For each event it provides:

- observation count
- RMSE
- MAE
- R²
- observed mean
- predicted mean
- mean residual
- prediction-level records
- observed spatial ground motion
- predicted spatial ground motion
- spatial residuals
- observed-vs-predicted comparison

This makes the final evaluation auditable rather than reducing the entire project to one metric.

---

## 11. Streamlit Application

The application is organized into five major sections:

### 🏠 Overview

High-level project objective, dataset scale, validation design, and headline results.

### 🌎 Earthquake Explorer

Interactive investigation of completely unseen test earthquakes.

### 📊 Model Comparison

Comparison of classical ML models and the final PyTorch MLP.

### 🔍 Interpretability

Permutation importance and model-response analysis.

### 🎯 PGA Prediction

Interactive inference using:

- magnitude
- depth
- epicentral distance
- hypocentral distance
- Vs30
- latitude
- longitude

The deployed inference pipeline uses the saved:

```text
pga_mlp_development.pt
pga_scaler.pkl
pga_imputer.pkl
```

with the same 13-feature ordering used during development.

---

## 12. Example Inference

Example input:

```text
Magnitude:             5.5
Depth:                 20 km
Epicentral distance:   100 km
Hypocentral distance:  105 km
Vs30:                  500 m/s
Latitude:              20°
Longitude:             78°
```

The application produces an estimated:

```text
log-PGA ≈ -0.0918
PGA      ≈ 0.8094
```

This is a machine-learning estimate based on the training distribution and should not be interpreted as an operational earthquake-warning or structural-safety prediction.

---

## 13. Project Structure

```text
EARTHQUAKE_ML_DL/
│
├── app/
│   ├── app.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
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
│   ├── pga_mlp_development.pt
│   ├── pga_scaler.pkl
│   └── pga_imputer.pkl
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_shakemap_processing.ipynb
│   ├── 03_eda_and_physical_analysis.ipynb
│   ├── 04_geological_feature_engineering.ipynb
│   ├── 05_feature_engineering_and_dataset_design.ipynb
│   └── 06_ml_baselines.ipynb
│
├── reports/
├── src/
├── README.md
└── requirements.txt
```

---

## 14. Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

---

## 15. Reproducibility

The project separates:

- raw data
- processed data
- trained model artifacts
- notebooks
- application code
- figures
- reports

The Streamlit inference layer does **not** retrain the model.

It loads the saved model, imputer, and scaler and reproduces the same feature-engineering logic used during model development.

---

## 16. Important Limitations

This project is a research/portfolio implementation rather than an operational seismic-hazard product.

Important limitations include:

- only 40 earthquake events are represented
- unseen-event performance is substantially harder than grouped development validation
- geographic coverage is finite
- site-condition coverage is incomplete
- model behavior outside the training distribution is uncertain
- residuals vary considerably by earthquake
- PGA estimates should not be used as structural-safety decisions
- model-response curves describe learned associations, not causality

These limitations are explicitly reported because reliable ML evaluation requires understanding where a model can fail.

---

## 17. Key Takeaways

### Modeling

The PyTorch MLP achieved the strongest grouped CV performance:

```text
RMSE = 0.2128
MAE  = 0.1777
R²   = 0.8015
```

### Generalization

Performance on completely unseen earthquakes:

```text
RMSE = 0.3113
MAE  = 0.2800
R²   = 0.6099
```

### Scientific interpretation

Depth, distance, source-distance interactions, and Vs30-related variables are among the strongest model contributors.

### Engineering

The final result is packaged as an interactive application rather than only a notebook.

---

## 18. What Makes This Project Different?

This project is not simply:

> "Train neural network → report R²."

Instead it combines:

```text
Large-scale data
+
Domain knowledge
+
Physical feature engineering
+
Multiple ML baselines
+
Earthquake-grouped validation
+
Unseen-event testing
+
Residual diagnostics
+
Spatial diagnostics
+
Model interpretability
+
Production-style inference
+
Interactive deployment
```

The central objective is therefore not just predictive accuracy, but **credible evaluation and interpretable scientific modeling**.

---

## 19. Disclaimer

This application is intended for research, educational, and portfolio purposes.

It provides machine-learning estimates of ground motion based on the feature distribution and earthquake events represented in the development data. It is **not an earthquake early-warning system, seismic alert system, structural-safety assessment, or substitute for engineering analysis**.
