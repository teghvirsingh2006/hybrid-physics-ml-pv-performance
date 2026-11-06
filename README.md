# Hybrid Physics–ML Model for PV Performance Under Partial Shading

> **Research Project | Aug–Oct 2026**

A research project investigating whether a **physics-guided machine learning model** can improve photovoltaic (PV) power prediction under changing irradiance, temperature, and partial-shading conditions.

The core idea is to use a physics-based PV model as the baseline and train a machine-learning model to learn the **residual error** between the physics prediction and observed/simulated power.

---

## Research Question

**In which irradiance, temperature, and partial-shading regimes does a physics-based PV model deviate from observed power, and can a hybrid residual ML model correct those errors?**

The objective is **not to replace the physics model with a black-box ML model**, but to combine physical knowledge with machine learning.

---

## Approach

The project compares three approaches:

### 1. Physics-only Model

A photovoltaic physics model provides the baseline prediction:

\[
P_{\text{physics}}
\]

The physics model is implemented using **pvlib** and a single-diode-based PV modelling approach.

---

### 2. Pure Machine Learning Model

A machine-learning model directly predicts PV power from environmental and system features:

\[
X \rightarrow P_{\text{ML}}
\]

Candidate models include:

- Scikit-learn models
- XGBoost
- Neural networks

---

### 3. Hybrid Residual Model

The hybrid model learns only the error made by the physics model:

\[
\Delta P =
P_{\text{observed}} - P_{\text{physics}}
\]

The ML model then predicts this residual:

\[
\Delta P_{\text{ML}} = f_\theta(X)
\]

and the final prediction becomes:

\[
P_{\text{hybrid}}
=
P_{\text{physics}} + \Delta P_{\text{ML}}
\]

This approach aims to retain the physical interpretability of the baseline while allowing ML to correct systematic modelling errors.

---

## Dataset

The project uses two complementary sources of data.

### PVDAQ System 2107

A real-world PV dataset from the NREL PVDAQ system.

The processed dataset contains approximately:

- **127,309 observations**
- 15-minute measurements
- 2017–2025 operating period
- Irradiance
- Temperature
- Wind speed
- DC power
- PV operating conditions

The full processed dataset is **not committed to GitHub** because of its file size.

A smaller sample is included in:

```text
data/sample/PVDAQ_2107_sample.csv
```

---

### Controlled PV Simulation Dataset

A controlled simulation dataset is used to investigate partial shading and operating conditions that are difficult to isolate in field measurements.

Current dataset:

```text
data/processed/PV_simulation_clean.csv
```

It contains variables including:

- Irradiance
- Temperature
- Shading fraction
- Global maximum power
- Local maximum power
- Physics-model power
- Physics residual
- Peak mismatch

---

## Current Repository Status

The repository is currently organized around data preprocessing, exploratory analysis, and research documentation.

```text
hybrid-physics-ml-pv-performance/
│
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── processed/
│   │   ├── PVDAQ_2107_clean.csv
│   │   ├── PVDAQ_residuals_clean.csv
│   │   └── PV_simulation_clean.csv
│   │
│   └── sample/
│       └── PVDAQ_2107_sample.csv
│
├── docs/
│   ├── Physics-guided-ML-for-PV-under-partial-shading.pptx
│   ├── PV_Hybrid_ML_Research_Project_Roadmap.pdf
│   ├── PV_Midsem_v2 (1).pptx
│   └── PV_Working_Paper.docx
│
├── results/
│   └── figures/
│       ├── fig1_dataset_overview.png
│       ├── fig2_cleaning_waterfall.png
│       ├── fig3_pvdaq_distributions.png
│       ├── fig4_residual_analysis.png
│       ├── fig5_error_heatmaps.png
│       ├── fig6_simulation_shading.png
│       ├── fig7_cross_dataset.png
│       ├── fig8_long_term_trends.png
│       └── fig9_summary_metrics.png
│
└── src/
    └── preprocessing/
        └── exploratory_analysis.py
```

---

## Current Results

The preprocessing and exploratory-analysis pipeline currently generates:

1. Dataset overview
2. Dataset cleaning/size analysis
3. PVDAQ feature distributions
4. Physics residual analysis
5. Irradiance–temperature error heatmaps
6. Controlled simulation shading analysis
7. Cross-dataset comparison
8. Long-term operating trends
9. Summary metrics

The exploratory analysis provides the foundation for the next stage of the project: implementing and benchmarking the physics, pure-ML, and hybrid residual models.

---

## Research Pipeline

The planned modelling pipeline is:

```text
                 PV Data
                    │
                    ▼
             Data Preprocessing
                    │
                    ▼
          Feature Engineering
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Physics Baseline       ML Models
      (pvlib)          Pure ML / XGBoost
          │                   │
          │                   ▼
          │             P_ML prediction
          │
          ▼
     P
