# 🌏 India Climate Data Science
### End-to-End Data Science Analysis of India's Climate System · 1950–2022

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-✅-blue)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-✅-orange)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Status](https://img.shields.io/badge/Status-M.Sc(AA)-6366f1)]()
[![Years](https://img.shields.io/badge/Analysis-1950--2022-0ea5e9)]()

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Research Objectives](#-research-objectives)
- [Key Findings at a Glance](#-key-findings-at-a-glance)
- [Project Structure](#-project-structure)
- [Data Sources](#-data-sources)
- [Environment & Packages](#-environment--packages)
- [Global Constants](#-global-constants)
- [Data Validation Results](#-data-validation-results)
- [Annual Climate Series](#-annual-climate-series-raw-data)
- [Feature Engineering](#-feature-engineering)
- [Data Quality Summary](#-data-quality-summary)
- [Trend Analysis — Full Results](#-trend-analysis--full-results)
- [Correlation Analysis — Full Matrix](#-correlation-analysis--full-matrix)
- [Principal Component Analysis](#-principal-component-analysis)
- [PCA Scores & Clustering](#-pca-scores--clustering)
- [Regression Modelling](#-regression-modelling)
- [Feature Importance](#-feature-importance)
- [Regression Diagnostics](#-regression-diagnostics)
- [Rainfall Classification](#-rainfall-classification)
- [ARIMA Forecasting](#-arima-forecasting)
- [IMD Diurnal Temperature Range](#-imd-diurnal-temperature-range)
- [Figures Generated](#-figures-generated)
- [Setup & Usage](#-setup--usage)
- [Dependencies](#-dependencies)
- [Reproducibility](#-reproducibility)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🌐 Overview

This project presents a **end-to-end data science analysis** of India's climate system over **73 years (1950–2022)**. It integrates five major climate data streams — global surface temperature anomalies, India annual mean temperature, India monsoon rainfall, ENSO/ONI indices, and IMD gridded Tmax/Tmin NetCDF files — into a single, fully reproducible Jupyter notebook.

The analysis covers statistical trend testing, dimensionality reduction, unsupervised clustering, supervised ML regression and classification, time-series forecasting, ENSO teleconnection analysis, and IMD diurnal temperature range decomposition — producing **16 publication-ready figures** and multiple result tables.

---
## ✨ Why This Repository?

This project converts research questions into measurable results. It produces validated datasets, statistical evidence, interpretable climate modes, benchmarked models, forecast tables, diagnostics, and publication-ready figures—making it suitable for a professional portfolio, thesis appendix, or reproducible research repository.
---

## 🎯 Research Objectives

1. Quantify long-term trends in India temperature and monsoon rainfall (1950–2022) using Mann-Kendall tests and Sen's slope estimator.
2. Decompose the multivariate climate system using PCA to identify dominant modes of variability.
3. Cluster years into distinct climate regimes using K-Means and Agglomerative Clustering.
4. Build and evaluate ML regression models (Ridge, Random Forest, XGBoost) to predict India's temperature anomaly using 46 lag/rolling features.
5. Classify monsoon seasons as above/below normal using Logistic Regression with time-series cross-validation.
6. Forecast India's temperature anomaly 10 years into the future (2023–2032) using ARIMA.
7. Quantify ENSO–India teleconnections through the Oceanic Niño Index.
8. Detect asymmetric warming through IMD Diurnal Temperature Range (DTR) trend analysis (1951–2014).

---

## 🔑 Key Findings at a Glance

| Finding | Value |
|---|---|
| India warming rate (Sen's slope) | **+0.01155 °C / year** |
| Global warming rate (Sen's slope) | **+0.01512 °C / year** |
| Global–India temperature correlation | **r = +0.824** |
| ONI Monsoon Mean – Rainfall correlation | **r = −0.452** |
| Variance explained by PC1 + PC2 | **79.89%** |
| Best regression model | **Ridge (R² = 0.810)** |
| ARIMA 10-yr forecast (2032) | **+0.613 °C anomaly** |
| DTR trend direction (1951–2014) | **Declining (asymmetric warming)** |
| India temperature baseline (1951–1980) | **25.2527 °C** |
| Rainfall baseline (1951–1980) | **324.522 mm** |

---

## 🗂️ Project Structure

```
climate_phd_project/
│
├── 📓 notebooks/
│   └── data_science_climate.ipynb              # Complete analysis (13 sections, 16 figures)
│
├── 📁 data/
│   ├── raw/
│   │   ├── global_temperature/
│   │   │   ├── gistemp_global_annual.csv        # NASA GISTEMP annual anomalies (13 KB)
│   │   │   └── gistemp_global_monthly.csv       # NASA GISTEMP monthly anomalies
│   │   ├── global_rainfall/
│   │   │   └── global_precipitation_source.txt  # Global precipitation (325 KB)
│   │   ├── global_comparison/
│   │   │   └── nasa_power_20_5N_77_5E_monthly.csv
│   │   └── imd_temperature/
│   │       ├── tmax/                            # IMD Tmax NetCDF (75 files)
│   │       └── tmin/                            # IMD Tmin NetCDF (75 files)
│   │
│   ├── processed/
│   │   ├── annual.csv                           # Master annual climate series (6 KB)
│   │   ├── india_official.csv                   # India official temperature (4 KB)
│   │   ├── india_rainfall_annual.csv            # Annual monsoon rainfall (8 KB)
│   │   ├── monthly.csv                          # Monthly climate records (81 KB)
│   │   ├── imd_temperature_csv/
│   │   │   ├── tmax_annual.csv
│   │   │   └── tmin_annual.csv
│   │   └── climate_project_outputs_final_thesis_complete/
│   │       ├── figures/                         # 16 publication-ready figures (140 DPI)
│   │       ├── tables/                          # All result tables (CSV)
│   │       ├── exports/                         # Excel workbooks
│   │       └── configuration.json              # Run configuration snapshot
│   │
│   └── external/
│       └── oni.csv                              # NOAA ONI index (19 KB)
│
├── 📊 outputs/
│   ├── figures/
│   ├── tables/
│   └── exports/
│
├── 🐍 src/
│   └── run_pipeline.py
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 📡 Data Sources

| # | Dataset | Provider | Format | Coverage | File Size |
|---|---|---|---|---|---|
| 1 | Global Surface Temp Anomaly | NASA GISTEMP | CSV | 1950–2022 | 13 KB |
| 2 | India Annual Mean Temperature | IMD Official | CSV | 1950–2022 | 4 KB |
| 3 | India Monsoon Rainfall | IMD | CSV | 1950–2022 | 8 KB |
| 4 | Monthly Climate Records | IMD / NASA | CSV | 1950–2022 | 81 KB |
| 5 | ONI / ENSO Index | NOAA CPC | CSV | 1950–2022 | 19 KB |
| 6 | IMD Tmax Gridded | IMD | NetCDF | 1951–2014 | 75 files |
| 7 | IMD Tmin Gridded | IMD | NetCDF | 1951–2014 | 75 files |
| 8 | NASA POWER Comparison | NASA | CSV | 1981–2022 | 276 bytes |
| 9 | Global Precipitation | External | TXT | 1950–2022 | 325 KB |

---

## ⚙️ Environment & Packages

All packages are imported with graceful fallbacks — the notebook runs even if optional packages are absent. Confirmed installed on run:

| # | Package | Status |
|---|---|---|
| 0 | xgboost | ✅ Installed |
| 1 | shap | ✅ Installed |
| 2 | pymannkendall | ✅ Installed |
| 3 | xarray | ✅ Installed |
| 4 | statsmodels | ✅ Installed |
| 5 | openpyxl | ✅ Installed |

---

## 🔧 Global Constants

```python
ANALYSIS_START_YEAR       = 1950
ANALYSIS_END_YEAR         = 2022
INDIA_BASE_START          = 1951    # Anomaly baseline start
INDIA_BASE_END            = 1980    # Anomaly baseline end
RAINFALL_BASE_START       = 1951
RAINFALL_BASE_END         = 1980
GLOBAL_ABSOLUTE_BASELINE  = 14.0   # °C offset for absolute temperature
FORECAST_HORIZON          = 10     # Years ahead
RANDOM_STATE              = 42
N_BOOTSTRAP               = 2000
FIGURE_DPI                = 140
FIGURE_SIZE               = (12, 7)
```

---

## ✅ Data Validation Results

| # | Key | File | Exists |
|---|---|---|---|
| 0 | global_file | `data/processed/annual.csv` | ✅ True |
| 1 | india_file | `data/processed/india_official.csv` | ✅ True |
| 2 | rainfall_file | `data/processed/india_rainfall_annual.csv` | ✅ True |
| 3 | imd_tmax_annual | `data/processed/imd_temperature_csv/tmax_annual.csv` | ✅ True |
| 4 | imd_tmin_annual | `data/processed/imd_temperature_csv/tmin_annual.csv` | ✅ True |
| 5 | oni_file | `data/external/oni.csv` | ✅ True |
| 6 | rainfall_nc_count | 75 NetCDF files | ✅ True |

---

## 📊 Annual Climate Series (Raw Data)

### Global Temperature Series (first 5 rows)

| # | Year | GlobalAnomaly (°C) | GlobalAbsoluteTemp (°C) |
|---|---|---|---|
| 0 | 1950 | −0.1750 | 13.8250 |
| 1 | 1951 | −0.0683 | 13.9317 |
| 2 | 1952 | +0.0108 | 14.0108 |
| 3 | 1953 | +0.0825 | 14.0825 |
| 4 | 1954 | −0.1317 | 13.8683 |

### India Temperature Series (first 5 rows)

| # | Year | IndiaTemp (°C) | IndiaTempAnomaly (°C) |
|---|---|---|---|
| 0 | 1950 | 24.88 | −0.3727 |
| 1 | 1951 | 25.33 | +0.0773 |
| 2 | 1952 | 25.41 | +0.1573 |
| 3 | 1953 | 25.58 | +0.3273 |
| 4 | 1954 | 25.24 | −0.0127 |

### India Rainfall Series (first 5 rows)

| # | Year | Rainfall (mm) | RainfallAnomaly (mm) |
|---|---|---|---|
| 0 | 1950 | 330.3488 | +5.8268 |
| 1 | 1951 | 283.1122 | −41.4098 |
| 2 | 1952 | 303.0451 | −21.4769 |
| 3 | 1953 | 332.2954 | +7.7734 |
| 4 | 1954 | 327.7124 | +3.1904 |

### ONI / ENSO Series (first 5 rows)

| # | Year | ONIAnnualMean | ONIMonsoonMean |
|---|---|---|---|
| 0 | 1950 | −0.8600 | −0.5500 |
| 1 | 1951 | +0.4308 | +0.7900 |
| 2 | 1952 | +0.1742 | +0.0175 |
| 3 | 1953 | +0.7133 | +0.7575 |
| 4 | 1954 | −0.4008 | −0.7200 |

### Master Merged Dataset (first 5 rows)

| Year | GlobalAnomaly | GlobalAbsoluteTemp | IndiaTemp | IndiaTempAnomaly | Rainfall | RainfallAnomaly | ONIAnnualMean | ONIMonsoonMean |
|---|---|---|---|---|---|---|---|---|
| 1950 | −0.1750 | 13.8250 | 24.88 | −0.3727 | 330.349 | +5.827 | −0.8600 | −0.5500 |
| 1951 | −0.0683 | 13.9317 | 25.33 | +0.0773 | 283.112 | −41.410 | +0.4308 | +0.7900 |
| 1952 | +0.0108 | 14.0108 | 25.41 | +0.1573 | 303.045 | −21.477 | +0.1742 | +0.0175 |
| 1953 | +0.0825 | 14.0825 | 25.58 | +0.3273 | 332.295 | +7.773 | +0.7133 | +0.7575 |
| 1954 | −0.1317 | 13.8683 | 25.24 | −0.0127 | 327.712 | +3.190 | −0.4008 | −0.7200 |

---

## 🛠️ Feature Engineering

For each of the 5 core variables, the following features are engineered — producing **46 total feature columns**:

| Feature Type | Column Suffix | Description |
|---|---|---|
| Lag 1 year | `_lag1` | Value from previous year |
| Lag 2 years | `_lag2` | Value from 2 years prior |
| First difference | `_diff1` | Year-over-year change |
| 3-year rolling mean | `_rolling3` | Short-term smoothed trend |
| 5-year rolling mean | `_rolling5` | Medium-term smoothed trend |
| Decade label | `Decade` | Decade grouping (e.g. 1950, 1960…) |
| Year index | `YearIndex` | 0-based integer index |

### Sample Feature-Engineered Row (Year 1954)

| Feature | Value |
|---|---|
| GlobalAnomaly | −0.1317 |
| GlobalAnomaly_lag1 | +0.0825 |
| GlobalAnomaly_lag2 | +0.0108 |
| GlobalAnomaly_diff1 | −0.2142 |
| GlobalAnomaly_rolling3 | −0.0128 |
| GlobalAnomaly_rolling5 | −0.0563 |
| IndiaTemp | 25.24 |
| IndiaTemp_lag1 | 25.58 |
| IndiaTemp_diff1 | −0.34 |
| IndiaTemp_rolling3 | 25.41 |
| IndiaTemp_rolling5 | 25.288 |
| Rainfall_lag1 | 332.295 |
| Rainfall_diff1 | −4.583 |
| ONIAnnualMean_lag1 | +0.7133 |
| ONIAnnualMean_rolling3 | +0.1622 |
| Decade | 1950 |
| YearIndex | 4 |

---

## 📋 Data Quality Summary

| Metric | Value |
|---|---|
| Number of years analysed | 73 |
| First year | 1950 |
| Last year | 2022 |
| Missing rainfall values | 0 |
| Duplicate years | 0 |
| India baseline mean (1951–1980) | **25.2527 °C** |
| Rainfall baseline mean (1951–1980) | **324.5220 mm** |

---

## 📈 Trend Analysis — Full Results

Mann-Kendall monotonic trend test + Sen's slope estimator for all 7 variables:

| # | Variable | Trend | p-value | Kendall's τ | Sen's Slope/yr | Linear Slope/yr | Sig. 5%? | N |
|---|---|---|---|---|---|---|---|---|
| 0 | GlobalAnomaly | **Increasing** | 0.000000e+00 | +0.782725 | +0.015121 | +0.015079 | ✅ Yes | 73 |
| 1 | IndiaTemp | **Increasing** | 1.131628e−11 | +0.546557 | +0.011545 | +0.010646 | ✅ Yes | 72 |
| 2 | IndiaTempAnomaly | **Increasing** | 1.131628e−11 | +0.546557 | +0.011545 | +0.010646 | ✅ Yes | 72 |
| 3 | Rainfall | No trend | 7.496685e−01 | +0.025875 | +0.045142 | +0.053545 | ❌ No | 73 |
| 4 | RainfallAnomaly | No trend | 7.496685e−01 | +0.025875 | +0.045142 | +0.053545 | ❌ No | 73 |
| 5 | ONIAnnualMean | No trend | 4.779496e−01 | −0.057078 | −0.002428 | −0.001631 | ❌ No | 73 |
| 6 | ONIMonsoonMean | No trend | 5.452916e−01 | −0.048706 | −0.002717 | −0.002323 | ❌ No | 73 |

> **Interpretation:** Only temperature variables show statistically significant monotonic increasing trends. Rainfall and ENSO indices show no significant directional trend over 73 years.

---

## 🔗 Correlation Analysis — Full Matrix

Pearson correlation matrix across all climate variables:

| | GlobalAnomaly | IndiaTemp | IndiaTempAnomaly | Rainfall | RainfallAnomaly | ONIAnnualMean | ONIMonsoonMean |
|---|---|---|---|---|---|---|---|
| **GlobalAnomaly** | 1.0000 | **+0.8240** | **+0.8240** | +0.0934 | +0.0934 | +0.0532 | −0.0144 |
| **IndiaTemp** | +0.8240 | 1.0000 | 1.0000 | −0.1463 | −0.1463 | +0.2237 | +0.0934 |
| **IndiaTempAnomaly** | +0.8240 | 1.0000 | 1.0000 | −0.1463 | −0.1463 | +0.2237 | +0.0934 |
| **Rainfall** | +0.0934 | −0.1463 | −0.1463 | 1.0000 | 1.0000 | −0.3077 | **−0.4521** |
| **RainfallAnomaly** | +0.0934 | −0.1463 | −0.1463 | 1.0000 | 1.0000 | −0.3077 | **−0.4521** |
| **ONIAnnualMean** | +0.0532 | +0.2237 | +0.2237 | −0.3077 | −0.3077 | 1.0000 | **+0.9130** |
| **ONIMonsoonMean** | −0.0144 | +0.0934 | +0.0934 | **−0.4521** | **−0.4521** | **+0.9130** | 1.0000 |

> **Key:** Global–India temperature (r=0.824, p<0.001); ONI Monsoon Mean–Rainfall (r=−0.452, p<0.001); ONI Annual–Monsoon correlation (r=0.913).

---

## 🔬 Principal Component Analysis

5 components fitted on standardised [GlobalAnomaly, IndiaTempAnomaly, RainfallAnomaly, ONIAnnualMean, ONIMonsoonMean].

### Explained Variance Table

| PC | Explained Variance Ratio | Cumulative Variance Ratio |
|---|---|---|
| PC1 | 45.0163% | 45.0163% |
| PC2 | 34.8749% | **79.8912%** |
| PC3 | 15.4869% | 95.3781% |
| PC4 | 3.6438% | 99.0219% |
| PC5 | 0.9781% | 100.0000% |

### Full PCA Loadings (all 5 PCs)

| Variable | PC1 | PC2 | PC3 | PC4 | PC5 |
|---|---|---|---|---|---|
| GlobalAnomaly | +0.1973 | **+0.6892** | +0.0638 | +0.6556 | −0.2282 |
| IndiaTempAnomaly | +0.3228 | **+0.6212** | −0.1416 | −0.6310 | +0.3027 |
| RainfallAnomaly | **−0.3848** | +0.1968 | **+0.8768** | −0.1150 | +0.1767 |
| ONIAnnualMean | **+0.5952** | −0.1723 | +0.3932 | −0.2554 | −0.6295 |
| ONIMonsoonMean | **+0.5955** | −0.2657 | +0.2291 | +0.3058 | +0.6548 |

> **PC1** = ENSO mode (high ONI loadings, negative rainfall loading). **PC2** = Global warming mode (high global & India temperature). **PC3** = Rainfall variability mode.

### PCA Scores — First 5 Years

| Year | PC1 | PC2 |
|---|---|---|
| 1950 | −2.2605 | −1.6400 |
| 1951 | +1.3332 | −1.7105 |
| 1952 | +0.2751 | −0.8757 |
| 1953 | +1.2571 | −0.6126 |
| 1954 | −1.5202 | −0.8960 |

---

## 🎯 PCA Scores & Clustering

K-Means (k=3) and Agglomerative Clustering on PC1 and PC2 scores:

| Year | PC1 | PC2 | K-Means Cluster | Agg Cluster |
|---|---|---|---|---|
| 1950 | −2.2605 | −1.6400 | 2 | 2 |
| 1951 | +1.3332 | −1.7105 | 0 | 1 |
| 1952 | +0.2751 | −0.8757 | 0 | 1 |
| 1953 | +1.2571 | −0.6126 | 0 | 1 |
| 1954 | −1.5202 | −0.8960 | 2 | 2 |

**Cluster interpretation:**
- **Cluster 0** — El Niño regime: High PC1, warm temperature anomaly, below-normal rainfall
- **Cluster 1** — Neutral regime: Near-zero PC1/PC2, moderate conditions
- **Cluster 2** — La Niña / cool regime: Low PC1, above-normal rainfall, cooler anomaly

---

## 🤖 Regression Modelling

**Target variable:** `IndiaTempAnomaly`  
**Cross-validation:** `TimeSeriesSplit(n_splits=5)` — strictly no data leakage  
**Total features:** 46 (lag, rolling, difference, base variables)  
**Preprocessing:** `SimpleImputer(strategy='median')` → `StandardScaler` → model  

### Model Comparison (mean across 5 time-series folds)

| # | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| 0 | **Ridge (α=1.0)** | **0.0540** | **0.0601** | **+0.810** |
| 1 | Random Forest | 0.1942 | 0.2271 | −1.608 |
| 2 | XGBoost | 0.2037 | 0.2386 | −1.831 |

> ✅ **Best model: Ridge Regression** — lowest MAE, highest R².

---

## 📊 Feature Importance

Random Forest permutation importance — top 15 features:

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | GlobalAnomaly_rolling5 | **0.122801** |
| 2 | YearIndex | 0.070143 |
| 3 | GlobalAnomaly | 0.067208 |
| 4 | GlobalAnomaly_rolling3 | 0.058529 |
| 5 | ONIAnnualMean | 0.053253 |
| 6 | GlobalAbsoluteTemp | 0.053224 |
| 7 | ONIAnnualMean_rolling3 | 0.034802 |
| 8 | ONIAnnualMean_lag1 | 0.014462 |
| 9 | ONIMonsoonMean_lag1 | 0.013194 |
| 10 | GlobalAnomaly_diff1 | 0.009989 |
| 11 | GlobalAnomaly_lag2 | 0.009743 |
| 12 | ONIMonsoonMean | 0.008128 |
| 13 | RainfallAnomaly_lag1 | 0.007456 |
| 14 | RainfallAnomaly | 0.007322 |
| 15 | Rainfall_lag1 | 0.007090 |

---

## 🔍 Regression Diagnostics

Ridge regression residual normality tests:

| # | Diagnostic | Value | Interpretation |
|---|---|---|---|
| 0 | Residual Mean | 0.106021 | Small positive bias |
| 1 | Residual Std | 0.218421 | Moderate spread |
| 2 | Shapiro-Wilk p-value | **0.432863** | ✅ Residuals are normally distributed |
| 3 | Jarque-Bera p-value | **0.754867** | ✅ Residuals are normally distributed |

---

## 🌧️ Rainfall Classification

**Task:** Classify each monsoon year as Above Normal vs. Below/Near Normal  
**Model:** Logistic Regression  
**Validation:** `TimeSeriesSplit(n_splits=5)`  

### Cross-Validation Results — Per Fold

| Fold | Accuracy | Balanced Accuracy | F1 Score |
|---|---|---|---|
| 1 | 0.5833 | 0.6875 | 0.6154 |
| 2 | **0.8333** | **0.8571** | **0.8333** |
| 3 | 0.5000 | 0.4375 | 0.6250 |
| 4 | 0.5000 | 0.5000 | 0.6667 |
| 5 | 0.5833 | 0.5833 | 0.7059 |
| **Mean** | **0.6000** | **0.6131** | **0.6893** |

---

## 🔮 ARIMA Forecasting

**Target:** `IndiaTempAnomaly`  
**Model:** ARIMA / SARIMAX (statsmodels)  
**Horizon:** 10 years (2023–2032)  

### Forecast Diagnostics

| # | Diagnostic | Value | Interpretation |
|---|---|---|---|
| 0 | ADF p-value | 0.992618 | ⚠️ Non-stationary — differencing applied |
| 1 | AIC | −14.435010 | Model fit quality |
| 2 | BIC | −7.646971 | Penalised complexity |
| 3 | Ljung-Box p-value | 0.042540 | Borderline residual autocorrelation |

### ARIMA 10-Year Forecast Table

| # | Year | Forecast India Temp Anomaly (°C) |
|---|---|---|
| 0 | 2022 (last observed) | +0.6303 |
| 1 | 2023 | +0.6176 |
| 2 | 2024 | +0.6141 |
| 3 | 2025 | +0.6132 |
| 4 | **2026** | **+0.6130** |
| 5 | 2027 | +0.6129 |
| 6 | 2028 | +0.6128 |
| 7 | 2029 | +0.6128 |
| 8 | 2030 | +0.6128 |
| 9 | 2031 | +0.6128 |
| 10 | 2032 | +0.6128 |

> **Interpretation:** India's temperature anomaly is projected to plateau at approximately **+0.613 °C** above the 1951–1980 baseline through 2032.

---

## 🌡️ IMD Diurnal Temperature Range

Based on 75 IMD NetCDF files (1951–2014). Values in °F as stored in IMD grid:

| # | Year | Tmax Annual (°F) | Tmin Annual (°F) | DTR (°F) |
|---|---|---|---|---|
| 0 | 1951 | 75.3563 | 71.1099 | **4.2464** |
| 1 | 1952 | 75.3672 | 71.1668 | 4.2003 |
| 2 | 1953 | 75.3268 | 71.2452 | 4.0815 |
| 3 | 1954 | 75.1838 | 71.1353 | 4.0485 |
| 4 | 1955 | 74.9123 | 70.7914 | 4.1209 |

**Key DTR Findings:**
- DTR shows a **long-term declining trend** from 1951 to 2014
- Tmin is rising **faster** than Tmax — classic greenhouse-gas-forced warming signature
- Nighttime minimum temperatures warming faster than daytime maxima (asymmetric warming)
- Consistent with global findings that DTR is narrowing under climate change

---

## 🖼️ Figures Generated

All figures saved at **140 DPI**, size **12 × 7 inches**, Seaborn `whitegrid` theme:

| # | Filename | Description |
|---|---|---|
| 01 | `01_temperature_anomaly_comparison` | Global vs. India temperature anomaly — dual time series with trend lines |
| 02 | `02_rainfall_anomaly_bar` | Annual monsoon rainfall anomaly — bar chart with ±1σ baseline bands |
| 03 | `03_oni_series` | ONI Annual Mean and Monsoon Mean — time series with El Niño/La Niña bands |
| 04 | `04_correlation_heatmap` | Pearson correlation heatmap with annotated r values |
| 05 | `05_pca_cluster_map` | PCA biplot (PC1 vs PC2) coloured by K-Means cluster |
| 06 | `06_pc1_loadings` | PC1 variable loadings — horizontal bar chart |
| 07 | `07_best_model_and_residuals` | Ridge regression: predicted vs. actual + residual distribution |
| 08 | `08_feature_importance` | Random Forest permutation importance — top 15 features |
| 09 | `09_rainfall_classification` | Confusion matrix — monsoon classification (Logistic Regression) |
| 10a | `10_arima_forecast` | ARIMA 10-year India temperature anomaly forecast with 95% CI |
| 10b | `10_forecast_results` | Forecast results summary comparison plot |
| 11a | `11_enso_relationships` | ONI vs. India temperature and rainfall scatter (coloured by decade) |
| 11b | `11_forecast_acf_pacf` | ACF and PACF of India temperature anomaly series |
| 12 | `12_rolling_mean_plots` | 5-year and 10-year rolling means for all 5 variables |
| 13a | `13_imd_dtr` | IMD Tmax, Tmin, and DTR trends 1951–2014 (°F) |
| 13b | `13_imd_dtr_celsius` | IMD Tmax, Tmin, and DTR trends 1951–2014 (°C) |

---

## ⚙️ Setup & Usage

### Prerequisites

- Python 3.9 or higher
- ~500 MB disk space for data and outputs
- Jupyter Lab or Jupyter Notebook

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/india-climate-data-science.git
cd india-climate-data-science
```

### Step 2 — Create a virtual environment

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Place your data files

```
data/processed/annual.csv
data/processed/india_official.csv
data/processed/india_rainfall_annual.csv
data/processed/imd_temperature_csv/tmax_annual.csv
data/processed/imd_temperature_csv/tmin_annual.csv
data/external/oni.csv
data/raw/imd_temperature/tmax/*.nc    ← 75 NetCDF files
data/raw/imd_temperature/tmin/*.nc    ← 75 NetCDF files
```

### Step 5 — Launch the notebook

```bash
jupyter lab
# Open: notebooks/data_science_climate.ipynb
# Kernel → Restart & Run All
```

### Step 6 — Find your outputs

```
data/processed/climate_project_outputs_final_thesis_complete/
├── figures/     ← 16 publication-ready PNG figures (140 DPI)
├── tables/      ← All result tables as CSV
└── exports/     ← Excel workbooks with all tables
```

---

## 📦 Dependencies

```
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
statsmodels
xgboost
shap
pymannkendall
xarray
netCDF4
openpyxl
jupyter
jupyterlab
nbformat
```

Full install command:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn statsmodels \
            xgboost shap pymannkendall xarray netCDF4 openpyxl jupyter jupyterlab nbformat
```

---

## 🧪 Reproducibility

| Setting | Value |
|---|---|
| Random seed | `RANDOM_STATE = 42` |
| Cross-validation | `TimeSeriesSplit(n_splits=5)` — no data leakage |
| Bootstrap samples | `N_BOOTSTRAP = 2000` |
| Anomaly baseline | 1951–1980 |
| Global absolute offset | 14.0 °C |
| Forecast horizon | 10 years |
| Figure DPI | 140 |
| Figure size | 12 × 7 inches |
| Seaborn theme | `whitegrid`, context=`talk` |

All random seeds are fixed. All outputs are fully deterministic given the same input data. A `configuration.json` snapshot is written to the output directory on every run.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **NASA GISTEMP** — Global Surface Temperature Analysis (GISS Surface Temperature Analysis v4)
- **India Meteorological Department (IMD)** — Gridded Tmax/Tmin NetCDF and annual rainfall data
- **NOAA Climate Prediction Center** — Oceanic Niño Index (ONI)
- **NASA POWER** — Satellite-derived climate comparison data

---
## Author

**Maharshi K Patel**
MSc Agriculture Analytics
[LinkedIn](https://www.linkedin.com/in/maharshikpmk2002)
---

<p align="center">
  <b>🌡️ 73 years · 5 data streams · 13 analysis sections · 16 figures · 46 engineered features</b><br>
  <b>Ridge R²=0.810 · PCA 79.89% variance · ARIMA forecast to 2032</b><br><br>
  <i>Built with 🐍 Python — for a warmer, better-understood planet.</i>
</p>
