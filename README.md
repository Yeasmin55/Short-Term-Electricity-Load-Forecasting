# ⚡ Short-Term Electricity Load Forecasting
### A Comparative Time-Series Analysis | UCI Electricity Load Diagrams 2011–2014

>  | **Author:** Yeasmin Akter | **Date:** April 2026

## 📌 Project Overview
This project develops and compares **six forecasting models across four model families** to predict hourly electricity demand over a **168-hour (one-week) horizon**, using real-world load data from 370 Portuguese electricity consumers (2011–2014).
Accurate short-term load forecasting directly reduces utility imbalance penalties, improves procurement scheduling, and lowers operational risk. Even a 1% reduction in forecast error across a large portfolio can represent **millions of dollars in annual savings**.
---
## 🏆 Key Results

| Model | MASE (Test) | RMSE (kW) | MAE (kW) |
|---|---|---|---|
| 🥇 Random Forest | **1.12** | 11,999 | 7,608 |
| Linear Regression | 1.99 | 21,871 | - |
| SARIMA | 3.80 | 36,212 | - |
| ETS (Holt-Winters) | 7.17 | - | - |
| Seasonal Naïve | ~3.2 | - | 24,598 |
| Naïve | 9.02 | - | 61,177 |

> **Random Forest achieves an 87.6% reduction in MAE over the Naïve baseline.**  
> MASE < 1.0 = beats naïve; lower is better.

---

## 📊 Dataset
| Attribute | Detail |
|---|---|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) |
| File | `LD2011_2014.txt` |
| Raw Resolution | 15-minute intervals |
| Time Span | January 2011 - December 2014 |
| Customers | 370 Portuguese electricity consumers |
| Aggregated Series | 35,064 hourly observations |
| Units | Kilowatts (kW) |
---
## 🧪 Model Families Compared

### Family 1 - Naïve Baselines
- **Naïve**: Carries last observed value forward (performance floor)
- **Seasonal Naïve (SNaïve)**: Uses same hour from the prior day

### Family 2 - Exponential Smoothing (ETS / Holt-Winters)
- Auto-selected via AIC: `ETS(A,Ad,A)` - additive error, damped trend, additive seasonality
- Box-Cox transformation applied (λ = −0.44) for variance stabilization

### Family 3 - Seasonal ARIMA (SARIMA)
- Final spec: `ARIMA(3,0,1)(2,1,1)[24]`  selected via AICc with full parameter search
- Seasonal period s = 24 hours; one seasonal difference (D=1)

### Family 4 - Machine Learning
- **Multiple Linear Regression** - interpretable lag-feature baseline
- **Random Forest** (200 trees) - captures nonlinear lag interactions
- Features: lags 1–24, lag 168 (weekly), sin/cos hour encodings, day-of-week indicators

---

## 🔧 Methodology

```
Raw 15-min data → Hourly Aggregation → Missing Value Imputation (forward-fill)
       ↓
Stationarity Testing (ADF) → Box-Cox Transformation (λ = −0.44)
       ↓
Train/Test Split (final 168 hrs held out) → Model Fitting → Residual Diagnostics
       ↓
Rolling-Origin Cross-Validation (14 folds) → Model Selection → 168-hr Forecast
```

### Preprocessing Steps
1. **Aggregation** - arithmetic mean of four 15-min intervals per hour
2. **Missing Values** - forward-fill (<0.3% of records affected)
3. **Stationarity** - ADF test: statistic = −4.95, p < 0.01 (level-stationary)
4. **Variance Stabilization** - Box-Cox with λ = −0.44
5. **Seasonal Differencing** - D = 1 at s = 24 (for SARIMA only)

### Evaluation Metrics
- **Primary:** MASE (Mean Absolute Scaled Error) - scale-independent, interpretable
- **Secondary:** RMSE, MAE, MAPE
- **Validation:** 14-fold rolling-origin cross-validation

---

## 📈 Key Findings

- The series shows strong **24-hour daily** and **168-hour weekly** seasonal patterns confirmed by ACF/PACF and STL decomposition
- **Random Forest** was the top performer, with lag_1, lag_24, lag_168, and hour-of-day encodings as the most important features
- **SARIMA** is the best interpretable statistical alternative; Ljung-Box residual diagnostics confirmed white-noise residuals in the refined model
- **ETS** underperformed relative to SARIMA on this hourly dataset — autoregressive structure captures the seasonality more effectively than exponential smoothing
- **Rolling-origin CV** confirmed performance rankings are consistent across forecast origins (no overfitting to one test window)

---

## 🛠 Tools & Technologies

![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)
![Time Series](https://img.shields.io/badge/Time%20Series-Analysis-orange) 
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)

| Tool | Usage |
|---|---|
| **R** (v4.5.1) | Primary analysis environment |
| `forecast` package | ETS, SARIMA, accuracy metrics |
| `randomForest` package | Random Forest model |
| `tseries` package | ADF stationarity testing |
| UCI ML Repository | Data source |

---

## 📂 Repository Structure

```
├── data/
│   └── LD2011_2014.txt          # Raw UCI dataset (download separately)
├── R/
│   ├── Short-Term Electricity Load Forecasting-Yeasmin Akter.R
├── outputs/
│   ├── figures/                 # All generated plots
│   └── forecast_table.csv       # 168-hour forecasts from all 6 models
├── Dashboard/
│   ├── App.r                 
│   ├── model_objects.r          
├── report/
│   └── Yeasmin_Akter_Short_Term_Load_Forecasting_Analysis.docx
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```r
install.packages(c("forecast", "randomForest", "tseries", "ggplot2", "dplyr"))
```

### Run the Analysis
```r
# 1. Download data from UCI repository and place in /data
# 2. Run scripts in order:
source("R/01_data_prep.R")
source("R/02_eda.R")
source("R/03_baseline_models.R")
source("R/04_ets_sarima.R")
source("R/05_ml_models.R")
source("R/06_evaluation.R")
```

---

## 📚 References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). [otexts.com/fpp3](https://otexts.com/fpp3/)
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
- Taylor, J. W. (2003). Short-term electricity demand forecasting using double seasonal exponential smoothing. *JORS*, 54(8), 799–805.
- Trindade, A. (2015). *ElectricityLoadDiagrams20112014* [Dataset]. UCI ML Repository. [doi.org/10.24432/C58C86](https://doi.org/10.24432/C58C86)

---

## 📄 License

This project was completed as part of DSCI 725 – Data Mining coursework. The dataset is publicly available via the UCI Machine Learning Repository under their usage terms.

---

*Feel free to open an issue or reach out with questions!*
