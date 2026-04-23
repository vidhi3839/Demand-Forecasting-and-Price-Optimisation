# Retail Demand Forecasting and Price Optimisation

**Dataset:** UCI Online Retail II — 1,067,371 transactions across 3,139 products (2009–2011)  
**Best Model:** PyTorch DNN — R²=0.9459, RMSE=21.21, MAE=11.01  
**Notebooks:** 4 sequential notebooks — EDA → Feature Engineering → Modelling → Price Optimisation

---

## Project Overview

This project builds a weekly retail demand forecasting system and price optimisation engine on real transactional data. The pipeline covers:

1. **EDA and Data Cleaning** — 1,067,371 raw transactions cleaned to 159,645 weekly product observations
2. **Feature Engineering** — 30 features: lag features, rolling statistics, cyclical calendar encoding, product-level elasticity
3. **Modelling** — 5 models compared: Linear Regression, Ridge, Lasso, SVR, PyTorch DNN
4. **Price Optimisation** — Revenue lift simulation and bootstrap uncertainty quantification

**Key Results:**

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **DNN (PyTorch)** | **0.9459** | **21.21** | **11.01** |
| Linear Regression | 0.8857 | 30.81 | 17.87 |
| Ridge (α=10) | 0.8858 | 30.81 | 17.86 |
| Lasso (α=0.0052) | 0.8858 | 30.80 | 17.85 |
| SVR (RBF) | 0.4746 | 66.07 | 22.58 |

**Price Optimisation:** DNN +24.2% simulated revenue lift via scipy.optimize  
**Uncertainty:** 10-model bootstrap ensemble — mean std=5.03 units vs RMSE=21.21

---

## Repository Structure

```
demand-forecasting/
├── README.md
├── 01_eda_and_cleaning.ipynb
├── 02_feature_engineering.ipynb
├── 03_modelling.ipynb
└── 04_price_optimisation.ipynb
```

---

## How to Replicate Results

All notebooks run on **Google Colab**. All imports are included at the top of each notebook — nothing extra to install.

### Step 1 — Get the Dataset

1. Go to: [https://archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
2. Download `online_retail_II.xlsx`
3. Upload it to Colab

### Step 2 — Enable GPU (NB03 and NB04 only)

```
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```

### Step 3 — Run Notebooks in Order

Notebooks must be run sequentially. Each notebook saves outputs to Google Drive that the next notebook loads.

#### NB01 — EDA and Data Cleaning
Loads both Excel sheets, removes cancellations and non-product codes, clips outliers at 99th percentile, aggregates to product-week level, performs EDA.

#### NB02 — Feature Engineering
Builds 30 features — lag (1/2/4-week), rolling statistics (4/8-week), change features, cyclical calendar encoding (month sin/cos), product-level elasticity proxy.

#### NB03 — Modelling
Temporal train/test split at 2011-08-15, trains 5 models, DNN regularisation experiment, model comparison.

#### NB04 — Price Optimisation
Loads trained DNN, runs price optimisation via scipy.optimize on 500 test rows, trains 10 bootstrap DNN models for uncertainty quantification.

---

## Key Technical Decisions

**Temporal split not random split** — Lag features use past demand values. Random splitting leaks future data into training. Split fixed at 2011-08-15: 118,955 training rows, 28,134 test rows covering Q4 holiday season.

**SVR trained on a subset** — Full SVR training on 118,955 rows is intractable (O(n²) kernel). A 10,000-row subset was used, which is itself evidence of SVR's scaling limitation.

**Cyclical encoding for month** — Raw month numbers treat December (12) and January (1) as far apart. Sine/cosine encoding places them adjacent on a unit circle, correctly representing the annual cycle.

**DNN beats linear models** — Linear models plateau at R²≈0.886. The DNN captures non-linear feature interactions that linear models can only approximate additively.

---

## Confirmed Results

After running all 4 notebooks, `results_summary.csv` in your Google Drive will contain the full model comparison. Key numbers:

- **DNN:** R²=0.9459, RMSE=21.21, MAE=11.01, Train-Test Gap=0.0151
- **Linear Regression:** R²=0.8857, RMSE=30.81, CV R²=0.849±0.031
- **SVR (10K subset):** R²=0.4746, RMSE=66.07, MAE=22.58
- **Revenue lift (DNN):** +24.2%
- **Bootstrap uncertainty:** mean std=5.03 units, median=3.11 units, max=163.69 units
