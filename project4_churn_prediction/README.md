# Customer Churn Prediction — XGBoost (class-weighted) + Threshold Tuning

Predicts telecom customer churn on the **IBM Telco Customer Churn** dataset
(7,043 customers, ~27% churn) using a class-weighted XGBoost model, threshold
tuning, feature-importance analysis, and predicted risk tiers for targeting.

---

## What This Project Does

Two phases:
- **`01_data_exploration.ipynb`** — EDA: churn distribution and churn rates by
  contract, internet service, tenure, and demographics.
- **`churn_prediction.py`** — the model: preprocessing, class-weighted XGBoost,
  threshold tuning, feature importance, and risk-tier segmentation.

All metrics are produced by the committed code on the real dataset (`random_state=42`).

---

## Results (real dataset, held-out test set)

| Metric | Value |
|---|---|
| ROC-AUC | **0.834** |
| Recall @ 0.50 (churners caught) | 78.3% (293 / 374) |
| F1-optimal threshold | 0.61 → recall 70.6%, precision 56.2% |
| Dominant driver | Contract type (~60% of feature importance) |
| High-Risk tier actual churn | **65%** |
| Low-Risk tier actual churn | **1%** |

**Headline:** the model separates risk cleanly — the top predicted-risk tier
churns at 65% versus 1% for the lowest, a 60x+ differential that's directly
usable for targeted retention spend. Contract type dominates: month-to-month
customers churn far more than one- or two-year contracts.

**On the data:** the pipeline drops 11 rows with blank `TotalCharges` (new
customers at tenure 0 — a known quirk of this dataset), leaving 7,032 modelled.

---

## Project Structure

```
project4_churn_prediction/
├── churn_prediction.py          # Model: preprocess → XGBoost → tune → tiers
├── 01_data_exploration.ipynb    # Phase 1 EDA
├── data/
│   └── telco_churn.csv          # IBM Telco dataset (included; public sample data)
├── outputs/
│   ├── eda_churn_overview.png
│   ├── threshold_tuning.png
│   ├── feature_importance.png
│   ├── risk_tiers.png
│   └── metrics_summary.csv
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
pip install -r requirements.txt
python churn_prediction.py     # data/telco_churn.csv is already included
```

Deterministic — re-running reproduces the committed charts and `metrics_summary.csv`.

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
```
