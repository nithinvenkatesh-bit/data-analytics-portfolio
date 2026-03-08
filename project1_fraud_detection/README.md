# Fraud Detection Pipeline

End-to-end fraud detection system built on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud rate). Covers data quality validation, statistical threshold tuning, KPI tracking, and production-style SQL monitoring queries.

---

## What This Project Does

Most fraud detection tutorials stop at model accuracy. This project goes further — it replicates the operational layer that sits around the model: threshold tuning to reduce false positives, KPI dashboards that track fraud rate and recovery rate over time, and SQL queries that flag threshold drift before it hits SLA.

---

## Project Structure

```
project1_fraud_detection/
│
├── fraud_detection_pipeline.py   # Main pipeline — EDA → features → model → KPIs
├── sql/
│   └── fraud_kpi_queries.sql     # Production-style SQL: KPI tracking, SLA compliance,
│                                 # threshold drift detection, chargeback anomalies
├── data/
│   └── creditcard.csv            # Download from Kaggle (link below — not committed)
├── outputs/                      # Auto-generated charts and KPI CSV
│   ├── eda_overview.png
│   ├── threshold_tuning.png
│   ├── feature_importance.png
│   └── kpi_summary.csv
└── requirements.txt
```

---

## Key Techniques

| Area | Approach |
|---|---|
| Data Quality | Row-level checks — duplicates, negative amounts, class validity |
| Class Imbalance | `class_weight='balanced'` on both models; stratified train/test split |
| Threshold Tuning | Grid search across 0.1–0.9 to maximise F1; quantifies false positive reduction |
| Models | Logistic Regression (interpretable baseline) + Random Forest (production) |
| KPI Tracking | Fraud detection rate, precision, false positive rate, ROC-AUC |
| SQL Monitoring | Threshold drift detection using 4-week rolling average; SLA compliance by priority tier |

---

## Results

| Metric | Default Threshold (0.50) | Tuned Threshold |
|---|---|---|
| F1 Score | ~0.84 | ~0.87 |
| False Positives | Higher | Reduced ~20% |
| Fraud Detection Rate | ~80% | ~85% |

> Exact numbers will vary slightly depending on dataset version downloaded from Kaggle.

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/nithinvenkatesh/portfolio.git
cd project1_fraud_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in the /data folder

# 4. Run the pipeline
python fraud_detection_pipeline.py
```

---

## SQL Queries

The `sql/fraud_kpi_queries.sql` file contains 5 production-style queries:

1. **Daily Fraud KPI Summary** — fraud rate, recovery rate, avg case turnaround by day
2. **Threshold Drift Detection** — flags accounts where fraud rate has drifted >1.5% from 4-week rolling average
3. **False Positive Analysis** — segments FP alerts by transaction type and amount bucket for root cause analysis
4. **SLA Compliance Tracking** — monitors case resolution against P1/P2/P3 SLA targets
5. **Chargeback Anomaly Detection** — surfaces accounts with abnormal chargeback ratios

These queries are written for SQL Server / Snowflake syntax. Minor adjustments needed for MySQL or PostgreSQL (e.g. `DATEADD` → `DATE_SUB`).

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```
