# Data Analytics Portfolio — Nithin Venkatesh

[![LinkedIn](https://img.shields.io/badge/LinkedIn-nthiv-blue)](https://www.linkedin.com/in/nthiv/)

Three end-to-end analytics projects covering fraud detection, customer segmentation, and e-commerce analytics. Each project is built on a public dataset, uses real analytical techniques, and is structured the way production work is — with data quality checks, documented decisions, and outputs a stakeholder can actually use.

---

## Projects

### 1. [Fraud Detection Pipeline](./project1_fraud_detection/)
**Stack:** Python (Scikit-learn, Pandas, NumPy), SQL (Snowflake/SQL Server syntax)

End-to-end fraud detection system on 284,807 credit card transactions. Covers row-level data quality validation, statistical threshold tuning to reduce false positives, Random Forest vs Logistic Regression comparison, and production-style SQL queries for KPI tracking, SLA compliance monitoring, and threshold drift detection.

**Key result:** Threshold tuning reduces false positive alerts by ~20% vs default 0.5 cutoff.

---

### 2. [Customer Segmentation — K-Means + PCA](./project2_customer_segmentation/)
**Stack:** Python (Scikit-learn, Pandas, Matplotlib), Tableau Public

Segments 4,300+ retail customers using RFM feature engineering and K-Means clustering. Optimal K selected via Elbow + Silhouette analysis. Segments validated visually with PCA and translated into business-ready profiles (Champions, At Risk, New Buyers, etc.) with revenue contribution and specific retention recommendations per segment.

**Key result:** 5 distinct customer segments identified; Champions (9% of customers) account for 34% of revenue.

---

### 3. [E-Commerce SQL Analytics](./project3_sql_analytics/)
**Stack:** SQL (Snowflake/SQL Server syntax), Python (Pandas, Seaborn), Tableau Public

Four analytical workstreams on 100K+ Olist orders: conversion funnel analysis with recoverable margin estimation, monthly cohort retention matrix, statistically rigorous A/B test framework (includes sample ratio mismatch check + Z-test in SQL), and operational KPI queries designed to feed a Tableau dashboard.

**Key result:** Cohort analysis reveals 30-day retention drops below 5% for most channels — repeat purchase strategies are highest-ROI opportunity.

---

## How This Portfolio Is Structured

Each project follows the same pattern:
1. **Raw data → quality checks** — problems are flagged explicitly, not silently dropped
2. **Analysis** — documented decisions, not just code
3. **Output** — charts, CSVs, or dashboard-ready exports a stakeholder can use
4. **README** — explains the analytical choices, not just how to run the code

---

## Contact

- **Email:** nithinvenkatesh203@gmail.com
- **LinkedIn:** [linkedin.com/in/nthiv](https://www.linkedin.com/in/nthiv/)
- **Location:** San Francisco, CA (Open to Relocate)
