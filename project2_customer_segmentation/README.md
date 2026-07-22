# E-Commerce Analytics (Olist) — Revenue, Cohort Retention & Vendor Delivery

Analytics on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(99,441 orders; 96,478 delivered). Monthly revenue trend, cohort-retention
analysis, and vendor delivery performance, exported for Tableau. **Currency is
Brazilian Reais (R$).** All figures are produced by the committed code on the real data.

---

## Headline Findings (real data)

- **Revenue** scaled from R$46K/month (late 2016) to a Black-Friday peak of
  **R$1.15M** (Nov 2017), holding ~R$1M/month through 2018; R$15.4M total across
  delivered orders.
- **Repeat purchase is near zero — 3.0%** of customers ever place a second
  order. Month-1 cohort retention is ~5%, collapsing to <1% by month 2. This is
  the defining trait of the Olist marketplace and the most important insight in
  the project: acquisition, not retention, drives this business.
- **Delivery**: 92% of delivered orders arrived on or before the estimated date,
  measured across 1,238 sellers with 10+ orders.

---

## What This Project Does

- **`data_prep_for_tableau.py`** — loads Olist, builds the master delivered-order
  table, and produces the revenue trend, cohort matrix, and vendor-delivery
  outputs plus Tableau CSVs and two charts.
- **`ecommerce_analytics.sql`** — Snowflake SQL. **Part A** (revenue, cohort,
  repeat rate, vendor delivery) runs on the real Olist tables. **Part B** (funnel,
  A/B-test Z-test) are clearly-labelled illustrative templates for a production
  analytics schema — Olist has no session, channel, or experiment data, so they
  demonstrate the pattern rather than producing Olist numbers.

---

## Project Structure

```
project3_ecommerce_analytics/
├── data_prep_for_tableau.py      # revenue + cohort + vendor delivery → charts + CSVs
├── ecommerce_analytics.sql       # Part A (real Olist) + Part B (illustrative)
├── data/                         # 9 Olist CSVs (Kaggle; not committed)
├── outputs/
│   ├── monthly_revenue_trend.png
│   ├── cohort_retention_heatmap.png
│   ├── tableau_monthly_kpis.csv
│   ├── tableau_cohort_retention.csv
│   ├── tableau_vendor_delivery.csv
│   └── tableau_orders.csv
└── requirements.txt
```

---

## Setup & Run

```bash
pip install -r requirements.txt
# place the 9 Olist CSVs in data/  (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
python data_prep_for_tableau.py
```

## Interview talking points
- **Uses `customer_unique_id`, not `customer_id`** — Olist issues a new
  `customer_id` per order, so retention must key on the person-level unique id.
- **The 3% repeat rate is the finding**, not a bug — a marketplace where nearly
  all revenue is first-time purchase changes the strategy from retention to acquisition.
- **Part A vs Part B SQL split** — the funnel/A/B templates are labelled as not
  running on Olist, because the public dataset has no session or experiment data.

## Requirements
```
pandas
numpy
matplotlib
```
