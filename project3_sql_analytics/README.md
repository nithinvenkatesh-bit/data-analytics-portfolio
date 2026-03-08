# E-Commerce SQL Analytics — Conversion Funnel, Cohort Retention & A/B Testing

End-to-end analytics project on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (100K+ orders). Covers conversion funnel analysis, monthly cohort retention, A/B test framework with statistical significance testing, and a Tableau Public dashboard built from exported CSVs.

---

## What This Project Does

Four distinct analytical workstreams that mirror real consulting deliverables:

1. **Conversion Funnel** — maps session-to-purchase drop-off by acquisition channel, estimates recoverable margin from cart abandonment
2. **Cohort Retention** — monthly retention matrix showing how different customer cohorts behave over 12 months
3. **A/B Test Framework** — statistically rigorous comparison of two discount strategies, including sample ratio mismatch check and Z-test for significance
4. **Operational KPIs** — weekly revenue, AOV, WoW growth, and vendor on-time delivery rates — exported for Tableau

---

## Project Structure

```
project3_sql_analytics/
│
├── ecommerce_analytics.sql       # All SQL: funnel, cohort, A/B test, KPI queries
├── data_prep_for_tableau.py      # Python: loads Olist data, builds cohort matrix,
│                                 # exports clean CSVs for Tableau Public
├── data/                         # Download from Kaggle (not committed)
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   └── olist_sellers_dataset.csv
├── outputs/                      # Auto-generated
│   ├── cohort_retention_heatmap.png
│   ├── tableau_orders.csv
│   ├── tableau_cohort_retention.csv
│   └── tableau_monthly_kpis.csv
└── requirements.txt
```

---

## SQL Query Overview

### Section 1 — Conversion Funnel
| Query | Purpose |
|---|---|
| `1a` Full funnel by channel | Conversion rate at each stage (view → cart → checkout → order) |
| `1b` Cart abandonment | Revenue lost pre-checkout vs at-checkout; recoverable margin estimate |

### Section 2 — Cohort Retention
| Query | Purpose |
|---|---|
| `2a` Monthly cohort matrix | Retention rate by acquisition month and months since first purchase |
| `2b` Repeat purchase rate | Which channels produce the highest-LTV repeat buyers |

### Section 3 — A/B Test Framework
| Query | Purpose |
|---|---|
| `3a` Sample ratio mismatch | Validates even split before looking at results |
| `3b` Primary metrics | Conversion rate, AOV, revenue by variant |
| `3c` Statistical significance | Z-test for proportions — flags result as significant or not |

### Section 4 — Operational KPIs
| Query | Purpose |
|---|---|
| `4a` Weekly revenue | Orders, revenue, AOV, WoW growth |
| `4b` Vendor delivery | On-time delivery rate and avg review score by seller |

---

## Tableau Public Dashboard

After running `data_prep_for_tableau.py`, connect Tableau Public to the three exported CSVs:

**Recommended views:**
- Monthly revenue trend with WoW growth overlay (`tableau_monthly_kpis.csv`)
- Cohort retention heatmap (`tableau_cohort_retention.csv`)
- Funnel chart by acquisition channel (`tableau_orders.csv`)
- Vendor delivery performance scatter (`tableau_orders.csv`)

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/nithinvenkatesh/portfolio.git
cd project3_sql_analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# Go to: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Place all CSV files in the /data folder

# 4. Run Python prep
python data_prep_for_tableau.py

# 5. SQL queries
# Run ecommerce_analytics.sql in your SQL client against the Olist schema
# Or adapt for SQLite using the CSV files directly
```

---

## Key Analytical Decisions Worth Discussing in Interviews

- **A/B test: sample ratio mismatch check runs first** — looking at conversion rates before validating the split is a common mistake that invalidates tests
- **Z-test is in SQL, not just Python** — shows the statistical logic, not just the output
- **Cohort analysis uses `customer_unique_id`** — Olist has separate `customer_id` per order; using unique ID is the correct approach for retention
- **Recoverable margin is estimated conservatively** — 15% winback rate is the lower bound of industry benchmarks, not a best-case number

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
openpyxl
```
