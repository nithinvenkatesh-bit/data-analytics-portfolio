# Customer Segmentation — K-Means Clustering + PCA

Segments 4,300+ retail customers into distinct behavioural profiles using RFM (Recency, Frequency, Monetary) feature engineering, K-Means clustering, and PCA validation. Each segment is profiled with revenue contribution and actionable business recommendations.

---

## What This Project Does

Raw transaction data doesn't tell you who your customers are. This project builds RFM features from invoice-level data, uses Elbow + Silhouette analysis to find the right number of clusters, validates the segments visually with PCA, and translates cluster numbers into business language — Champions, Loyal Customers, At Risk, etc. — with a specific recommendation for each.

---

## Project Structure

```
project2_customer_segmentation/
│
├── customer_segmentation.py     # Full pipeline — RFM → clustering → PCA → profiles
├── data/
│   └── online_retail_II.xlsx    # Download from UCI (link below — not committed)
├── outputs/                     # Auto-generated on run
│   ├── optimal_k_selection.png  # Elbow + Silhouette charts
│   ├── pca_validation.png       # 2D PCA cluster plot
│   ├── revenue_by_segment.png   # Revenue contribution by segment
│   ├── segment_summary.csv      # Raw cluster means
│   ├── segment_profiles.csv     # Labelled segments + recommendations
│   └── customers_with_segments.csv
└── requirements.txt
```

---

## Key Techniques

| Area | Approach |
|---|---|
| Feature Engineering | RFM (Recency, Frequency, Monetary) from raw invoice transactions |
| Data Quality | Row-level checks — nulls, negative quantities, cancelled invoices removed |
| Scaling | StandardScaler before clustering (K-Means is distance-based) |
| Optimal K | Elbow Method + Silhouette Score — both evaluated together |
| Clustering | K-Means with `n_init=10`, `random_state=42` for reproducibility |
| Validation | PCA to 2 components — visual separation confirms real cluster structure |
| Output | Human-readable segment labels + revenue analysis + business recommendations |

---

## Sample Output — Segment Profiles

| Segment | Label | Customers | Avg Recency | Avg Frequency | Avg Spend |
|---|---|---|---|---|---|
| 0 | Champions | 412 | 18 days | 14.2 | $3,847 |
| 1 | Loyal Customers | 891 | 45 days | 7.8 | $1,203 |
| 2 | At Risk / Churned | 673 | 220 days | 3.1 | $489 |
| 3 | New / One-Time Buyers | 1,204 | 92 days | 1.4 | $312 |
| 4 | Potential Loyalists | 1,140 | 61 days | 4.3 | $748 |

> Exact values vary by dataset version. Run the pipeline to get your results.

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/nithinvenkatesh/portfolio.git
cd project2_customer_segmentation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# Go to: https://archive.ics.uci.edu/dataset/502/online+retail+ii
# Place online_retail_II.xlsx in the /data folder

# 4. Run
python customer_segmentation.py
```

---

## Tableau Dashboard

The `outputs/customers_with_segments.csv` file is designed to connect directly to Tableau Public for interactive exploration.

**Recommended views to build:**
- Segment size and revenue contribution (bar chart)
- RFM scatter plot coloured by segment
- Recency vs Monetary heatmap by segment
- Geographic breakdown if country data is included

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
```
