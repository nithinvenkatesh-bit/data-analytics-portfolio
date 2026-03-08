"""
E-Commerce Analytics — Python Data Prep & Cohort Visualisation
==============================================================
Loads the Olist dataset, runs the cohort retention analysis,
and exports a clean CSV for Tableau Public dashboard.

Dataset: Brazilian E-Commerce (Olist) — Kaggle
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Author: Nithin Venkatesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ─── 1. DATA LOADING ────────────────────────────────────────────────────────

def load_olist(data_dir: str = 'data/') -> dict:
    """Load all Olist CSV files into a dict of DataFrames."""
    files = {
        'orders':        'olist_orders_dataset.csv',
        'order_items':   'olist_order_items_dataset.csv',
        'customers':     'olist_customers_dataset.csv',
        'products':      'olist_products_dataset.csv',
        'reviews':       'olist_order_reviews_dataset.csv',
        'sellers':       'olist_sellers_dataset.csv',
    }
    dfs = {}
    for key, fname in files.items():
        dfs[key] = pd.read_csv(f"{data_dir}{fname}")
        print(f"[LOAD] {key}: {len(dfs[key]):,} rows")
    return dfs


# ─── 2. DATA PREPARATION ────────────────────────────────────────────────────

def prepare_master(dfs: dict) -> pd.DataFrame:
    """
    Join orders, items, customers into a single analysis-ready table.
    Filter to delivered orders only — undelivered orders skew revenue metrics.
    """
    orders = dfs['orders'].copy()
    orders = orders[orders['order_status'] == 'delivered']

    # Parse dates
    for col in ['order_purchase_timestamp', 'order_delivered_customer_date',
                'order_estimated_delivery_date']:
        orders[col] = pd.to_datetime(orders[col])

    # Join customers
    master = orders.merge(dfs['customers'], on='customer_id', how='left')

    # Join order items — aggregate to order level
    items_agg = dfs['order_items'].groupby('order_id').agg(
        total_revenue=('price', 'sum'),
        total_freight=('freight_value', 'sum'),
        item_count=('order_item_id', 'count')
    ).reset_index()
    items_agg['order_value'] = items_agg['total_revenue'] + items_agg['total_freight']

    master = master.merge(items_agg, on='order_id', how='left')

    # Delivery performance flag
    master['delivered_on_time'] = (
        master['order_delivered_customer_date'] <=
        master['order_estimated_delivery_date']
    ).astype(int)

    print(f"\n[MASTER] {len(master):,} delivered orders | "
          f"{master['customer_unique_id'].nunique():,} unique customers")
    return master


# ─── 3. COHORT RETENTION ANALYSIS ───────────────────────────────────────────

def build_cohort_matrix(master: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly cohort retention matrix.
    Exported to CSV for Tableau heatmap visualisation.
    """
    master['order_month'] = master['order_purchase_timestamp'].dt.to_period('M')

    # First purchase month per customer
    first_purchase = master.groupby('customer_unique_id')['order_month'].min().reset_index()
    first_purchase.columns = ['customer_unique_id', 'cohort_month']

    master = master.merge(first_purchase, on='customer_unique_id')
    master['months_since_acquisition'] = (
        master['order_month'] - master['cohort_month']
    ).apply(lambda x: x.n)

    cohort_data = master.groupby(
        ['cohort_month', 'months_since_acquisition']
    )['customer_unique_id'].nunique().reset_index()
    cohort_data.columns = ['cohort_month', 'months_since_acquisition', 'customers']

    # Pivot to matrix
    cohort_matrix = cohort_data.pivot(
        index='cohort_month',
        columns='months_since_acquisition',
        values='customers'
    )

    # Convert to retention rates
    cohort_sizes = cohort_matrix[0]
    retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0).round(4)

    print(f"\n[COHORT] Matrix shape: {retention_matrix.shape}")
    return retention_matrix, cohort_matrix


def plot_cohort_heatmap(retention_matrix: pd.DataFrame) -> None:
    """Heatmap of monthly retention rates — designed for portfolio presentation."""
    # Limit to first 12 months for readability
    plot_data = retention_matrix.iloc[:, :12].head(18)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        plot_data,
        annot=True,
        fmt='.0%',
        cmap='Blues',
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Retention Rate'}
    )
    ax.set_title('Monthly Cohort Retention Matrix\n(Row = Acquisition Month, Col = Months Since First Purchase)',
                 fontsize=13, pad=15)
    ax.set_xlabel('Months Since First Purchase')
    ax.set_ylabel('Cohort (Acquisition Month)')
    plt.tight_layout()
    plt.savefig('outputs/cohort_retention_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[COHORT] Heatmap saved → outputs/cohort_retention_heatmap.png")


# ─── 4. TABLEAU EXPORT ──────────────────────────────────────────────────────

def export_for_tableau(master: pd.DataFrame,
                       retention_matrix: pd.DataFrame) -> None:
    """
    Export clean, structured CSVs for Tableau Public dashboard.
    Three separate views: orders, cohort matrix, delivery performance.
    """
    # Orders summary for revenue/funnel views
    orders_export = master[[
        'order_id', 'customer_unique_id', 'order_purchase_timestamp',
        'order_month', 'order_value', 'item_count',
        'delivered_on_time', 'customer_state'
    ]].copy()
    orders_export['order_month'] = orders_export['order_month'].astype(str)
    orders_export.to_csv('outputs/tableau_orders.csv', index=False)

    # Cohort matrix (long format — easier for Tableau)
    cohort_long = retention_matrix.reset_index().melt(
        id_vars='cohort_month',
        var_name='months_since_acquisition',
        value_name='retention_rate'
    )
    cohort_long['cohort_month'] = cohort_long['cohort_month'].astype(str)
    cohort_long.to_csv('outputs/tableau_cohort_retention.csv', index=False)

    # Monthly KPIs
    monthly = master.groupby('order_month').agg(
        orders=('order_id', 'count'),
        revenue=('order_value', 'sum'),
        avg_order_value=('order_value', 'mean'),
        unique_customers=('customer_unique_id', 'nunique'),
        on_time_rate=('delivered_on_time', 'mean')
    ).reset_index()
    monthly['order_month'] = monthly['order_month'].astype(str)
    monthly.to_csv('outputs/tableau_monthly_kpis.csv', index=False)

    print("[TABLEAU] Exports saved:")
    print("  → outputs/tableau_orders.csv")
    print("  → outputs/tableau_cohort_retention.csv")
    print("  → outputs/tableau_monthly_kpis.csv")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    import os
    os.makedirs('outputs', exist_ok=True)

    print("=" * 60)
    print("  E-COMMERCE SQL ANALYTICS — PYTHON PREP")
    print("=" * 60)

    dfs = load_olist()
    master = prepare_master(dfs)

    retention_matrix, cohort_matrix = build_cohort_matrix(master)
    plot_cohort_heatmap(retention_matrix)

    export_for_tableau(master, retention_matrix)

    print("\n[DONE] All outputs ready for Tableau Public")
    print("Connect Tableau to the /outputs CSV files to build your dashboard.")
    print("=" * 60)


if __name__ == "__main__":
    main()
