"""
Customer Segmentation — K-Means Clustering + PCA
=================================================
Segments retail customers into distinct behavioural profiles
using K-Means clustering validated by PCA and Silhouette analysis.

Dataset: Online Retail II (UCI Machine Learning Repository)
https://archive.ics.uci.edu/dataset/502/online+retail+ii

Author: Nithin Venkatesh
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ─── 1. DATA LOADING & QUALITY CHECKS ───────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load transaction data and apply row-level quality checks."""
    df = pd.read_excel(filepath, sheet_name='Year 2010-2011')

    print(f"[LOAD] Raw rows: {len(df):,}")

    # Row-level quality checks
    before = len(df)
    df = df.dropna(subset=['Customer ID', 'Description'])
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df = df[~df['Invoice'].astype(str).str.startswith('C')]  # Remove cancellations
    df['Revenue'] = df['Quantity'] * df['Price']

    print(f"[QUALITY] Removed {before - len(df):,} invalid rows")
    print(f"[QUALITY] Clean rows: {len(df):,} | Unique customers: {df['Customer ID'].nunique():,}")
    return df


# ─── 2. RFM FEATURE ENGINEERING ─────────────────────────────────────────────

def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build RFM (Recency, Frequency, Monetary) features per customer.
    RFM is the industry standard for customer segmentation —
    captures behavioural dimensions that map to distinct segments.
    """
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('Customer ID').agg(
        Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency = ('Invoice',     'nunique'),
        Monetary  = ('Revenue',     'sum')
    ).reset_index()

    print(f"\n[RFM] Built features for {len(rfm):,} customers")
    print(rfm.describe().round(2))

    return rfm


# ─── 3. FEATURE SCALING ─────────────────────────────────────────────────────

def scale_features(rfm: pd.DataFrame) -> tuple:
    """
    Standardise RFM features before clustering.
    K-Means uses Euclidean distance — unscaled features bias toward
    high-magnitude variables like Monetary.
    """
    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])
    return rfm_scaled, features


# ─── 4. OPTIMAL K SELECTION ─────────────────────────────────────────────────

def find_optimal_k(rfm_scaled: np.ndarray, k_range: range = range(2, 11)) -> int:
    """
    Use Elbow Method + Silhouette Score together to select optimal K.
    Silhouette score validates cluster separation — not just inertia minimisation.
    """
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(rfm_scaled, labels))

    # Plot both metrics side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(k_range), inertias, marker='o', color='#2563EB', linewidth=2)
    axes[0].set_title('Elbow Method — Inertia vs K')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')

    axes[1].plot(list(k_range), silhouettes, marker='o', color='#16A34A', linewidth=2)
    axes[1].set_title('Silhouette Score vs K')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')

    plt.tight_layout()
    plt.savefig('outputs/optimal_k_selection.png', dpi=150, bbox_inches='tight')
    plt.close()

    optimal_k = list(k_range)[silhouettes.index(max(silhouettes))]
    print(f"\n[K SELECTION] Optimal K = {optimal_k} "
          f"(Silhouette: {max(silhouettes):.4f})")
    return optimal_k


# ─── 5. K-MEANS CLUSTERING ──────────────────────────────────────────────────

def run_kmeans(rfm: pd.DataFrame, rfm_scaled: np.ndarray,
               k: int) -> pd.DataFrame:
    """Fit K-Means and assign segment labels to each customer."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm['Segment'] = km.fit_predict(rfm_scaled)

    segment_summary = rfm.groupby('Segment').agg(
        Count     = ('Customer ID', 'count'),
        Recency   = ('Recency',   'mean'),
        Frequency = ('Frequency', 'mean'),
        Monetary  = ('Monetary',  'mean')
    ).round(2)

    print(f"\n[CLUSTERING] Segment Summary:")
    print(segment_summary)
    segment_summary.to_csv('outputs/segment_summary.csv')

    return rfm, km


# ─── 6. PCA VALIDATION ──────────────────────────────────────────────────────

def validate_with_pca(rfm_scaled: np.ndarray,
                      segments: pd.Series) -> None:
    """
    Reduce to 2 components for visual validation.
    Well-separated clusters in PCA space confirm K-Means found real structure.
    """
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(rfm_scaled)

    explained = pca.explained_variance_ratio_
    print(f"\n[PCA] Variance explained: PC1={explained[0]:.1%}, "
          f"PC2={explained[1]:.1%}, Total={sum(explained):.1%}")

    pca_df = pd.DataFrame({
        'PC1': components[:, 0],
        'PC2': components[:, 1],
        'Segment': segments
    })

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#2563EB', '#16A34A', '#EF4444', '#F59E0B', '#8B5CF6']
    for seg in sorted(pca_df['Segment'].unique()):
        mask = pca_df['Segment'] == seg
        ax.scatter(
            pca_df.loc[mask, 'PC1'],
            pca_df.loc[mask, 'PC2'],
            label=f'Segment {seg}',
            alpha=0.5, s=20,
            color=colors[seg % len(colors)]
        )
    ax.set_title(f'PCA Cluster Validation\n'
                 f'(Variance explained: {sum(explained):.1%})')
    ax.set_xlabel(f'PC1 ({explained[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({explained[1]:.1%} variance)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/pca_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[PCA] Chart saved → outputs/pca_validation.png")


# ─── 7. SEGMENT PROFILING & BUSINESS RECOMMENDATIONS ────────────────────────

def profile_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable labels and actionable recommendations per segment.
    Translates cluster numbers into language business stakeholders can act on.
    """
    summary = rfm.groupby('Segment').agg(
        Customer_Count = ('Customer ID', 'count'),
        Avg_Recency    = ('Recency',     'mean'),
        Avg_Frequency  = ('Frequency',   'mean'),
        Avg_Monetary   = ('Monetary',    'mean'),
        Total_Revenue  = ('Monetary',    'sum')
    ).round(2).reset_index()

    # Assign labels based on RFM profile
    def label_segment(row):
        if row['Avg_Recency'] < 30 and row['Avg_Frequency'] > 10:
            return 'Champions'
        elif row['Avg_Recency'] < 60 and row['Avg_Monetary'] > summary['Avg_Monetary'].median():
            return 'Loyal Customers'
        elif row['Avg_Recency'] > 180:
            return 'At Risk / Churned'
        elif row['Avg_Frequency'] <= 2:
            return 'New / One-Time Buyers'
        else:
            return 'Potential Loyalists'

    summary['Label'] = summary.apply(label_segment, axis=1)

    recommendations = {
        'Champions':            'Reward with loyalty programs; use for referrals and new product testing.',
        'Loyal Customers':      'Upsell premium products; offer early access to new inventory.',
        'At Risk / Churned':    'Win-back campaign: personalised discount + re-engagement email sequence.',
        'New / One-Time Buyers':'Nurture with onboarding sequence; incentivise second purchase within 30 days.',
        'Potential Loyalists':  'Frequency programs: reward 3rd and 5th purchases to build habit.'
    }

    summary['Recommendation'] = summary['Label'].map(recommendations)

    print("\n[SEGMENTS] Final Profiles:")
    print(summary[['Segment', 'Label', 'Customer_Count',
                   'Avg_Recency', 'Avg_Frequency',
                   'Avg_Monetary', 'Total_Revenue']].to_string(index=False))

    summary.to_csv('outputs/segment_profiles.csv', index=False)
    print("[SEGMENTS] Saved → outputs/segment_profiles.csv")

    # Revenue contribution chart
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#2563EB', '#16A34A', '#EF4444', '#F59E0B', '#8B5CF6']
    bars = ax.barh(summary['Label'], summary['Total_Revenue'],
                   color=colors[:len(summary)])
    ax.set_title('Revenue Contribution by Segment')
    ax.set_xlabel('Total Revenue ($)')
    for bar, val in zip(bars, summary['Total_Revenue']):
        ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height() / 2,
                f'${val:,.0f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('outputs/revenue_by_segment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[SEGMENTS] Revenue chart saved → outputs/revenue_by_segment.png")

    return summary


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    import os
    os.makedirs('outputs', exist_ok=True)

    print("=" * 60)
    print("  CUSTOMER SEGMENTATION — K-MEANS + PCA")
    print("=" * 60)

    df = load_and_clean('data/online_retail_II.xlsx')
    rfm = build_rfm(df)
    rfm_scaled, features = scale_features(rfm)

    optimal_k = find_optimal_k(rfm_scaled)
    rfm, model = run_kmeans(rfm, rfm_scaled, optimal_k)

    validate_with_pca(rfm_scaled, rfm['Segment'])
    profile_segments(rfm)

    rfm.to_csv('outputs/customers_with_segments.csv', index=False)
    print("\n[DONE] All outputs saved to /outputs")
    print("=" * 60)


if __name__ == "__main__":
    main()
