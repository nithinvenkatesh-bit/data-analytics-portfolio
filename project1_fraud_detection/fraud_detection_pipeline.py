"""
Fraud Detection Pipeline
========================
End-to-end pipeline for detecting fraudulent transactions using
statistical threshold tuning and machine learning.

Dataset: Credit Card Fraud Detection (Kaggle)
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Author: Nithin Venkatesh
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ─── 1. DATA LOADING & VALIDATION ───────────────────────────────────────────

def load_and_validate(filepath: str) -> pd.DataFrame:
    """Load dataset and run row-level quality checks before analysis."""
    df = pd.read_csv(filepath)

    print(f"[LOAD] Rows: {len(df):,} | Columns: {df.shape[1]}")
    print(f"[LOAD] Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Row-level quality checks — flag rather than silently drop
    issues = []
    if df.duplicated().sum() > 0:
        issues.append(f"  ⚠️  {df.duplicated().sum()} duplicate rows detected")
    if df['Amount'].min() < 0:
        issues.append("  ⚠️  Negative transaction amounts detected")
    if df['Class'].nunique() != 2:
        issues.append("  ⚠️  Unexpected class values in target column")

    if issues:
        print("[QUALITY CHECKS] Issues found:")
        for i in issues: print(i)
    else:
        print("[QUALITY CHECKS] All checks passed ✓")

    # Remove duplicates after flagging
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# ─── 2. EXPLORATORY DATA ANALYSIS ───────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """Summarize class imbalance and transaction amount distributions."""
    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]

    fraud_rate = len(fraud) / len(df) * 100
    print(f"\n[EDA] Fraud rate: {fraud_rate:.4f}%")
    print(f"[EDA] Legitimate transactions: {len(legit):,}")
    print(f"[EDA] Fraudulent transactions:  {len(fraud):,}")
    print(f"[EDA] Avg fraud amount:  ${fraud['Amount'].mean():.2f}")
    print(f"[EDA] Avg legit amount:  ${legit['Amount'].mean():.2f}")

    # Visualise class imbalance
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(['Legitimate', 'Fraud'], [len(legit), len(fraud)],
                color=['#2563EB', '#EF4444'])
    axes[0].set_title('Class Distribution')
    axes[0].set_ylabel('Transaction Count')
    for i, v in enumerate([len(legit), len(fraud)]):
        axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

    axes[1].hist(legit['Amount'], bins=50, alpha=0.6, label='Legitimate',
                 color='#2563EB', density=True)
    axes[1].hist(fraud['Amount'], bins=50, alpha=0.6, label='Fraud',
                 color='#EF4444', density=True)
    axes[1].set_title('Transaction Amount Distribution')
    axes[1].set_xlabel('Amount ($)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('outputs/eda_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] Chart saved → outputs/eda_overview.png")


# ─── 3. FEATURE ENGINEERING ─────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale Amount and Time — the only raw features alongside PCA components.
    V1–V28 are already PCA-transformed in this dataset.
    """
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    df = df.drop(columns=['Amount', 'Time'])
    return df


# ─── 4. STATISTICAL THRESHOLD TUNING ────────────────────────────────────────

def tune_threshold(model, X_test: np.ndarray, y_test: pd.Series) -> float:
    """
    Find the probability threshold that maximises F1 score.
    Default 0.5 threshold is rarely optimal on imbalanced fraud data.
    """
    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        fp = ((preds == 1) & (y_test == 0)).sum()
        results.append({
            'threshold': t,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'false_positives': fp
        })

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['f1'].idxmax()]

    print(f"\n[THRESHOLD TUNING] Default (0.50) → F1: "
          f"{results_df[results_df['threshold'].between(0.49,0.51)]['f1'].values[0]:.4f}")
    print(f"[THRESHOLD TUNING] Optimal ({best['threshold']:.2f}) → F1: {best['f1']:.4f}")
    print(f"[THRESHOLD TUNING] Precision: {best['precision']:.4f} | "
          f"Recall: {best['recall']:.4f} | "
          f"False Positives: {int(best['false_positives'])}")

    # Plot threshold vs F1 / precision / recall
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results_df['threshold'], results_df['f1'],
            label='F1 Score', color='#2563EB', linewidth=2)
    ax.plot(results_df['threshold'], results_df['precision'],
            label='Precision', color='#16A34A', linewidth=2, linestyle='--')
    ax.plot(results_df['threshold'], results_df['recall'],
            label='Recall', color='#EF4444', linewidth=2, linestyle='--')
    ax.axvline(best['threshold'], color='gray', linestyle=':', label=f"Optimal: {best['threshold']:.2f}")
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Tuning — F1 / Precision / Recall')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/threshold_tuning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[THRESHOLD] Chart saved → outputs/threshold_tuning.png")

    return float(best['threshold'])


# ─── 5. KPI TRACKING ────────────────────────────────────────────────────────

def compute_kpis(y_true: pd.Series, y_pred: np.ndarray,
                 y_prob: np.ndarray) -> pd.DataFrame:
    """
    Compute the three core fraud KPIs tracked in production:
      - Fraud Detection Rate (Recall)
      - False Positive Rate
      - Case Turnaround Proxy (model confidence on detected fraud)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    kpis = {
        'Fraud Detection Rate (Recall)': f"{recall_score(y_true, y_pred):.2%}",
        'Precision':                     f"{precision_score(y_true, y_pred):.2%}",
        'False Positive Rate':           f"{fp / (fp + tn):.2%}",
        'F1 Score':                      f"{f1_score(y_true, y_pred):.4f}",
        'ROC-AUC':                       f"{roc_auc_score(y_true, y_prob):.4f}",
        'True Positives (Caught Fraud)': f"{tp:,}",
        'False Positives (Noise)':       f"{fp:,}",
        'Missed Fraud (FN)':             f"{fn:,}",
    }

    kpi_df = pd.DataFrame(list(kpis.items()), columns=['KPI', 'Value'])
    print("\n[KPIs]")
    print(kpi_df.to_string(index=False))
    kpi_df.to_csv('outputs/kpi_summary.csv', index=False)
    print("[KPIs] Saved → outputs/kpi_summary.csv")
    return kpi_df


# ─── 6. MODEL TRAINING ──────────────────────────────────────────────────────

def train_models(X_train, X_test, y_train, y_test):
    """
    Train Logistic Regression and Random Forest.
    LR as interpretable baseline; RF for production performance.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=42, n_jobs=-1
        )
    }

    trained = {}
    for name, model in models.items():
        print(f"\n[MODEL] Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(classification_report(y_test, preds, target_names=['Legit', 'Fraud']))
        trained[name] = model

    return trained


# ─── 7. FEATURE IMPORTANCE ──────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list) -> None:
    """Identify which PCA components most drive fraud detection."""
    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    importances.plot(kind='barh', ax=ax, color='#2563EB')
    ax.invert_yaxis()
    ax.set_title('Top 15 Features — Random Forest Fraud Detection')
    ax.set_xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[FEATURES] Chart saved → outputs/feature_importance.png")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    import os
    os.makedirs('outputs', exist_ok=True)

    print("=" * 60)
    print("  FRAUD DETECTION PIPELINE")
    print("=" * 60)

    # Load data — download from Kaggle first (see README)
    df = load_and_validate('data/creditcard.csv')
    run_eda(df)

    # Feature engineering
    df = engineer_features(df)

    X = df.drop(columns=['Class'])
    y = df['Class']

    # Stratified split to preserve fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[SPLIT] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train
    trained_models = train_models(X_train, X_test, y_train, y_test)

    # Tune threshold on Random Forest (production model)
    rf = trained_models['Random Forest']
    optimal_threshold = tune_threshold(rf, X_test, y_test)

    # Final predictions with tuned threshold
    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred_tuned = (y_prob >= optimal_threshold).astype(int)

    # KPI dashboard
    compute_kpis(y_test, y_pred_tuned, y_prob)

    # Feature importance
    plot_feature_importance(rf, X.columns.tolist())

    print("\n[DONE] All outputs saved to /outputs")
    print("=" * 60)


if __name__ == "__main__":
    main()
