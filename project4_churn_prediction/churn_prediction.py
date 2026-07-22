"""
Customer Churn Prediction - XGBoost (class-weighted) + threshold tuning
Dataset: IBM Telco Customer Churn (7,043 customers)
Author: Nithin Venkatesh

All metrics produced by this script on the real dataset. random_state=42.
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             f1_score, confusion_matrix, roc_curve)
from xgboost import XGBClassifier

FIG_BG, AX_BG, FG, GRID = "#0F172A", "#111C30", "#E5E7EB", "#1E293B"
BLUE, GREEN, RED, AMBER, PURPLE = "#2563EB", "#16A34A", "#EF4444", "#F59E0B", "#8B5CF6"

def _style(ax):
    ax.set_facecolor(AX_BG); ax.figure.set_facecolor(FIG_BG)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG); ax.title.set_color(FG)
    for s in ax.spines.values(): s.set_color(GRID)

def load_clean(fp):
    df = pd.read_csv(fp)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    n0 = len(df); df = df.dropna(subset=['TotalCharges'])
    print(f"[CLEAN] dropped {n0-len(df)} blank-TotalCharges rows | {len(df):,} customers")
    print(f"[CLEAN] churn rate: {(df.Churn=='Yes').mean():.1%}")
    return df

def eda_chart(df):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    vc = df['Churn'].value_counts()
    ax[0].bar(['Retained','Churned'], [vc.get('No',0), vc.get('Yes',0)], color=[BLUE, RED])
    ax[0].set_title('Churn Distribution'); ax[0].set_ylabel('Customers')
    for i,v in enumerate([vc.get('No',0), vc.get('Yes',0)]):
        ax[0].text(i, v, f"{v:,}", ha='center', va='bottom', color=FG, fontweight='bold')
    cc = df.groupby('Contract')['Churn'].apply(lambda x:(x=='Yes').mean()*100).reindex(
        ['Month-to-month','One year','Two year'])
    ax[1].bar(cc.index, cc.values, color=[RED, AMBER, GREEN])
    ax[1].set_title('Churn Rate by Contract'); ax[1].set_ylabel('Churn %')
    for i,v in enumerate(cc.values):
        ax[1].text(i, v, f"{v:.0f}%", ha='center', va='bottom', color=FG, fontweight='bold')
    for a in ax: _style(a)
    plt.tight_layout(); plt.savefig('outputs/eda_churn_overview.png', dpi=150, bbox_inches='tight', facecolor=FIG_BG); plt.close()
    print("[EDA] outputs/eda_churn_overview.png")

def main():
    os.makedirs('outputs', exist_ok=True)
    df = load_clean('data/telco_churn.csv')
    eda_chart(df)

    y = (df['Churn']=='Yes').astype(int)
    X = pd.get_dummies(df.drop(columns=['customerID','Churn']), drop_first=True)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    spw = (ytr==0).sum()/(ytr==1).sum()
    print(f"[SPLIT] train {len(Xtr):,} | test {len(Xte):,} | scale_pos_weight {spw:.2f}")

    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          scale_pos_weight=spw, eval_metric='logloss', random_state=42)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, p)

    # threshold tuning
    ths = np.round(np.arange(0.10,0.90,0.01),2)
    rows=[{'t':t,'recall':recall_score(yte,(p>=t)),'precision':precision_score(yte,(p>=t),zero_division=0),
           'f1':f1_score(yte,(p>=t),zero_division=0)} for t in ths]
    r = pd.DataFrame(rows); best = r.loc[r.f1.idxmax()]
    rec05 = recall_score(yte,(p>=0.5)); caught=int(((p>=0.5)&(yte==1)).sum()); tot=int(yte.sum())

    print(f"\n[RESULTS] ROC-AUC {auc:.3f}")
    print(f"[RESULTS] @0.50 recall {rec05:.1%} (caught {caught}/{tot}) | "
          f"F1-opt t={best.t:.2f} recall {best.recall:.1%} precision {best.precision:.1%}")

    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(r.t,r.f1,color=BLUE,lw=2,label='F1')
    ax.plot(r.t,r.precision,color=GREEN,lw=2,ls='--',label='Precision')
    ax.plot(r.t,r.recall,color=RED,lw=2,ls='--',label='Recall')
    ax.axvline(best.t,color=AMBER,ls=':',lw=2,label=f'F1-optimal: {best.t:.2f}')
    ax.set_title(f'Threshold Tuning (ROC-AUC {auc:.2f})'); ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
    ax.legend(facecolor=AX_BG,edgecolor=GRID,labelcolor=FG,loc='lower center'); _style(ax)
    plt.tight_layout(); plt.savefig('outputs/threshold_tuning.png',dpi=150,bbox_inches='tight',facecolor=FIG_BG); plt.close()

    # feature importance
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
    contract_share = pd.Series(model.feature_importances_, index=X.columns)
    cshare = contract_share[[c for c in X.columns if c.startswith('Contract')]].sum()/contract_share.sum()
    fig,ax=plt.subplots(figsize=(10,6))
    ax.barh(imp.index, imp.values, color=BLUE); ax.invert_yaxis()
    ax.set_title(f'Top Churn Drivers (Contract = {cshare:.0%} of importance)'); ax.set_xlabel('Feature Importance')
    _style(ax); plt.tight_layout(); plt.savefig('outputs/feature_importance.png',dpi=150,bbox_inches='tight',facecolor=FIG_BG); plt.close()

    # risk tiers
    dt = pd.DataFrame({'p':p,'y':yte.values})
    dt['tier']=pd.qcut(dt.p,5,labels=['Low','Low-Mid','Mid','Mid-High','High'])
    tr = dt.groupby('tier').y.mean()*100
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(tr.index, tr.values, color=[GREEN,'#65A30D',AMBER,'#F97316',RED])
    ax.set_title(f'Actual Churn Rate by Predicted Risk Tier (High {tr["High"]:.0f}% vs Low {tr["Low"]:.0f}%)')
    ax.set_ylabel('Actual Churn %')
    for i,v in enumerate(tr.values): ax.text(i,v,f"{v:.0f}%",ha='center',va='bottom',color=FG,fontweight='bold')
    _style(ax); plt.tight_layout(); plt.savefig('outputs/risk_tiers.png',dpi=150,bbox_inches='tight',facecolor=FIG_BG); plt.close()

    pd.DataFrame({'Metric':['ROC-AUC','Recall@0.50','Precision@0.50','F1-opt threshold',
                            'Recall@F1-opt','Precision@F1-opt','Contract importance share',
                            'High-tier churn','Low-tier churn'],
                  'Value':[f"{auc:.3f}",f"{rec05:.1%}",f"{precision_score(yte,(p>=0.5)):.1%}",
                           f"{best.t:.2f}",f"{best.recall:.1%}",f"{best.precision:.1%}",
                           f"{cshare:.1%}",f"{tr['High']:.1f}%",f"{tr['Low']:.1f}%"]}).to_csv(
                  'outputs/metrics_summary.csv',index=False)
    print(f"[RESULTS] Contract importance {cshare:.0%} | risk tiers High {tr['High']:.0f}% vs Low {tr['Low']:.0f}%")
    print("[DONE] outputs/ written")

if __name__=='__main__':
    main()
