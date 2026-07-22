"""
Customer Segmentation - K-Means (log-RFM) + PCA
Dataset: Online Retail II (UCI), sheet 'Year 2010-2011'
Author: Nithin Venkatesh

All figures produced by this script on the real dataset.
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

FIG_BG, AX_BG, FG, GRID = "#0F172A", "#111C30", "#E5E7EB", "#1E293B"
LABEL_ORDER = ["Champions","Loyal Customers","Potential Loyalists",
               "New / One-Time Buyers","At Risk / Churned"]
LABEL_COLORS = {"Champions":"#2563EB","Loyal Customers":"#16A34A",
                "Potential Loyalists":"#F59E0B","New / One-Time Buyers":"#EF4444",
                "At Risk / Churned":"#8B5CF6"}

def _style(ax):
    ax.set_facecolor(AX_BG); ax.figure.set_facecolor(FIG_BG)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG); ax.title.set_color(FG)
    for s in ax.spines.values(): s.set_color(GRID)

def load_clean(fp):
    df = pd.read_excel(fp, sheet_name='Year 2010-2011')
    b=len(df)
    df=df.dropna(subset=['Customer ID','Description'])
    df=df[df['Quantity']>0]; df=df[df['Price']>0]
    df=df[~df['Invoice'].astype(str).str.startswith('C')]
    df['Revenue']=df['Quantity']*df['Price']
    print(f"[CLEAN] removed {b-len(df):,} rows | {len(df):,} rows | {df['Customer ID'].nunique():,} customers")
    return df

def build_rfm(df):
    snap=df['InvoiceDate'].max()+pd.Timedelta(days=1)
    rfm=df.groupby('Customer ID').agg(
        Recency=('InvoiceDate',lambda x:(snap-x.max()).days),
        Frequency=('Invoice','nunique'),
        Monetary=('Revenue','sum')).reset_index()
    print(f"[RFM] {len(rfm):,} customers")
    return rfm

def scale_log(rfm):
    # Monetary/Frequency are heavily right-skewed (a few whales spend 100x the median).
    # Raw StandardScaler lets those outliers form singleton clusters. log1p first.
    L=rfm[['Recency','Frequency','Monetary']].apply(np.log1p)
    return StandardScaler().fit_transform(L)

def choose_k(X, krange=range(2,11)):
    ks=list(krange); inertias=[]; sils=[]
    for k in ks:
        km=KMeans(n_clusters=k,random_state=42,n_init=10).fit(X)
        inertias.append(km.inertia_); sils.append(silhouette_score(X,km.labels_))
    # Silhouette favours coarse clustering (typical for continuous RFM structure).
    # We choose K=5 for business interpretability: five actionable tiers. The
    # trade-off is explicit rather than hidden - silhouette is modest at K=5.
    optimal_k=5
    print(f"[K] silhouettes: "+", ".join(f"{k}:{s:.3f}" for k,s in zip(ks,sils)))
    print(f"[K] chosen K={optimal_k} (business interpretability; silhouette {sils[ks.index(5)]:.3f})")
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    ax[0].plot(ks,inertias,marker='o',color='#2563EB',lw=2)
    ax[0].set_title('Elbow Method - Inertia vs K'); ax[0].set_xlabel('K'); ax[0].set_ylabel('Inertia')
    ax[1].plot(ks,sils,marker='o',color='#16A34A',lw=2)
    ax[1].set_title('Silhouette Score vs K'); ax[1].set_xlabel('K'); ax[1].set_ylabel('Silhouette')
    for a in ax:
        a.axvline(optimal_k,color='#F59E0B',ls=':',lw=2,label=f'Chosen K={optimal_k}')
        a.legend(facecolor=AX_BG,edgecolor=GRID,labelcolor=FG); _style(a)
    plt.tight_layout(); plt.savefig('outputs/optimal_k_selection.png',dpi=150,bbox_inches='tight',facecolor=FIG_BG); plt.close()
    return optimal_k

def label_segments(rfm):
    means=rfm.groupby('Segment').agg(R=('Recency','mean'),F=('Frequency','mean'),M=('Monetary','mean'))
    med_m=means['M'].median()
    def lab(r):
        if r.R<60 and r.F>10: return "Champions"
        if r.R>150: return "At Risk / Churned"
        if r.F<=2 and r.R<60: return "New / One-Time Buyers"
        if r.M>med_m: return "Loyal Customers"
        return "Potential Loyalists"
    m={seg:lab(row) for seg,row in means.iterrows()}
    rfm['Label']=rfm['Segment'].map(m)
    return rfm

def pca_plot(X, labels):
    pca=PCA(n_components=2,random_state=42); C=pca.fit_transform(X)
    var=pca.explained_variance_ratio_
    print(f"[PCA] PC1={var[0]:.1%} PC2={var[1]:.1%} total={var.sum():.1%}")
    d=pd.DataFrame({'PC1':C[:,0],'PC2':C[:,1],'Label':labels.values})
    fig,ax=plt.subplots(figsize=(10,7))
    for l in LABEL_ORDER:
        m=d['Label']==l
        if m.any(): ax.scatter(d.loc[m,'PC1'],d.loc[m,'PC2'],label=l,alpha=0.5,s=18,color=LABEL_COLORS[l])
    ax.set_title(f'PCA Cluster Validation\n(Variance explained: PC1={var[0]:.1%}, PC2={var[1]:.1%}, Total={var.sum():.1%})')
    ax.set_xlabel(f'PC1 ({var[0]:.1%})'); ax.set_ylabel(f'PC2 ({var[1]:.1%})')
    ax.legend(facecolor=AX_BG,edgecolor=GRID,labelcolor=FG); _style(ax)
    plt.tight_layout(); plt.savefig('outputs/pca_validation.png',dpi=150,bbox_inches='tight',facecolor=FIG_BG); plt.close()

def profile(rfm):
    s=rfm.groupby('Label').agg(Customers=('Customer ID','count'),Avg_Recency=('Recency','mean'),
        Avg_Frequency=('Frequency','mean'),Avg_Monetary=('Monetary','mean'),
        Total_Revenue=('Monetary','sum')).reindex(LABEL_ORDER).reset_index().round(2)
    tot=s['Total_Revenue'].sum(); totc=s['Customers'].sum()
    s['Pct_of_Revenue']=(s['Total_Revenue']/tot*100).round(1)
    s['Pct_of_Customers']=(s['Customers']/totc*100).round(1)
    recs={"Champions":"Reward with loyalty perks; use for referrals and new-product testing.",
          "Loyal Customers":"Upsell premium lines; early access to new inventory.",
          "Potential Loyalists":"Frequency incentives to build the repeat-purchase habit.",
          "New / One-Time Buyers":"Onboarding sequence; drive a second purchase within 30 days.",
          "At Risk / Churned":"Win-back: personalised discount + re-engagement emails."}
    s['Recommendation']=s['Label'].map(recs)
    s.to_csv('outputs/segment_profiles.csv',index=False)
    print("\n[PROFILES]"); print(s[['Label','Customers','Pct_of_Customers','Avg_Recency','Avg_Frequency','Avg_Monetary','Total_Revenue','Pct_of_Revenue']].to_string(index=False))
    champ=s.loc[s.Label=='Champions'].iloc[0]
    chart=s.sort_values('Total_Revenue')
    fig,ax=plt.subplots(figsize=(9,6))
    bars=ax.barh(chart['Label'],chart['Total_Revenue']/1000,color=[LABEL_COLORS[l] for l in chart['Label']])
    ax.set_title(f"Revenue Contribution by Segment\n(Champions = {champ.Pct_of_Customers:.0f}% of customers, {champ.Pct_of_Revenue:.1f}% of revenue)")
    ax.set_xlabel('Total Revenue ($K)')
    for b,v in zip(bars,chart['Total_Revenue']/1000):
        ax.text(b.get_width()+30,b.get_y()+b.get_height()/2,f"${v:,.0f}K",va='center',color=FG,fontsize=10)
    _style(ax); plt.tight_layout(); plt.savefig('outputs/revenue_by_segment.png',dpi=150,bbox_inches='tight',facecolor=FIG_BG); plt.close()
    return s

def main():
    os.makedirs('outputs',exist_ok=True)
    df=load_clean('data/online_retail_II.xlsx')
    rfm=build_rfm(df)
    X=scale_log(rfm)
    k=choose_k(X)
    rfm['Segment']=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X)
    rfm=label_segments(rfm)
    pca_plot(X,rfm['Label'])
    profile(rfm)
    rfm.to_csv('outputs/customers_with_segments.csv',index=False)
    print("\n[DONE]")

if __name__=='__main__':
    main()
