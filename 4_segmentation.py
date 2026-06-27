"""
=====================================================
BANKING PROJECT - STEP 4: Customer Segmentation
=====================================================
Unsupervised Learning — K-Means Clustering with
PCA visualization and segment profiling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, joblib

from sklearn.preprocessing  import StandardScaler
from sklearn.cluster        import KMeans, AgglomerativeClustering
from sklearn.decomposition  import PCA
from sklearn.metrics        import silhouette_score, davies_bouldin_score

os.makedirs("outputs/segmentation", exist_ok=True)

SEGMENT_NAMES = {
    0: "Low-Value Savers",
    1: "High-Value Active",
    2: "Regular Spenders",
    3: "Occasional Users",
}
SEGMENT_COLORS = ["#42A5F5", "#EF5350", "#66BB6A", "#FFA726"]


def build_rfm(df):
    """Build RFM + extra features for clustering."""
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    snapshot_date = df["transaction_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency       = ("transaction_date",  lambda x: (snapshot_date - x.max()).days),
        frequency     = ("transaction_id",    "count"),
        monetary      = ("transaction_amount","sum"),
        avg_amount    = ("transaction_amount","mean"),
        max_amount    = ("transaction_amount","max"),
        n_types       = ("transaction_type",  "nunique"),
        n_weekends    = ("is_weekend",        "sum"),
        n_large_txns  = ("is_large_txn",      "sum"),
        active_months = ("month",             "nunique"),
    ).reset_index()
    rfm["monetary"]  = rfm["monetary"].round(2)
    rfm["avg_amount"]= rfm["avg_amount"].round(2)
    return rfm


def find_optimal_k(X_scaled, k_range=range(2, 9)):
    inertias, silhouettes, db_scores = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Optimal K Selection", fontsize=13, fontweight="bold")

    axes[0].plot(k_range, inertias, marker="o", color="#42A5F5", lw=2)
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")

    axes[1].plot(k_range, silhouettes, marker="o", color="#66BB6A", lw=2)
    axes[1].set_title("Silhouette Score ↑")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette")

    axes[2].plot(k_range, db_scores, marker="o", color="#EF5350", lw=2)
    axes[2].set_title("Davies-Bouldin ↓")
    axes[2].set_xlabel("K"); axes[2].set_ylabel("DB Score")

    plt.tight_layout()
    plt.savefig("outputs/segmentation/optimal_k.png", dpi=150, bbox_inches="tight")
    plt.close()

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"  Best K = {best_k} (Silhouette = {max(silhouettes):.4f})")
    return best_k


def plot_clusters(rfm, X_scaled, labels, best_k):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    rfm["pca1"]    = X_pca[:, 0]
    rfm["pca2"]    = X_pca[:, 1]
    rfm["cluster"] = labels

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Customer Segmentation", fontsize=14, fontweight="bold")

    # 1. PCA scatter
    for k in range(best_k):
        mask = rfm["cluster"] == k
        axes[0].scatter(rfm.loc[mask, "pca1"], rfm.loc[mask, "pca2"],
                        c=SEGMENT_COLORS[k % len(SEGMENT_COLORS)],
                        label=SEGMENT_NAMES.get(k, f"Seg {k}"), alpha=0.6, s=25)
    axes[0].set_title("PCA Cluster Visualization")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axes[0].legend(fontsize=8)

    # 2. Frequency vs Monetary
    for k in range(best_k):
        mask = rfm["cluster"] == k
        axes[1].scatter(rfm.loc[mask, "frequency"], rfm.loc[mask, "monetary"],
                        c=SEGMENT_COLORS[k % len(SEGMENT_COLORS)],
                        label=SEGMENT_NAMES.get(k, f"Seg {k}"), alpha=0.6, s=25)
    axes[1].set_title("Frequency vs Monetary")
    axes[1].set_xlabel("Frequency (# Transactions)")
    axes[1].set_ylabel("Monetary (Total Spend)")
    axes[1].legend(fontsize=8)

    # 3. Cluster size
    sizes = rfm["cluster"].value_counts().sort_index()
    axes[2].bar([SEGMENT_NAMES.get(k, f"Seg {k}") for k in sizes.index],
                sizes.values,
                color=[SEGMENT_COLORS[k % len(SEGMENT_COLORS)] for k in sizes.index])
    axes[2].set_title("Customers per Segment")
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig("outputs/segmentation/clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Saved outputs/segmentation/clusters.png")
    return rfm


def plot_radar(rfm, best_k, feature_cols):
    """Radar chart per segment."""
    cluster_means = rfm.groupby("cluster")[feature_cols].mean()
    cluster_norm  = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)

    N = len(feature_cols)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, axes = plt.subplots(1, best_k, figsize=(4 * best_k, 4), subplot_kw=dict(polar=True))
    fig.suptitle("Segment Profiles (Radar Chart)", fontsize=13, fontweight="bold")

    for k in range(best_k):
        ax = axes[k] if best_k > 1 else axes
        values = cluster_norm.loc[k].tolist() + [cluster_norm.loc[k].tolist()[0]]
        ax.plot(angles, values, color=SEGMENT_COLORS[k], lw=2)
        ax.fill(angles, values, alpha=0.25, color=SEGMENT_COLORS[k])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_cols, size=8)
        ax.set_title(SEGMENT_NAMES.get(k, f"Seg {k}"), size=9, pad=15)

    plt.tight_layout()
    plt.savefig("outputs/segmentation/radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Saved outputs/segmentation/radar.png")


# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  BANKING PROJECT — CUSTOMER SEGMENTATION")
    print("=" * 50)

    df  = pd.read_csv("data/transaction_data.csv")
    rfm = build_rfm(df)
    print(f"\nRFM table shape: {rfm.shape}")
    print(rfm.describe().T.to_string())

    feature_cols = ["recency", "frequency", "monetary", "avg_amount",
                    "n_types", "n_weekends", "n_large_txns", "active_months"]
    X = rfm[feature_cols].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nFinding optimal K...")
    best_k = find_optimal_k(X_scaled)
    print(f"  ✅ Saved outputs/segmentation/optimal_k.png")

    print(f"\nFitting K-Means with K={best_k}...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil  = silhouette_score(X_scaled, labels)
    db   = davies_bouldin_score(X_scaled, labels)
    print(f"  Silhouette Score  : {sil:.4f}")
    print(f"  Davies-Bouldin    : {db:.4f}")

    rfm = plot_clusters(rfm, X_scaled, labels, best_k)
    plot_radar(rfm, best_k, feature_cols)

    # Segment summary
    seg_summary = rfm.groupby("cluster")[feature_cols].mean().round(2)
    seg_summary.index = [SEGMENT_NAMES.get(k, f"Seg {k}") for k in seg_summary.index]
    print(f"\nSegment Profiles:\n{seg_summary.to_string()}")
    seg_summary.to_csv("outputs/segmentation/segment_profiles.csv")

    rfm.to_csv("outputs/segmentation/customer_segments.csv", index=False)
    joblib.dump((kmeans, scaler), "models/segmentation_kmeans.pkl")
    print("\n✅ Segment labels saved to outputs/segmentation/customer_segments.csv")
    print("✅ Model saved to models/segmentation_kmeans.pkl")
