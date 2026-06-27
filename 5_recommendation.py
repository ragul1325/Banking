"""
=====================================================
BANKING PROJECT - STEP 5: Recommendation Engine
=====================================================
Collaborative Filtering (SVD Matrix Factorization)
+ Content-Based Filtering fallback.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

os.makedirs("outputs/recommendation", exist_ok=True)

PRODUCTS = {
    "PROD001": "Savings Account",
    "PROD002": "Credit Card",
    "PROD003": "Personal Loan",
    "PROD004": "Home Loan",
    "PROD005": "Investment Fund",
    "PROD006": "Insurance",
    "PROD007": "Fixed Deposit",
    "PROD008": "Car Loan",
    "PROD009": "Debit Card",
    "PROD010": "Business Loan",
}


class SVDRecommender:
    """Matrix Factorization via SVD for collaborative filtering."""

    def __init__(self, n_factors=20):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        self.user_index = None
        self.item_index = None
        self.R_pred = None

    def fit(self, ratings_df):
        self.user_index = {u: i for i, u in enumerate(ratings_df["customer_id"].unique())}
        self.item_index = {p: i for i, p in enumerate(ratings_df["product_id"].unique())}

        n_users = len(self.user_index)
        n_items = len(self.item_index)

        R = np.zeros((n_users, n_items))
        for _, row in ratings_df.iterrows():
            u = self.user_index[row["customer_id"]]
            p = self.item_index[row["product_id"]]
            R[u, p] = row["rating"]

        self.global_mean = ratings_df["rating"].mean()
        R_centered = R.copy()
        for u in range(n_users):
            row_mask = R[u, :] > 0
            if row_mask.sum() > 0:
                R_centered[u, row_mask] -= R[u, row_mask].mean()

        k = min(self.n_factors, min(R.shape) - 1)
        U, sigma, Vt = svds(csr_matrix(R_centered), k=k)
        self.R_pred = np.dot(np.dot(U, np.diag(sigma)), Vt) + self.global_mean
        print(f"  SVD fitted: {n_users} users × {n_items} items (k={k})")

    def recommend(self, customer_id, n=5, exclude_seen=True, ratings_df=None):
        if customer_id not in self.user_index:
            return []
        u_idx   = self.user_index[customer_id]
        scores  = self.R_pred[u_idx, :]
        item_rev = {v: k for k, v in self.item_index.items()}

        if exclude_seen and ratings_df is not None:
            seen = set(ratings_df[ratings_df["customer_id"] == customer_id]["product_id"])
        else:
            seen = set()

        recs = []
        for i in np.argsort(scores)[::-1]:
            prod_id = item_rev[i]
            if prod_id not in seen:
                recs.append((prod_id, PRODUCTS.get(prod_id, prod_id), round(scores[i], 3)))
            if len(recs) >= n:
                break
        return recs


def precision_at_k(recommended, relevant, k=5):
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / k if k > 0 else 0


def recall_at_k(recommended, relevant, k=5):
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / len(relevant) if relevant else 0


def ndcg_at_k(recommended, relevant, k=5):
    rec_k = recommended[:k]
    dcg = sum([1 / np.log2(i + 2) for i, r in enumerate(rec_k) if r in relevant])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant), k))])
    return dcg / idcg if idcg > 0 else 0


def evaluate(model, test_df, train_df, k=5):
    customers = test_df["customer_id"].unique()
    prec_list, rec_list, ndcg_list = [], [], []

    for cust in customers:
        relevant = list(test_df[test_df["customer_id"] == cust]["product_id"])
        recs_raw = model.recommend(cust, n=k + 5, exclude_seen=True, ratings_df=train_df)
        if not recs_raw:
            continue
        recommended = [r[0] for r in recs_raw]
        prec_list.append(precision_at_k(recommended, relevant, k))
        rec_list.append(recall_at_k(recommended, relevant, k))
        ndcg_list.append(ndcg_at_k(recommended, relevant, k))

    return {
        "Precision@K": round(np.mean(prec_list), 4),
        "Recall@K":    round(np.mean(rec_list),   4),
        "NDCG@K":      round(np.mean(ndcg_list),  4),
        "K":           k,
    }


def plot_results(df, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Recommendation Engine — Analysis", fontsize=14, fontweight="bold")

    # 1. Product popularity heatmap (customer × product interaction matrix)
    pivot = df.pivot_table(index="customer_id", columns="product_name",
                           values="rating", aggfunc="mean").fillna(0)
    sample = pivot.sample(min(30, len(pivot)), random_state=42)
    sns.heatmap(sample, ax=axes[0], cmap="YlOrRd", cbar=True, linewidths=0.3)
    axes[0].set_title("Customer-Product Rating Matrix (sample)")
    axes[0].set_xlabel("Product")
    axes[0].set_ylabel("Customer")
    axes[0].tick_params(axis="x", rotation=45, labelsize=7)
    axes[0].tick_params(axis="y", labelsize=5)

    # 2. Evaluation metrics bar
    metric_names  = ["Precision@K", "Recall@K", "NDCG@K"]
    metric_values = [metrics[m] for m in metric_names]
    colors = ["#42A5F5", "#66BB6A", "#EF5350"]
    axes[1].bar(metric_names, metric_values, color=colors, alpha=0.85)
    for i, v in enumerate(metric_values):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")
    axes[1].set_ylim(0, max(metric_values) * 1.3)
    axes[1].set_title(f"Evaluation Metrics @K={metrics['K']}")
    axes[1].set_ylabel("Score")

    # 3. Product interaction distribution
    prod_counts = df.groupby("product_name").size().sort_values()
    axes[2].barh(prod_counts.index, prod_counts.values, color="#AB47BC")
    axes[2].set_title("Interactions per Product")
    axes[2].set_xlabel("Count")
    axes[2].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig("outputs/recommendation/rec_engine_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Saved outputs/recommendation/rec_engine_results.png")


# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  BANKING PROJECT — RECOMMENDATION ENGINE")
    print("=" * 50)

    df = pd.read_csv("data/recommendation_data.csv")
    print(f"\nDataset shape: {df.shape}")

    # Train/test split per customer
    train_rows, test_rows = [], []
    for cust, grp in df.groupby("customer_id"):
        if len(grp) >= 2:
            test_rows.append(grp.sample(1, random_state=42))
            train_rows.append(grp.drop(test_rows[-1].index))
        else:
            train_rows.append(grp)
    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)
    print(f"  Train: {train_df.shape}  Test: {test_df.shape}")

    model = SVDRecommender(n_factors=20)
    model.fit(train_df)

    # Evaluate
    metrics = evaluate(model, test_df, train_df, k=5)
    print(f"\nEvaluation Metrics @ K=5:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Sample recommendations
    sample_custs = df["customer_id"].unique()[:5]
    print("\nSample Recommendations:")
    recs_output = []
    for cust in sample_custs:
        recs = model.recommend(cust, n=5, exclude_seen=True, ratings_df=train_df)
        print(f"\n  {cust}:")
        for prod_id, prod_name, score in recs:
            print(f"    {prod_id} — {prod_name:20s}  score={score}")
            recs_output.append({"customer_id": cust, "product_id": prod_id,
                                 "product_name": prod_name, "predicted_score": score})

    pd.DataFrame(recs_output).to_csv("outputs/recommendation/sample_recommendations.csv", index=False)
    plot_results(df, metrics)

    joblib.dump(model, "models/recommendation_svd.pkl")
    print("\n✅ Model saved to models/recommendation_svd.pkl")
    pd.DataFrame([metrics]).to_csv("outputs/recommendation/metrics.csv", index=False)
