"""
=====================================================
BANKING PROJECT - STEP 2: Exploratory Data Analysis
=====================================================
Generates EDA plots for all three datasets.
Output saved to outputs/eda/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

os.makedirs("outputs/eda", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ─────────────────────────────────────────────────
# A. LOAN DEFAULT EDA
# ─────────────────────────────────────────────────
def eda_loan(df):
    print("\n📊 LOAN DEFAULT EDA")
    print(df.describe().T.to_string())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"Default rate: {df['repayment_status'].mean():.1%}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Loan Default — Exploratory Data Analysis", fontsize=16, fontweight="bold")

    # 1. Default rate
    counts = df["repayment_status"].value_counts()
    axes[0, 0].pie(counts, labels=["No Default", "Default"],
                   autopct="%1.1f%%", colors=["#4CAF50", "#F44336"], startangle=90)
    axes[0, 0].set_title("Default Rate")

    # 2. Credit score distribution
    df.groupby("repayment_status")["credit_score"].plot(kind="kde", ax=axes[0, 1],
                                                        legend=True)
    axes[0, 1].set_title("Credit Score by Default Status")
    axes[0, 1].set_xlabel("Credit Score")
    axes[0, 1].legend(["No Default", "Default"])

    # 3. Income distribution
    df.groupby("repayment_status")["income"].plot(kind="kde", ax=axes[0, 2], legend=True)
    axes[0, 2].set_title("Income by Default Status")
    axes[0, 2].set_xlabel("Income")
    axes[0, 2].legend(["No Default", "Default"])

    # 4. Loan amount vs interest rate
    colors = df["repayment_status"].map({0: "#4CAF50", 1: "#F44336"})
    axes[1, 0].scatter(df["loan_amount"], df["interest_rate"], c=colors, alpha=0.3, s=15)
    axes[1, 0].set_title("Loan Amount vs Interest Rate")
    axes[1, 0].set_xlabel("Loan Amount")
    axes[1, 0].set_ylabel("Interest Rate")

    # 5. Debt-to-Income
    df.groupby("repayment_status")["debt_to_income"].plot(kind="kde", ax=axes[1, 1])
    axes[1, 1].set_title("Debt-to-Income Ratio")
    axes[1, 1].legend(["No Default", "Default"])

    # 6. Correlation heatmap
    num_cols = ["age", "income", "credit_score", "loan_amount",
                "interest_rate", "debt_to_income", "num_late_payments", "repayment_status"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, ax=axes[1, 2], annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, annot_kws={"size": 7})
    axes[1, 2].set_title("Correlation Heatmap")
    axes[1, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("outputs/eda/loan_eda.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Saved outputs/eda/loan_eda.png")


# ─────────────────────────────────────────────────
# B. TRANSACTION / SEGMENTATION EDA
# ─────────────────────────────────────────────────
def eda_transactions(df):
    print("\n📊 TRANSACTION EDA")
    print(df.describe().T.to_string())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Transaction Data — Exploratory Data Analysis", fontsize=16, fontweight="bold")

    # 1. Transaction type distribution
    tx_counts = df["transaction_type"].value_counts()
    axes[0, 0].bar(tx_counts.index, tx_counts.values, color=sns.color_palette("muted"))
    axes[0, 0].set_title("Transaction Type Distribution")
    axes[0, 0].set_xlabel("Type")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis="x", rotation=30)

    # 2. Transaction amount distribution
    df["transaction_amount"].hist(bins=40, ax=axes[0, 1], color="#42A5F5", edgecolor="white")
    axes[0, 1].set_title("Transaction Amount Distribution")
    axes[0, 1].set_xlabel("Amount")

    # 3. Average amount by type
    avg_amt = df.groupby("transaction_type")["transaction_amount"].mean().sort_values()
    avg_amt.plot(kind="barh", ax=axes[0, 2], color="#66BB6A")
    axes[0, 2].set_title("Average Amount by Transaction Type")

    # 4. Transactions by month
    monthly = df.groupby("month").size()
    axes[1, 0].plot(monthly.index, monthly.values, marker="o", color="#EF5350", linewidth=2)
    axes[1, 0].set_title("Transactions per Month")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Count")

    # 5. Weekday vs Weekend
    wd_counts = df["is_weekend"].value_counts()
    axes[1, 1].bar(["Weekday", "Weekend"], wd_counts.values, color=["#5C6BC0", "#FFA726"])
    axes[1, 1].set_title("Weekday vs Weekend Transactions")

    # 6. Top customers by transaction volume
    top_custs = df.groupby("customer_id")["transaction_amount"].sum().nlargest(10)
    top_custs.plot(kind="bar", ax=axes[1, 2], color="#AB47BC")
    axes[1, 2].set_title("Top 10 Customers by Volume")
    axes[1, 2].set_xlabel("Customer ID")
    axes[1, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("outputs/eda/transaction_eda.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Saved outputs/eda/transaction_eda.png")


# ─────────────────────────────────────────────────
# C. RECOMMENDATION EDA
# ─────────────────────────────────────────────────
def eda_recommendation(df):
    print("\n📊 RECOMMENDATION EDA")
    print(df.describe().T.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Product Interaction — Exploratory Data Analysis", fontsize=16, fontweight="bold")

    # 1. Interaction type
    int_counts = df["interaction_type"].value_counts()
    axes[0].pie(int_counts, labels=int_counts.index, autopct="%1.1f%%",
                colors=sns.color_palette("pastel"), startangle=90)
    axes[0].set_title("Interaction Types")

    # 2. Most popular products
    prod_counts = df.groupby("product_name").size().sort_values(ascending=True)
    prod_counts.plot(kind="barh", ax=axes[1], color="#29B6F6")
    axes[1].set_title("Product Popularity")
    axes[1].set_xlabel("Interactions")

    # 3. Rating distribution
    df["rating"].hist(bins=10, ax=axes[2], color="#FF7043", edgecolor="white")
    axes[2].set_title("Rating Distribution")
    axes[2].set_xlabel("Rating")

    plt.tight_layout()
    plt.savefig("outputs/eda/recommendation_eda.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Saved outputs/eda/recommendation_eda.png")


# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  BANKING PROJECT — EDA")
    print("=" * 50)
    loan_df  = pd.read_csv("data/loan_data.csv")
    trans_df = pd.read_csv("data/transaction_data.csv")
    rec_df   = pd.read_csv("data/recommendation_data.csv")

    eda_loan(loan_df)
    eda_transactions(trans_df)
    eda_recommendation(rec_df)
    print("\nAll EDA plots saved to outputs/eda/")
