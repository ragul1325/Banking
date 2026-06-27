"""
=====================================================
BANKING PROJECT - STEP 1: Synthetic Data Generation
=====================================================
Generates three datasets:
  A. Loan Default Prediction
  B. Customer Segmentation (Transactions)
  C. Product Recommendation Engine
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

os.makedirs("data", exist_ok=True)

N_CUSTOMERS = 2000

# ─────────────────────────────────────────────────
# A. LOAN DEFAULT PREDICTION DATASET
# ─────────────────────────────────────────────────
def generate_loan_data(n=N_CUSTOMERS):
    records = []
    for i in range(1, n + 1):
        age           = random.randint(22, 65)
        income        = round(random.uniform(20000, 150000), 2)
        credit_score  = random.randint(300, 850)
        loan_amount   = round(random.uniform(1000, 50000), 2)
        interest_rate = round(random.uniform(3.5, 25.0), 2)
        loan_term     = random.choice([12, 24, 36, 48, 60])

        # Derived / engineered features
        debt_to_income    = round(loan_amount / income, 4)
        monthly_payment   = round((loan_amount * (interest_rate / 1200)) /
                                  (1 - (1 + interest_rate / 1200) ** -loan_term), 2)
        payment_to_income = round(monthly_payment / (income / 12), 4)
        employment_years  = random.randint(0, 30)
        num_open_accounts = random.randint(1, 10)
        num_late_payments = random.randint(0, 5)

        # Default probability logic (realistic)
        default_prob = 0.1
        if credit_score < 500:        default_prob += 0.35
        elif credit_score < 600:      default_prob += 0.20
        if debt_to_income > 0.5:      default_prob += 0.20
        if payment_to_income > 0.4:   default_prob += 0.15
        if num_late_payments > 2:     default_prob += 0.15
        if income < 30000:            default_prob += 0.10
        default_prob = min(default_prob, 0.95)

        repayment_status = int(random.random() < default_prob)

        records.append({
            "customer_id":        f"CUST{i:04d}",
            "age":                age,
            "income":             income,
            "credit_score":       credit_score,
            "loan_amount":        loan_amount,
            "interest_rate":      interest_rate,
            "loan_term":          loan_term,
            "debt_to_income":     debt_to_income,
            "monthly_payment":    monthly_payment,
            "payment_to_income":  payment_to_income,
            "employment_years":   employment_years,
            "num_open_accounts":  num_open_accounts,
            "num_late_payments":  num_late_payments,
            "repayment_status":   repayment_status,  # 1=default, 0=no default
        })

    df = pd.DataFrame(records)
    # Introduce ~3% missing values for realism
    for col in ["income", "credit_score", "employment_years"]:
        mask = np.random.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan
    df.to_csv("data/loan_data.csv", index=False)
    print(f"✅ Loan data saved: {df.shape}  | Default rate: {df['repayment_status'].mean():.1%}")
    return df


# ─────────────────────────────────────────────────
# B. CUSTOMER SEGMENTATION (TRANSACTION) DATASET
# ─────────────────────────────────────────────────
def generate_transaction_data(n_customers=N_CUSTOMERS):
    tx_types = ["deposit", "withdrawal", "transfer", "bill_payment", "investment"]
    records  = []
    tx_id    = 1

    start_date = datetime(2023, 1, 1)
    end_date   = datetime(2024, 12, 31)

    for i in range(1, n_customers + 1):
        cust_id = f"CUST{i:04d}"
        n_txns  = random.randint(5, 50)
        for _ in range(n_txns):
            tx_date   = start_date + timedelta(days=random.randint(0, 730))
            tx_type   = random.choice(tx_types)
            tx_amount = round(random.uniform(10, 10000), 2)

            # Derived features
            is_weekend      = int(tx_date.weekday() >= 5)
            month           = tx_date.month
            quarter         = (month - 1) // 3 + 1
            transaction_hour = random.randint(0, 23)
            is_large_txn    = int(tx_amount > 5000)

            records.append({
                "transaction_id":    f"TXN{tx_id:06d}",
                "customer_id":       cust_id,
                "transaction_amount": tx_amount,
                "transaction_type":  tx_type,
                "transaction_date":  tx_date.strftime("%Y-%m-%d"),
                "transaction_hour":  transaction_hour,
                "is_weekend":        is_weekend,
                "month":             month,
                "quarter":           quarter,
                "is_large_txn":      is_large_txn,
            })
            tx_id += 1

    df = pd.DataFrame(records)
    df.to_csv("data/transaction_data.csv", index=False)
    print(f"✅ Transaction data saved: {df.shape}")
    return df


# ─────────────────────────────────────────────────
# C. RECOMMENDATION ENGINE DATASET
# ─────────────────────────────────────────────────
def generate_recommendation_data(n_customers=N_CUSTOMERS):
    products = {
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
    interaction_types = ["viewed", "clicked", "applied", "purchased"]
    records = []
    start_date = datetime(2023, 1, 1)

    for i in range(1, n_customers + 1):
        cust_id   = f"CUST{i:04d}"
        n_inter   = random.randint(3, 20)
        prods     = random.sample(list(products.keys()), min(n_inter, len(products)))
        for prod in prods:
            int_type = random.choice(interaction_types)
            int_date = start_date + timedelta(days=random.randint(0, 730))
            # Rating proxy
            rating_map = {"viewed": 1, "clicked": 2, "applied": 3, "purchased": 5}
            rating = rating_map[int_type] + random.choice([-0.5, 0, 0.5])
            rating = max(1, min(5, rating))
            records.append({
                "customer_id":      cust_id,
                "product_id":       prod,
                "product_name":     products[prod],
                "interaction_type": int_type,
                "interaction_date": int_date.strftime("%Y-%m-%d"),
                "rating":           round(rating, 1),
            })

    df = pd.DataFrame(records)
    df.to_csv("data/recommendation_data.csv", index=False)
    print(f"✅ Recommendation data saved: {df.shape}")
    return df


# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  BANKING PROJECT — DATA GENERATION")
    print("=" * 50)
    loan_df   = generate_loan_data()
    trans_df  = generate_transaction_data()
    rec_df    = generate_recommendation_data()
    print("\nAll datasets generated successfully in /data/")
