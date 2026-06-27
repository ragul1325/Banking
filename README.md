# Predictive Analytics & Recommendation Systems in Banking

## Project Overview
This project implements three ML systems on synthetic banking data:
1. **Loan Default Prediction** (Supervised Learning)
2. **Customer Segmentation** (Unsupervised Learning)
3. **Product Recommendation Engine** (Collaborative Filtering)

---

## Setup & Run

```bash
pip install -r requirements.txt
python run_all.py          # Run complete pipeline
```

Or run each step individually:
```bash
python 1_generate_data.py   # Generate synthetic datasets
python 2_eda.py             # Exploratory Data Analysis
python 3_loan_default.py    # Loan Default Prediction
python 4_segmentation.py    # Customer Segmentation
python 5_recommendation.py  # Recommendation Engine
```

---

## Project Structure

```
banking_project/
├── data/                          # Generated datasets
│   ├── loan_data.csv              # 2000 rows, 14 features
│   ├── transaction_data.csv       # ~55k transactions
│   └── recommendation_data.csv   # ~13k interactions
│
├── outputs/
│   ├── eda/                       # EDA plots (3 PNG files)
│   ├── loan/                      # Metrics CSV + plots
│   ├── segmentation/              # Cluster plots + CSVs
│   └── recommendation/            # Rec engine plots + CSVs
│
├── models/                        # Saved trained models
│   ├── loan_default_rf.pkl        # Random Forest pipeline
│   ├── segmentation_kmeans.pkl    # KMeans + scaler
│   └── recommendation_svd.pkl    # SVD recommender
│
├── 1_generate_data.py
├── 2_eda.py
├── 3_loan_default.py
├── 4_segmentation.py
├── 5_recommendation.py
├── run_all.py
├── requirements.txt
└── README.md
```

---

## Results Summary

### A. Loan Default Prediction

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|----------|-----------|--------|-------|---------|
| Logistic Regression | 0.6775   | 0.6020    | 0.6982 | 0.6466| 0.7378  |
| Random Forest       | 0.6875   | 0.6571    | 0.5444 | 0.5955| 0.7395  |
| Gradient Boosting   | 0.6775   | 0.6266    | 0.5858 | 0.6055| 0.7392  |

**Best Model:** Random Forest (ROC-AUC = 0.74)

Top Features: `num_late_payments`, `credit_score`, `debt_to_income`, `payment_to_income`

### B. Customer Segmentation

| Metric           | Score  |
|------------------|--------|
| Silhouette Score | 0.4009 |
| Davies-Bouldin   | 1.0244 |
| Clusters Found   | 2      |

| Segment            | Recency | Frequency | Monetary   | Interpretation             |
|--------------------|---------|-----------|------------|----------------------------|
| Low-Value Savers   | 21 days | 37 txns   | $184,391   | Recent, active, high value |
| High-Value Active  | 61 days | 14 txns   | $68,257    | Infrequent, lower value    |

### C. Recommendation Engine

| Metric       | Score  |
|--------------|--------|
| Precision@5  | 0.1805 |
| Recall@5     | 0.9025 |
| NDCG@5       | 0.6348 |

Algorithm: SVD Matrix Factorization (k=9 latent factors)

---

## Feature Engineering (Derived Attributes)

**Loan Data:**
- `debt_to_income` = loan_amount / income
- `monthly_payment` = computed via amortization formula
- `payment_to_income` = monthly_payment / (income/12)

**Transaction Data:**
- `is_weekend` = binary flag
- `quarter` = derived from month
- `is_large_txn` = transaction_amount > 5000

**Segmentation (RFM):**
- `recency` = days since last transaction
- `frequency` = total transaction count
- `monetary` = total spend
- `avg_amount`, `n_types`, `n_weekends`, `n_large_txns`, `active_months`

---

## Business Insights

1. **Loan Risk**: Customers with credit score < 600 and >2 late payments have 70%+ default probability.
2. **Segments**: ~60% of customers are highly active (Low-Value Savers) — ideal for upselling.
3. **Products**: Investment Fund and Savings Account are most recommended — align with cross-sell campaigns.

---

## Deployment Notes

- Models saved as `.pkl` files via `joblib`
- Load and predict with:
```python
import joblib, pandas as pd
model = joblib.load('models/loan_default_rf.pkl')
pred = model.predict(new_customer_df[FEATURES])
```
