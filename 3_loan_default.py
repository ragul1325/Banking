"""
=====================================================
BANKING PROJECT - STEP 3: Loan Default Prediction
=====================================================
Supervised Learning — Logistic Regression, Random
Forest, and Gradient Boosting compared.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics          import (accuracy_score, precision_score, recall_score,
                                      f1_score, roc_auc_score, classification_report,
                                      confusion_matrix, roc_curve)

os.makedirs("outputs/loan", exist_ok=True)
os.makedirs("models",       exist_ok=True)

FEATURES = [
    "age", "income", "credit_score", "loan_amount", "interest_rate",
    "loan_term", "debt_to_income", "monthly_payment", "payment_to_income",
    "employment_years", "num_open_accounts", "num_late_payments",
]
TARGET = "repayment_status"


def build_preprocessor():
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return ColumnTransformer([("num", num_transformer, FEATURES)])


def evaluate_model(name, model, X_test, y_test, results):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred),    4),
        "F1":        round(f1_score(y_test, y_pred),        4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba),  4),
    }
    results.append(metrics)
    print(f"\n  {name}:")
    print(f"    Accuracy={metrics['Accuracy']}  Precision={metrics['Precision']}  "
          f"Recall={metrics['Recall']}  F1={metrics['F1']}  AUC={metrics['ROC-AUC']}")
    return y_pred, y_proba


def plot_results(models_dict, X_test, y_test, results_df, preprocessor):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Loan Default Prediction — Model Evaluation", fontsize=14, fontweight="bold")

    # 1. Metrics comparison bar chart
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    colors = ["#42A5F5", "#66BB6A", "#EF5350"]
    for idx, (name, _) in enumerate(models_dict.items()):
        row = results_df[results_df["Model"] == name].iloc[0]
        vals = [row[m] for m in metrics_to_plot]
        axes[0].bar(x + idx * width, vals, width, label=name, color=colors[idx], alpha=0.85)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(metrics_to_plot)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Metrics Comparison")
    axes[0].legend()
    axes[0].set_ylabel("Score")

    # 2. ROC curves
    X_test_t = preprocessor.transform(X_test)
    for (name, model), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("ROC Curves")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()

    # 3. Confusion matrix for best model (Random Forest)
    best_model = models_dict["Random Forest"]
    y_pred = best_model.predict(X_test_t)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    axes[2].set_title("Confusion Matrix — Random Forest")
    axes[2].set_ylabel("Actual")
    axes[2].set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig("outputs/loan/model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Saved outputs/loan/model_evaluation.png")


def plot_feature_importance(rf_model, preprocessor):
    importances = rf_model.named_steps["clf"].feature_importances_
    fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(fi_df["Feature"], fi_df["Importance"], color="#42A5F5")
    plt.title("Feature Importance — Random Forest")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("outputs/loan/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Saved outputs/loan/feature_importance.png")


# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  BANKING PROJECT — LOAN DEFAULT PREDICTION")
    print("=" * 50)

    df = pd.read_csv("data/loan_data.csv")
    X  = df[FEATURES]
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    }

    results = []
    fitted_models = {}

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_t, y_train)
        fitted_models[name] = clf
        evaluate_model(name, clf, X_test_t, y_test, results)

    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/loan/metrics.csv", index=False)
    print(f"\n📋 Results Table:\n{results_df.to_string(index=False)}")

    # Build full pipeline for best model
    best_pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", fitted_models["Random Forest"]),
    ])

    # Wrap fitted models with preprocessor for plotting
    class WrappedModel:
        def __init__(self, clf): self.clf = clf
        def predict(self, X): return self.clf.predict(X)
        def predict_proba(self, X): return self.clf.predict_proba(X)

    wrapped = {n: WrappedModel(c) for n, c in fitted_models.items()}
    plot_results(wrapped, X_test, y_test, results_df, preprocessor)

    # Feature importance for RF
    rf_pipeline = Pipeline([("pre", preprocessor), ("clf", fitted_models["Random Forest"])])
    rf_pipeline.fit(X_train, y_train)
    plot_feature_importance(rf_pipeline, preprocessor)

    joblib.dump(rf_pipeline, "models/loan_default_rf.pkl")
    print("\n✅ Best model saved to models/loan_default_rf.pkl")

    # Cross-validation
    print("\nCross-Validation (5-fold) for Random Forest:")
    cv_scores = cross_val_score(rf_pipeline, X, y, cv=StratifiedKFold(5), scoring="roc_auc")
    print(f"  AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
