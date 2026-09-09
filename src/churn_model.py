"""
churn_model.py
==============
Step 13: Customer Churn Prediction Model and Risk Scoring.

Provides an end-to-end Machine Learning pipeline using Logistic Regression
to predict customer churn probability and assign operational risk tiers.

Pipeline components:
- Data validation and loading for customer-level churn data
- Stratified train/test splitting (80/20, random_state=42)
- Scikit-learn Pipeline with ColumnTransformer:
    * StandardScaler for numerical features
    * OneHotEncoder(drop='first', handle_unknown='ignore') for categorical features
    * LogisticRegression(max_iter=1000, random_state=42)
- Model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- Customer-level probability prediction and Risk Tier assignment:
    * Low Risk   : Probability < 0.30
    * Medium Risk: 0.30 <= Probability < 0.60
    * High Risk  : Probability >= 0.60
- Feature coefficient and odds ratio extraction
- Visualization generators (Confusion Matrix & ROC Curve)
- Model and evaluation metric serialization (joblib & json)

Portfolio Disclaimer:
    This model is trained on a synthetic customer-level portfolio dataset
    to demonstrate an end-to-end ML workflow and is not intended for
    live production automated decisioning.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Default Feature Schema
NUMERICAL_FEATURES: List[str] = [
    "MonthlyFee",
    "TenureMonths",
    "LastLoginDaysAgo",
    "SupportTickets",
    "MonthlyUsageHours",
    "PaymentFailures",
    "DiscountPercent",
    "AutoRenew",
]

CATEGORICAL_FEATURES: List[str] = [
    "Plan",
    "ContractType",
]

FEATURE_COLS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COL: str = "Churned"
CUSTOMER_ID_COL: str = "CustomerID"
EXCLUDED_COLS: List[str] = ["CustomerID", "SignupDate", "CancellationDate"]

# Risk Tiers
RISK_LOW_THRESHOLD: float = 0.30
RISK_HIGH_THRESHOLD: float = 0.60


def load_churn_data(
    file_path: Union[str, Path],
    required_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load customer churn CSV dataset and validate schema integrity.

    Parameters
    ----------
    file_path : str or Path
        Path to the customer churn CSV file.
    required_cols : list of str, optional
        List of required columns. Defaults to FEATURE_COLS + [TARGET_COL, CUSTOMER_ID_COL].

    Returns
    -------
    pd.DataFrame
        Validated churn dataset.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If dataset is empty, missing required columns, or contains invalid target values.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Customer churn data file not found at: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Customer churn data file is empty: {path}")

    if required_cols is None:
        required_cols = FEATURE_COLS + [TARGET_COL, CUSTOMER_ID_COL]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Customer churn dataset missing required columns: {missing}")

    # Validate target column binary values
    if TARGET_COL in df.columns:
        unique_targets = set(df[TARGET_COL].dropna().unique())
        if not unique_targets.issubset({0, 1}):
            raise ValueError(
                f"Target column '{TARGET_COL}' must only contain binary values {0, 1}, got: {unique_targets}"
            )

    return df


def split_churn_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split churn dataset into train and test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Source churn dataframe.
    feature_cols : list of str, optional
        Feature columns to include. Defaults to FEATURE_COLS.
    target_col : str, default 'Churned'
        Target variable name.
    test_size : float, default 0.20
        Fraction of data reserved for testing.
    random_state : int, default 42
        Seed for reproducibility.
    stratify : bool, default True
        Whether to perform stratified sampling based on the target column.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for train/test split: {missing}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )
    return X_train, X_test, y_train, y_test


def build_churn_pipeline(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    random_state: int = 42,
    class_weight: Optional[Union[str, dict]] = None,
    max_iter: int = 1000,
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with ColumnTransformer preprocessing
    and a LogisticRegression classifier.

    Parameters
    ----------
    numerical_features : list of str, optional
        Numerical columns for StandardScaler. Defaults to NUMERICAL_FEATURES.
    categorical_features : list of str, optional
        Categorical columns for OneHotEncoder. Defaults to CATEGORICAL_FEATURES.
    random_state : int, default 42
        Random state for LogisticRegression.
    class_weight : str or dict, optional
        Class weight setting (e.g. 'balanced').
    max_iter : int, default 1000
        Maximum iterations for logistic solver convergence.

    Returns
    -------
    Pipeline
        Unfitted scikit-learn pipeline.
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    # 1. Feature Preprocessing (Encapsulated inside the pipeline to eliminate data leakage):
    # - Numerical features: Standardized (mean=0, variance=1)
    # - Categorical features: One-hot encoded with first category dropped (avoid multicollinearity)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="drop",  # Exclude non-predictive identifiers (e.g., CustomerID)
    )

    # 2. Complete Pipeline: Preprocessing + Logistic Regression Classifier
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state,
                    class_weight=class_weight,
                    max_iter=max_iter,
                ),
            ),
        ]
    )
    return pipeline


def train_churn_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Fit the churn pipeline on the training dataset.

    Parameters
    ----------
    pipeline : Pipeline
        The scikit-learn pipeline to train.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.

    Returns
    -------
    Pipeline
        Fitted pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_churn_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluate the trained churn pipeline on test data.

    Calculates Accuracy, Precision, Recall, F1 Score, ROC-AUC,
    and Confusion Matrix components.

    Parameters
    ----------
    pipeline : Pipeline
        Trained model pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test ground-truth target vector.

    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix": cm.tolist(),
        },
        "test_sample_size": int(len(y_test)),
        "test_churn_count": int(y_test.sum()),
        "test_active_count": int(len(y_test) - y_test.sum()),
    }


def assign_churn_risk(
    probability: float,
    low_threshold: float = RISK_LOW_THRESHOLD,
    high_threshold: float = RISK_HIGH_THRESHOLD,
) -> str:
    """
    Map predicted churn probability to an operational risk tier.

    Parameters
    ----------
    probability : float
        Predicted churn probability in [0.0, 1.0].
    low_threshold : float, default 0.30
        Upper bound for Low risk.
    high_threshold : float, default 0.60
        Lower bound for High risk.

    Returns
    -------
    str
        'Low', 'Medium', or 'High'
    """
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"Probability must be between 0.0 and 1.0, got: {probability}")

    if probability < low_threshold:
        return "Low"
    elif probability < high_threshold:
        return "Medium"
    else:
        return "High"


def predict_churn_probabilities(
    pipeline: Pipeline,
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    customer_id_col: str = CUSTOMER_ID_COL,
) -> pd.DataFrame:
    """
    Generate churn probabilities and risk classifications for all customers.

    Parameters
    ----------
    pipeline : Pipeline
        Trained churn pipeline.
    df : pd.DataFrame
        Customer dataset containing features.
    feature_cols : list of str, optional
        Features expected by pipeline. Defaults to FEATURE_COLS.
    customer_id_col : str, default 'CustomerID'
        Unique customer ID column.

    Returns
    -------
    pd.DataFrame
        Customer risk scoring dataframe sorted by ChurnProbability descending.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X = df[feature_cols].copy()
    probabilities = pipeline.predict_proba(X)[:, 1]

    scored_df = df.copy()
    scored_df["ChurnProbability"] = np.round(probabilities, 4)
    scored_df["RiskLevel"] = [assign_churn_risk(p) for p in probabilities]

    if TARGET_COL in scored_df.columns:
        scored_df = scored_df.rename(columns={TARGET_COL: "ActualChurned"})

    # Order columns cleanly
    lead_cols = [c for c in [customer_id_col, "Plan", "ContractType", "TenureMonths", "MonthlyFee", "ChurnProbability", "RiskLevel"] if c in scored_df.columns]
    other_cols = [c for c in scored_df.columns if c not in lead_cols]
    final_cols = lead_cols + other_cols

    scored_df = scored_df[final_cols].sort_values(by="ChurnProbability", ascending=False).reset_index(drop=True)
    return scored_df


def get_feature_coefficients(
    pipeline: Pipeline,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract learned feature coefficients and odds ratios from the LogisticRegression classifier.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted model pipeline.
    numerical_features : list of str, optional
        Numerical feature list.
    categorical_features : list of str, optional
        Categorical feature list.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Feature', 'Coefficient', 'OddsRatio', 'Direction', 'ImpactRank'].
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    classifier: LogisticRegression = pipeline.named_steps["classifier"]
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    # Clean feature names (remove prefixes like num__ or cat__)
    cleaned_names = []
    for name in feature_names:
        if name.startswith("num__"):
            cleaned_names.append(name.replace("num__", ""))
        elif name.startswith("cat__"):
            cleaned_names.append(name.replace("cat__", ""))
        else:
            cleaned_names.append(name)

    coefs = classifier.coef_[0]
    odds_ratios = np.exp(coefs)

    coef_df = pd.DataFrame({
        "Feature": cleaned_names,
        "Coefficient": np.round(coefs, 4),
        "OddsRatio": np.round(odds_ratios, 4),
    })

    coef_df["Direction"] = coef_df["Coefficient"].apply(
        lambda x: "Increases Risk (+)" if x > 0 else ("Decreases Risk (-)" if x < 0 else "Neutral")
    )
    coef_df["AbsCoefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values(by="AbsCoefficient", ascending=False).reset_index(drop=True)
    coef_df["ImpactRank"] = range(1, len(coef_df) + 1)
    coef_df = coef_df.drop(columns=["AbsCoefficient"])

    return coef_df


def save_churn_model(
    pipeline: Pipeline,
    output_path: Union[str, Path] = "models/churn_model.joblib",
) -> Path:
    """
    Serialize the trained pipeline to disk via joblib.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline.
    output_path : str or Path
        Destination filepath.

    Returns
    -------
    Path
        Absolute or resolved Path to saved artifact.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_saved_churn_model(
    model_path: Union[str, Path] = "models/churn_model.joblib",
) -> Pipeline:
    """
    Load a serialized pipeline from disk.

    Parameters
    ----------
    model_path : str or Path
        Path to joblib file.

    Returns
    -------
    Pipeline
        Loaded model pipeline.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)


def save_churn_metrics(
    metrics: Dict[str, Any],
    output_path: Union[str, Path] = "models/churn_model_metrics.json",
) -> Path:
    """
    Save model evaluation metrics to a JSON file.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary.
    output_path : str or Path
        Destination filepath.

    Returns
    -------
    Path
        Path to saved JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path


def generate_confusion_matrix_chart(
    y_test: Union[pd.Series, np.ndarray, List[int]],
    y_pred: Union[pd.Series, np.ndarray, List[int]],
    output_path: Union[str, Path] = "charts/churn_confusion_matrix.png",
) -> Path:
    """
    Generate and save a styled Confusion Matrix heatmap chart.

    Parameters
    ----------
    y_test : array-like
        True binary target values.
    y_pred : array-like
        Predicted binary target values.
    output_path : str or Path
        Destination PNG path.

    Returns
    -------
    Path
        Path to saved chart PNG.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fa")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    labels = ["Active (0)", "Churned (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")

    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Customer Churn Confusion Matrix\n(Test Set Evaluation)", fontsize=12, fontweight="bold", pad=12)

    # Label values in cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > thresh else "#1e293b"
            cell_name = ""
            if i == 0 and j == 0:
                cell_name = "\n(TN)"
            elif i == 0 and j == 1:
                cell_name = "\n(FP)"
            elif i == 1 and j == 0:
                cell_name = "\n(FN)"
            elif i == 1 and j == 1:
                cell_name = "\n(TP)"
            ax.text(
                j, i, f"{val}{cell_name}",
                ha="center", va="center",
                color=color, fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_roc_curve_chart(
    y_test: Union[pd.Series, np.ndarray, List[int]],
    y_proba: Union[pd.Series, np.ndarray, List[float]],
    output_path: Union[str, Path] = "charts/churn_roc_curve.png",
) -> Path:
    """
    Generate and save a styled ROC-AUC curve chart.

    Parameters
    ----------
    y_test : array-like
        True binary target values.
    y_proba : array-like
        Predicted probabilities for positive class (Churn=1).
    output_path : str or Path
        Destination PNG path.

    Returns
    -------
    Path
        Path to saved chart PNG.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    try:
        auc_score = roc_auc_score(y_test, y_proba)
    except Exception:
        auc_score = 0.5

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fa")

    ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"Logistic Regression (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_title("Customer Churn ROC Curve\n(Portfolio Logistic Regression Model)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
