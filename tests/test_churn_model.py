"""
test_churn_model.py
===================
Unit and integration tests for Step 13: Customer Churn Prediction Model and Risk Scoring.
30 comprehensive test cases verifying data loading, splitting, pipeline training,
evaluation metrics, risk scoring, feature analysis, chart generation, and model persistence.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.churn_model import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    TARGET_COL,
    CUSTOMER_ID_COL,
    load_churn_data,
    split_churn_data,
    build_churn_pipeline,
    train_churn_model,
    evaluate_churn_model,
    assign_churn_risk,
    predict_churn_probabilities,
    get_feature_coefficients,
    save_churn_model,
    load_saved_churn_model,
    save_churn_metrics,
    generate_confusion_matrix_chart,
    generate_roc_curve_chart,
)


# ── FIXTURES ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_churn_df() -> pd.DataFrame:
    """Generate a small deterministic churn dataframe for testing."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "CustomerID": [f"CUST-{i:04d}" for i in range(1, n + 1)],
        "SignupDate": ["2023-01-01"] * n,
        "Plan": np.random.choice(["Basic", "Pro", "Enterprise"], size=n),
        "MonthlyFee": np.random.uniform(20.0, 200.0, size=n).round(2),
        "TenureMonths": np.random.randint(1, 36, size=n),
        "LastLoginDaysAgo": np.random.randint(1, 100, size=n),
        "SupportTickets": np.random.randint(0, 10, size=n),
        "MonthlyUsageHours": np.random.uniform(1.0, 150.0, size=n).round(2),
        "PaymentFailures": np.random.randint(0, 5, size=n),
        "DiscountPercent": np.random.uniform(0.0, 30.0, size=n).round(2),
        "ContractType": np.random.choice(["Monthly", "Annual"], size=n),
        "AutoRenew": np.random.randint(0, 2, size=n),
        "Churned": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    })


@pytest.fixture
def trained_pipeline(sample_churn_df: pd.DataFrame):
    """Return a fitted pipeline on sample churn data."""
    X_train, _, y_train, _ = split_churn_data(sample_churn_df, test_size=0.25, random_state=42)
    pipe = build_churn_pipeline(random_state=42)
    return train_churn_model(pipe, X_train, y_train)


# ── 1. DATA LOADING TESTS ─────────────────────────────────────────────────────

def test_load_churn_data_success(tmp_path: Path, sample_churn_df: pd.DataFrame):
    csv_file = tmp_path / "churn.csv"
    sample_churn_df.to_csv(csv_file, index=False)
    loaded = load_churn_data(csv_file)
    assert len(loaded) == len(sample_churn_df)
    assert TARGET_COL in loaded.columns
    assert CUSTOMER_ID_COL in loaded.columns


def test_load_churn_data_file_not_found():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_churn_data("non_existent_file.csv")


def test_load_churn_data_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises((ValueError, pd.errors.EmptyDataError)):
        load_churn_data(empty_file)


def test_load_churn_data_missing_columns(tmp_path: Path):
    bad_df = pd.DataFrame({"CustomerID": ["CUST-001"], "Plan": ["Basic"]})
    bad_file = tmp_path / "bad.csv"
    bad_df.to_csv(bad_file, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_churn_data(bad_file)


def test_load_churn_data_invalid_target_values(tmp_path: Path, sample_churn_df: pd.DataFrame):
    sample_churn_df.loc[0, "Churned"] = 99
    bad_file = tmp_path / "invalid_target.csv"
    sample_churn_df.to_csv(bad_file, index=False)
    with pytest.raises(ValueError, match="binary values"):
        load_churn_data(bad_file)


# ── 2. DATA SPLITTING TESTS ───────────────────────────────────────────────────

def test_split_churn_data_shapes(sample_churn_df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_churn_data(sample_churn_df, test_size=0.20, random_state=42)
    assert len(X_train) == 48
    assert len(X_test) == 12
    assert len(y_train) == 48
    assert len(y_test) == 12
    assert list(X_train.columns) == FEATURE_COLS


def test_split_churn_data_stratification(sample_churn_df: pd.DataFrame):
    _, _, y_train, y_test = split_churn_data(sample_churn_df, test_size=0.20, random_state=42, stratify=True)
    orig_prop = sample_churn_df["Churned"].mean()
    train_prop = y_train.mean()
    test_prop = y_test.mean()
    assert abs(train_prop - orig_prop) < 0.05
    assert abs(test_prop - orig_prop) < 0.10


def test_split_churn_data_reproducibility(sample_churn_df: pd.DataFrame):
    X_tr1, X_te1, _, _ = split_churn_data(sample_churn_df, test_size=0.20, random_state=42)
    X_tr2, X_te2, _, _ = split_churn_data(sample_churn_df, test_size=0.20, random_state=42)
    pd.testing.assert_frame_equal(X_tr1, X_tr2)
    pd.testing.assert_frame_equal(X_te1, X_te2)


def test_split_churn_data_missing_column(sample_churn_df: pd.DataFrame):
    df_no_target = sample_churn_df.drop(columns=["Churned"])
    with pytest.raises(ValueError, match="Missing columns"):
        split_churn_data(df_no_target)


# ── 3. PIPELINE & TRAINING TESTS ──────────────────────────────────────────────

def test_build_churn_pipeline_structure():
    pipe = build_churn_pipeline()
    assert isinstance(pipe, Pipeline)
    assert "preprocessor" in pipe.named_steps
    assert "classifier" in pipe.named_steps


def test_build_churn_pipeline_custom_features():
    num_feats = ["MonthlyFee", "TenureMonths"]
    cat_feats = ["Plan"]
    pipe = build_churn_pipeline(numerical_features=num_feats, categorical_features=cat_feats)
    assert pipe is not None


def test_train_churn_model_fit(sample_churn_df: pd.DataFrame):
    X_train, _, y_train, _ = split_churn_data(sample_churn_df, test_size=0.20, random_state=42)
    pipe = build_churn_pipeline(random_state=42)
    fitted = train_churn_model(pipe, X_train, y_train)
    assert hasattr(fitted.named_steps["classifier"], "coef_")
    assert hasattr(fitted.named_steps["classifier"], "classes_")


# ── 4. EVALUATION TESTS ───────────────────────────────────────────────────────

def test_evaluate_churn_model_metrics_keys(trained_pipeline: Pipeline, sample_churn_df: pd.DataFrame):
    _, X_test, _, y_test = split_churn_data(sample_churn_df, test_size=0.25, random_state=42)
    metrics = evaluate_churn_model(trained_pipeline, X_test, y_test)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics


def test_evaluate_churn_model_metrics_ranges(trained_pipeline: Pipeline, sample_churn_df: pd.DataFrame):
    _, X_test, _, y_test = split_churn_data(sample_churn_df, test_size=0.25, random_state=42)
    metrics = evaluate_churn_model(trained_pipeline, X_test, y_test)
    for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        assert 0.0 <= metrics[m] <= 1.0


def test_evaluate_churn_model_confusion_matrix_sum(trained_pipeline: Pipeline, sample_churn_df: pd.DataFrame):
    _, X_test, _, y_test = split_churn_data(sample_churn_df, test_size=0.25, random_state=42)
    metrics = evaluate_churn_model(trained_pipeline, X_test, y_test)
    cm = metrics["confusion_matrix"]
    total = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
    assert total == len(y_test)


# ── 5. RISK TIER SCORING TESTS ────────────────────────────────────────────────

def test_assign_churn_risk_low():
    assert assign_churn_risk(0.0) == "Low"
    assert assign_churn_risk(0.15) == "Low"
    assert assign_churn_risk(0.299) == "Low"


def test_assign_churn_risk_medium():
    assert assign_churn_risk(0.30) == "Medium"
    assert assign_churn_risk(0.45) == "Medium"
    assert assign_churn_risk(0.599) == "Medium"


def test_assign_churn_risk_high():
    assert assign_churn_risk(0.60) == "High"
    assert assign_churn_risk(0.85) == "High"
    assert assign_churn_risk(1.00) == "High"


def test_assign_churn_risk_boundary_values():
    assert assign_churn_risk(0.30) == "Medium"
    assert assign_churn_risk(0.60) == "High"


def test_assign_churn_risk_invalid_values():
    with pytest.raises(ValueError):
        assign_churn_risk(-0.01)
    with pytest.raises(ValueError):
        assign_churn_risk(1.05)


# ── 6. PROBABILITY PREDICTION & RISK EXPORT TESTS ─────────────────────────────

def test_predict_churn_probabilities_output_structure(trained_pipeline: Pipeline, sample_churn_df: pd.DataFrame):
    scored = predict_churn_probabilities(trained_pipeline, sample_churn_df)
    assert "CustomerID" in scored.columns
    assert "ChurnProbability" in scored.columns
    assert "RiskLevel" in scored.columns
    assert len(scored) == len(sample_churn_df)


def test_predict_churn_probabilities_sorted_descending(trained_pipeline: Pipeline, sample_churn_df: pd.DataFrame):
    scored = predict_churn_probabilities(trained_pipeline, sample_churn_df)
    probs = scored["ChurnProbability"].tolist()
    assert probs == sorted(probs, reverse=True)


def test_predict_churn_probabilities_probabilities_in_range(trained_pipeline: Pipeline, sample_churn_df: pd.DataFrame):
    scored = predict_churn_probabilities(trained_pipeline, sample_churn_df)
    assert scored["ChurnProbability"].between(0.0, 1.0).all()


# ── 7. FEATURE COEFFICIENT ANALYSIS TESTS ─────────────────────────────────────

def test_get_feature_coefficients_structure(trained_pipeline: Pipeline):
    coef_df = get_feature_coefficients(trained_pipeline)
    assert "Feature" in coef_df.columns
    assert "Coefficient" in coef_df.columns
    assert "OddsRatio" in coef_df.columns
    assert "Direction" in coef_df.columns
    assert "ImpactRank" in coef_df.columns
    assert len(coef_df) > 0


def test_get_feature_coefficients_ranking(trained_pipeline: Pipeline):
    coef_df = get_feature_coefficients(trained_pipeline)
    assert coef_df["ImpactRank"].tolist() == list(range(1, len(coef_df) + 1))


# ── 8. PERSISTENCE TESTS ──────────────────────────────────────────────────────

def test_save_and_load_churn_model(tmp_path: Path, trained_pipeline: Pipeline, sample_churn_df: pd.DataFrame):
    model_path = tmp_path / "test_model.joblib"
    save_churn_model(trained_pipeline, model_path)
    assert model_path.exists()

    loaded = load_saved_churn_model(model_path)
    assert isinstance(loaded, Pipeline)

    X_test = sample_churn_df[FEATURE_COLS]
    orig_preds = trained_pipeline.predict(X_test)
    loaded_preds = loaded.predict(X_test)
    np.testing.assert_array_equal(orig_preds, loaded_preds)


def test_save_churn_metrics_json(tmp_path: Path):
    metrics = {"accuracy": 0.88, "precision": 0.85, "recall": 0.80, "f1_score": 0.82, "roc_auc": 0.91}
    json_path = tmp_path / "metrics.json"
    save_churn_metrics(metrics, json_path)
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["accuracy"] == 0.88


# ── 9. CHART GENERATION TESTS ─────────────────────────────────────────────────

def test_generate_confusion_matrix_chart(tmp_path: Path):
    y_test = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 0, 1, 0, 0, 1]
    chart_path = tmp_path / "cm_test.png"
    out = generate_confusion_matrix_chart(y_test, y_pred, chart_path)
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_roc_curve_chart(tmp_path: Path):
    y_test = [0, 0, 1, 1, 0, 1]
    y_proba = [0.1, 0.2, 0.8, 0.4, 0.3, 0.9]
    chart_path = tmp_path / "roc_test.png"
    out = generate_roc_curve_chart(y_test, y_proba, chart_path)
    assert out.exists()
    assert out.stat().st_size > 0


# ── 10. LEAKAGE & INTEGRITY TESTS ─────────────────────────────────────────────

def test_pipeline_no_data_leakage_on_id_or_date(trained_pipeline: Pipeline):
    coef_df = get_feature_coefficients(trained_pipeline)
    features = coef_df["Feature"].tolist()
    assert "CustomerID" not in features
    assert "SignupDate" not in features
    assert "CancellationDate" not in features
    assert "Churned" not in features
