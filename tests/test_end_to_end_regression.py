"""
test_end_to_end_regression.py
=============================
Step 15: Final Testing, Code Cleanup, Regression Validation, and Quality Hardening.
High-value regression tests verifying mathematical correctness across all historical
financial metrics, cash-flow roll-forwards, statistical forecasts, and ML predictions.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.config import FINANCIAL_DATA_PATH, CHURN_DATA_PATH
from src.data_loader import load_financial_data
from src.revenue_dashboard import calculate_revenue_metrics, generate_revenue_summary
from src.cac_calculator import calculate_cac_metrics, generate_cac_summary
from src.churn_retention_analyzer import calculate_churn_retention_metrics, generate_churn_retention_summary
from src.cash_flow_analyzer import analyze_cash_flow, generate_cash_flow_summary
from src.cash_flow_forecaster import build_cash_flow_forecast, generate_forecast_summary
from src.churn_model import (
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
)
from src.pipeline import run_full_pipeline


# ── 1. REVENUE REGRESSION TESTS ───────────────────────────────────────────────

def test_revenue_core_regression():
    df = load_financial_data(FINANCIAL_DATA_PATH)
    rev_df = calculate_revenue_metrics(df)
    rev_sum = generate_revenue_summary(rev_df)

    assert rev_df.loc[0, "MRR"] == 60000.0
    assert rev_df.loc[5, "MRR"] == 97000.0
    assert rev_df.loc[5, "ARR"] == 1164000.0
    assert rev_sum["latest_mrr"] == 97000.0
    assert rev_sum["latest_arr"] == 1164000.0
    assert pytest.approx(rev_sum["overall_mrr_growth"], abs=0.1) == 61.67


# ── 2. CAC REGRESSION TESTS ───────────────────────────────────────────────────

def test_cac_core_regression():
    df = load_financial_data(FINANCIAL_DATA_PATH)
    cac_df = calculate_cac_metrics(df)
    cac_sum = generate_cac_summary(df)

    assert cac_df.loc[0, "CAC"] == 125.00
    assert pytest.approx(cac_df.loc[5, "CAC"], abs=0.01) == 90.91
    assert pytest.approx(cac_sum["overall_cac_change"], abs=0.1) == -27.27


# ── 3. CUSTOMER BASE ROLL-FORWARD REGRESSION TESTS ─────────────────────────────

def test_customer_base_reconstruction_regression():
    df = load_financial_data(FINANCIAL_DATA_PATH)
    churn_df = calculate_churn_retention_metrics(df)

    expected_ending_customers = [105, 225, 367, 522, 700, 890]
    actual_ending_customers = churn_df["EndingCustomers"].tolist()
    assert actual_ending_customers == expected_ending_customers


# ── 4. CHURN & RETENTION REGRESSION TESTS ─────────────────────────────────────

def test_churn_retention_core_regression():
    df = load_financial_data(FINANCIAL_DATA_PATH)
    churn_df = calculate_churn_retention_metrics(df)
    churn_sum = generate_churn_retention_summary(df)

    assert pytest.approx(churn_df.loc[5, "ChurnRatePercent"], abs=0.01) == 4.29
    assert pytest.approx(churn_df.loc[5, "RetentionRatePercent"], abs=0.01) == 95.71
    assert pytest.approx(churn_sum["latest_churn_rate"], abs=0.01) == 4.29
    assert pytest.approx(churn_sum["latest_retention_rate"], abs=0.01) == 95.71


# ── 5. CASH FLOW HISTORICAL REGRESSION TESTS ──────────────────────────────────

def test_cash_flow_core_regression():
    df = load_financial_data(FINANCIAL_DATA_PATH)
    cf_df = analyze_cash_flow(df, opening_cash_balance=100000.0)

    # January
    assert cf_df.loc[0, "CashInflow"] == 60000.0
    assert cf_df.loc[0, "TotalCashOutflow"] == 50000.0
    assert cf_df.loc[0, "NetCashFlow"] == 10000.0
    assert cf_df.loc[0, "OpeningCashBalance"] == 100000.0
    assert cf_df.loc[0, "ClosingCashBalance"] == 110000.0

    # February
    assert cf_df.loc[1, "CashInflow"] == 68000.0
    assert cf_df.loc[1, "TotalCashOutflow"] == 52200.0
    assert cf_df.loc[1, "NetCashFlow"] == 15800.0
    assert cf_df.loc[1, "OpeningCashBalance"] == 110000.0
    assert cf_df.loc[1, "ClosingCashBalance"] == 125800.0

    # Roll-forward relationship across all historical months
    for i in range(1, len(cf_df)):
        assert cf_df.loc[i, "OpeningCashBalance"] == cf_df.loc[i - 1, "ClosingCashBalance"]
        assert cf_df.loc[i, "ClosingCashBalance"] == cf_df.loc[i, "OpeningCashBalance"] + cf_df.loc[i, "NetCashFlow"]


# ── 6. CASH FLOW 6-MONTH FORECAST REGRESSION TESTS ─────────────────────────────

def test_cash_flow_forecast_relationships_regression():
    df = load_financial_data(FINANCIAL_DATA_PATH)
    cf_df = analyze_cash_flow(df)
    fc_df = build_cash_flow_forecast(cf_df, horizon=6)
    fc_sum = generate_forecast_summary(fc_df)

    assert len(fc_df) == 6
    assert fc_sum["forecast_months"] == 6
    assert fc_df.loc[0, "ForecastOpeningCashBalance"] == cf_df["ClosingCashBalance"].iloc[-1]

    # Non-negative component constraints & mathematical relationships
    for i, row in fc_df.iterrows():
        assert row["ForecastCashInflow"] >= 0.0
        assert row["ForecastMarketingSpend"] >= 0.0
        assert row["ForecastOperatingExpenses"] >= 0.0
        assert row["ForecastPayrollExpenses"] >= 0.0
        assert row["ForecastOtherExpenses"] >= 0.0
        expected_outflow = (
            row["ForecastMarketingSpend"]
            + row["ForecastOperatingExpenses"]
            + row["ForecastPayrollExpenses"]
            + row["ForecastOtherExpenses"]
        )
        assert pytest.approx(row["ForecastTotalCashOutflow"], abs=0.01) == expected_outflow
        assert pytest.approx(row["ForecastNetCashFlow"], abs=0.01) == row["ForecastCashInflow"] - expected_outflow

    # Roll-forward continuity in forecast
    for i in range(1, len(fc_df)):
        assert fc_df.loc[i, "ForecastOpeningCashBalance"] == fc_df.loc[i - 1, "ForecastClosingCashBalance"]


# ── 7. CHURN ML REGRESSION & LEAKAGE TESTS ────────────────────────────────────

def test_churn_ml_pipeline_and_leakage_regression(tmp_path: Path):
    churn_df = load_churn_data(CHURN_DATA_PATH)
    assert len(churn_df) == 500

    X_train, X_test, y_train, y_test = split_churn_data(churn_df, test_size=0.20, random_state=42)
    pipeline = build_churn_pipeline(random_state=42)
    train_churn_model(pipeline, X_train, y_train)

    # Verification of evaluation metrics
    metrics = evaluate_churn_model(pipeline, X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    # Verification of risk scoring
    scored = predict_churn_probabilities(pipeline, churn_df)
    assert len(scored) == 500
    assert set(scored["RiskLevel"].unique()).issubset({"Low", "Medium", "High"})

    # Verification of model serialization & round-trip consistency
    saved_model_path = tmp_path / "model.joblib"
    save_churn_model(pipeline, saved_model_path)
    loaded_pipe = load_saved_churn_model(saved_model_path)
    np.testing.assert_array_equal(pipeline.predict(X_test), loaded_pipe.predict(X_test))

    # Anti-leakage verification
    coef_df = get_feature_coefficients(pipeline)
    features = coef_df["Feature"].tolist()
    assert "CustomerID" not in features
    assert "SignupDate" not in features
    assert "CancellationDate" not in features
    assert "Churned" not in features


# ── 8. RISK TIER THRESHOLD REGRESSION ─────────────────────────────────────────

def test_risk_tier_boundary_thresholds_regression():
    assert assign_churn_risk(0.0) == "Low"
    assert assign_churn_risk(0.2999) == "Low"
    assert assign_churn_risk(0.30) == "Medium"
    assert assign_churn_risk(0.5999) == "Medium"
    assert assign_churn_risk(0.60) == "High"
    assert assign_churn_risk(1.0) == "High"


# ── 9. FULL PIPELINE SUCCESS REGRESSION TEST ──────────────────────────────────

def test_full_pipeline_end_to_end_regression(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.read_csv(FINANCIAL_DATA_PATH).to_csv(data_dir / "saas_financial_data.csv", index=False)
    pd.read_csv(CHURN_DATA_PATH).to_csv(data_dir / "customer_churn_data.csv", index=False)

    res = run_full_pipeline(project_root=tmp_path, run_churn_ml=True, verbose=False)
    assert res["overall_status"] == "SUCCESS"
    assert res["errors_count"] == 0
    assert res["warnings_count"] == 0
    assert res["artifact_validation"]["all_required_valid"]
