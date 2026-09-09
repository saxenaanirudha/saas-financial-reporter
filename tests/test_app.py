"""
test_app.py
===========
Tests for Streamlit Browser Dashboard helper functions and loaders (app.py).
"""

import pytest
import pandas as pd
from pathlib import Path
import app
from src.config import FINANCIAL_DATA_PATH, CHURN_DATA_PATH


def test_load_financial_analytics_returns_expected_structure():
    """Verify that load_financial_analytics returns all required keys."""
    data = app.load_financial_analytics(FINANCIAL_DATA_PATH)
    assert data is not None
    required_keys = [
        "raw_df",
        "revenue_df",
        "revenue_summary",
        "cac_df",
        "cac_summary",
        "churn_df",
        "churn_summary",
        "mom_comp",
        "first_comp",
        "trend_summary",
        "cash_flow_df",
        "cash_flow_summary",
        "forecast_df",
        "forecast_summary",
        "dashboard_data",
    ]
    for k in required_keys:
        assert k in data, f"Missing key {k} in load_financial_analytics output"


def test_load_financial_analytics_missing_file_returns_none(tmp_path):
    """Verify that missing financial file returns None gracefully."""
    missing_file = tmp_path / "non_existent.csv"
    data = app.load_financial_analytics(missing_file)
    assert data is None


def test_load_churn_intelligence_returns_expected_structure():
    """Verify that load_churn_intelligence returns metrics, scored_df, and counts."""
    data = app.load_churn_intelligence(CHURN_DATA_PATH)
    assert data is not None
    assert "metrics" in data
    assert "scored_df" in data
    assert "risk_counts" in data
    assert data["metrics"]["accuracy"] > 0.80
    assert len(data["scored_df"]) == 500


def test_load_project_artifacts_discovers_outputs():
    """Verify artifact discovery identifies existing files and returns structure."""
    artifacts = app.load_project_artifacts()
    assert "pdf_report" in artifacts
    assert "financial_charts" in artifacts
    assert "ml_artifacts" in artifacts
    assert len(artifacts["financial_charts"]) == 10


def test_forecast_summary_keys_match_app_expectations():
    """Verify that forecast_summary dictionary contains all expected keys used by app.py."""
    data = app.load_financial_analytics(FINANCIAL_DATA_PATH)
    assert data is not None
    fc_s = data["forecast_summary"]
    required_forecast_keys = [
        "forecast_start_month",
        "forecast_end_month",
        "starting_forecast_cash_balance",
        "ending_forecast_cash_balance",
        "lowest_forecast_closing_cash_balance",
        "highest_forecast_closing_cash_balance",
        "forecast_months",
        "interpretation",
    ]
    for key in required_forecast_keys:
        assert key in fc_s, f"Expected key '{key}' missing from forecast_summary"


def test_render_charts_execute_successfully():
    """Verify that all matplotlib chart helper functions in app.py execute without errors."""
    import matplotlib.pyplot as plt

    data = app.load_financial_analytics(FINANCIAL_DATA_PATH)
    assert data is not None

    fig_rev = app.render_mrr_arr_trend(data["revenue_df"])
    assert fig_rev is not None
    plt.close(fig_rev)

    fig_cac = app.render_cac_growth_trend(data["cac_df"], data["revenue_df"])
    assert fig_cac is not None
    plt.close(fig_cac)

    fig_churn = app.render_customer_churn_trend(data["churn_df"])
    assert fig_churn is not None
    plt.close(fig_churn)

    fig_cf, fig_bal = app.render_cash_forecast_charts(
        data["cash_flow_df"], data["forecast_df"]
    )
    assert fig_cf is not None
    assert fig_bal is not None
    plt.close(fig_cf)
    plt.close(fig_bal)

