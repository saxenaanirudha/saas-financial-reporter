"""
Tests for Step 11: Consolidated Financial Reporting Dashboard Module
=============================================================================
Validates:
1. Complete dashboard data structure creation and schema.
2. Accurate extraction of Revenue, CAC, Churn, Cash Flow, Forecast, and KPI data.
3. Correct reporting and forecast period boundaries.
4. Deterministic Overall Business Status evaluation.
5. On-disk artifact discovery (PDF report and PNG charts).
6. Resilient handling when optional charts or PDF files are missing.
7. Graceful degradation when historical forecast data is unavailable (<2 months).
8. Validation error on empty DataFrames.
9. Immutability of input DataFrames and dictionaries.
10. Proper console formatting and printing without errors.
=============================================================================
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.reporting_dashboard import (
    build_dashboard_data,
    determine_overall_business_status,
    get_report_artifacts,
    format_consolidated_dashboard,
    display_dashboard,
    TRACKED_CHARTS,
)


@pytest.fixture
def sample_full_df():
    """Returns a realistic 6-month SaaS financial DataFrame."""
    return pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025", "Mar-2025", "Apr-2025", "May-2025", "Jun-2025"],
        "MarketingSpend": [5000, 6000, 7500, 8000, 9000, 10000],
        "NewCustomers": [50, 60, 75, 80, 90, 110],
        "ChurnedCustomers": [5, 6, 8, 7, 9, 10],
        "MRR": [50000, 58000, 67000, 76000, 86000, 97000],
        "OperatingExpenses": [15000, 16000, 17000, 18000, 19000, 20000],
        "PayrollExpenses": [20000, 21000, 22000, 23000, 24000, 25000],
        "OtherExpenses": [3000, 3200, 3400, 3500, 3600, 3800],
    })


@pytest.fixture
def sample_single_month_df():
    """Returns a single-month SaaS financial DataFrame."""
    return pd.DataFrame({
        "Month": ["Jan-2025"],
        "MarketingSpend": [5000],
        "NewCustomers": [50],
        "ChurnedCustomers": [5],
        "MRR": [50000],
        "OperatingExpenses": [15000],
        "PayrollExpenses": [20000],
        "OtherExpenses": [3000],
    })


def test_build_dashboard_data_structure(sample_full_df):
    """Test that build_dashboard_data returns a dictionary with all required root keys."""
    data = build_dashboard_data(sample_full_df)
    
    expected_keys = {
        "reporting_period",
        "forecast_period",
        "business_overview",
        "revenue",
        "cac",
        "churn_retention",
        "cash_flow",
        "forecast",
        "kpi_movements",
        "executive_narrative",
        "artifacts",
    }
    assert expected_keys.issubset(data.keys())


def test_reporting_period_and_forecast_period(sample_full_df):
    """Test that reporting and forecast period boundaries match the data."""
    data = build_dashboard_data(sample_full_df, forecast_horizon=6)
    
    rep = data["reporting_period"]
    assert rep["first_month"] == "Jan-2025"
    assert rep["latest_month"] == "Jun-2025"
    assert rep["total_months"] == 6

    fc = data["forecast_period"]
    assert fc["available"] is True
    assert fc["start_month"] == "Jul-2025"
    assert fc["end_month"] == "Dec-2025"
    assert fc["horizon"] == 6


def test_revenue_section_values(sample_full_df):
    """Test that the revenue section contains correct values from the existing revenue summary."""
    data = build_dashboard_data(sample_full_df)
    rev = data["revenue"]
    
    assert rev["latest_month"] == "Jun-2025"
    assert rev["latest_mrr"] == 97000.0
    assert rev["latest_arr"] == 97000.0 * 12
    assert rev["starting_mrr"] == 50000.0
    assert rev["net_mrr_change"] == 47000.0
    assert pytest.approx(rev["overall_mrr_growth"], 0.01) == 94.0


def test_cac_section_values(sample_full_df):
    """Test that CAC section contains expected values."""
    data = build_dashboard_data(sample_full_df)
    cac = data["cac"]
    
    assert cac["latest_marketing_spend"] == 10000.0
    assert cac["latest_new_customers"] == 110
    assert pytest.approx(cac["latest_cac"], 0.01) == 10000.0 / 110
    assert "interpretation" in cac


def test_churn_retention_section_values(sample_full_df):
    """Test that churn & retention section contains expected roll-forward values."""
    data = build_dashboard_data(sample_full_df)
    churn = data["churn_retention"]
    
    assert churn["new_customers"] == 110
    assert churn["churned_customers"] == 10
    assert churn["ending_customers"] > 0
    assert "latest_churn_rate" in churn
    assert "latest_retention_rate" in churn
    assert "interpretation" in churn


def test_cash_flow_section_values(sample_full_df):
    """Test that cash flow section contains expected operating cash metrics."""
    data = build_dashboard_data(sample_full_df, opening_cash_balance=100000.0)
    cf = data["cash_flow"]
    
    assert cf["latest_cash_inflow"] == 97000.0
    assert cf["latest_total_cash_outflow"] == (10000 + 20000 + 25000 + 3800)
    assert cf["latest_net_cash_flow"] == 97000.0 - (10000 + 20000 + 25000 + 3800)
    assert cf["starting_cash_balance"] == 100000.0
    assert "interpretation" in cf


def test_forecast_section_values(sample_full_df):
    """Test that the 6-month forecast section contains expected projection summaries."""
    data = build_dashboard_data(sample_full_df, forecast_horizon=6)
    fc = data["forecast"]
    
    assert fc is not None
    assert fc["forecast_months"] == 6
    assert fc["forecast_start_month"] == "Jul-2025"
    assert fc["forecast_end_month"] == "Dec-2025"
    assert "ending_forecast_cash_balance" in fc
    assert "lowest_forecast_closing_cash_balance" in fc
    assert "highest_forecast_closing_cash_balance" in fc


def test_kpi_movements_included(sample_full_df):
    """Test that core KPI comparison rows are extracted and populated."""
    data = build_dashboard_data(sample_full_df)
    kpis = data["kpi_movements"]
    
    assert isinstance(kpis, list)
    assert len(kpis) > 0
    kpi_names = [row["KPI"] for row in kpis]
    assert "MRR" in kpi_names
    assert "ARR" in kpi_names
    assert "CAC" in kpi_names


def test_deterministic_overall_business_status(sample_full_df):
    """Test that overall business status evaluation is deterministic."""
    data1 = build_dashboard_data(sample_full_df)
    data2 = build_dashboard_data(sample_full_df)
    
    assert data1["business_overview"]["overall_status"] == data2["business_overview"]["overall_status"]
    assert data1["business_overview"]["overall_status"] in ["Positive", "Mixed", "Stable", "Needs Attention"]


def test_business_status_edge_cases():
    """Test edge cases for determine_overall_business_status."""
    empty_df = pd.DataFrame()
    assert determine_overall_business_status(empty_df, {}) == "Insufficient Data"

    # Synthetic worsening KPI comparison
    worsening_df = pd.DataFrame({
        "KPI": ["MRR", "ARR", "CAC", "Churn Rate %"],
        "Status": ["Worsened", "Worsened", "Worsened", "Worsened"],
    })
    negative_cf = {"latest_net_cash_flow": -10000.0}
    assert determine_overall_business_status(worsening_df, negative_cf) == "Needs Attention"


def test_artifact_discovery_existing_files(tmp_path):
    """Test that existing PDF report and charts are detected accurately."""
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()
    for _, fname in TRACKED_CHARTS:
        f = charts_dir / fname
        f.write_text("dummy chart content")

    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_text("dummy pdf content")

    artifacts = get_report_artifacts(charts_dir=charts_dir, pdf_path=pdf_file)
    
    assert artifacts["pdf_report"]["exists"] is True
    assert artifacts["pdf_report"]["size_bytes"] > 0
    assert artifacts["total_charts_found"] == len(TRACKED_CHARTS)
    assert artifacts["total_charts_tracked"] == len(TRACKED_CHARTS)


def test_artifact_discovery_missing_files(tmp_path):
    """Test that missing PDF and charts do not crash and are marked properly."""
    empty_charts = tmp_path / "empty_charts"
    empty_charts.mkdir()
    missing_pdf = tmp_path / "non_existent.pdf"

    artifacts = get_report_artifacts(charts_dir=empty_charts, pdf_path=missing_pdf)
    
    assert artifacts["pdf_report"]["exists"] is False
    assert artifacts["pdf_report"]["size_bytes"] == 0
    assert artifacts["total_charts_found"] == 0
    assert artifacts["total_charts_tracked"] == len(TRACKED_CHARTS)


def test_missing_optional_artifacts_format_dashboard(sample_full_df, tmp_path):
    """Test that format_consolidated_dashboard handles missing artifacts without crashing."""
    empty_charts = tmp_path / "empty_charts"
    empty_charts.mkdir()
    missing_pdf = tmp_path / "non_existent.pdf"

    data = build_dashboard_data(
        sample_full_df,
        charts_dir=empty_charts,
        pdf_path=missing_pdf,
    )
    output = format_consolidated_dashboard(data)
    
    assert "SAAS FINANCIAL REPORTING DASHBOARD" in output
    assert "PDF Investor Report : Unavailable" in output
    assert "[MISSING]" in output


def test_single_month_data_forecast_unavailable(sample_single_month_df):
    """Test dashboard handles single-month data gracefully where forecast cannot be computed."""
    data = build_dashboard_data(sample_single_month_df)
    
    assert data["forecast_period"]["available"] is False
    assert data["forecast"] is None
    
    output = format_consolidated_dashboard(data)
    assert "Projected Forecast Period   : Unavailable" in output
    assert "Forecast Status            : Unavailable" in output


def test_empty_dataframe_raises_error():
    """Test that building dashboard data from an empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="Cannot build reporting dashboard from an empty DataFrame"):
        build_dashboard_data(pd.DataFrame())

    with pytest.raises(ValueError, match="Cannot build reporting dashboard from an empty DataFrame"):
        build_dashboard_data(None)


def test_input_dataframe_immutability(sample_full_df):
    """Test that passing DataFrame to build_dashboard_data does not modify the original DataFrame."""
    df_copy = sample_full_df.copy(deep=True)
    _ = build_dashboard_data(sample_full_df)
    
    pd.testing.assert_frame_equal(sample_full_df, df_copy)


def test_display_dashboard_prints_output(sample_full_df, capsys):
    """Test that display_dashboard prints the expected header and sections to stdout."""
    data = build_dashboard_data(sample_full_df)
    display_dashboard(data)
    
    captured = capsys.readouterr()
    assert "SAAS FINANCIAL REPORTING DASHBOARD" in captured.out
    assert "1. BUSINESS OVERVIEW" in captured.out
    assert "2. RECURRING REVENUE PERFORMANCE" in captured.out
    assert "3. CUSTOMER ACQUISITION (CAC)" in captured.out
    assert "4. CUSTOMER CHURN & RETENTION" in captured.out
    assert "5. HISTORICAL OPERATING CASH FLOW" in captured.out
    assert "6. SIX-MONTH CASH FLOW FORECAST" in captured.out
    assert "7. PERIOD-OVER-PERIOD CORE KPI MOVEMENTS" in captured.out
    assert "8. GENERATED ARTIFACTS & OUTPUT FILES" in captured.out
