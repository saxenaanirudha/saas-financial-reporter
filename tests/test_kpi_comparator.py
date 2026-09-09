"""
Unit Tests for src/kpi_comparator.py
=============================================================================
Tests KPI snapshot assembly, period comparisons (Latest vs Previous, Latest vs First),
percentage difference calculation, percentage point rate metrics, direction rules,
zero denominator safety, single-row handling, and trend summary generation.
"""

import pytest
import numpy as np
import pandas as pd
from src.kpi_comparator import (
    build_kpi_snapshot,
    compare_periods,
    compare_latest_vs_previous,
    compare_latest_vs_first,
    generate_trend_summary,
    format_comparison_table,
    _determine_status,
)


@pytest.fixture
def sample_df():
    """
    Returns standard 6-month financial DataFrame.
    """
    return pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025", "Mar-2025", "Apr-2025", "May-2025", "Jun-2025"],
        "NewCustomers": [120, 140, 160, 180, 200, 220],
        "ChurnedCustomers": [15, 20, 18, 25, 22, 30],
        "MRR": [60000, 68000, 75000, 82000, 90000, 97000],
        "MarketingSpend": [15000, 16000, 17000, 18000, 19000, 20000],
    })


def test_build_kpi_snapshot_contains_expected_fields(sample_df):
    """
    Test 1: KPI snapshot contains all calculated metrics from steps 3, 4, and 5.
    """
    snapshot = build_kpi_snapshot(sample_df)

    expected_cols = [
        "Month",
        "MRR",
        "ARR",
        "MRRGrowthPct",
        "CAC",
        "BeginningCustomers",
        "EndingCustomers",
        "ChurnRatePercent",
        "RetentionRatePercent",
    ]
    for col in expected_cols:
        assert col in snapshot.columns


def test_latest_vs_previous_selects_correct_months(sample_df):
    """
    Test 2: Latest vs Previous compares Jun-2025 (latest) vs May-2025 (previous).
    """
    comp = compare_latest_vs_previous(sample_df)

    assert list(comp["Latest Month"].unique()) == ["Jun-2025"]
    assert list(comp["Comparison Month"].unique()) == ["May-2025"]


def test_latest_vs_first_selects_correct_months(sample_df):
    """
    Test 3: Latest vs First compares Jun-2025 (latest) vs Jan-2025 (first).
    """
    comp = compare_latest_vs_first(sample_df)

    assert list(comp["Latest Month"].unique()) == ["Jun-2025"]
    assert list(comp["Comparison Month"].unique()) == ["Jan-2025"]


def test_mrr_absolute_difference_and_percentage_change(sample_df):
    """
    Test 4 & 5: MRR absolute diff and percentage change calculations.
    May MRR = 90,000, Jun MRR = 97,000 -> Diff = +7,000, Pct = +7.78%
    """
    comp = compare_latest_vs_previous(sample_df)
    mrr_row = comp[comp["KPI"] == "MRR"].iloc[0]

    assert mrr_row["Comparison Value"] == 90000
    assert mrr_row["Latest Value"] == 97000
    assert mrr_row["Absolute Change"] == 7000
    assert round(mrr_row["Change % / pp"], 2) == 7.78
    assert mrr_row["Status"] == "Improved"


def test_cac_decrease_classified_as_improved(sample_df):
    """
    Test 6: CAC decrease is classified as Improved (lower is better).
    May CAC = 95.00, Jun CAC = 90.91 -> Diff = -4.09 -> Improved
    """
    comp = compare_latest_vs_previous(sample_df)
    cac_row = comp[comp["KPI"] == "CAC"].iloc[0]

    assert round(cac_row["Absolute Change"], 2) == -4.09
    assert round(cac_row["Change % / pp"], 2) == -4.31
    assert cac_row["Status"] == "Improved"


def test_churn_decrease_and_increase_classification():
    """
    Test 7 & 8: Churn decrease is Improved; Churn increase is Worsened.
    """
    assert _determine_status("lower_is_better", abs_change=-2.5, pct_or_pp_change=-2.5, is_rate=True) == "Improved"
    assert _determine_status("lower_is_better", abs_change=0.08, pct_or_pp_change=0.08, is_rate=True) == "Worsened"


def test_retention_increase_classified_as_improved():
    """
    Test 9: Retention rate increase is classified as Improved (higher is better).
    """
    assert _determine_status("higher_is_better", abs_change=1.5, pct_or_pp_change=1.5, is_rate=True) == "Improved"
    assert _determine_status("higher_is_better", abs_change=-0.5, pct_or_pp_change=-0.5, is_rate=True) == "Worsened"


def test_rate_differences_use_percentage_points(sample_df):
    """
    Test 10: Rate-based KPIs (Churn, Retention, MRR Growth) use percentage points.
    """
    comp = compare_latest_vs_previous(sample_df)

    churn_row = comp[comp["KPI"] == "Churn Rate %"].iloc[0]
    ret_row = comp[comp["KPI"] == "Retention Rate %"].iloc[0]

    assert bool(churn_row["Is Rate"]) is True
    assert bool(ret_row["Is Rate"]) is True
    assert round(churn_row["Change % / pp"], 2) == 0.07
    assert round(ret_row["Change % / pp"], 2) == -0.07



def test_zero_baseline_does_not_cause_division_by_zero():
    """
    Test 11: Zero baseline produces NaN / N/A without division-by-zero crashes.
    """
    df_zero_baseline = pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025"],
        "NewCustomers": [0, 100],
        "ChurnedCustomers": [0, 10],
        "MRR": [0, 50000],
        "MarketingSpend": [0, 10000],
    })

    comp = compare_latest_vs_previous(df_zero_baseline)
    mrr_row = comp[comp["KPI"] == "MRR"].iloc[0]

    assert pd.isna(mrr_row["Change % / pp"])


def test_na_first_month_churn_handled_safely(sample_df):
    """
    Test 12: Comparing Latest vs First Month handles Month 1 NaN churn/retention safely.
    """
    comp = compare_latest_vs_first(sample_df)
    churn_row = comp[comp["KPI"] == "Churn Rate %"].iloc[0]

    assert pd.isna(churn_row["Comparison Value"])
    assert pd.isna(churn_row["Change % / pp"])
    assert churn_row["Status"] == "N/A"


def test_single_row_dataset_handled_clearly():
    """
    Test 13: Single row dataset returns graceful insufficient data handling.
    """
    single_df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "NewCustomers": [100],
        "ChurnedCustomers": [10],
        "MRR": [50000],
        "MarketingSpend": [10000],
    })

    comp = compare_latest_vs_previous(single_df)
    assert comp.iloc[0]["Status"] == "Insufficient Data"

    summary = generate_trend_summary(single_df)
    assert summary["latest_vs_previous_available"] is False
    assert summary["overall_status"] == "Insufficient Data"


def test_input_dataframe_not_mutated(sample_df):
    """
    Test 14: Input DataFrame is not mutated by comparator functions.
    """
    original_cols = list(sample_df.columns)
    compare_latest_vs_previous(sample_df)
    compare_latest_vs_first(sample_df)

    assert list(sample_df.columns) == original_cols


def test_trend_summary_generated_from_actual_results(sample_df):
    """
    Test 15: Trend summary generates realistic narrative and overall status.
    """
    summary = generate_trend_summary(sample_df)

    assert summary["latest_vs_previous_available"] is True
    assert summary["overall_status"] == "Positive"
    assert len(summary["summary_statements"]) >= 5
    assert "MRR increased by 7.78%" in summary["summary_statements"][0]


def test_format_comparison_table(sample_df):
    """
    Tests console formatting function.
    """
    comp = compare_latest_vs_previous(sample_df)
    formatted = format_comparison_table(comp)

    assert "KPI" in formatted.columns
    assert "Previous" in formatted.columns
    assert "Latest" in formatted.columns
    assert "Change" in formatted.columns
    assert "Change % / pp" in formatted.columns
    assert "Status" in formatted.columns

    mrr_row = formatted[formatted["KPI"] == "MRR"].iloc[0]
    assert mrr_row["Previous"] == "$90,000"
    assert mrr_row["Latest"] == "$97,000"
    assert mrr_row["Change"] == "$+7,000"
    assert mrr_row["Change % / pp"] == "+7.78%"

