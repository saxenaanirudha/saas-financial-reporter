"""
Unit Tests for src/churn_retention_analyzer.py
=============================================================================
Tests customer base balances, churn rates, retention rates, MoM churn change
in percentage points, edge cases, and summary generation.
"""

import pytest
import numpy as np
import pandas as pd
from src.churn_retention_analyzer import (
    calculate_customer_base,
    calculate_churn_retention,
    calculate_churn_retention_metrics,
    generate_churn_retention_summary,
    format_churn_retention_table,
)


@pytest.fixture
def sample_df():
    """
    Returns the standard 6-month sample financial dataset.
    """
    return pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025", "Mar-2025", "Apr-2025", "May-2025", "Jun-2025"],
        "NewCustomers": [120, 140, 160, 180, 200, 220],
        "ChurnedCustomers": [15, 20, 18, 25, 22, 30],
        "MRR": [60000, 68000, 75000, 82000, 90000, 97000],
        "MarketingSpend": [15000, 16000, 17000, 18000, 19000, 20000],
    })


def test_beginning_and_ending_balances(sample_df):
    """
    Test 1 & 2 & 3: Tests that beginning, ending customer balances and roll-forwards
    match expected values for all months.
    """
    base_df = calculate_customer_base(sample_df)

    expected_beg = [0, 105, 225, 367, 522, 700]
    expected_end = [105, 225, 367, 522, 700, 890]

    assert list(base_df["BeginningCustomers"]) == expected_beg
    assert list(base_df["EndingCustomers"]) == expected_end


def test_first_month_churn_and_retention_are_nan(sample_df):
    """
    Test 4 & 5: First month has Beginning Customers = 0, so churn rate and retention
    rate should be NaN (N/A).
    """
    metrics_df = calculate_churn_retention_metrics(sample_df)

    assert pd.isna(metrics_df.iloc[0]["ChurnRatePercent"])
    assert pd.isna(metrics_df.iloc[0]["RetentionRatePercent"])
    assert pd.isna(metrics_df.iloc[0]["ChurnChangePP"])


def test_churn_rate_calculated_correctly(sample_df):
    """
    Test 6: Tests churn rate calculations against manual expected values:
    Feb: 20 / 105 * 100 ≈ 19.0476%
    Mar: 18 / 225 * 100 = 8.0%
    """
    metrics_df = calculate_churn_retention_metrics(sample_df)

    assert round(metrics_df.iloc[1]["ChurnRatePercent"], 2) == 19.05
    assert round(metrics_df.iloc[2]["ChurnRatePercent"], 2) == 8.00
    assert round(metrics_df.iloc[3]["ChurnRatePercent"], 2) == 6.81
    assert round(metrics_df.iloc[4]["ChurnRatePercent"], 2) == 4.21
    assert round(metrics_df.iloc[5]["ChurnRatePercent"], 2) == 4.29


def test_retention_rate_calculated_correctly(sample_df):
    """
    Test 7: Tests retention rate calculations:
    Retention % = 100 - Churn Rate %
    """
    metrics_df = calculate_churn_retention_metrics(sample_df)

    assert round(metrics_df.iloc[1]["RetentionRatePercent"], 2) == 80.95
    assert round(metrics_df.iloc[2]["RetentionRatePercent"], 2) == 92.00
    assert round(metrics_df.iloc[3]["RetentionRatePercent"], 2) == 93.19
    assert round(metrics_df.iloc[4]["RetentionRatePercent"], 2) == 95.79
    assert round(metrics_df.iloc[5]["RetentionRatePercent"], 2) == 95.71


def test_churn_change_pp_calculated_correctly(sample_df):
    """
    Test 8: Tests month-over-month Churn Change in percentage points (pp).
    Mar Churn (8.00) - Feb Churn (19.05) = -11.05 pp
    Jun Churn (4.29) - May Churn (4.21) = +0.07 pp
    """
    metrics_df = calculate_churn_retention_metrics(sample_df)

    assert pd.isna(metrics_df.iloc[0]["ChurnChangePP"])
    assert pd.isna(metrics_df.iloc[1]["ChurnChangePP"])  # Prev month churn was NaN
    assert round(metrics_df.iloc[2]["ChurnChangePP"], 2) == -11.05
    assert round(metrics_df.iloc[3]["ChurnChangePP"], 2) == -1.19
    assert round(metrics_df.iloc[4]["ChurnChangePP"], 2) == -2.60
    assert round(metrics_df.iloc[5]["ChurnChangePP"], 2) == 0.07


def test_summary_values(sample_df):
    """
    Test 9 & 14: Tests generated summary dictionary and final customer base of 890.
    """
    summary = generate_churn_retention_summary(sample_df)

    assert summary["latest_month"] == "Jun-2025"
    assert summary["beginning_customers"] == 700
    assert summary["new_customers"] == 220
    assert summary["churned_customers"] == 30
    assert summary["ending_customers"] == 890
    assert summary["latest_customer_base"] == 890
    assert round(summary["latest_churn_rate"], 2) == 4.29
    assert round(summary["latest_retention_rate"], 2) == 95.71
    assert round(summary["previous_churn_rate"], 2) == 4.21
    assert round(summary["latest_churn_change_pp"], 2) == 0.07
    assert summary["total_new_customers"] == 1020
    assert summary["total_churned_customers"] == 130
    assert "increased" in summary["interpretation"]


def test_single_row_dataset():
    """
    Test 10 & 11: Single row dataset produces Beginning=0, Ending=New-Churn,
    Churn Rate = NaN, and summary runs without division-by-zero errors.
    """
    single_row_df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "NewCustomers": [100],
        "ChurnedCustomers": [10],
        "MRR": [50000],
        "MarketingSpend": [10000],
    })

    metrics = calculate_churn_retention_metrics(single_row_df)
    assert metrics.iloc[0]["BeginningCustomers"] == 0
    assert metrics.iloc[0]["EndingCustomers"] == 90
    assert pd.isna(metrics.iloc[0]["ChurnRatePercent"])

    summary = generate_churn_retention_summary(single_row_df)
    assert summary["ending_customers"] == 90
    assert summary["latest_churn_rate"] is None
    assert summary["previous_churn_rate"] is None


def test_impossible_negative_ending_balance_rejected():
    """
    Test 12: If ChurnedCustomers > Beginning + New, a clear ValueError is raised.
    """
    invalid_df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "NewCustomers": [10],
        "ChurnedCustomers": [25],  # Churned exceeds New when Beginning is 0
        "MRR": [50000],
        "MarketingSpend": [10000],
    })

    with pytest.raises(ValueError) as excinfo:
        calculate_customer_base(invalid_df)

    assert "Churned customers (25) cannot exceed" in str(excinfo.value)


def test_input_dataframe_not_mutated(sample_df):
    """
    Test 13: Verifies input DataFrame is not mutated.
    """
    original_cols = list(sample_df.columns)
    calculate_churn_retention_metrics(sample_df)

    assert list(sample_df.columns) == original_cols
    assert "BeginningCustomers" not in sample_df.columns


def test_format_table_output(sample_df):
    """
    Tests formatting of table display strings.
    """
    metrics_df = calculate_churn_retention_metrics(sample_df)
    formatted = format_churn_retention_table(metrics_df)

    assert formatted.iloc[0]["Churn Rate"] == "N/A"
    assert formatted.iloc[0]["Retention Rate"] == "N/A"
    assert formatted.iloc[0]["Churn Change (pp)"] == "N/A"

    assert formatted.iloc[1]["Churn Rate"] == "19.05%"
    assert formatted.iloc[1]["Retention Rate"] == "80.95%"
    assert formatted.iloc[2]["Churn Change (pp)"] == "-11.05 pp"
    assert formatted.iloc[5]["Churn Change (pp)"] == "+0.07 pp"

