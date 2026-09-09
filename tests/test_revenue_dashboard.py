"""
Unit tests for SaaS Revenue Dashboard module (Step 3).
"""

import numpy as np
import pandas as pd
import pytest

from src.revenue_dashboard import (
    calculate_arr,
    calculate_mrr_growth,
    calculate_revenue_metrics,
    generate_revenue_summary,
    format_revenue_dashboard,
)


@pytest.fixture
def sample_financial_df():
    """Returns the standard 6-month sample SaaS dataset."""
    return pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025", "Mar-2025", "Apr-2025", "May-2025", "Jun-2025"],
        "NewCustomers": [120, 140, 160, 180, 200, 220],
        "ChurnedCustomers": [15, 20, 18, 25, 22, 30],
        "MRR": [60000, 68000, 75000, 82000, 90000, 97000],
        "MarketingSpend": [15000, 16000, 17000, 18000, 19000, 20000]
    })


def test_arr_calculated_correctly(sample_financial_df):
    """Test 1: ARR should strictly equal MRR * 12 for every row."""
    arr_series = calculate_arr(sample_financial_df)
    expected_arr = [720000, 816000, 900000, 984000, 1080000, 1164000]
    assert arr_series.tolist() == expected_arr


def test_mrr_growth_calculated_correctly(sample_financial_df):
    """Test 2: Month-over-month MRR growth percentages match expected values."""
    growth_series = calculate_mrr_growth(sample_financial_df)

    # First month is NaN
    assert np.isnan(growth_series.iloc[0])

    # Feb: (68000 - 60000) / 60000 * 100 = 13.333...
    assert pytest.approx(growth_series.iloc[1], 0.01) == 13.33
    # Mar: (75000 - 68000) / 68000 * 100 = 10.294...
    assert pytest.approx(growth_series.iloc[2], 0.01) == 10.29
    # Apr: (82000 - 75000) / 75000 * 100 = 9.333...
    assert pytest.approx(growth_series.iloc[3], 0.01) == 9.33
    # May: (90000 - 82000) / 82000 * 100 = 9.756...
    assert pytest.approx(growth_series.iloc[4], 0.01) == 9.76
    # Jun: (97000 - 90000) / 90000 * 100 = 7.777...
    assert pytest.approx(growth_series.iloc[5], 0.01) == 7.78


def test_first_month_growth_is_nan(sample_financial_df):
    """Test 3: First month has no previous month, so growth is NaN."""
    growth_series = calculate_mrr_growth(sample_financial_df)
    assert pd.isna(growth_series.iloc[0])


def test_revenue_summary_latest_metrics(sample_financial_df):
    """Test 4: Revenue summary accurately reports latest month numbers."""
    summary = generate_revenue_summary(sample_financial_df)
    assert summary["latest_month"] == "Jun-2025"
    assert summary["latest_mrr"] == 97000.0
    assert summary["latest_arr"] == 1164000.0
    assert summary["previous_mrr"] == 90000.0
    assert pytest.approx(summary["latest_mrr_growth"], 0.01) == 7.78


def test_overall_mrr_growth(sample_financial_df):
    """Test 5: Overall growth across all recorded periods (60k -> 97k = +61.67%)."""
    summary = generate_revenue_summary(sample_financial_df)
    assert summary["starting_mrr"] == 60000.0
    assert summary["net_mrr_change"] == 37000.0
    assert pytest.approx(summary["overall_mrr_growth"], 0.01) == 61.67


def test_input_dataframe_not_mutated(sample_financial_df):
    """Test 6: calculate_revenue_metrics must not mutate original DataFrame."""
    original_cols = list(sample_financial_df.columns)
    _ = calculate_revenue_metrics(sample_financial_df)
    assert list(sample_financial_df.columns) == original_cols


def test_single_row_dataset_handling():
    """Test 7: A single-row dataset behaves properly without errors."""
    single_row_df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "NewCustomers": [120],
        "ChurnedCustomers": [15],
        "MRR": [60000],
        "MarketingSpend": [15000]
    })
    metrics_df = calculate_revenue_metrics(single_row_df)
    assert len(metrics_df) == 1
    assert metrics_df["ARR"].iloc[0] == 720000
    assert pd.isna(metrics_df["MRRGrowthPct"].iloc[0])

    summary = generate_revenue_summary(single_row_df)
    assert summary["latest_mrr"] == 60000.0
    assert summary["previous_mrr"] is None
    assert summary["latest_mrr_growth"] is None
    assert summary["overall_mrr_growth"] == 0.0


def test_zero_previous_mrr_handled_safely():
    """Test 8: Zero previous-month MRR is handled safely without division by zero."""
    zero_prev_df = pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025"],
        "NewCustomers": [0, 100],
        "ChurnedCustomers": [0, 5],
        "MRR": [0, 50000],
        "MarketingSpend": [5000, 10000]
    })
    growth_series = calculate_mrr_growth(zero_prev_df)
    # Feb growth dividing by 0 should produce NaN rather than crashing with ZeroDivisionError
    assert pd.isna(growth_series.iloc[1])


def test_format_revenue_dashboard(sample_financial_df):
    """Test 9: Formatter outputs formatted strings with $ and % signs."""
    metrics_df = calculate_revenue_metrics(sample_financial_df)
    formatted_df = format_revenue_dashboard(metrics_df)
    assert formatted_df["MRR"].iloc[0] == "$60,000"
    assert formatted_df["ARR"].iloc[0] == "$720,000"
    assert formatted_df["MRR Growth %"].iloc[0] == "N/A"
    assert formatted_df["MRR Growth %"].iloc[1] == "13.33%"
