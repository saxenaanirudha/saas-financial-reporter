"""
Unit tests for the Customer Acquisition Cost (CAC) Calculator module (Step 4).
"""

import numpy as np
import pandas as pd
import pytest

from src.cac_calculator import (
    calculate_cac,
    calculate_cac_change,
    calculate_cac_metrics,
    generate_cac_summary,
    format_cac_table,
)


@pytest.fixture
def sample_df():
    """Standard 6-month SaaS sample dataset."""
    return pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025", "Mar-2025", "Apr-2025", "May-2025", "Jun-2025"],
        "NewCustomers": [120, 140, 160, 180, 200, 220],
        "ChurnedCustomers": [15, 20, 18, 25, 22, 30],
        "MRR": [60000, 68000, 75000, 82000, 90000, 97000],
        "MarketingSpend": [15000, 16000, 17000, 18000, 19000, 20000],
    })


def test_cac_calculates_correctly(sample_df):
    """Test 1: CAC equals MarketingSpend / NewCustomers for each month."""
    cac_series = calculate_cac(sample_df)
    assert pytest.approx(cac_series.iloc[0], 0.01) == 125.00   # 15000 / 120
    assert pytest.approx(cac_series.iloc[1], 0.01) == 114.29   # 16000 / 140
    assert pytest.approx(cac_series.iloc[2], 0.01) == 106.25   # 17000 / 160
    assert pytest.approx(cac_series.iloc[3], 0.01) == 100.00   # 18000 / 180
    assert pytest.approx(cac_series.iloc[4], 0.01) == 95.00    # 19000 / 200
    assert pytest.approx(cac_series.iloc[5], 0.01) == 90.91    # 20000 / 220


def test_cac_change_calculates_correctly(sample_df):
    """Test 2: MoM CAC Change % is calculated correctly for each month."""
    cac_series = calculate_cac(sample_df)
    change_series = calculate_cac_change(cac_series)

    # Feb: (114.29 - 125.00) / 125.00 * 100 ≈ -8.57%
    assert pytest.approx(change_series.iloc[1], 0.01) == -8.57
    # Mar: (106.25 - 114.29) / 114.29 * 100 ≈ -7.03%
    assert pytest.approx(change_series.iloc[2], 0.01) == -7.03
    # Jun: (90.91 - 95.00) / 95.00 * 100 ≈ -4.31%
    assert pytest.approx(change_series.iloc[5], 0.01) == -4.31


def test_first_month_cac_change_is_nan(sample_df):
    """Test 3: First month has no prior month, so CAC Change % is NaN."""
    cac_series = calculate_cac(sample_df)
    change_series = calculate_cac_change(cac_series)
    assert pd.isna(change_series.iloc[0])


def test_latest_cac_summary_is_correct(sample_df):
    """Test 4: Summary correctly reports the latest month's CAC and change."""
    summary = generate_cac_summary(sample_df)
    assert summary["latest_month"] == "Jun-2025"
    assert pytest.approx(summary["latest_cac"], 0.01) == 90.91
    assert pytest.approx(summary["previous_cac"], 0.01) == 95.00
    assert pytest.approx(summary["latest_cac_change"], 0.01) == -4.31


def test_overall_cac_change_is_correct(sample_df):
    """Test 5: Overall CAC change from first to latest month ≈ -27.27%."""
    summary = generate_cac_summary(sample_df)
    assert summary["starting_cac"] == pytest.approx(125.00, 0.01)
    assert summary["overall_cac_change"] == pytest.approx(-27.27, 0.01)
    assert summary["interpretation"] == "Customer acquisition efficiency improved."


def test_zero_new_customers_no_division_error():
    """Test 6: NewCustomers = 0 must return NaN without raising an error."""
    df_zero = pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025"],
        "NewCustomers": [0, 100],
        "ChurnedCustomers": [0, 10],
        "MRR": [0, 50000],
        "MarketingSpend": [5000, 10000],
    })
    cac_series = calculate_cac(df_zero)
    assert pd.isna(cac_series.iloc[0])      # Jan: 5000 / 0 → NaN
    assert pytest.approx(cac_series.iloc[1], 0.01) == 100.0


def test_zero_previous_cac_does_not_crash():
    """Test 7: Previous CAC = 0 must produce NaN in change %, not a crash."""
    cac_series = pd.Series([0.0, 100.0])
    change_series = calculate_cac_change(cac_series)
    assert pd.isna(change_series.iloc[1])   # 0 as denominator → NaN


def test_single_row_dataset(sample_df):
    """Test 8: A single-row dataset computes CAC normally; CAC Change is NaN."""
    single_row = sample_df.iloc[:1].reset_index(drop=True)
    metrics_df = calculate_cac_metrics(single_row)
    assert len(metrics_df) == 1
    assert pytest.approx(metrics_df["CAC"].iloc[0], 0.01) == 125.00
    assert pd.isna(metrics_df["CACChangePct"].iloc[0])

    summary = generate_cac_summary(single_row)
    assert summary["previous_cac"] is None
    assert summary["latest_cac_change"] is None


def test_input_dataframe_not_mutated(sample_df):
    """Test 9: calculate_cac_metrics must not add columns to the original DataFrame."""
    original_cols = list(sample_df.columns)
    _ = calculate_cac_metrics(sample_df)
    assert list(sample_df.columns) == original_cols


def test_full_sample_dataset_produces_expected_cac(sample_df):
    """Test 10: Full 6-month dataset returns the complete expected CAC series."""
    metrics_df = calculate_cac_metrics(sample_df)
    expected_cac = [125.00, 114.29, 106.25, 100.00, 95.00, 90.91]
    for i, expected in enumerate(expected_cac):
        assert pytest.approx(metrics_df["CAC"].iloc[i], 0.01) == expected
