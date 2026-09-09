"""
Unit Tests for src/chart_generator.py
=============================================================================
Tests financial chart generation, file outputs, automatic directory creation,
empty DataFrame handling, NaN safety, and non-empty PNG file outputs.
"""

import pytest
from pathlib import Path
import pandas as pd
from src.chart_generator import (
    generate_mrr_chart,
    generate_arr_chart,
    generate_cac_chart,
    generate_churn_retention_chart,
    generate_customer_base_chart,
    generate_mrr_growth_chart,
    generate_all_charts,
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


def test_generate_mrr_chart(sample_df, tmp_path):
    """
    Test 1: MRR chart is created successfully and is non-empty.
    """
    chart_path = generate_mrr_chart(sample_df, tmp_path)

    assert chart_path.exists()
    assert chart_path.name == "mrr_trend.png"
    assert chart_path.stat().st_size > 0


def test_generate_arr_chart(sample_df, tmp_path):
    """
    Test 2: ARR chart is created successfully and is non-empty.
    """
    chart_path = generate_arr_chart(sample_df, tmp_path)

    assert chart_path.exists()
    assert chart_path.name == "arr_trend.png"
    assert chart_path.stat().st_size > 0


def test_generate_cac_chart(sample_df, tmp_path):
    """
    Test 3: CAC chart is created successfully and is non-empty.
    """
    chart_path = generate_cac_chart(sample_df, tmp_path)

    assert chart_path.exists()
    assert chart_path.name == "cac_trend.png"
    assert chart_path.stat().st_size > 0


def test_generate_churn_retention_chart(sample_df, tmp_path):
    """
    Test 4 & 11: Churn/Retention chart is created safely even with NaN first-month values.
    """
    chart_path = generate_churn_retention_chart(sample_df, tmp_path)

    assert chart_path.exists()
    assert chart_path.name == "churn_retention_trend.png"
    assert chart_path.stat().st_size > 0


def test_generate_customer_base_chart(sample_df, tmp_path):
    """
    Test 5: Customer Base Growth chart is created successfully.
    """
    chart_path = generate_customer_base_chart(sample_df, tmp_path)

    assert chart_path.exists()
    assert chart_path.name == "customer_base_growth.png"
    assert chart_path.stat().st_size > 0


def test_generate_mrr_growth_chart(sample_df, tmp_path):
    """
    Test 6: MRR Growth Rate chart is created successfully.
    """
    chart_path = generate_mrr_growth_chart(sample_df, tmp_path)

    assert chart_path.exists()
    assert chart_path.name == "mrr_growth_rate.png"
    assert chart_path.stat().st_size > 0


def test_generate_all_charts(sample_df, tmp_path):
    """
    Test 7 & 13: generate_all_charts creates all 6 PNG files and all are non-empty.
    """
    chart_dict = generate_all_charts(sample_df, tmp_path)

    expected_keys = [
        "mrr_trend",
        "arr_trend",
        "cac_trend",
        "churn_retention_trend",
        "customer_base_growth",
        "mrr_growth_rate",
    ]

    for key in expected_keys:
        assert key in chart_dict
        path = chart_dict[key]
        assert path.exists()
        assert path.stat().st_size > 0


def test_output_directory_created_automatically(sample_df, tmp_path):
    """
    Test 8: Output directory is automatically created if it does not exist.
    """
    nested_dir = tmp_path / "nested" / "sub_charts"
    assert not nested_dir.exists()

    generate_all_charts(sample_df, nested_dir)
    assert nested_dir.exists()


def test_empty_dataframe_raises_value_error(tmp_path):
    """
    Test 9: Passing an empty DataFrame raises a clear ValueError.
    """
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError) as excinfo:
        generate_mrr_chart(empty_df, tmp_path)

    assert "Cannot generate financial charts from an empty DataFrame" in str(excinfo.value)


def test_missing_required_column_raises_error(tmp_path):
    """
    Test 10: Missing foundational raw columns raises ValueError from calculation layer.
    """
    invalid_df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "MRR": [60000],
        # Missing NewCustomers, ChurnedCustomers, MarketingSpend
    })

    with pytest.raises(ValueError):
        generate_all_charts(invalid_df, tmp_path)


def test_input_dataframe_not_mutated(sample_df, tmp_path):
    """
    Test 12: Input DataFrame is not mutated during chart generation.
    """
    original_cols = list(sample_df.columns)
    generate_all_charts(sample_df, tmp_path)

    assert list(sample_df.columns) == original_cols
    assert "BeginningCustomers" not in sample_df.columns
