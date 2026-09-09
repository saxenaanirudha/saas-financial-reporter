"""
Unit tests for SaaS data loader and validation module (Step 2 & Step 8).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import (
    load_financial_data,
    validate_financial_data,
    DataValidationError,
    REQUIRED_COLUMNS,
)


@pytest.fixture
def sample_valid_df():
    """Returns a valid SaaS financial DataFrame with enhanced expense schema."""
    return pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025", "Mar-2025"],
        "NewCustomers": [120, 140, 160],
        "ChurnedCustomers": [15, 20, 18],
        "MRR": [60000, 68000, 75000],
        "MarketingSpend": [15000, 16000, 17000],
        "OperatingExpenses": [12000, 12500, 13000],
        "PayrollExpenses": [20000, 20500, 21000],
        "OtherExpenses": [3000, 3200, 3500],
    })


def test_valid_dataset_loads_successfully(sample_valid_df):
    """Test 1: Valid dataset passes validation and returns correct row count."""
    df_cleaned = validate_financial_data(sample_valid_df)
    assert len(df_cleaned) == 3
    assert list(df_cleaned.columns) == REQUIRED_COLUMNS


def test_missing_required_column_is_rejected(sample_valid_df):
    """Test 2: Missing required column raises DataValidationError."""
    df_missing_col = sample_valid_df.drop(columns=["MRR"])
    with pytest.raises(DataValidationError, match="Missing required column"):
        validate_financial_data(df_missing_col)


def test_missing_expense_column_is_rejected(sample_valid_df):
    """Test 3: Missing new expense column (e.g. OperatingExpenses) raises DataValidationError."""
    df_missing_expense = sample_valid_df.drop(columns=["OperatingExpenses"])
    with pytest.raises(DataValidationError, match="Missing required column.*OperatingExpenses"):
        validate_financial_data(df_missing_expense)


def test_missing_null_value_is_rejected(sample_valid_df):
    """Test 4: Missing/null values raise DataValidationError."""
    df_with_null = sample_valid_df.copy()
    df_with_null.loc[1, "MarketingSpend"] = None
    with pytest.raises(DataValidationError, match="missing/null values"):
        validate_financial_data(df_with_null)


def test_null_expense_value_is_rejected(sample_valid_df):
    """Test 5: Null values in expense fields raise DataValidationError."""
    df_with_null = sample_valid_df.copy()
    df_with_null.loc[0, "PayrollExpenses"] = None
    with pytest.raises(DataValidationError, match="missing/null values"):
        validate_financial_data(df_with_null)


def test_negative_numeric_value_is_rejected(sample_valid_df):
    """Test 6: Negative numeric values raise DataValidationError."""
    df_with_negative = sample_valid_df.copy()
    df_with_negative.loc[0, "NewCustomers"] = -10
    with pytest.raises(DataValidationError, match="negative values"):
        validate_financial_data(df_with_negative)


def test_negative_expense_value_is_rejected(sample_valid_df):
    """Test 7: Negative expense values raise DataValidationError."""
    df_with_neg_expense = sample_valid_df.copy()
    df_with_neg_expense.loc[0, "OtherExpenses"] = -500
    with pytest.raises(DataValidationError, match="negative values"):
        validate_financial_data(df_with_neg_expense)


def test_duplicate_month_is_rejected(sample_valid_df):
    """Test 8: Duplicate months raise DataValidationError."""
    df_duplicates = pd.DataFrame({
        "Month": ["Jan-2025", "Jan-2025"],
        "NewCustomers": [120, 130],
        "ChurnedCustomers": [15, 10],
        "MRR": [60000, 62000],
        "MarketingSpend": [15000, 15500],
        "OperatingExpenses": [12000, 12500],
        "PayrollExpenses": [20000, 20500],
        "OtherExpenses": [3000, 3200],
    })
    with pytest.raises(DataValidationError, match="duplicate month"):
        validate_financial_data(df_duplicates)


def test_invalid_month_format_is_rejected(sample_valid_df):
    """Test 9: Invalid month date format raises DataValidationError."""
    df_invalid_date = sample_valid_df.copy()
    df_invalid_date.loc[0, "Month"] = "NotAMonth"
    with pytest.raises(DataValidationError, match="invalid date"):
        validate_financial_data(df_invalid_date)


def test_missing_file_is_handled_correctly(tmp_path):
    """Test 10: Attempting to load a non-existent file raises FileNotFoundError."""
    non_existent_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_financial_data(non_existent_file)


def test_chronological_sorting():
    """Test 11: Unordered months are sorted chronologically."""
    df_unsorted = pd.DataFrame({
        "Month": ["Mar-2025", "Jan-2025", "Feb-2025"],
        "NewCustomers": [160, 120, 140],
        "ChurnedCustomers": [18, 15, 20],
        "MRR": [75000, 60000, 68000],
        "MarketingSpend": [17000, 15000, 16000],
        "OperatingExpenses": [13000, 12000, 12500],
        "PayrollExpenses": [21000, 20000, 20500],
        "OtherExpenses": [3500, 3000, 3200],
    })
    df_sorted = validate_financial_data(df_unsorted)
    assert df_sorted["Month"].tolist() == ["Jan-2025", "Feb-2025", "Mar-2025"]
