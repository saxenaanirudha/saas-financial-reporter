"""
Data Loader & Validation Module
=============================================================================
This module handles loading SaaS financial data from CSV files and performing
strict data validation to ensure consistency and correctness for downstream
analytics and financial reporting.
=============================================================================
"""

from pathlib import Path
from typing import Union
import pandas as pd


class DataValidationError(Exception):
    """Custom exception raised when financial data fails validation rules."""
    pass


# Define the columns strictly required for financial reporting
REQUIRED_COLUMNS = [
    "Month",
    "NewCustomers",
    "ChurnedCustomers",
    "MRR",
    "MarketingSpend",
    "OperatingExpenses",
    "PayrollExpenses",
    "OtherExpenses",
]

# Numeric columns that must contain non-negative numbers
NUMERIC_COLUMNS = [
    "NewCustomers",
    "ChurnedCustomers",
    "MRR",
    "MarketingSpend",
    "OperatingExpenses",
    "PayrollExpenses",
    "OtherExpenses",
]


def validate_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates the structure, types, and values of the SaaS financial DataFrame.

    Validation Rules:
    1. The DataFrame must not be empty.
    2. All required columns must be present.
    3. No null/missing values are allowed.
    4. Numeric columns must contain valid, non-negative numbers (>= 0).
    5. The 'Month' column must contain valid date formats (e.g., 'Jan-2025').
    6. No duplicate month entries are allowed.
    7. Data is sorted chronologically.

    Args:
        df: Input pandas DataFrame to validate.

    Returns:
        pd.DataFrame: Validated and chronologically sorted DataFrame with a clean index.

    Raises:
        DataValidationError: If any validation rule fails.
    """
    if df is None or df.empty:
        raise DataValidationError("Validation failed: The provided dataset is empty.")

    # 1. Verify required columns exist
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise DataValidationError(
            f"Validation failed: Missing required column(s): {', '.join(missing_columns)}"
        )

    # Work on a copy to avoid mutating the original DataFrame unexpectedly
    df_clean = df.copy()

    # 2. Check for missing/null values in required columns
    null_counts = df_clean[REQUIRED_COLUMNS].isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        null_details = ", ".join(f"{col} ({count} nulls)" for col, count in columns_with_nulls.items())
        raise DataValidationError(
            f"Validation failed: Dataset contains missing/null values in: {null_details}"
        )

    # 3. Validate numeric columns (must be numeric and non-negative)
    for col in NUMERIC_COLUMNS:
        # Convert to numeric, turning invalid values into NaN for easy detection
        numeric_series = pd.to_numeric(df_clean[col], errors="coerce")
        if numeric_series.isnull().any():
            raise DataValidationError(
                f"Validation failed: Column '{col}' contains non-numeric values."
            )

        # Check for negative numbers
        if (numeric_series < 0).any():
            negative_rows = df_clean[numeric_series < 0][col].tolist()
            raise DataValidationError(
                f"Validation failed: Column '{col}' contains negative values: {negative_rows}. "
                "All financial and customer counts must be >= 0."
            )

        df_clean[col] = numeric_series

    # 4. Validate 'Month' column & parse dates
    try:
        # Accepts 'Jan-2025', 'Feb-2025', etc.
        parsed_dates = pd.to_datetime(df_clean["Month"], format="%b-%Y", errors="raise")
    except Exception:
        # If standard %b-%Y fails, try flexible date parsing
        try:
            parsed_dates = pd.to_datetime(df_clean["Month"], format="mixed", errors="coerce")
            if parsed_dates.isnull().any():
                raise DataValidationError(
                    "Validation failed: Column 'Month' contains invalid date formats. "
                    "Expected format like 'Jan-2025' or 'YYYY-MM'."
                )
        except Exception as err:
            raise DataValidationError(
                f"Validation failed: Column 'Month' contains invalid date formats ({err})."
            )

    # 5. Check for duplicate months
    if parsed_dates.duplicated().any():
        duplicate_months = df_clean["Month"][parsed_dates.duplicated()].tolist()
        raise DataValidationError(
            f"Validation failed: Dataset contains duplicate month entries: {duplicate_months}"
        )

    # 6. Sort data chronologically and reset index
    df_clean["_ParsedDate"] = parsed_dates
    df_clean = df_clean.sort_values(by="_ParsedDate").drop(columns=["_ParsedDate"]).reset_index(drop=True)

    return df_clean


def load_financial_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads and validates SaaS financial data from a CSV file.

    Args:
        file_path: Path to the CSV file (string or Path object).

    Returns:
        pd.DataFrame: Validated and cleaned pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        DataValidationError: If the CSV is empty, unreadable, or fails validation.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Financial data file not found at: {path.resolve()}")

    if not path.is_file():
        raise DataValidationError(f"Specified path is not a file: {path.resolve()}")

    try:
        df = pd.read_csv(path)
    except Exception as err:
        raise DataValidationError(f"Failed to read CSV file '{path.name}': {err}")

    return validate_financial_data(df)
