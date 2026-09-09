"""
Customer Acquisition Cost (CAC) Calculator Module
=============================================================================
Calculates how much marketing spend is required to acquire each new customer
and tracks whether acquisition efficiency improves over time.

Formula:
    CAC = MarketingSpend / NewCustomers

CAC Month-over-Month Change %:
    CAC Change % = ((Current CAC - Previous CAC) / Previous CAC) * 100

A declining CAC trend is generally positive — it means the business is
acquiring customers more efficiently over time.
=============================================================================
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


# =============================================================================
# CAC Calculation Functions
# =============================================================================

def calculate_cac(df: pd.DataFrame) -> pd.Series:
    """
    Calculates Customer Acquisition Cost (CAC) for each month.
    CAC = MarketingSpend / NewCustomers

    Edge Cases:
    - NewCustomers = 0: Returns NaN (dividing by zero has no meaningful result).

    Args:
        df: Validated DataFrame containing 'MarketingSpend' and 'NewCustomers' columns.

    Returns:
        pd.Series: CAC values (float or NaN).
    """
    required = {"MarketingSpend", "NewCustomers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")

    # Replace 0 in NewCustomers with NaN before dividing to avoid ZeroDivisionError
    # (acquiring 0 customers with non-zero marketing spend is mathematically undefined as unit cost)
    new_customers_safe = df["NewCustomers"].replace(0, np.nan)
    return df["MarketingSpend"] / new_customers_safe


def calculate_cac_change(cac_series: pd.Series) -> pd.Series:
    """
    Calculates Month-over-Month (MoM) CAC Change Percentage.
    CAC Change % = ((Current CAC - Previous CAC) / Previous CAC) * 100

    Edge Cases:
    - First month: Returns NaN (no preceding month to compare against).
    - Previous CAC = 0: Returns NaN to prevent division by zero.

    Args:
        cac_series: pd.Series of calculated CAC values in chronological order.

    Returns:
        pd.Series: MoM CAC change rates in percent (float or NaN).
    """
    # Shift previous CAC value for period-over-period comparison
    previous_cac = cac_series.shift(1)

    # Safe division: NaN where previous_cac is 0 or NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        change_series = ((cac_series - previous_cac) / previous_cac) * 100.0

    # Replace any inf/-inf (from 0-division) with NaN
    change_series = change_series.replace([np.inf, -np.inf], np.nan)
    return change_series


def calculate_cac_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces an enriched DataFrame with CAC and MoM CAC Change % columns.
    Does NOT mutate the original input DataFrame.

    Args:
        df: Validated financial DataFrame.

    Returns:
        pd.DataFrame: Copy of DataFrame with added 'CAC' and 'CACChangePct' columns.
    """
    # Work on a defensive copy
    cac_df = df.copy()
    cac_df["CAC"] = calculate_cac(cac_df)
    cac_df["CACChangePct"] = calculate_cac_change(cac_df["CAC"])
    return cac_df


def generate_cac_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a concise management-level CAC summary for the reporting period.

    Args:
        df: Validated financial DataFrame.

    Returns:
        dict: Summary containing latest/previous/starting CAC and overall efficiency trend.
    """
    if df.empty:
        raise ValueError("Cannot generate CAC summary from an empty DataFrame.")

    metrics_df = calculate_cac_metrics(df)

    first_row = metrics_df.iloc[0]
    latest_row = metrics_df.iloc[-1]

    starting_cac: Optional[float] = (
        float(first_row["CAC"]) if pd.notnull(first_row["CAC"]) else None
    )
    latest_cac: Optional[float] = (
        float(latest_row["CAC"]) if pd.notnull(latest_row["CAC"]) else None
    )

    # Overall CAC change: (Latest - Starting) / Starting * 100
    if starting_cac is not None and starting_cac != 0 and latest_cac is not None:
        overall_cac_change = ((latest_cac - starting_cac) / starting_cac) * 100.0
    else:
        overall_cac_change = None

    # Determine efficiency interpretation:
    # A negative change in CAC means acquisition costs went down (improved efficiency)
    if overall_cac_change is None:
        interpretation = "Overall CAC change cannot be determined."
    elif overall_cac_change < 0:
        interpretation = "Customer acquisition efficiency improved."
    elif overall_cac_change > 0:
        interpretation = "Customer acquisition cost increased."
    else:
        interpretation = "CAC remained stable."

    # Previous month stats (only available when >= 2 months of data exist)
    if len(metrics_df) > 1:
        prev_row = metrics_df.iloc[-2]
        previous_cac: Optional[float] = (
            float(prev_row["CAC"]) if pd.notnull(prev_row["CAC"]) else None
        )
        latest_change_val = latest_row["CACChangePct"]
        latest_cac_change: Optional[float] = (
            float(latest_change_val) if pd.notnull(latest_change_val) else None
        )
    else:
        previous_cac = None
        latest_cac_change = None

    return {
        "starting_month": str(first_row["Month"]),
        "starting_cac": starting_cac,
        "latest_month": str(latest_row["Month"]),
        "latest_marketing_spend": float(latest_row["MarketingSpend"]),
        "latest_new_customers": int(latest_row["NewCustomers"]),
        "latest_cac": latest_cac,
        "previous_cac": previous_cac,
        "latest_cac_change": latest_cac_change,
        "overall_cac_change": overall_cac_change,
        "interpretation": interpretation,
        "total_months": len(metrics_df),
    }


def format_cac_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats numeric CAC metrics into readable string representations
    for terminal display.

    Args:
        df: DataFrame produced by calculate_cac_metrics().

    Returns:
        pd.DataFrame: Formatted DataFrame ready for display.
    """
    display_df = pd.DataFrame()
    display_df["Month"] = df["Month"]
    display_df["New Customers"] = df["NewCustomers"].apply(lambda x: f"{int(x):,}")
    display_df["Marketing Spend"] = df["MarketingSpend"].apply(lambda x: f"${x:,.0f}")
    display_df["CAC"] = df["CAC"].apply(
        lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A"
    )
    display_df["CAC Change %"] = df["CACChangePct"].apply(
        lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
    )
    return display_df
