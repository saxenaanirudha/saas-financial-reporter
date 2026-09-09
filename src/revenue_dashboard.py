"""
Revenue Dashboard Module
=============================================================================
Calculates key SaaS recurring revenue metrics from validated financial data:
- Monthly Recurring Revenue (MRR)
- Annual Recurring Revenue (ARR = MRR * 12)
- Month-over-Month (MoM) MRR Growth Rate (%)
- Overall Revenue Movement & Executive Summary
=============================================================================
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


# =============================================================================
# Recurring Revenue Calculation Functions
# =============================================================================

def calculate_arr(df: pd.DataFrame) -> pd.Series:
    """
    Calculates Annual Recurring Revenue (ARR).
    ARR = MRR * 12

    Args:
        df: DataFrame containing the 'MRR' column.

    Returns:
        pd.Series: Calculated ARR values.
    """
    if "MRR" not in df.columns:
        raise ValueError("DataFrame must contain 'MRR' column to calculate ARR.")
    
    # ARR is annualized MRR (12-month extrapolation)
    return df["MRR"] * 12


def calculate_mrr_growth(df: pd.DataFrame) -> pd.Series:
    """
    Calculates Month-over-Month (MoM) MRR Growth Percentage.
    Growth % = ((Current MRR - Previous MRR) / Previous MRR) * 100

    Edge Cases:
    - First month: Returns NaN (no preceding month to compare against).
    - Previous MRR is 0: Returns NaN to prevent DivisionByZero errors.

    Args:
        df: DataFrame containing the 'MRR' column, ordered chronologically.

    Returns:
        pd.Series: MoM growth rates in percent (float or NaN).
    """
    if "MRR" not in df.columns:
        raise ValueError("DataFrame must contain 'MRR' column to calculate MRR growth.")

    # Shift MRR by 1 row to align previous month with current month
    previous_mrr = df["MRR"].shift(1)

    # Safe division: if previous_mrr is 0 or NaN, result will be NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        growth_series = ((df["MRR"] - previous_mrr) / previous_mrr) * 100.0

    # Replace infinite values (caused by division by 0) with NaN
    growth_series = growth_series.replace([np.inf, -np.inf], np.nan)

    return growth_series


def calculate_revenue_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces an enriched DataFrame with calculated revenue metrics (ARR and MRR Growth %).
    Does not mutate the original input DataFrame.

    Args:
        df: Validated financial DataFrame.

    Returns:
        pd.DataFrame: Copy of DataFrame containing 'Month', 'MRR', 'ARR', 'MRRGrowthPct'.
    """
    # Work on a defensive copy to prevent side effects on caller DataFrame
    revenue_df = df.copy()
    revenue_df["ARR"] = calculate_arr(revenue_df)
    revenue_df["MRRGrowthPct"] = calculate_mrr_growth(revenue_df)
    return revenue_df


def generate_revenue_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates high-level executive summary metrics from the revenue dataset.

    Args:
        df: Validated financial DataFrame (or revenue metrics DataFrame).

    Returns:
        dict: Summary containing latest month, latest MRR/ARR, growth rates,
              and overall net revenue expansion.
    """
    if df.empty:
        raise ValueError("Cannot generate summary from an empty DataFrame.")

    # Ensure metrics are calculated
    metrics_df = calculate_revenue_metrics(df)

    # Extract boundaries (baseline month vs latest reporting month)
    first_row = metrics_df.iloc[0]
    latest_row = metrics_df.iloc[-1]

    first_mrr = float(first_row["MRR"])
    latest_mrr = float(latest_row["MRR"])
    latest_arr = float(latest_row["ARR"])

    # Net dollar change and overall growth from the very first recorded month
    net_mrr_change = latest_mrr - first_mrr
    if first_mrr > 0:
        overall_mrr_growth = (net_mrr_change / first_mrr) * 100.0
    else:
        overall_mrr_growth = None

    # Previous month stats for period-over-period tracking (if at least 2 months exist)
    if len(metrics_df) > 1:
        prev_row = metrics_df.iloc[-2]
        prev_mrr: Optional[float] = float(prev_row["MRR"])
        latest_growth_val = latest_row["MRRGrowthPct"]
        latest_mrr_growth: Optional[float] = (
            float(latest_growth_val) if pd.notnull(latest_growth_val) else None
        )
    else:
        prev_mrr = None
        latest_mrr_growth = None

    return {
        "starting_month": str(first_row["Month"]),
        "starting_mrr": first_mrr,
        "latest_month": str(latest_row["Month"]),
        "latest_mrr": latest_mrr,
        "latest_arr": latest_arr,
        "previous_mrr": prev_mrr,
        "latest_mrr_growth": latest_mrr_growth,
        "net_mrr_change": net_mrr_change,
        "overall_mrr_growth": overall_mrr_growth,
        "total_months": len(metrics_df)
    }


def format_revenue_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats numeric revenue metrics into investor-friendly string representations
    (currency commas and percentage symbols) for terminal/report display.

    Args:
        df: DataFrame produced by calculate_revenue_metrics.

    Returns:
        pd.DataFrame: Formatted DataFrame ready for presentation.
    """
    display_df = pd.DataFrame()
    display_df["Month"] = df["Month"]
    display_df["MRR"] = df["MRR"].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A")
    display_df["ARR"] = df["ARR"].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A")
    display_df["MRR Growth %"] = df["MRRGrowthPct"].apply(
        lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
    )
    return display_df
