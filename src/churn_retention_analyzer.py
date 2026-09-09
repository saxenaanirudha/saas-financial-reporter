"""
Customer Churn & Retention Analyzer Module
=============================================================================
Calculates customer base balances, churn rates, retention rates, and
month-over-month churn movement (in percentage points) across reporting periods.

Customer Base Roll-Forward Logic:
    Beginning Customers (Month 1) = 0
    Ending Customers = Beginning Customers + New Customers - Churned Customers
    Beginning Customers (Month N) = Ending Customers (Month N-1)

Churn & Retention Formulas:
    Churn Rate % = (Churned Customers / Beginning Customers) * 100
    Retention Rate % = 100 - Churn Rate %
    Churn Change (pp) = Current Churn Rate % - Previous Available Churn Rate %

Limitations:
    Works on aggregated monthly totals. In production, subscriber-level logs
    or explicit opening/closing balance records are preferred.
=============================================================================
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


# =============================================================================
# Customer Base Roll-Forward & Retention Calculation Functions
# =============================================================================

def calculate_customer_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstructs monthly Beginning and Ending customer balances chronologically.

    Roll-forward formula:
        Month 1 Beginning = 0
        Month N Beginning = Month N-1 Ending
        Ending = Beginning + NewCustomers - ChurnedCustomers

    Raises:
        ValueError: If required columns are missing or if ChurnedCustomers
                    exceeds available customers (Beginning + New).

    Args:
        df: Validated financial DataFrame containing 'NewCustomers' and 'ChurnedCustomers'.

    Returns:
        pd.DataFrame: Copy of DataFrame with added 'BeginningCustomers' and 'EndingCustomers'.
    """
    required = {"NewCustomers", "ChurnedCustomers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")

    data = df.copy()
    num_rows = len(data)

    beginning_customers = np.zeros(num_rows, dtype=int)
    ending_customers = np.zeros(num_rows, dtype=int)

    # Invariant: Month 1 starts from baseline 0 (reconstructed historical base)
    current_beg = 0

    for i in range(num_rows):
        new_cust = int(data.iloc[i]["NewCustomers"])
        churn_cust = int(data.iloc[i]["ChurnedCustomers"])

        # Maximum subscribers available to churn during the period
        available_cust = current_beg + new_cust

        # Conservation invariant: Churned subscribers cannot exceed total available subscribers
        if churn_cust > available_cust:
            month_label = data.iloc[i].get("Month", f"Row {i+1}")
            raise ValueError(
                f"Churned customers ({churn_cust}) cannot exceed the customers "
                f"available during the month ({available_cust}) at {month_label}."
            )

        current_end = available_cust - churn_cust

        beginning_customers[i] = current_beg
        ending_customers[i] = current_end

        # Roll forward this month's closing customer base into next month's opening balance
        current_beg = current_end

    data["BeginningCustomers"] = beginning_customers
    data["EndingCustomers"] = ending_customers
    return data


def calculate_churn_retention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Churn Rate %, Retention Rate %, and MoM Churn Change (pp).

    Edge Cases:
    - Month 1 (Beginning Customers = 0): Churn Rate = NaN, Retention Rate = NaN.
    - Churn Change (pp) is NaN when current or previous churn rate is unavailable.

    Args:
        df: DataFrame containing customer base columns (or raw validated DataFrame).

    Returns:
        pd.DataFrame: Copy of DataFrame with churn and retention metrics added.
    """
    # Ensure customer base balances are reconstructed first
    if "BeginningCustomers" not in df.columns or "EndingCustomers" not in df.columns:
        data = calculate_customer_base(df)
    else:
        data = df.copy()

    num_rows = len(data)

    churn_rate = np.full(num_rows, np.nan, dtype=float)
    retention_rate = np.full(num_rows, np.nan, dtype=float)

    for i in range(num_rows):
        beg = data.iloc[i]["BeginningCustomers"]
        churned = data.iloc[i]["ChurnedCustomers"]

        # Churn rate is defined relative to the beginning active customer base
        if beg > 0:
            c_rate = (churned / beg) * 100.0
            r_rate = 100.0 - c_rate
            churn_rate[i] = c_rate
            retention_rate[i] = r_rate

    data["ChurnRatePercent"] = churn_rate
    data["RetentionRatePercent"] = retention_rate

    # Calculate month-over-month Churn Change in Percentage Points (pp)
    # (e.g. 5.0% to 4.0% is -1.0 percentage point change, not -20% relative change)
    previous_churn = data["ChurnRatePercent"].shift(1)
    data["ChurnChangePP"] = data["ChurnRatePercent"] - previous_churn

    return data


def calculate_churn_retention_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: calculates customer base balances, churn rates, retention
    rates, and percentage point changes.

    Args:
        df: Validated financial DataFrame.

    Returns:
        pd.DataFrame: Copy of DataFrame enriched with all Step 5 metrics.
    """
    return calculate_churn_retention(df)


def generate_churn_retention_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a concise management-level Churn & Retention summary.

    Args:
        df: Validated financial DataFrame.

    Returns:
        dict: Structured summary dictionary with metric values and business interpretation.
    """
    if df.empty:
        raise ValueError("Cannot generate churn summary from an empty DataFrame.")

    metrics_df = calculate_churn_retention_metrics(df)

    latest_row = metrics_df.iloc[-1]

    latest_churn: Optional[float] = (
        float(latest_row["ChurnRatePercent"])
        if pd.notnull(latest_row["ChurnRatePercent"])
        else None
    )
    latest_retention: Optional[float] = (
        float(latest_row["RetentionRatePercent"])
        if pd.notnull(latest_row["RetentionRatePercent"])
        else None
    )

    # Determine previous available churn rate
    if len(metrics_df) > 1:
        prev_row = metrics_df.iloc[-2]
        prev_churn: Optional[float] = (
            float(prev_row["ChurnRatePercent"])
            if pd.notnull(prev_row["ChurnRatePercent"])
            else None
        )
        latest_change_pp_val = latest_row["ChurnChangePP"]
        latest_churn_change_pp: Optional[float] = (
            float(latest_change_pp_val)
            if pd.notnull(latest_change_pp_val)
            else None
        )
    else:
        prev_churn = None
        latest_churn_change_pp = None

    # Determine business interpretation comparing latest churn vs previous churn
    if latest_churn is None or prev_churn is None:
        interpretation = "Insufficient historical churn data for trend comparison."
    elif latest_churn < prev_churn:
        interpretation = "Latest customer churn improved compared with the previous month."
    elif latest_churn > prev_churn:
        interpretation = "Latest customer churn increased compared with the previous month."
    else:
        interpretation = "Latest customer churn remained stable."

    return {
        "latest_month": str(latest_row["Month"]),
        "beginning_customers": int(latest_row["BeginningCustomers"]),
        "new_customers": int(latest_row["NewCustomers"]),
        "churned_customers": int(latest_row["ChurnedCustomers"]),
        "ending_customers": int(latest_row["EndingCustomers"]),
        "latest_churn_rate": latest_churn,
        "latest_retention_rate": latest_retention,
        "previous_churn_rate": prev_churn,
        "latest_churn_change_pp": latest_churn_change_pp,
        "total_new_customers": int(metrics_df["NewCustomers"].sum()),
        "total_churned_customers": int(metrics_df["ChurnedCustomers"].sum()),
        "latest_customer_base": int(latest_row["EndingCustomers"]),
        "interpretation": interpretation,
    }


def format_churn_retention_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats numeric customer churn and retention metrics into readable strings
    for terminal display.

    Args:
        df: DataFrame produced by calculate_churn_retention_metrics().

    Returns:
        pd.DataFrame: Formatted DataFrame ready for display.
    """
    display_df = pd.DataFrame()
    display_df["Month"] = df["Month"]
    display_df["Beginning"] = df["BeginningCustomers"].astype(str)
    display_df["New"] = df["NewCustomers"].astype(str)
    display_df["Churned"] = df["ChurnedCustomers"].astype(str)
    display_df["Ending"] = df["EndingCustomers"].astype(str)

    display_df["Churn Rate"] = df["ChurnRatePercent"].apply(
        lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
    )
    display_df["Retention Rate"] = df["RetentionRatePercent"].apply(
        lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
    )
    display_df["Churn Change (pp)"] = df["ChurnChangePP"].apply(
        lambda x: f"{x:+.2f} pp" if pd.notnull(x) else "N/A"
    )

    return display_df
