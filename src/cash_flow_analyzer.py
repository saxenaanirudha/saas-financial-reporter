"""
Cash Flow Analyzer Module (Step 8)
=============================================================================
Calculates historical monthly cash inflows, operating cash outflows, net cash
flows, and rolling cash balances for SaaS businesses.

Simplified Cash Flow Modeling Assumptions:
1. Cash Inflow is modeled as Monthly Recurring Revenue (MRR).
2. Cash Outflow consists of the sum of:
   - MarketingSpend
   - OperatingExpenses
   - PayrollExpenses
   - OtherExpenses
3. Net Cash Flow = Cash Inflow - Total Cash Outflow.
4. Cash Balance Roll-Forward:
   - Month 1 Opening Cash Balance = Configured Starting Cash Balance (Default: $100,000).
   - Month N Opening Cash Balance = Month N-1 Closing Cash Balance.
   - Closing Cash Balance = Opening Cash Balance + Net Cash Flow.

Important Modeling Note:
This is a simplified operating cash-flow model designed for portfolio analytics
and financial reporting automation. It is NOT a full GAAP/IFRS-compliant accounting
engine. In enterprise SaaS operations, recognized revenue, invoiced revenue, and
collected cash often differ due to payment terms, deferred revenue, and receivables.

Opening Cash Balance Policy:
Negative opening cash balances are permitted to represent pre-existing debt or
overdraft/deficit positions.
=============================================================================
"""

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd

# Default initial cash position for sample dataset
DEFAULT_OPENING_CASH_BALANCE: float = 100000.0

# Required expense columns for outflow calculations
EXPENSE_COLUMNS = [
    "MarketingSpend",
    "OperatingExpenses",
    "PayrollExpenses",
    "OtherExpenses",
]


def calculate_cash_inflow(df: pd.DataFrame) -> pd.Series:
    """
    Calculates monthly cash inflow from recurring revenue (MRR).

    Args:
        df: DataFrame containing the 'MRR' column.

    Returns:
        pd.Series: Monthly cash inflow series.

    Raises:
        ValueError: If 'MRR' column is missing.
    """
    if "MRR" not in df.columns:
        raise ValueError("DataFrame must contain 'MRR' column to calculate cash inflow.")
    return df["MRR"].astype(float)


def calculate_cash_outflow(df: pd.DataFrame) -> pd.Series:
    """
    Calculates total monthly operating cash outflow by summing all expense categories.

    Total Cash Outflow = MarketingSpend + OperatingExpenses + PayrollExpenses + OtherExpenses

    Args:
        df: DataFrame containing all required expense columns.

    Returns:
        pd.Series: Total monthly cash outflow series.

    Raises:
        ValueError: If any required expense column is missing.
    """
    missing = [col for col in EXPENSE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required expense column(s) for outflow calculation: {', '.join(missing)}"
        )

    outflow = (
        df["MarketingSpend"].astype(float)
        + df["OperatingExpenses"].astype(float)
        + df["PayrollExpenses"].astype(float)
        + df["OtherExpenses"].astype(float)
    )
    return outflow


def calculate_net_cash_flow(df: pd.DataFrame) -> pd.Series:
    """
    Calculates monthly Net Cash Flow.

    Net Cash Flow = Cash Inflow - Total Cash Outflow

    Args:
        df: DataFrame containing 'MRR' and expense columns (or pre-calculated 'CashInflow' and 'TotalCashOutflow').

    Returns:
        pd.Series: Monthly net cash flow series (positive, zero, or negative).
    """
    if "CashInflow" in df.columns and "TotalCashOutflow" in df.columns:
        inflow = df["CashInflow"].astype(float)
        outflow = df["TotalCashOutflow"].astype(float)
    else:
        inflow = calculate_cash_inflow(df)
        outflow = calculate_cash_outflow(df)

    return inflow - outflow


def calculate_cash_balances(
    df: pd.DataFrame,
    opening_cash_balance: float = DEFAULT_OPENING_CASH_BALANCE,
) -> pd.DataFrame:
    """
    Calculates month-by-month Opening and Closing cash balances chronologically.

    Roll-Forward Logic:
        Month 1:
            Opening Cash Balance = Initial Configured Opening Cash Balance
            Closing Cash Balance = Opening Cash Balance + Net Cash Flow
        Month N:
            Opening Cash Balance = Month N-1 Closing Cash Balance
            Closing Cash Balance = Opening Cash Balance + Net Cash Flow

    Args:
        df: DataFrame containing 'NetCashFlow' column (or columns to compute it).
        opening_cash_balance: Initial cash balance at the start of Month 1.

    Returns:
        pd.DataFrame: DataFrame copy with 'OpeningCashBalance' and 'ClosingCashBalance'.
    """
    if df.empty:
        raise ValueError("Cannot calculate cash balances for an empty DataFrame.")

    data = df.copy()

    if "NetCashFlow" not in data.columns:
        data["NetCashFlow"] = calculate_net_cash_flow(data)

    num_rows = len(data)
    opening_balances = np.zeros(num_rows, dtype=float)
    closing_balances = np.zeros(num_rows, dtype=float)

    current_opening = float(opening_cash_balance)

    for i in range(num_rows):
        net_flow = float(data.iloc[i]["NetCashFlow"])
        current_closing = current_opening + net_flow

        opening_balances[i] = current_opening
        closing_balances[i] = current_closing

        # Next month's opening is this month's closing
        current_opening = current_closing

    data["OpeningCashBalance"] = opening_balances
    data["ClosingCashBalance"] = closing_balances

    return data


def analyze_cash_flow(
    df: pd.DataFrame,
    opening_cash_balance: float = DEFAULT_OPENING_CASH_BALANCE,
) -> pd.DataFrame:
    """
    Master function: calculates complete historical cash flow metrics.
    Enriches DataFrame with CashInflow, TotalCashOutflow, NetCashFlow,
    OpeningCashBalance, and ClosingCashBalance without mutating the input DataFrame.

    Args:
        df: Validated financial DataFrame.
        opening_cash_balance: Starting cash balance for the first month.

    Returns:
        pd.DataFrame: Enriched DataFrame with all cash flow columns.
    """
    if df.empty:
        raise ValueError("Cannot analyze cash flow for an empty DataFrame.")

    data = df.copy()
    data["CashInflow"] = calculate_cash_inflow(data)
    data["TotalCashOutflow"] = calculate_cash_outflow(data)
    data["NetCashFlow"] = data["CashInflow"] - data["TotalCashOutflow"]
    data = calculate_cash_balances(data, opening_cash_balance=opening_cash_balance)

    return data


def generate_cash_flow_summary(
    df: pd.DataFrame,
    opening_cash_balance: float = DEFAULT_OPENING_CASH_BALANCE,
) -> Dict[str, Any]:
    """
    Generates high-level executive summary metrics for historical cash flow.

    Args:
        df: Validated financial DataFrame (or enriched cash flow DataFrame).
        opening_cash_balance: Starting cash balance.

    Returns:
        dict: Structured summary dictionary containing:
            - latest_month
            - latest_cash_inflow
            - latest_total_cash_outflow
            - latest_net_cash_flow
            - latest_opening_cash_balance
            - latest_closing_cash_balance
            - starting_cash_balance
            - total_historical_cash_inflow
            - total_historical_cash_outflow
            - cumulative_net_cash_flow
            - latest_cash_position
            - interpretation
            - total_months
    """
    if df.empty:
        raise ValueError("Cannot generate cash flow summary from an empty DataFrame.")

    if (
        "CashInflow" not in df.columns
        or "TotalCashOutflow" not in df.columns
        or "NetCashFlow" not in df.columns
        or "OpeningCashBalance" not in df.columns
        or "ClosingCashBalance" not in df.columns
    ):
        metrics_df = analyze_cash_flow(df, opening_cash_balance=opening_cash_balance)
    else:
        metrics_df = df.copy()

    first_row = metrics_df.iloc[0]
    latest_row = metrics_df.iloc[-1]

    latest_net_flow = float(latest_row["NetCashFlow"])
    total_inflow = float(metrics_df["CashInflow"].sum())
    total_outflow = float(metrics_df["TotalCashOutflow"].sum())
    cumulative_net = float(metrics_df["NetCashFlow"].sum())
    starting_balance = float(first_row["OpeningCashBalance"])
    latest_closing_balance = float(latest_row["ClosingCashBalance"])

    # Neutral interpretation without claiming GAAP profitability
    if latest_net_flow > 0:
        interpretation = "Latest month generated positive operating cash flow under the simplified model."
    elif latest_net_flow < 0:
        interpretation = "Latest month generated negative operating cash flow under the simplified model."
    else:
        interpretation = "Latest month was cash-flow neutral under the simplified model."

    return {
        "latest_month": str(latest_row["Month"]),
        "latest_cash_inflow": float(latest_row["CashInflow"]),
        "latest_total_cash_outflow": float(latest_row["TotalCashOutflow"]),
        "latest_net_cash_flow": latest_net_flow,
        "latest_opening_cash_balance": float(latest_row["OpeningCashBalance"]),
        "latest_closing_cash_balance": latest_closing_balance,
        "starting_cash_balance": starting_balance,
        "total_historical_cash_inflow": total_inflow,
        "total_historical_cash_outflow": total_outflow,
        "cumulative_net_cash_flow": cumulative_net,
        "latest_cash_position": latest_closing_balance,
        "interpretation": interpretation,
        "total_months": len(metrics_df),
    }


def format_cash_flow_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats numeric cash flow metrics into investor-friendly string representations
    for terminal or report display.

    Format:
        Month      Inflow    Outflow    Net Cash Flow    Opening Cash    Closing Cash

    Args:
        df: Enriched cash flow DataFrame.

    Returns:
        pd.DataFrame: Formatted presentation DataFrame.
    """
    display_df = pd.DataFrame()
    display_df["Month"] = df["Month"]
    display_df["Inflow"] = df["CashInflow"].apply(lambda x: f"${x:,.0f}")
    display_df["Outflow"] = df["TotalCashOutflow"].apply(lambda x: f"${x:,.0f}")
    display_df["Net Cash Flow"] = df["NetCashFlow"].apply(
        lambda x: f"+${x:,.0f}" if x > 0 else (f"-${abs(x):,.0f}" if x < 0 else "$0")
    )
    display_df["Opening Cash"] = df["OpeningCashBalance"].apply(
        lambda x: f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"
    )
    display_df["Closing Cash"] = df["ClosingCashBalance"].apply(
        lambda x: f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"
    )
    return display_df
