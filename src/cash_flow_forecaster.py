"""
Cash Flow Forecaster Module (Step 9)
=============================================================================
Forecasts the next 6 months of SaaS cash flow (inflows, expense components,
total operating outflow, net cash flow, and closing cash balances) using
linear regression on historical monthly trends.

Forecasting Methodology:
1. Component-by-Component Forecasting:
   - Historical Cash Inflow (MRR)
   - Marketing Spend
   - Operating Expenses
   - Payroll Expenses
   - Other Expenses
   Each component is fitted individually against chronological month sequence indices (1..N).
2. Derived Totals:
   - Forecast Total Outflow = Sum of forecast expense components
   - Forecast Net Cash Flow = Forecast Cash Inflow - Forecast Total Outflow
3. Rolling Cash Balances:
   - Month 1 Forecast Opening Cash = Last Historical Closing Cash Balance
   - Month N Forecast Opening Cash = Month N-1 Forecast Closing Cash Balance
   - Forecast Closing Cash = Forecast Opening Cash + Forecast Net Cash Flow
4. Non-Negative Floor:
   - Revenue and expense forecast components are floored at 0 (minimum 0).
   - Net Cash Flow and Closing Cash Balance are NOT floored (they can be negative).

Limitations & Assumptions:
- Uses simple linear trend regression.
- Assumes historical linear patterns continue into the future.
- Does NOT account for seasonality, pricing adjustments, market shocks, or lump-sum funding.
- This is an illustrative portfolio forecast, NOT a guarantee of future financial performance.
=============================================================================
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.cash_flow_analyzer import (
    analyze_cash_flow,
    DEFAULT_OPENING_CASH_BALANCE,
    EXPENSE_COLUMNS,
)

# Standard forecast horizon (months)
DEFAULT_FORECAST_HORIZON: int = 6


def generate_future_month_labels(latest_month_str: str, horizon: int = DEFAULT_FORECAST_HORIZON) -> List[str]:
    """
    Programmatically generates future calendar month labels (format '%b-%Y')
    starting immediately after the latest historical month.

    Handles year rollovers correctly (e.g., 'Dec-2025' -> 'Jan-2026').

    Args:
        latest_month_str: Latest historical month label (e.g., 'Jun-2025').
        horizon: Number of future months to generate.

    Returns:
        List[str]: List of formatted future month strings.
    """
    try:
        start_date = pd.to_datetime(latest_month_str, format="%b-%Y")
    except Exception:
        start_date = pd.to_datetime(latest_month_str)

    future_dates = [start_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
    return [d.strftime("%b-%Y") for d in future_dates]


def forecast_series(
    historical_values: np.ndarray,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    floor_at_zero: bool = True,
) -> np.ndarray:
    """
    Fits a linear regression model to historical numeric series and predicts
    the next `horizon` values.

    Args:
        historical_values: 1D numpy array or pd.Series of historical numbers (len >= 2).
        horizon: Number of future periods to forecast.
        floor_at_zero: If True, clamps negative predictions to 0.0.

    Returns:
        np.ndarray: Predicted values for the future periods.

    Raises:
        ValueError: If fewer than 2 historical points are provided.
    """
    n = len(historical_values)
    if n < 2:
        raise ValueError(
            f"At least 2 historical data points are required to fit a linear regression trend (got {n})."
        )

    # Independent variable: Chronological Month index 1..N
    X_train = np.arange(1, n + 1).reshape(-1, 1)
    y_train = np.asarray(historical_values, dtype=float)

    # Future independent variable: (N+1)..(N+horizon)
    X_future = np.arange(n + 1, n + horizon + 1).reshape(-1, 1)

    # Fit Ordinary Least Squares (OLS) linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future values based on historical trajectory
    predictions = model.predict(X_future)

    # Clamping policy: Revenue and operational expenses cannot be negative in the real world
    if floor_at_zero:
        predictions = np.maximum(0.0, predictions)

    return predictions


def build_cash_flow_forecast(
    df: pd.DataFrame,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    opening_cash_balance: float = DEFAULT_OPENING_CASH_BALANCE,
) -> pd.DataFrame:
    """
    Generates a 6-month cash flow forecast by forecasting individual components
    and rolling forward cash balances.

    Args:
        df: Validated historical financial DataFrame.
        horizon: Number of future months to forecast (default 6).
        opening_cash_balance: Starting cash balance for historical analysis.

    Returns:
        pd.DataFrame: Forecast DataFrame containing:
            - Month
            - ForecastCashInflow
            - ForecastMarketingSpend
            - ForecastOperatingExpenses
            - ForecastPayrollExpenses
            - ForecastOtherExpenses
            - ForecastTotalCashOutflow
            - ForecastNetCashFlow
            - ForecastOpeningCashBalance
            - ForecastClosingCashBalance

    Raises:
        ValueError: If df is empty, has < 2 rows, or is missing required columns.
    """
    if df is None or df.empty:
        raise ValueError("Cannot generate cash flow forecast from an empty DataFrame.")

    if len(df) < 2:
        raise ValueError(
            f"At least 2 historical months are required to forecast cash flow (got {len(df)})."
        )

    # Ensure historical cash flow metrics and closing balance are available
    if "ClosingCashBalance" not in df.columns or "CashInflow" not in df.columns:
        hist_df = analyze_cash_flow(df, opening_cash_balance=opening_cash_balance)
    else:
        hist_df = df.copy()

    latest_hist_month = str(hist_df["Month"].iloc[-1])
    last_closing_cash = float(hist_df["ClosingCashBalance"].iloc[-1])

    # 1. Generate future month labels
    future_months = generate_future_month_labels(latest_hist_month, horizon=horizon)

    # 2. Forecast each component separately
    # Cash Inflow (from MRR or CashInflow)
    inflow_series = hist_df["CashInflow"] if "CashInflow" in hist_df.columns else hist_df["MRR"]
    forecast_inflow = forecast_series(inflow_series.values, horizon=horizon, floor_at_zero=True)

    # Expense components
    forecast_expenses = {}
    for col in EXPENSE_COLUMNS:
        if col not in hist_df.columns:
            raise ValueError(f"Missing required historical column '{col}' for cash flow forecasting.")
        forecast_expenses[col] = forecast_series(hist_df[col].values, horizon=horizon, floor_at_zero=True)

    # 3. Derive Total Outflow and Net Cash Flow
    forecast_total_outflow = (
        forecast_expenses["MarketingSpend"]
        + forecast_expenses["OperatingExpenses"]
        + forecast_expenses["PayrollExpenses"]
        + forecast_expenses["OtherExpenses"]
    )

    forecast_net_cash_flow = forecast_inflow - forecast_total_outflow

    # 4. Roll forward forecast cash balances
    forecast_opening_balances = np.zeros(horizon, dtype=float)
    forecast_closing_balances = np.zeros(horizon, dtype=float)

    current_opening = last_closing_cash

    for i in range(horizon):
        net_flow = forecast_net_cash_flow[i]
        current_closing = current_opening + net_flow

        forecast_opening_balances[i] = current_opening
        forecast_closing_balances[i] = current_closing

        current_opening = current_closing

    # Assemble forecast DataFrame
    forecast_df = pd.DataFrame({
        "Month": future_months,
        "ForecastCashInflow": forecast_inflow,
        "ForecastMarketingSpend": forecast_expenses["MarketingSpend"],
        "ForecastOperatingExpenses": forecast_expenses["OperatingExpenses"],
        "ForecastPayrollExpenses": forecast_expenses["PayrollExpenses"],
        "ForecastOtherExpenses": forecast_expenses["OtherExpenses"],
        "ForecastTotalCashOutflow": forecast_total_outflow,
        "ForecastNetCashFlow": forecast_net_cash_flow,
        "ForecastOpeningCashBalance": forecast_opening_balances,
        "ForecastClosingCashBalance": forecast_closing_balances,
    })

    return forecast_df


def generate_forecast_summary(forecast_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates an executive summary of the 6-month cash flow forecast.

    Args:
        forecast_df: DataFrame produced by build_cash_flow_forecast().

    Returns:
        dict: Summary metrics and neutral business trend interpretation.
    """
    if forecast_df is None or forecast_df.empty:
        raise ValueError("Cannot generate summary from an empty forecast DataFrame.")

    first_row = forecast_df.iloc[0]
    last_row = forecast_df.iloc[-1]

    starting_cash = float(first_row["ForecastOpeningCashBalance"])
    ending_cash = float(last_row["ForecastClosingCashBalance"])
    lowest_cash = float(forecast_df["ForecastClosingCashBalance"].min())
    highest_cash = float(forecast_df["ForecastClosingCashBalance"].max())

    # Formulate neutral trend interpretation
    cash_diff = ending_cash - starting_cash
    if cash_diff > 1.0:
        interpretation = "Forecast indicates an increasing cash position under the historical trend assumptions."
    elif cash_diff < -1.0:
        interpretation = "Forecast indicates a declining cash position under the historical trend assumptions."
    else:
        interpretation = "Forecast indicates a broadly stable cash position under the historical trend assumptions."

    return {
        "forecast_start_month": str(first_row["Month"]),
        "forecast_end_month": str(last_row["Month"]),
        "first_forecast_cash_inflow": float(first_row["ForecastCashInflow"]),
        "last_forecast_cash_inflow": float(last_row["ForecastCashInflow"]),
        "first_forecast_net_cash_flow": float(first_row["ForecastNetCashFlow"]),
        "last_forecast_net_cash_flow": float(last_row["ForecastNetCashFlow"]),
        "starting_forecast_cash_balance": starting_cash,
        "ending_forecast_cash_balance": ending_cash,
        "lowest_forecast_closing_cash_balance": lowest_cash,
        "highest_forecast_closing_cash_balance": highest_cash,
        "forecast_months": len(forecast_df),
        "interpretation": interpretation,
    }


def format_cash_flow_forecast_table(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats numeric forecast metrics into investor-friendly string representations
    for terminal or report display.

    Format:
        Month      Inflow    Outflow    Net Cash Flow    Opening Cash    Closing Cash

    Args:
        forecast_df: DataFrame produced by build_cash_flow_forecast().

    Returns:
        pd.DataFrame: Formatted presentation DataFrame.
    """
    display_df = pd.DataFrame()
    display_df["Month"] = forecast_df["Month"]
    display_df["Inflow"] = forecast_df["ForecastCashInflow"].apply(lambda x: f"${x:,.0f}")
    display_df["Outflow"] = forecast_df["ForecastTotalCashOutflow"].apply(lambda x: f"${x:,.0f}")
    display_df["Net Cash Flow"] = forecast_df["ForecastNetCashFlow"].apply(
        lambda x: f"+${x:,.0f}" if x > 0 else (f"-${abs(x):,.0f}" if x < 0 else "$0")
    )
    display_df["Opening Cash"] = forecast_df["ForecastOpeningCashBalance"].apply(
        lambda x: f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"
    )
    display_df["Closing Cash"] = forecast_df["ForecastClosingCashBalance"].apply(
        lambda x: f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"
    )
    return display_df
