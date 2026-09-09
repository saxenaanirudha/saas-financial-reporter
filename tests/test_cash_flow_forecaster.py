"""
Unit Tests for Cash Flow Forecaster Module (Step 9)
=============================================================================
Tests:
1. Forecast returns exactly six future months.
2. Forecast begins after the latest historical month.
3. Month/year rollover works correctly (e.g., Dec-2025 -> Jan-2026).
4. Cash inflow forecast is generated and positive.
5. Expense component forecasts are generated.
6. Total cash outflow equals sum of forecast expense components.
7. Net cash flow equals inflow minus outflow.
8. First forecast opening cash equals latest historical closing cash balance.
9. Forecast cash balances roll forward correctly.
10. Negative component predictions are floored at zero.
11. Negative net cash flow is preserved if produced.
12. Negative closing cash balance is preserved.
13. Original input DataFrame is not mutated.
14. One-month historical dataset raises ValueError (< 2 points).
15. Empty historical dataset is rejected with ValueError.
16. Forecast summary values match calculated results.
17. Forecast charts are generated and non-empty.
18. Format cash flow forecast table produces expected presentation format.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.cash_flow_forecaster import (
    generate_future_month_labels,
    forecast_series,
    build_cash_flow_forecast,
    generate_forecast_summary,
    format_cash_flow_forecast_table,
    DEFAULT_FORECAST_HORIZON,
)
from src.chart_generator import (
    generate_cash_flow_forecast_chart,
    generate_cash_balance_forecast_chart,
    generate_all_charts,
)
from src.cash_flow_analyzer import analyze_cash_flow


@pytest.fixture
def sample_historical_df():
    """Returns standard 6-month historical SaaS financial DataFrame."""
    return pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025", "Mar-2025", "Apr-2025", "May-2025", "Jun-2025"],
        "NewCustomers": [120, 140, 160, 180, 200, 220],
        "ChurnedCustomers": [15, 20, 18, 25, 22, 30],
        "MRR": [60000, 68000, 75000, 82000, 90000, 97000],
        "MarketingSpend": [15000, 16000, 17000, 18000, 19000, 20000],
        "OperatingExpenses": [12000, 12500, 13000, 13500, 14000, 14500],
        "PayrollExpenses": [20000, 20500, 21000, 21500, 22000, 22500],
        "OtherExpenses": [3000, 3200, 3500, 3600, 3800, 4000],
    })


def test_forecast_returns_exactly_six_months(sample_historical_df):
    """Test 1: Forecast produces exactly 6 future months."""
    forecast_df = build_cash_flow_forecast(sample_historical_df)
    assert len(forecast_df) == 6


def test_forecast_begins_after_latest_historical_month(sample_historical_df):
    """Test 2: Forecast months begin immediately after Jun-2025."""
    forecast_df = build_cash_flow_forecast(sample_historical_df)
    expected_months = ["Jul-2025", "Aug-2025", "Sep-2025", "Oct-2025", "Nov-2025", "Dec-2025"]
    assert list(forecast_df["Month"]) == expected_months


def test_month_year_rollover():
    """Test 3: Date generator handles December to January year rollover correctly."""
    future_months = generate_future_month_labels("Dec-2025", horizon=3)
    assert future_months == ["Jan-2026", "Feb-2026", "Mar-2026"]


def test_cash_inflow_forecast_generated(sample_historical_df):
    """Test 4: Cash inflow trend is forecasted and increasing."""
    forecast_df = build_cash_flow_forecast(sample_historical_df)
    inflows = forecast_df["ForecastCashInflow"].values
    assert len(inflows) == 6
    assert (inflows > 97000).all()
    # Check monotonic increase given historical positive trend
    assert (np.diff(inflows) > 0).all()


def test_expense_components_forecasted(sample_historical_df):
    """Test 5: All 4 expense components are forecasted individually."""
    forecast_df = build_cash_flow_forecast(sample_historical_df)

    assert "ForecastMarketingSpend" in forecast_df.columns
    assert "ForecastOperatingExpenses" in forecast_df.columns
    assert "ForecastPayrollExpenses" in forecast_df.columns
    assert "ForecastOtherExpenses" in forecast_df.columns

    # Verify marketing spend continues from $20k with +$1k/month trend
    assert np.isclose(forecast_df["ForecastMarketingSpend"].iloc[0], 21000.0)


def test_total_cash_outflow_is_sum_of_expense_components(sample_historical_df):
    """Test 6: Forecast total outflow equals sum of forecasted expenses."""
    forecast_df = build_cash_flow_forecast(sample_historical_df)

    expected_outflows = (
        forecast_df["ForecastMarketingSpend"]
        + forecast_df["ForecastOperatingExpenses"]
        + forecast_df["ForecastPayrollExpenses"]
        + forecast_df["ForecastOtherExpenses"]
    )
    assert np.allclose(forecast_df["ForecastTotalCashOutflow"], expected_outflows)


def test_net_cash_flow_equals_inflow_minus_outflow(sample_historical_df):
    """Test 7: Forecast Net Cash Flow = Forecast Inflow - Forecast Outflow."""
    forecast_df = build_cash_flow_forecast(sample_historical_df)
    expected_net = forecast_df["ForecastCashInflow"] - forecast_df["ForecastTotalCashOutflow"]
    assert np.allclose(forecast_df["ForecastNetCashFlow"], expected_net)


def test_first_forecast_opening_cash_equals_last_historical_closing(sample_historical_df):
    """Test 8: First forecast opening cash matches Jun-2025 historical closing cash balance ($238,900)."""
    hist_enriched = analyze_cash_flow(sample_historical_df, opening_cash_balance=100000.0)
    last_closing = hist_enriched["ClosingCashBalance"].iloc[-1]

    forecast_df = build_cash_flow_forecast(sample_historical_df, opening_cash_balance=100000.0)
    assert forecast_df["ForecastOpeningCashBalance"].iloc[0] == last_closing
    assert forecast_df["ForecastOpeningCashBalance"].iloc[0] == 238900.0


def test_forecast_cash_balances_roll_forward(sample_historical_df):
    """Test 9: Forecast cash balances roll forward correctly across all 6 months."""
    forecast_df = build_cash_flow_forecast(sample_historical_df, opening_cash_balance=100000.0)

    for i in range(len(forecast_df) - 1):
        assert np.isclose(
            forecast_df["ForecastOpeningCashBalance"].iloc[i + 1],
            forecast_df["ForecastClosingCashBalance"].iloc[i],
        )

    for i in range(len(forecast_df)):
        assert np.isclose(
            forecast_df["ForecastClosingCashBalance"].iloc[i],
            forecast_df["ForecastOpeningCashBalance"].iloc[i] + forecast_df["ForecastNetCashFlow"].iloc[i],
        )


def test_negative_component_prediction_floored_at_zero():
    """Test 10: Declining series that goes below zero is floored at 0.0."""
    declining_series = np.array([1000.0, 500.0, 100.0])  # Trend drops below 0 next month
    pred = forecast_series(declining_series, horizon=3, floor_at_zero=True)
    assert (pred >= 0.0).all()
    assert pred[-1] == 0.0


def test_negative_net_cash_flow_preserved():
    """Test 11: Negative net cash flow is preserved and not floored."""
    df_heavy_burn = pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025"],
        "MRR": [10000, 10000],
        "MarketingSpend": [20000, 20000],
        "OperatingExpenses": [15000, 15000],
        "PayrollExpenses": [25000, 25000],
        "OtherExpenses": [5000, 5000],
    })
    forecast_df = build_cash_flow_forecast(df_heavy_burn, opening_cash_balance=50000.0)
    # Inflow = 10,000, Outflow = 65,000, Net = -55,000
    assert (forecast_df["ForecastNetCashFlow"] < 0).all()
    assert np.isclose(forecast_df["ForecastNetCashFlow"].iloc[0], -55000.0)


def test_negative_closing_cash_balance_preserved():
    """Test 12: Negative closing cash balance (depletion) is preserved."""
    df_burning = pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025"],
        "MRR": [10000, 10000],
        "MarketingSpend": [20000, 20000],
        "OperatingExpenses": [10000, 10000],
        "PayrollExpenses": [10000, 10000],
        "OtherExpenses": [5000, 5000],
    })
    # Starting with only $10,000, burning $35,000/month
    forecast_df = build_cash_flow_forecast(df_burning, opening_cash_balance=10000.0)
    assert forecast_df["ForecastClosingCashBalance"].iloc[-1] < 0


def test_input_dataframe_not_mutated(sample_historical_df):
    """Test 13: Input DataFrame is not mutated."""
    cols_before = list(sample_historical_df.columns)
    build_cash_flow_forecast(sample_historical_df)
    assert list(sample_historical_df.columns) == cols_before


def test_insufficient_historical_data_rejected():
    """Test 14: Single row dataset raises ValueError (requires >= 2 points)."""
    single_df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "MRR": [50000],
        "MarketingSpend": [10000],
        "OperatingExpenses": [10000],
        "PayrollExpenses": [15000],
        "OtherExpenses": [2000],
    })
    with pytest.raises(ValueError, match="At least 2 historical months are required"):
        build_cash_flow_forecast(single_df)


def test_empty_dataframe_rejected():
    """Test 15: Empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="empty DataFrame"):
        build_cash_flow_forecast(pd.DataFrame())


def test_forecast_summary_values_are_correct(sample_historical_df):
    """Test 16: Forecast summary metrics match DataFrame contents."""
    forecast_df = build_cash_flow_forecast(sample_historical_df, opening_cash_balance=100000.0)
    summary = generate_forecast_summary(forecast_df)

    assert summary["forecast_start_month"] == "Jul-2025"
    assert summary["forecast_end_month"] == "Dec-2025"
    assert summary["forecast_months"] == 6
    assert summary["starting_forecast_cash_balance"] == 238900.0
    assert summary["ending_forecast_cash_balance"] > 238900.0
    assert summary["interpretation"] == "Forecast indicates an increasing cash position under the historical trend assumptions."


def test_format_cash_flow_forecast_table(sample_historical_df):
    """Test 17: Presentation table formatting works properly."""
    forecast_df = build_cash_flow_forecast(sample_historical_df, opening_cash_balance=100000.0)
    formatted = format_cash_flow_forecast_table(forecast_df)

    assert list(formatted.columns) == ["Month", "Inflow", "Outflow", "Net Cash Flow", "Opening Cash", "Closing Cash"]
    assert formatted.iloc[0]["Month"] == "Jul-2025"
    assert "$" in formatted.iloc[0]["Inflow"]
    assert "$" in formatted.iloc[0]["Opening Cash"]


def test_forecast_charts_generated_successfully(sample_historical_df, tmp_path):
    """Test 18: Forecast charts are created and non-empty."""
    hist_df = analyze_cash_flow(sample_historical_df, opening_cash_balance=100000.0)
    forecast_df = build_cash_flow_forecast(hist_df, opening_cash_balance=100000.0)

    cf_chart = generate_cash_flow_forecast_chart(hist_df, forecast_df, tmp_path)
    assert cf_chart.exists()
    assert cf_chart.name == "cash_flow_forecast.png"
    assert cf_chart.stat().st_size > 0

    bal_chart = generate_cash_balance_forecast_chart(hist_df, forecast_df, tmp_path)
    assert bal_chart.exists()
    assert bal_chart.name == "cash_balance_forecast.png"
    assert bal_chart.stat().st_size > 0

    all_charts = generate_all_charts(sample_historical_df, tmp_path, forecast_df=forecast_df)
    assert "cash_flow_forecast" in all_charts
    assert "cash_balance_forecast" in all_charts
    assert all_charts["cash_flow_forecast"].exists()
    assert all_charts["cash_balance_forecast"].exists()
