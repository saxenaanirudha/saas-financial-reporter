"""
Unit Tests for Cash Flow Analyzer Module (Step 8)
=============================================================================
Tests:
1. Cash inflow equals MRR.
2. Total cash outflow calculates correctly.
3. Net cash flow calculates correctly.
4. First month opening cash balance is correct.
5. First month closing cash balance is correct.
6. Closing balance rolls forward correctly.
7. Previous closing balance becomes next opening balance.
8. Positive net cash flow works.
9. Negative net cash flow works.
10. Zero expenses work.
11. Zero opening cash balance works.
12. Negative opening cash balance works (deficit/overdraft position policy).
13. One-row dataset works.
14. Original input DataFrame is not mutated.
15. Summary values are correct and match expected calculations.
16. Missing expense columns raise ValueError.
17. Format cash flow table produces properly formatted output.
18. Cash flow charts are created and non-empty.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.cash_flow_analyzer import (
    calculate_cash_inflow,
    calculate_cash_outflow,
    calculate_net_cash_flow,
    calculate_cash_balances,
    analyze_cash_flow,
    generate_cash_flow_summary,
    format_cash_flow_table,
    DEFAULT_OPENING_CASH_BALANCE,
)
from src.chart_generator import (
    generate_net_cash_flow_chart,
    generate_cash_balance_chart,
    generate_all_charts,
)


@pytest.fixture
def sample_cash_flow_df():
    """
    Standard 6-month SaaS financial DataFrame with expenses.
    """
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


def test_cash_inflow_equals_mrr(sample_cash_flow_df):
    """Test 1: Cash Inflow equals MRR for all months."""
    inflow = calculate_cash_inflow(sample_cash_flow_df)
    assert list(inflow) == [60000.0, 68000.0, 75000.0, 82000.0, 90000.0, 97000.0]


def test_total_cash_outflow_calculates_correctly(sample_cash_flow_df):
    """Test 2: Total cash outflow equals sum of marketing, operating, payroll, and other expenses."""
    outflow = calculate_cash_outflow(sample_cash_flow_df)
    expected_outflows = [
        15000 + 12000 + 20000 + 3000,  # 50,000
        16000 + 12500 + 20500 + 3200,  # 52,200
        17000 + 13000 + 21000 + 3500,  # 54,500
        18000 + 13500 + 21500 + 3600,  # 56,600
        19000 + 14000 + 22000 + 3800,  # 58,800
        20000 + 14500 + 22500 + 4000,  # 61,000
    ]
    assert list(outflow) == expected_outflows


def test_net_cash_flow_calculates_correctly(sample_cash_flow_df):
    """Test 3: Net Cash Flow = Inflow - Outflow."""
    net_flow = calculate_net_cash_flow(sample_cash_flow_df)
    expected_net = [
        60000 - 50000,  # +10,000
        68000 - 52200,  # +15,800
        75000 - 54500,  # +20,500
        82000 - 56600,  # +25,400
        90000 - 58800,  # +31,200
        97000 - 61000,  # +36,000
    ]
    assert list(net_flow) == expected_net


def test_opening_and_closing_cash_balances_first_month(sample_cash_flow_df):
    """Test 4 & 5: First month opening balance equals initial configured balance, closing equals opening + net."""
    result = analyze_cash_flow(sample_cash_flow_df, opening_cash_balance=100000.0)
    assert result.iloc[0]["OpeningCashBalance"] == 100000.0
    assert result.iloc[0]["ClosingCashBalance"] == 110000.0


def test_cash_balance_roll_forward(sample_cash_flow_df):
    """Test 6 & 7: Closing balance rolls forward to next month's opening balance."""
    result = analyze_cash_flow(sample_cash_flow_df, opening_cash_balance=100000.0)

    # Check roll-forward linkage for every consecutive pair of months
    for i in range(len(result) - 1):
        assert result.iloc[i + 1]["OpeningCashBalance"] == result.iloc[i]["ClosingCashBalance"]

    # Verify all closing balances
    expected_closing = [110000.0, 125800.0, 146300.0, 171700.0, 202900.0, 238900.0]
    assert list(result["ClosingCashBalance"]) == expected_closing


def test_positive_net_cash_flow():
    """Test 8: Positive net cash flow increases cash balance."""
    df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "MRR": [50000],
        "MarketingSpend": [5000],
        "OperatingExpenses": [5000],
        "PayrollExpenses": [10000],
        "OtherExpenses": [1000],
    })
    result = analyze_cash_flow(df, opening_cash_balance=20000.0)
    assert result.iloc[0]["NetCashFlow"] == 29000.0
    assert result.iloc[0]["ClosingCashBalance"] == 49000.0


def test_negative_net_cash_flow_and_depletion():
    """Test 9: Negative net cash flow decreases cash balance and handles negative closing balance."""
    df = pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025"],
        "MRR": [20000, 20000],
        "MarketingSpend": [15000, 15000],
        "OperatingExpenses": [10000, 10000],
        "PayrollExpenses": [20000, 20000],
        "OtherExpenses": [5000, 5000],
    })
    # Total outflow = 50,000, Net = -30,000 each month
    result = analyze_cash_flow(df, opening_cash_balance=40000.0)

    assert result.iloc[0]["NetCashFlow"] == -30000.0
    assert result.iloc[0]["ClosingCashBalance"] == 10000.0

    assert result.iloc[1]["OpeningCashBalance"] == 10000.0
    assert result.iloc[1]["NetCashFlow"] == -30000.0
    assert result.iloc[1]["ClosingCashBalance"] == -20000.0


def test_zero_expenses_work():
    """Test 10: Zero expenses are allowed and handled correctly."""
    df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "MRR": [30000],
        "MarketingSpend": [0],
        "OperatingExpenses": [0],
        "PayrollExpenses": [0],
        "OtherExpenses": [0],
    })
    result = analyze_cash_flow(df, opening_cash_balance=10000.0)
    assert result.iloc[0]["TotalCashOutflow"] == 0.0
    assert result.iloc[0]["NetCashFlow"] == 30000.0
    assert result.iloc[0]["ClosingCashBalance"] == 40000.0


def test_zero_opening_cash_balance():
    """Test 11: Zero opening cash balance works as expected."""
    df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "MRR": [15000],
        "MarketingSpend": [5000],
        "OperatingExpenses": [2000],
        "PayrollExpenses": [3000],
        "OtherExpenses": [1000],
    })
    result = analyze_cash_flow(df, opening_cash_balance=0.0)
    assert result.iloc[0]["OpeningCashBalance"] == 0.0
    assert result.iloc[0]["NetCashFlow"] == 4000.0
    assert result.iloc[0]["ClosingCashBalance"] == 4000.0


def test_negative_opening_cash_balance():
    """Test 12: Negative opening cash balance is supported (representing overdraft/debt)."""
    df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "MRR": [10000],
        "MarketingSpend": [2000],
        "OperatingExpenses": [1000],
        "PayrollExpenses": [2000],
        "OtherExpenses": [1000],
    })
    # Outflow = 6,000, Net = +4,000, Opening = -50,000, Closing = -46,000
    result = analyze_cash_flow(df, opening_cash_balance=-50000.0)
    assert result.iloc[0]["OpeningCashBalance"] == -50000.0
    assert result.iloc[0]["NetCashFlow"] == 4000.0
    assert result.iloc[0]["ClosingCashBalance"] == -46000.0


def test_single_row_dataset_works():
    """Test 13: One-row dataset processes smoothly."""
    df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "MRR": [60000],
        "MarketingSpend": [15000],
        "OperatingExpenses": [12000],
        "PayrollExpenses": [20000],
        "OtherExpenses": [3000],
    })
    result = analyze_cash_flow(df, opening_cash_balance=100000.0)
    assert len(result) == 1
    assert result.iloc[0]["ClosingCashBalance"] == 110000.0

    summary = generate_cash_flow_summary(df, opening_cash_balance=100000.0)
    assert summary["latest_month"] == "Jan-2025"
    assert summary["total_months"] == 1
    assert summary["latest_net_cash_flow"] == 10000.0


def test_input_dataframe_not_mutated(sample_cash_flow_df):
    """Test 14: Input DataFrame is not mutated during cash flow analysis."""
    original_cols = list(sample_cash_flow_df.columns)
    analyze_cash_flow(sample_cash_flow_df)
    assert list(sample_cash_flow_df.columns) == original_cols
    assert "NetCashFlow" not in sample_cash_flow_df.columns


def test_summary_values_are_correct(sample_cash_flow_df):
    """Test 15: Summary dictionary values match calculations."""
    summary = generate_cash_flow_summary(sample_cash_flow_df, opening_cash_balance=100000.0)

    assert summary["latest_month"] == "Jun-2025"
    assert summary["latest_cash_inflow"] == 97000.0
    assert summary["latest_total_cash_outflow"] == 61000.0
    assert summary["latest_net_cash_flow"] == 36000.0
    assert summary["latest_opening_cash_balance"] == 202900.0
    assert summary["latest_closing_cash_balance"] == 238900.0
    assert summary["starting_cash_balance"] == 100000.0
    assert summary["total_historical_cash_inflow"] == 472000.0
    assert summary["total_historical_cash_outflow"] == 333100.0
    assert summary["cumulative_net_cash_flow"] == 138900.0
    assert summary["latest_cash_position"] == 238900.0
    assert summary["interpretation"] == "Latest month generated positive operating cash flow under the simplified model."
    assert summary["total_months"] == 6


def test_missing_expense_columns_raise_error():
    """Test 16: Missing expense columns raise ValueError."""
    incomplete_df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "MRR": [60000],
        "MarketingSpend": [15000],
        # Missing OperatingExpenses, PayrollExpenses, OtherExpenses
    })
    with pytest.raises(ValueError, match="missing required expense column"):
        calculate_cash_outflow(incomplete_df)


def test_empty_dataframe_raises_error():
    """Test 17: Empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty DataFrame"):
        analyze_cash_flow(empty_df)


def test_format_cash_flow_table(sample_cash_flow_df):
    """Test 18: Formats cash flow metrics into readable string table."""
    enriched = analyze_cash_flow(sample_cash_flow_df, opening_cash_balance=100000.0)
    formatted = format_cash_flow_table(enriched)

    assert list(formatted.columns) == ["Month", "Inflow", "Outflow", "Net Cash Flow", "Opening Cash", "Closing Cash"]
    assert formatted.iloc[0]["Inflow"] == "$60,000"
    assert formatted.iloc[0]["Outflow"] == "$50,000"
    assert formatted.iloc[0]["Net Cash Flow"] == "+$10,000"
    assert formatted.iloc[0]["Opening Cash"] == "$100,000"
    assert formatted.iloc[0]["Closing Cash"] == "$110,000"


def test_cash_flow_charts_generated_successfully(sample_cash_flow_df, tmp_path):
    """Test 19: Net cash flow and cash balance charts are created and non-empty."""
    net_chart = generate_net_cash_flow_chart(sample_cash_flow_df, tmp_path)
    assert net_chart.exists()
    assert net_chart.name == "net_cash_flow_trend.png"
    assert net_chart.stat().st_size > 0

    bal_chart = generate_cash_balance_chart(sample_cash_flow_df, tmp_path)
    assert bal_chart.exists()
    assert bal_chart.name == "cash_balance_trend.png"
    assert bal_chart.stat().st_size > 0

    all_charts = generate_all_charts(sample_cash_flow_df, tmp_path)
    assert "net_cash_flow_trend" in all_charts
    assert "cash_balance_trend" in all_charts
    assert all_charts["net_cash_flow_trend"].exists()
    assert all_charts["cash_balance_trend"].exists()
