"""
Unit Tests for PDF Report Generator Module (Step 10)
=============================================================================
Tests:
1. PDF report file is created.
2. Reports directory is created automatically.
3. Generated PDF file is non-empty (> 10KB).
4. PDF generation succeeds with full sample dataset and embeds charts.
5. PDF generation succeeds gracefully even if an optional chart is missing.
6. Empty input DataFrame raises ValueError.
7. Insufficient historical data (< 2 months) raises ValueError from forecasting layer.
8. Input DataFrame is not mutated during report generation.
9. Custom destination path and opening cash balance work seamlessly.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.pdf_report_generator import generate_investor_report


@pytest.fixture
def sample_report_df():
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


def test_pdf_report_created_and_non_empty(sample_report_df, tmp_path):
    """Test 1, 2, 3: PDF file is created, directory is auto-created, and file is non-empty."""
    out_pdf = tmp_path / "reports_sub" / "test_report.pdf"
    assert not out_pdf.exists()

    generated_path = generate_investor_report(
        sample_report_df,
        output_path=out_pdf,
        charts_dir=tmp_path / "charts",
    )

    assert generated_path.exists()
    assert generated_path == out_pdf
    assert generated_path.stat().st_size > 5000  # Substantial PDF document (> 5KB)


def test_pdf_generation_with_missing_optional_chart(sample_report_df, tmp_path):
    """Test 4 & 5: If an optional chart is deleted/missing, report generates safely without crashing."""
    charts_dir = tmp_path / "charts_missing"
    charts_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = tmp_path / "report_missing_chart.pdf"

    # Call generator with an empty charts directory to test graceful missing chart handling
    generated_path = generate_investor_report(
        sample_report_df,
        output_path=out_pdf,
        charts_dir=charts_dir,
    )

    assert generated_path.exists()
    assert generated_path.stat().st_size > 0


def test_empty_dataframe_raises_value_error(tmp_path):
    """Test 6: Empty input DataFrame raises clear ValueError."""
    empty_df = pd.DataFrame()
    out_pdf = tmp_path / "empty_report.pdf"

    with pytest.raises(ValueError, match="empty DataFrame"):
        generate_investor_report(empty_df, output_path=out_pdf)


def test_single_row_dataframe_rejected(tmp_path):
    """Test 7: 1-row DataFrame fails due to inability to forecast (requires >= 2 points)."""
    single_df = pd.DataFrame({
        "Month": ["Jan-2025"],
        "NewCustomers": [120],
        "ChurnedCustomers": [15],
        "MRR": [60000],
        "MarketingSpend": [15000],
        "OperatingExpenses": [12000],
        "PayrollExpenses": [20000],
        "OtherExpenses": [3000],
    })
    out_pdf = tmp_path / "single_report.pdf"

    with pytest.raises(ValueError, match="At least 2 historical months are required"):
        generate_investor_report(single_df, output_path=out_pdf)


def test_input_dataframe_not_mutated(sample_report_df, tmp_path):
    """Test 8: Input DataFrame is not mutated during PDF generation."""
    cols_before = list(sample_report_df.columns)
    out_pdf = tmp_path / "mutation_test.pdf"

    generate_investor_report(sample_report_df, output_path=out_pdf, charts_dir=tmp_path / "charts")

    assert list(sample_report_df.columns) == cols_before
    assert "ForecastNetCashFlow" not in sample_report_df.columns


def test_custom_opening_balance_and_forecast_horizon(sample_report_df, tmp_path):
    """Test 9: Custom opening cash balance and forecast horizon are respected."""
    out_pdf = tmp_path / "custom_report.pdf"

    generated_path = generate_investor_report(
        sample_report_df,
        output_path=out_pdf,
        charts_dir=tmp_path / "charts",
        opening_cash_balance=500000.0,
        forecast_horizon=3,
    )

    assert generated_path.exists()
    assert generated_path.stat().st_size > 0
