"""
Financial Charts & KPI Visualizations Module
=============================================================================
Converts calculated financial KPIs into clear, publication-ready PNG charts
for management review and automated investor reporting.

Architecture Rule:
- Does NOT recalculate financial KPIs independently.
- Reuses calculation functions from revenue_dashboard, cac_calculator, and
  churn_retention_analyzer.
- Closes matplotlib figures cleanly after saving to prevent resource leaks.
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")  # Enforce non-interactive backend for headless execution

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Union, Dict, List, Optional
import pandas as pd
import numpy as np

# Import business calculation modules to enrich input DataFrame if needed
from src.revenue_dashboard import calculate_revenue_metrics
from src.cac_calculator import calculate_cac_metrics
from src.churn_retention_analyzer import calculate_churn_retention_metrics
from src.cash_flow_analyzer import analyze_cash_flow


def _ensure_enriched_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to ensure all required KPI columns exist in the DataFrame.
    Reuses existing step 3, step 4, step 5, and step 8 modules without recalculating from scratch.

    Raises:
        ValueError: If input DataFrame is empty or missing foundational raw columns.
    """
    if df.empty:
        raise ValueError("Cannot generate financial charts from an empty DataFrame.")

    data = df.copy()

    # Enrich revenue metrics if missing
    if "MRR" not in data.columns or "ARR" not in data.columns or "MRRGrowthPct" not in data.columns:
        data = calculate_revenue_metrics(data)

    # Enrich CAC metrics if missing
    if "CAC" not in data.columns:
        data = calculate_cac_metrics(data)

    # Enrich Churn & Retention metrics if missing
    if "EndingCustomers" not in data.columns or "ChurnRatePercent" not in data.columns:
        data = calculate_churn_retention_metrics(data)

    # Enrich Cash Flow metrics if expense columns are present and cash flow metrics missing
    if "OperatingExpenses" in data.columns and ("NetCashFlow" not in data.columns or "ClosingCashBalance" not in data.columns):
        data = analyze_cash_flow(data)

    return data


def _setup_figure(title: str, xlabel: str = "Month", ylabel: str = ""):
    """
    Helper to set up a standardized clean plot style.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig, ax


def generate_mrr_chart(df: pd.DataFrame, output_dir: Union[str, Path] = "charts") -> Path:
    """
    Generates and saves the Monthly Recurring Revenue (MRR) Trend chart.

    File: output_dir / mrr_trend.png
    """
    data = _ensure_enriched_dataframe(df)
    out_path = Path(output_dir) / "mrr_trend.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = _setup_figure("Monthly Recurring Revenue (MRR) Trend", ylabel="MRR ($)")

    ax.plot(data["Month"], data["MRR"], marker="o", color="#1f77b4", linewidth=2.5, markersize=6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Add data point value labels
    for i, txt in enumerate(data["MRR"]):
        ax.annotate(f"${txt:,.0f}", (data["Month"].iloc[i], txt), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    return out_path


def generate_arr_chart(df: pd.DataFrame, output_dir: Union[str, Path] = "charts") -> Path:
    """
    Generates and saves the Annual Recurring Revenue (ARR) Trend chart.

    File: output_dir / arr_trend.png
    """
    data = _ensure_enriched_dataframe(df)
    out_path = Path(output_dir) / "arr_trend.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = _setup_figure("Annual Recurring Revenue (ARR) Trend", ylabel="ARR ($)")

    ax.plot(data["Month"], data["ARR"], marker="s", color="#2ca02c", linewidth=2.5, markersize=6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    for i, txt in enumerate(data["ARR"]):
        ax.annotate(f"${txt:,.0f}", (data["Month"].iloc[i], txt), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    return out_path


def generate_cac_chart(df: pd.DataFrame, output_dir: Union[str, Path] = "charts") -> Path:
    """
    Generates and saves the Customer Acquisition Cost (CAC) Trend chart.

    File: output_dir / cac_trend.png
    """
    data = _ensure_enriched_dataframe(df)
    out_path = Path(output_dir) / "cac_trend.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = _setup_figure("Customer Acquisition Cost (CAC) Trend", ylabel="CAC ($)")

    ax.plot(data["Month"], data["CAC"], marker="D", color="#d62728", linewidth=2.5, markersize=6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:,.2f}" if pd.notna(x) else ""))

    for i, val in enumerate(data["CAC"]):
        if pd.notna(val):
            ax.annotate(f"${val:,.2f}", (data["Month"].iloc[i], val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    return out_path


def generate_churn_retention_chart(df: pd.DataFrame, output_dir: Union[str, Path] = "charts") -> Path:
    """
    Generates and saves the Customer Churn & Retention Trend chart.

    File: output_dir / churn_retention_trend.png
    Handles month 1 NaNs safely without plotting fake 0% data.
    """
    data = _ensure_enriched_dataframe(df)
    out_path = Path(output_dir) / "churn_retention_trend.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = _setup_figure("Customer Churn & Retention Trend", ylabel="Percentage (%)")

    # Plot retention rate
    ax.plot(data["Month"], data["RetentionRatePercent"], marker="o", color="#2ca02c", linewidth=2.5, label="Retention Rate (%)")
    # Plot churn rate
    ax.plot(data["Month"], data["ChurnRatePercent"], marker="x", color="#d62728", linewidth=2.5, linestyle="--", label="Churn Rate (%)")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.0f}%"))
    ax.legend(loc="upper left", frameon=True)

    # Label data points safely ignoring NaNs
    for i in range(len(data)):
        month = data["Month"].iloc[i]
        c_val = data["ChurnRatePercent"].iloc[i]
        r_val = data["RetentionRatePercent"].iloc[i]

        if pd.notna(r_val):
            ax.annotate(f"{r_val:.1f}%", (month, r_val), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color="#2ca02c")
        if pd.notna(c_val):
            ax.annotate(f"{c_val:.1f}%", (month, c_val), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8, color="#d62728")

    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    return out_path


def generate_customer_base_chart(df: pd.DataFrame, output_dir: Union[str, Path] = "charts") -> Path:
    """
    Generates and saves the Active Customer Base Growth chart.

    File: output_dir / customer_base_growth.png
    """
    data = _ensure_enriched_dataframe(df)
    out_path = Path(output_dir) / "customer_base_growth.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = _setup_figure("Active Customer Base Growth", ylabel="Ending Customers")

    ax.plot(data["Month"], data["EndingCustomers"], marker="^", color="#9467bd", linewidth=2.5, markersize=7)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))

    for i, txt in enumerate(data["EndingCustomers"]):
        ax.annotate(f"{int(txt):,}", (data["Month"].iloc[i], txt), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    return out_path


def generate_mrr_growth_chart(df: pd.DataFrame, output_dir: Union[str, Path] = "charts") -> Path:
    """
    Generates and saves the Month-over-Month MRR Growth % chart.

    File: output_dir / mrr_growth_rate.png
    """
    data = _ensure_enriched_dataframe(df)
    out_path = Path(output_dir) / "mrr_growth_rate.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = _setup_figure("Month-over-Month MRR Growth", ylabel="MRR Growth Rate (%)")

    ax.plot(data["Month"], data["MRRGrowthPct"], marker="o", color="#ff7f0e", linewidth=2.5, markersize=6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.1f}%" if pd.notna(x) else ""))

    for i, val in enumerate(data["MRRGrowthPct"]):
        if pd.notna(val):
            ax.annotate(f"{val:.2f}%", (data["Month"].iloc[i], val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    return out_path


def generate_net_cash_flow_chart(df: pd.DataFrame, output_dir: Union[str, Path] = "charts") -> Path:
    """
    Generates and saves the Net Cash Flow Trend chart.

    File: output_dir / net_cash_flow_trend.png
    """
    data = _ensure_enriched_dataframe(df)
    if "NetCashFlow" not in data.columns:
        raise ValueError("DataFrame must contain 'NetCashFlow' or expense columns to generate cash flow chart.")

    out_path = Path(output_dir) / "net_cash_flow_trend.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = _setup_figure("Monthly Net Cash Flow Trend", ylabel="Net Cash Flow ($)")

    colors = ["#2ca02c" if val >= 0 else "#d62728" for val in data["NetCashFlow"]]
    ax.bar(data["Month"], data["NetCashFlow"], color=colors, width=0.55, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"))

    for i, val in enumerate(data["NetCashFlow"]):
        prefix = "+$" if val > 0 else ("-$" if val < 0 else "$")
        y_pos = val if val >= 0 else val
        xy_offset = (0, 6) if val >= 0 else (0, -14)
        ax.annotate(
            f"{prefix}{abs(val):,.0f}",
            (data["Month"].iloc[i], y_pos),
            textcoords="offset points",
            xytext=xy_offset,
            ha="center",
            fontsize=9,
            fontweight="bold" if val >= 0 else "normal",
        )

    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    return out_path


def generate_cash_balance_chart(df: pd.DataFrame, output_dir: Union[str, Path] = "charts") -> Path:
    """
    Generates and saves the Cash Balance Trend chart.

    File: output_dir / cash_balance_trend.png
    """
    data = _ensure_enriched_dataframe(df)
    if "ClosingCashBalance" not in data.columns:
        raise ValueError("DataFrame must contain 'ClosingCashBalance' or expense columns to generate cash balance chart.")

    out_path = Path(output_dir) / "cash_balance_trend.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = _setup_figure("Cash Balance Growth Trend", ylabel="Closing Cash Balance ($)")

    ax.plot(data["Month"], data["ClosingCashBalance"], marker="o", color="#17becf", linewidth=2.5, markersize=6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"))

    for i, txt in enumerate(data["ClosingCashBalance"]):
        prefix = "$" if txt >= 0 else "-$"
        ax.annotate(
            f"{prefix}{abs(txt):,.0f}",
            (data["Month"].iloc[i], txt),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    return out_path


def generate_cash_flow_forecast_chart(
    hist_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    output_dir: Union[str, Path] = "charts",
) -> Path:
    """
    Generates and saves the Cash Flow Forecast chart comparing historical
    and projected monthly Net Cash Flow.

    File: output_dir / cash_flow_forecast.png
    """
    out_path = Path(output_dir) / "cash_flow_forecast.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = _setup_figure("Cash Flow Forecast (Historical vs 6-Month Projected)", ylabel="Net Cash Flow ($)")

    hist_months = list(hist_df["Month"])
    hist_net = list(hist_df["NetCashFlow"])
    fc_months = list(forecast_df["Month"])
    fc_net = list(forecast_df["ForecastNetCashFlow"])

    all_months = hist_months + fc_months
    n_hist = len(hist_months)

    # Plot historical net cash flow
    hist_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in hist_net]
    ax.bar(np.arange(n_hist), hist_net, color=hist_colors, width=0.55, alpha=0.85, edgecolor="black", label="Historical Net Cash Flow")

    # Plot forecast net cash flow
    fc_colors = ["#1f77b4" if v >= 0 else "#ff7f0e" for v in fc_net]
    ax.bar(
        np.arange(n_hist, n_hist + len(fc_months)),
        fc_net,
        color=fc_colors,
        width=0.55,
        alpha=0.85,
        edgecolor="black",
        hatch="//",
        label="Forecast Net Cash Flow",
    )

    # Visual separator between historical and forecast
    ax.axvline(x=n_hist - 0.5, color="#7f7f7f", linestyle="--", linewidth=1.5, label="Forecast Boundary")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)

    ax.set_xticks(np.arange(len(all_months)))
    ax.set_xticklabels(all_months, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"))
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    # Annotate forecast values
    for i, val in enumerate(fc_net):
        idx = n_hist + i
        prefix = "+$" if val >= 0 else "-$"
        ax.annotate(
            f"{prefix}{abs(val):,.0f}",
            (idx, val),
            textcoords="offset points",
            xytext=(0, 6 if val >= 0 else -14),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    return out_path


def generate_cash_balance_forecast_chart(
    hist_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    output_dir: Union[str, Path] = "charts",
) -> Path:
    """
    Generates and saves the Cash Balance Forecast chart comparing historical
    and projected Closing Cash Balances.

    File: output_dir / cash_balance_forecast.png
    """
    out_path = Path(output_dir) / "cash_balance_forecast.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = _setup_figure("Cash Balance Forecast (Historical vs 6-Month Projected)", ylabel="Closing Cash Balance ($)")

    hist_months = list(hist_df["Month"])
    hist_bal = list(hist_df["ClosingCashBalance"])
    fc_months = list(forecast_df["Month"])
    fc_bal = list(forecast_df["ForecastClosingCashBalance"])

    all_months = hist_months + fc_months
    n_hist = len(hist_months)

    # Plot historical trajectory
    ax.plot(
        np.arange(n_hist),
        hist_bal,
        marker="o",
        color="#17becf",
        linewidth=2.5,
        markersize=6,
        label="Historical Cash Balance",
    )

    # Connect seamlessly from last historical to first forecast
    bridge_x = [n_hist - 1] + list(np.arange(n_hist, n_hist + len(fc_months)))
    bridge_y = [hist_bal[-1]] + fc_bal

    ax.plot(
        bridge_x,
        bridge_y,
        marker="s",
        color="#e377c2",
        linewidth=2.5,
        linestyle="--",
        markersize=6,
        label="Forecast Cash Balance",
    )

    # Visual separator between historical and forecast
    ax.axvline(x=n_hist - 0.5, color="#7f7f7f", linestyle="--", linewidth=1.5, label="Forecast Boundary")

    ax.set_xticks(np.arange(len(all_months)))
    ax.set_xticklabels(all_months, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"))
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    for i, val in enumerate(fc_bal):
        idx = n_hist + i
        prefix = "$" if val >= 0 else "-$"
        ax.annotate(
            f"{prefix}{abs(val):,.0f}",
            (idx, val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    return out_path


def generate_all_charts(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "charts",
    forecast_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    """
    Master function: generates all financial KPI and forecast charts and saves them to disk.

    Args:
        df: Financial DataFrame (raw validated or enriched).
        output_dir: Target directory path for PNG charts.
        forecast_df: Optional 6-month forecast DataFrame.

    Returns:
        dict: Mapping of chart metric name to generated Path.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Ensure DataFrame is enriched prior to passing to individual chart functions
    enriched_df = _ensure_enriched_dataframe(df)

    chart_paths = {
        "mrr_trend": generate_mrr_chart(enriched_df, target_dir),
        "arr_trend": generate_arr_chart(enriched_df, target_dir),
        "cac_trend": generate_cac_chart(enriched_df, target_dir),
        "churn_retention_trend": generate_churn_retention_chart(enriched_df, target_dir),
        "customer_base_growth": generate_customer_base_chart(enriched_df, target_dir),
        "mrr_growth_rate": generate_mrr_growth_chart(enriched_df, target_dir),
    }

    # Conditionally generate cash flow charts if cash flow data exists
    if "NetCashFlow" in enriched_df.columns or "OperatingExpenses" in enriched_df.columns:
        chart_paths["net_cash_flow_trend"] = generate_net_cash_flow_chart(enriched_df, target_dir)
        chart_paths["cash_balance_trend"] = generate_cash_balance_chart(enriched_df, target_dir)

        # Generate forecast charts if forecast_df is provided or can be computed
        if forecast_df is not None:
            chart_paths["cash_flow_forecast"] = generate_cash_flow_forecast_chart(enriched_df, forecast_df, target_dir)
            chart_paths["cash_balance_forecast"] = generate_cash_balance_forecast_chart(enriched_df, forecast_df, target_dir)
        elif len(enriched_df) >= 2 and "OperatingExpenses" in enriched_df.columns:
            from src.cash_flow_forecaster import build_cash_flow_forecast
            fc_df = build_cash_flow_forecast(enriched_df)
            chart_paths["cash_flow_forecast"] = generate_cash_flow_forecast_chart(enriched_df, fc_df, target_dir)
            chart_paths["cash_balance_forecast"] = generate_cash_balance_forecast_chart(enriched_df, fc_df, target_dir)

    return chart_paths
