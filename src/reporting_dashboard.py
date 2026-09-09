"""
Consolidated Financial Reporting Dashboard Module (Step 11)
=============================================================================
Provides a unified, executive-level reporting dashboard that consolidates all
core SaaS financial pillars into a single structured view:
1. Business Overview
2. Recurring Revenue Performance
3. Customer Acquisition Efficiency (CAC)
4. Customer Churn & Retention Dynamics
5. Historical Operating Cash Flow
6. Six-Month Cash Flow Forecast
7. KPI Movement & Trend Summary
8. Generated Artifact Discovery & Verification

Architecture Rules:
- Reuses calculation outputs from existing modules without formula duplication.
- Consolidates, formats, and renders clean PowerShell-compatible console output.
- Discovers on-disk artifacts (PDF reports and PNG charts) safely.
- Employs deterministic, explainable overall business health classification.
=============================================================================
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from src.revenue_dashboard import (
    calculate_revenue_metrics,
    generate_revenue_summary,
)
from src.cac_calculator import (
    calculate_cac_metrics,
    generate_cac_summary,
)
from src.churn_retention_analyzer import (
    calculate_churn_retention_metrics,
    generate_churn_retention_summary,
)
from src.kpi_comparator import (
    compare_latest_vs_previous,
    generate_trend_summary,
)
from src.cash_flow_analyzer import (
    analyze_cash_flow,
    generate_cash_flow_summary,
    DEFAULT_OPENING_CASH_BALANCE,
)
from src.cash_flow_forecaster import (
    build_cash_flow_forecast,
    generate_forecast_summary,
    DEFAULT_FORECAST_HORIZON,
)

# Standard list of charts tracked by the reporting dashboard
TRACKED_CHARTS = [
    ("Monthly Recurring Revenue (MRR) Trend", "mrr_trend.png"),
    ("Annual Recurring Revenue (ARR) Trend", "arr_trend.png"),
    ("Customer Acquisition Cost (CAC) Trend", "cac_trend.png"),
    ("Customer Churn & Retention Trend", "churn_retention_trend.png"),
    ("Active Customer Base Growth", "customer_base_growth.png"),
    ("Month-over-Month MRR Growth Rate", "mrr_growth_rate.png"),
    ("Monthly Net Cash Flow Trend", "net_cash_flow_trend.png"),
    ("Cash Balance Growth Trend", "cash_balance_trend.png"),
    ("Cash Flow Forecast (Historical vs Projected)", "cash_flow_forecast.png"),
    ("Cash Balance Forecast (Historical vs Projected)", "cash_balance_forecast.png"),
]


def determine_overall_business_status(
    kpi_comp_df: pd.DataFrame,
    cash_flow_sum: Dict[str, Any],
    forecast_sum: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Evaluates the overall business health status deterministically based on
    core KPI movements and cash flow indicators.

    Status Categories:
    - "Positive": Dominant favorable KPI movements and positive net cash flow.
    - "Mixed": Balance of improving and worsening indicators.
    - "Stable": Minimal net movements with stable retention and revenue.
    - "Needs Attention": Unfavorable cash drain, rising CAC, and worsening churn.

    Args:
        kpi_comp_df: DataFrame of Latest vs Previous KPI comparisons.
        cash_flow_sum: Cash flow summary dictionary.
        forecast_sum: Optional forecast summary dictionary.

    Returns:
        str: High-level business status string.
    """
    if kpi_comp_df.empty:
        return "Insufficient Data"

    statuses = kpi_comp_df["Status"].tolist()
    improved_count = sum(1 for s in statuses if s in ("Improved", "Higher Volume", "Favorable"))
    worsened_count = sum(1 for s in statuses if s in ("Worsened", "Unfavorable"))

    is_cash_flow_positive = cash_flow_sum.get("latest_net_cash_flow", 0) > 0
    is_forecast_positive = (
        forecast_sum.get("ending_forecast_cash_balance", 0)
        >= forecast_sum.get("starting_forecast_cash_balance", 0)
        if forecast_sum
        else True
    )

    if improved_count >= 4 and is_cash_flow_positive and is_forecast_positive:
        return "Positive"
    elif improved_count > worsened_count and is_cash_flow_positive:
        return "Positive"
    elif worsened_count > improved_count and not is_cash_flow_positive:
        return "Needs Attention"
    elif improved_count == worsened_count:
        return "Stable"
    else:
        return "Mixed"


def get_report_artifacts(
    charts_dir: Union[str, Path] = "charts",
    pdf_path: Union[str, Path] = "reports/saas_financial_report.pdf",
) -> Dict[str, Any]:
    """
    Discovers and validates the presence of generated PDF reports and PNG charts on disk.

    Args:
        charts_dir: Directory where charts are saved.
        pdf_path: Path to the generated PDF report.

    Returns:
        dict: Detailed mapping of discovered and verified artifacts.
    """
    charts_path = Path(charts_dir)
    pdf_file = Path(pdf_path)

    pdf_info = {
        "path": str(pdf_file),
        "exists": pdf_file.exists() and pdf_file.stat().st_size > 0,
        "size_bytes": pdf_file.stat().st_size if pdf_file.exists() else 0,
    }

    discovered_charts = []
    for title, filename in TRACKED_CHARTS:
        c_path = charts_path / filename
        exists = c_path.exists() and c_path.stat().st_size > 0
        discovered_charts.append({
            "title": title,
            "filename": filename,
            "path": str(c_path),
            "exists": exists,
            "size_bytes": c_path.stat().st_size if exists else 0,
        })

    return {
        "pdf_report": pdf_info,
        "charts": discovered_charts,
        "total_charts_found": sum(1 for c in discovered_charts if c["exists"]),
        "total_charts_tracked": len(TRACKED_CHARTS),
    }


def build_dashboard_data(
    df: pd.DataFrame,
    opening_cash_balance: float = DEFAULT_OPENING_CASH_BALANCE,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    charts_dir: Union[str, Path] = "charts",
    pdf_path: Union[str, Path] = "reports/saas_financial_report.pdf",
) -> Dict[str, Any]:
    """
    Assembles a complete, structured dashboard data dictionary by querying existing
    analytical modules.

    Args:
        df: Validated SaaS financial DataFrame.
        opening_cash_balance: Configured starting cash balance.
        forecast_horizon: Number of months to forecast.
        charts_dir: Directory where charts are located.
        pdf_path: Destination path for the PDF report.

    Returns:
        dict: Consolidated structured dashboard data.

    Raises:
        ValueError: If df is empty or missing foundational columns.
    """
    if df is None or df.empty:
        raise ValueError("Cannot build reporting dashboard from an empty DataFrame.")

    # 1. Query existing analytical modules
    revenue_df = calculate_revenue_metrics(df)
    rev_summary = generate_revenue_summary(revenue_df)

    cac_df = calculate_cac_metrics(df)
    cac_summary = generate_cac_summary(cac_df)

    churn_df = calculate_churn_retention_metrics(df)
    churn_summary = generate_churn_retention_summary(churn_df)

    mom_comp = compare_latest_vs_previous(df)
    trend_summary = generate_trend_summary(df)

    cash_flow_df = analyze_cash_flow(df, opening_cash_balance=opening_cash_balance)
    cash_flow_sum = generate_cash_flow_summary(cash_flow_df, opening_cash_balance=opening_cash_balance)

    # Handle forecast if sufficient historical data exists (>= 2 months)
    if len(df) >= 2:
        forecast_df = build_cash_flow_forecast(
            cash_flow_df,
            horizon=forecast_horizon,
            opening_cash_balance=opening_cash_balance,
        )
        forecast_sum = generate_forecast_summary(forecast_df)
        forecast_available = True
    else:
        forecast_df = None
        forecast_sum = None
        forecast_available = False

    # 2. Determine Overall Business Health Status
    overall_status = determine_overall_business_status(
        mom_comp,
        cash_flow_sum,
        forecast_sum,
    )

    # 3. Discover On-Disk Artifacts
    artifacts = get_report_artifacts(charts_dir=charts_dir, pdf_path=pdf_path)

    # 4. Extract core KPI movement rows for concise dashboard display
    core_kpis = ["MRR", "ARR", "CAC", "Churn Rate %", "Retention Rate %", "Ending Customers"]
    filtered_comp = mom_comp[mom_comp["KPI"].isin(core_kpis)].to_dict(orient="records")

    return {
        "reporting_period": {
            "first_month": str(df["Month"].iloc[0]),
            "latest_month": str(df["Month"].iloc[-1]),
            "total_months": len(df),
        },
        "forecast_period": {
            "available": forecast_available,
            "start_month": forecast_sum["forecast_start_month"] if forecast_sum else "N/A",
            "end_month": forecast_sum["forecast_end_month"] if forecast_sum else "N/A",
            "horizon": forecast_horizon,
        },
        "business_overview": {
            "latest_mrr": rev_summary["latest_mrr"],
            "latest_arr": rev_summary["latest_arr"],
            "latest_customers": churn_summary["ending_customers"],
            "latest_cac": cac_summary["latest_cac"],
            "latest_churn_rate": churn_summary["latest_churn_rate"],
            "latest_retention_rate": churn_summary["latest_retention_rate"],
            "latest_closing_cash": cash_flow_sum["latest_closing_cash_balance"],
            "forecast_ending_cash": forecast_sum["ending_forecast_cash_balance"] if forecast_sum else None,
            "overall_status": overall_status,
        },
        "revenue": rev_summary,
        "cac": cac_summary,
        "churn_retention": churn_summary,
        "cash_flow": cash_flow_sum,
        "forecast": forecast_sum,
        "kpi_movements": filtered_comp,
        "executive_narrative": trend_summary["narrative"],
        "artifacts": artifacts,
    }


def format_consolidated_dashboard(data: Dict[str, Any]) -> str:
    """
    Renders the structured dashboard data into a clean, professional,
    PowerShell-compatible text layout for console presentation.

    Args:
        data: Structured dashboard dictionary returned by build_dashboard_data().

    Returns:
        str: Formatted multi-line dashboard string.
    """
    lines = []
    w = 67
    div = "=" * w
    sub_div = "-" * w

    lines.append(div)
    lines.append("              SAAS FINANCIAL REPORTING DASHBOARD")
    lines.append(div)

    # 1. REPORTING PERIODS & OVERALL STATUS
    rep = data["reporting_period"]
    fc = data["forecast_period"]
    ov = data["business_overview"]

    lines.append(f"Historical Reporting Period : {rep['first_month']} to {rep['latest_month']} ({rep['total_months']} Months)")
    if fc["available"]:
        lines.append(f"Projected Forecast Period   : {fc['start_month']} to {fc['end_month']} ({fc['horizon']} Months)")
    else:
        lines.append("Projected Forecast Period   : Unavailable (Requires >= 2 historical months)")
    lines.append(f"Overall Business Status     : [{ov['overall_status'].upper()}]")
    lines.append(sub_div)

    # 2. SECTION 1: BUSINESS OVERVIEW
    lines.append("1. BUSINESS OVERVIEW")
    lines.append(sub_div)
    lines.append(f"  Monthly Recurring Revenue (MRR) : ${ov['latest_mrr']:,.0f}")
    lines.append(f"  Annual Recurring Revenue (ARR)  : ${ov['latest_arr']:,.0f}")
    lines.append(f"  Active Customer Base            : {ov['latest_customers']:,} accounts")
    cac_str = f"${ov['latest_cac']:,.2f}" if ov["latest_cac"] is not None else "N/A"
    lines.append(f"  Customer Acquisition Cost (CAC) : {cac_str}")
    churn_str = f"{ov['latest_churn_rate']:.2f}%" if ov["latest_churn_rate"] is not None else "N/A"
    lines.append(f"  Monthly Customer Churn Rate     : {churn_str}")
    ret_str = f"{ov['latest_retention_rate']:.2f}%" if ov["latest_retention_rate"] is not None else "N/A"
    lines.append(f"  Monthly Customer Retention Rate : {ret_str}")
    lines.append(f"  Closing Operating Cash Balance  : ${ov['latest_closing_cash']:,.0f}")
    if ov["forecast_ending_cash"] is not None:
        lines.append(f"  6-Month Projected Cash Balance  : ${ov['forecast_ending_cash']:,.0f}")
    lines.append(sub_div)

    # 3. SECTION 2: REVENUE PERFORMANCE
    rev = data["revenue"]
    lines.append("2. RECURRING REVENUE PERFORMANCE")
    lines.append(sub_div)
    lines.append(f"  Latest MRR          : ${rev['latest_mrr']:,.0f}")
    lines.append(f"  Latest ARR          : ${rev['latest_arr']:,.0f}")
    mom_str = f"{rev['latest_mrr_growth']:.2f}%" if rev["latest_mrr_growth"] is not None else "N/A"
    lines.append(f"  Latest MoM Growth   : {mom_str}")
    ov_growth_str = f"{rev['overall_mrr_growth']:.2f}%" if rev["overall_mrr_growth"] is not None else "N/A"
    lines.append(f"  Overall MRR Growth  : {ov_growth_str}")
    lines.append(f"  Net MRR Added       : ${rev['net_mrr_change']:+,.0f} (from ${rev['starting_mrr']:,.0f})")
    lines.append(sub_div)

    # 4. SECTION 3: CUSTOMER ACQUISITION (CAC)
    cac = data["cac"]
    lines.append("3. CUSTOMER ACQUISITION (CAC)")
    lines.append(sub_div)
    lines.append(f"  Latest Mktg. Spend  : ${cac['latest_marketing_spend']:,.0f}")
    lines.append(f"  Latest New Customers: {cac['latest_new_customers']:,}")
    lines.append(f"  Latest CAC          : ${cac['latest_cac']:,.2f}" if cac["latest_cac"] is not None else "  Latest CAC          : N/A")
    lines.append(f"  Previous Month CAC  : ${cac['previous_cac']:,.2f}" if cac["previous_cac"] is not None else "  Previous Month CAC  : N/A")
    cac_chg_str = f"{cac['latest_cac_change']:+.2f}%" if cac["latest_cac_change"] is not None else "N/A"
    lines.append(f"  Latest CAC Change   : {cac_chg_str}")
    ov_cac_str = f"{cac['overall_cac_change']:+.2f}%" if cac["overall_cac_change"] is not None else "N/A"
    lines.append(f"  Overall CAC Change  : {ov_cac_str}")
    lines.append(f"  Efficiency Status   : {cac['interpretation']}")
    lines.append(sub_div)

    # 5. SECTION 4: CHURN & RETENTION
    churn = data["churn_retention"]
    lines.append("4. CUSTOMER CHURN & RETENTION")
    lines.append(sub_div)
    lines.append(f"  Beginning Customers : {churn['beginning_customers']:,}")
    lines.append(f"  New Additions       : +{churn['new_customers']:,}")
    lines.append(f"  Churned Accounts    : -{churn['churned_customers']:,}")
    lines.append(f"  Ending Customers    : {churn['ending_customers']:,}")
    lines.append(f"  Monthly Churn Rate  : {churn['latest_churn_rate']:.2f}%" if churn["latest_churn_rate"] is not None else "  Monthly Churn Rate  : N/A")
    lines.append(f"  Retention Rate      : {churn['latest_retention_rate']:.2f}%" if churn["latest_retention_rate"] is not None else "  Retention Rate      : N/A")
    chg_pp_str = f"{churn['latest_churn_change_pp']:+.2f} pp" if churn["latest_churn_change_pp"] is not None else "N/A"
    lines.append(f"  MoM Churn Change    : {chg_pp_str}")
    lines.append(f"  Dynamics Status     : {churn['interpretation']}")
    lines.append("  [Note: Customer base is reconstructed from aggregated monthly totals]")
    lines.append(sub_div)

    # 6. SECTION 5: CASH FLOW
    cf = data["cash_flow"]
    lines.append("5. HISTORICAL OPERATING CASH FLOW")
    lines.append(sub_div)
    lines.append(f"  Latest Cash Inflow (MRR)   : ${cf['latest_cash_inflow']:,.0f}")
    lines.append(f"  Latest Total Cash Outflow  : ${cf['latest_total_cash_outflow']:,.0f}")
    net_sign = "+" if cf["latest_net_cash_flow"] >= 0 else "-"
    lines.append(f"  Latest Monthly Net Flow    : {net_sign}${abs(cf['latest_net_cash_flow']):,.0f}")
    lines.append(f"  Opening Cash Balance       : ${cf['latest_opening_cash_balance']:,.0f}")
    lines.append(f"  Closing Cash Balance       : ${cf['latest_closing_cash_balance']:,.0f}")
    cum_sign = "+" if cf["cumulative_net_cash_flow"] >= 0 else "-"
    lines.append(f"  Cumulative Net Cash Flow   : {cum_sign}${abs(cf['cumulative_net_cash_flow']):,.0f}")
    lines.append(f"  Cash Interpretation        : {cf['interpretation']}")
    lines.append(sub_div)

    # 7. SECTION 6: SIX-MONTH FORECAST
    lines.append("6. SIX-MONTH CASH FLOW FORECAST (PROJECTION)")
    lines.append(sub_div)
    if fc["available"] and data["forecast"]:
        f_sum = data["forecast"]
        lines.append(f"  Forecast Window            : {f_sum['forecast_start_month']} to {f_sum['forecast_end_month']} ({f_sum['forecast_months']} Months)")
        lines.append(f"  Starting Forecast Balance  : ${f_sum['starting_forecast_cash_balance']:,.0f}")
        lines.append(f"  Ending Forecast Balance    : ${f_sum['ending_forecast_cash_balance']:,.0f}")
        lines.append(f"  Lowest Projected Balance   : ${f_sum['lowest_forecast_closing_cash_balance']:,.0f}")
        lines.append(f"  Highest Projected Balance  : ${f_sum['highest_forecast_closing_cash_balance']:,.0f}")
        fn_sign = "+" if f_sum["first_forecast_net_cash_flow"] >= 0 else "-"
        lines.append(f"  First Month Net Cash Flow  : {fn_sign}${abs(f_sum['first_forecast_net_cash_flow']):,.0f}")
        ln_sign = "+" if f_sum["last_forecast_net_cash_flow"] >= 0 else "-"
        lines.append(f"  Last Month Net Cash Flow   : {ln_sign}${abs(f_sum['last_forecast_net_cash_flow']):,.0f}")
        lines.append(f"  Projected Outlook          : {f_sum['interpretation']}")
    else:
        lines.append("  Forecast Status            : Unavailable")
    lines.append(sub_div)

    # 8. SECTION 7: KPI MOVEMENT SUMMARY
    lines.append("7. PERIOD-OVER-PERIOD CORE KPI MOVEMENTS (MoM)")
    lines.append(sub_div)
    lines.append(f"  {'KPI':<18} {'Previous':<12} {'Latest':<12} {'Change':<12} {'Status'}")
    lines.append(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for row in data["kpi_movements"]:
        kpi_name = str(row.get("KPI", ""))
        prev_val = row.get("Comparison Value")
        latest_val = row.get("Latest Value")
        chg_str = row.get("Formatted Change", "")
        status_val = str(row.get("Status", "N/A"))

        # Format values cleanly
        if "MRR" in kpi_name or "ARR" in kpi_name:
            p_str = f"${prev_val:,.0f}" if pd.notnull(prev_val) else "N/A"
            l_str = f"${latest_val:,.0f}" if pd.notnull(latest_val) else "N/A"
        elif "CAC" in kpi_name:
            p_str = f"${prev_val:,.2f}" if pd.notnull(prev_val) else "N/A"
            l_str = f"${latest_val:,.2f}" if pd.notnull(latest_val) else "N/A"
        elif "Rate" in kpi_name:
            p_str = f"{prev_val:.2f}%" if pd.notnull(prev_val) else "N/A"
            l_str = f"{latest_val:.2f}%" if pd.notnull(latest_val) else "N/A"
        elif "Customers" in kpi_name:
            p_str = f"{int(prev_val):,}" if pd.notnull(prev_val) else "N/A"
            l_str = f"{int(latest_val):,}" if pd.notnull(latest_val) else "N/A"
        else:
            p_str = str(prev_val) if pd.notnull(prev_val) else "N/A"
            l_str = str(latest_val) if pd.notnull(latest_val) else "N/A"

        if not chg_str:
            abs_chg = row.get("Absolute Change")
            if pd.notnull(abs_chg):
                chg_str = f"{abs_chg:+,.0f}" if abs_chg == int(abs_chg) else f"{abs_chg:+,.2f}"
            else:
                chg_str = "N/A"

        lines.append(f"  {kpi_name:<18} {p_str:<12} {l_str:<12} {chg_str:<12} {status_val}")

    lines.append(f"\n  Executive Narrative:\n  {data['executive_narrative']}")
    lines.append(sub_div)

    # 9. SECTION 8: GENERATED ARTIFACTS
    art = data["artifacts"]
    lines.append("8. GENERATED ARTIFACTS & OUTPUT FILES")
    lines.append(sub_div)
    pdf = art["pdf_report"]
    if pdf["exists"]:
        lines.append(f"  PDF Investor Report : {pdf['path']} ({pdf['size_bytes']/1024:.1f} KB)")
    else:
        lines.append(f"  PDF Investor Report : Unavailable ({pdf['path']})")

    lines.append(f"\n  Financial Charts Discovered ({art['total_charts_found']}/{art['total_charts_tracked']}):")
    for c in art["charts"]:
        if c["exists"]:
            lines.append(f"    [FOUND] {c['title']} -> {c['path']} ({c['size_bytes']/1024:.1f} KB)")
        else:
            lines.append(f"    [MISSING] {c['title']} -> {c['path']}")

    lines.append(div)
    return "\n".join(lines)


def display_dashboard(dashboard_data: Dict[str, Any]) -> None:
    """
    Renders and prints the consolidated financial reporting dashboard to stdout.

    Args:
        dashboard_data: Structured dashboard dictionary.
    """
    output_str = format_consolidated_dashboard(dashboard_data)
    print("\n" + output_str)
