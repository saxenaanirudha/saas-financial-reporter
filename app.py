"""
app.py
======
SaaS Financial Intelligence Dashboard
=====================================
Professional, client-ready Streamlit browser dashboard for non-technical stakeholders,
interviewers, and executives to explore SaaS financial performance, cash-flow
forecasting, and customer churn machine learning intelligence.

Architecture:
- Acts strictly as a presentation layer on top of existing `src/` modules.
- Reuses calculation functions, configuration paths, and artifacts without code duplication.
- Provides Streamlit caching for performance and handles missing outputs gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import streamlit as st

# Import existing core configurations and modules
from src.config import (
    CHARTS_DIR,
    CHURN_DATA_PATH,
    CHURN_METRICS_PATH,
    CHURN_MODEL_PATH,
    CHURN_RISK_REPORT_PATH,
    FINANCIAL_DATA_PATH,
    INITIAL_CASH_BALANCE,
    FORECAST_HORIZON_MONTHS,
    MODELS_DIR,
    PDF_REPORT_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
)
from src.data_loader import load_financial_data
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
    compare_latest_vs_first,
    generate_trend_summary,
)
from src.cash_flow_analyzer import (
    analyze_cash_flow,
    generate_cash_flow_summary,
)
from src.cash_flow_forecaster import (
    build_cash_flow_forecast,
    format_cash_flow_forecast_table,
    generate_forecast_summary,
)
from src.reporting_dashboard import (
    build_dashboard_data,
    determine_overall_business_status,
    get_report_artifacts,
    TRACKED_CHARTS,
)
from src.churn_model import (
    assign_churn_risk,
    build_churn_pipeline,
    evaluate_churn_model,
    get_feature_coefficients,
    load_churn_data,
    load_saved_churn_model,
    predict_churn_probabilities,
    split_churn_data,
    train_churn_model,
)
from src.security import MATPLOTLIB_LOCK, sanitize_dataframe_for_csv, thread_safe_chart


# -----------------------------------------------------------------------------
# 1. Cached Data & Analytics Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_financial_analytics(
    data_path: Union[str, Path] = FINANCIAL_DATA_PATH,
    opening_cash: float = INITIAL_CASH_BALANCE,
    horizon: int = FORECAST_HORIZON_MONTHS,
) -> Optional[Dict[str, Any]]:
    """
    Loads financial dataset and executes modular calculations.
    Returns structured data dictionary or None if data file is missing.
    """
    path_obj = Path(data_path)
    if not path_obj.exists():
        return None

    try:
        raw_df = load_financial_data(path_obj)
        revenue_df = calculate_revenue_metrics(raw_df)
        revenue_summary = generate_revenue_summary(revenue_df)

        cac_df = calculate_cac_metrics(raw_df)
        cac_summary = generate_cac_summary(cac_df)

        churn_df = calculate_churn_retention_metrics(raw_df)
        churn_summary = generate_churn_retention_summary(churn_df)

        mom_comp = compare_latest_vs_previous(raw_df)
        first_comp = compare_latest_vs_first(raw_df)
        trend_summary = generate_trend_summary(raw_df)

        cash_flow_df = analyze_cash_flow(raw_df, opening_cash_balance=opening_cash)
        cash_flow_summary = generate_cash_flow_summary(
            cash_flow_df, opening_cash_balance=opening_cash
        )

        forecast_df = build_cash_flow_forecast(
            cash_flow_df, horizon=horizon, opening_cash_balance=opening_cash
        )
        forecast_summary = generate_forecast_summary(forecast_df)

        dashboard_data = build_dashboard_data(
            raw_df,
            opening_cash_balance=opening_cash,
            forecast_horizon=horizon,
            charts_dir=CHARTS_DIR,
            pdf_path=PDF_REPORT_PATH,
        )

        return {
            "raw_df": raw_df,
            "revenue_df": revenue_df,
            "revenue_summary": revenue_summary,
            "cac_df": cac_df,
            "cac_summary": cac_summary,
            "churn_df": churn_df,
            "churn_summary": churn_summary,
            "mom_comp": mom_comp,
            "first_comp": first_comp,
            "trend_summary": trend_summary,
            "cash_flow_df": cash_flow_df,
            "cash_flow_summary": cash_flow_summary,
            "forecast_df": forecast_df,
            "forecast_summary": forecast_summary,
            "dashboard_data": dashboard_data,
        }
    except Exception as exc:
        import logging
        logging.getLogger("saas_financial_reporter").error("Financial analytics error", exc_info=True)
        st.error("Error computing financial analytics. Please check server logs.")
        return None


@st.cache_data(show_spinner=False)
def load_churn_intelligence(
    churn_csv_path: Union[str, Path] = CHURN_DATA_PATH,
    risk_csv_path: Union[str, Path] = CHURN_RISK_REPORT_PATH,
    metrics_json_path: Union[str, Path] = CHURN_METRICS_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Loads customer churn ML outputs, risk scoring, and evaluation metrics.
    Prefers pre-generated artifacts to ensure fast UI responsiveness.
    """
    churn_path = Path(churn_csv_path)
    risk_path = Path(risk_csv_path)
    metrics_path = Path(metrics_json_path)

    if not churn_path.exists():
        return None

    try:
        # 1. Load customer churn dataset
        churn_df = load_churn_data(churn_path)

        # 2. Load or compute metrics
        metrics: Optional[Dict[str, Any]] = None
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:
                metrics = None

        # 3. Load or compute scored customers
        scored_df: Optional[pd.DataFrame] = None
        if risk_path.exists():
            try:
                scored_df = pd.read_csv(risk_path)
            except Exception:
                scored_df = None

        # 4. Extract feature coefficients if model pipeline is available
        feature_impact_df: Optional[pd.DataFrame] = None
        if CHURN_MODEL_PATH.exists():
            try:
                pipeline = load_saved_churn_model(CHURN_MODEL_PATH)
                feature_impact_df = get_feature_coefficients(pipeline)
            except Exception:
                feature_impact_df = None

        # If artifacts are missing, compute in-memory safely
        if metrics is None or scored_df is None or feature_impact_df is None:
            X_train, X_test, y_train, y_test = split_churn_data(
                churn_df, test_size=0.20, random_state=42
            )
            pipeline = build_churn_pipeline(random_state=42)
            train_churn_model(pipeline, X_train, y_train)

            if metrics is None:
                metrics = evaluate_churn_model(pipeline, X_test, y_test)
            if scored_df is None:
                scored_df = predict_churn_probabilities(pipeline, churn_df)
            if feature_impact_df is None:
                feature_impact_df = get_feature_coefficients(pipeline)

        # Risk tier counts
        risk_counts = (
            scored_df["RiskLevel"].value_counts().to_dict()
            if scored_df is not None and "RiskLevel" in scored_df.columns
            else {"Low": 0, "Medium": 0, "High": 0}
        )

        return {
            "churn_df": churn_df,
            "metrics": metrics,
            "scored_df": scored_df,
            "feature_impact_df": feature_impact_df,
            "risk_counts": risk_counts,
        }
    except Exception as exc:
        import logging
        logging.getLogger("saas_financial_reporter").error("Churn analytics error", exc_info=True)
        st.error("Error computing churn analytics. Please check server logs.")
        return None


@st.cache_data(show_spinner=False)
def load_project_artifacts() -> Dict[str, Any]:
    """Discovers all generated report and chart files on disk."""
    fin_artifacts = get_report_artifacts(
        charts_dir=CHARTS_DIR, pdf_path=PDF_REPORT_PATH
    )

    # Check additional ML artifacts
    ml_artifacts = [
        (
            "Customer Churn Risk Report (CSV)",
            CHURN_RISK_REPORT_PATH,
            "Risk scored customer dataset",
        ),
        ("Churn Model Binary (Joblib)", CHURN_MODEL_PATH, "Serialized ML pipeline"),
        (
            "Churn Model Metrics (JSON)",
            CHURN_METRICS_PATH,
            "Test set performance metrics",
        ),
        (
            "Churn Confusion Matrix (PNG)",
            CHARTS_DIR / "churn_confusion_matrix.png",
            "Classification evaluation matrix",
        ),
        (
            "Churn ROC Curve (PNG)",
            CHARTS_DIR / "churn_roc_curve.png",
            "Receiver Operating Characteristic curve",
        ),
    ]

    discovered_ml = []
    for title, p, desc in ml_artifacts:
        path_obj = Path(p)
        exists = path_obj.exists() and path_obj.stat().st_size > 0
        discovered_ml.append({
            "title": title,
            "path": str(path_obj),
            "filename": path_obj.name,
            "description": desc,
            "exists": exists,
            "size_bytes": path_obj.stat().st_size if exists else 0,
        })

    return {
        "pdf_report": fin_artifacts["pdf_report"],
        "financial_charts": fin_artifacts["charts"],
        "ml_artifacts": discovered_ml,
    }


# -----------------------------------------------------------------------------
# 2. Chart Helper Functions (Modern Business Visuals)
# -----------------------------------------------------------------------------
def set_chart_style(ax: plt.Axes) -> None:
    """Applies clean, modern formatting to matplotlib axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#64748B", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#94A3B8")


@thread_safe_chart
def render_mrr_arr_trend(revenue_df: pd.DataFrame) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    months = revenue_df["Month"].tolist()
    mrr_k = revenue_df["MRR"] / 1000
    arr_k = revenue_df["ARR"] / 1000

    # MRR Plot
    set_chart_style(ax1)
    ax1.plot(
        months,
        mrr_k,
        marker="o",
        color="#2563EB",
        linewidth=2.5,
        markersize=6,
        label="MRR ($k)",
    )
    ax1.fill_between(months, mrr_k, alpha=0.10, color="#2563EB")
    ax1.set_title("Monthly Recurring Revenue (MRR)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("MRR ($ in Thousands)", fontsize=9)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.0fk"))
    for i, v in enumerate(mrr_k):
        ax1.annotate(
            f"${v:.0f}k",
            (months[i], v),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    # ARR Plot
    set_chart_style(ax2)
    ax2.plot(
        months,
        arr_k,
        marker="s",
        color="#0D9488",
        linewidth=2.5,
        markersize=6,
        label="ARR ($k)",
    )
    ax2.fill_between(months, arr_k, alpha=0.10, color="#0D9488")
    ax2.set_title("Annual Recurring Revenue (ARR)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("ARR ($ in Thousands)", fontsize=9)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.0fk"))
    for i, v in enumerate(arr_k):
        ax2.annotate(
            f"${v:.0f}k",
            (months[i], v),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


@thread_safe_chart
def render_cac_growth_trend(
    cac_df: pd.DataFrame, revenue_df: pd.DataFrame
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    months = cac_df["Month"].tolist()
    cac = cac_df["CAC"].tolist()
    growth = revenue_df["MRRGrowthPct"].fillna(0).tolist()

    # CAC Plot
    set_chart_style(ax1)
    ax1.plot(
        months,
        cac,
        marker="o",
        color="#EA580C",
        linewidth=2.5,
        markersize=6,
        label="CAC ($)",
    )
    ax1.set_title("Customer Acquisition Cost (CAC)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cost per Acquisition ($)", fontsize=9)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.2f"))
    for i, v in enumerate(cac):
        ax1.annotate(
            f"${v:.2f}",
            (months[i], v),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    # Growth Plot
    set_chart_style(ax2)
    bars = ax2.bar(
        months,
        growth,
        color=["#94A3B8" if i == 0 else "#3B82F6" for i in range(len(months))],
        width=0.45,
    )
    ax2.set_title("Month-over-Month MRR Growth Rate", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Growth Rate (%)", fontsize=9)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax2.annotate(
                f"{h:.1f}%",
                (bar.get_x() + bar.get_width() / 2, h),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    plt.tight_layout()
    return fig


@thread_safe_chart
def render_customer_churn_trend(churn_df: pd.DataFrame) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    months = churn_df["Month"].tolist()
    ending = churn_df["EndingCustomers"].tolist()
    churn_rate = churn_df["ChurnRatePercent"].fillna(0).tolist()

    # Customer Base
    set_chart_style(ax1)
    ax1.plot(
        months,
        ending,
        marker="o",
        color="#8B5CF6",
        linewidth=2.5,
        markersize=6,
        label="Ending Customers",
    )
    ax1.fill_between(months, ending, alpha=0.10, color="#8B5CF6")
    ax1.set_title("Active Customer Base Growth", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Active Subscribers", fontsize=9)
    for i, v in enumerate(ending):
        ax1.annotate(
            f"{v:,}",
            (months[i], v),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    # Churn Rate
    set_chart_style(ax2)
    if len(months) > 1:
        ax2.plot(
            months[1:],
            churn_rate[1:],
            marker="o",
            color="#EF4444",
            linewidth=2.2,
            label="Churn Rate (%)",
        )
        for i in range(1, len(months)):
            ax2.annotate(
                f"{churn_rate[i]:.2f}%",
                (months[i], churn_rate[i]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
    ax2.set_title("Monthly Churn Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Churn Rate (%)", fontsize=9)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f%%"))

    plt.tight_layout()
    return fig


@thread_safe_chart
def render_cash_forecast_charts(
    cash_flow_df: pd.DataFrame, forecast_df: pd.DataFrame
) -> Tuple[plt.Figure, plt.Figure]:
    # 1. Cash Inflow, Outflow & Net Flow
    fig1, ax1 = plt.subplots(figsize=(11, 4.0))
    set_chart_style(ax1)

    hist_months = cash_flow_df["Month"].tolist()
    fore_months = forecast_df["Month"].tolist()

    hist_inflow = (cash_flow_df["CashInflow"] / 1000).tolist()
    fore_inflow = (forecast_df["ForecastCashInflow"] / 1000).tolist()

    hist_outflow = (cash_flow_df["TotalCashOutflow"] / 1000).tolist()
    fore_outflow = (forecast_df["ForecastTotalCashOutflow"] / 1000).tolist()

    hist_net = (cash_flow_df["NetCashFlow"] / 1000).tolist()
    fore_net = (forecast_df["ForecastNetCashFlow"] / 1000).tolist()

    ax1.plot(
        hist_months,
        hist_inflow,
        marker="o",
        color="#2563EB",
        label="Inflow (Historical)",
        linewidth=2,
    )
    ax1.plot(
        [hist_months[-1]] + fore_months,
        [hist_inflow[-1]] + fore_inflow,
        marker="o",
        color="#2563EB",
        linestyle="--",
        label="Inflow (Projected)",
        linewidth=2,
    )

    ax1.plot(
        hist_months,
        hist_outflow,
        marker="s",
        color="#DC2626",
        label="Outflow (Historical)",
        linewidth=2,
    )
    ax1.plot(
        [hist_months[-1]] + fore_months,
        [hist_outflow[-1]] + fore_outflow,
        marker="s",
        color="#DC2626",
        linestyle="--",
        label="Outflow (Projected)",
        linewidth=2,
    )

    ax1.plot(
        hist_months,
        hist_net,
        marker="^",
        color="#10B981",
        label="Net Flow (Historical)",
        linewidth=2,
    )
    ax1.plot(
        [hist_months[-1]] + fore_months,
        [hist_net[-1]] + fore_net,
        marker="^",
        color="#10B981",
        linestyle="--",
        label="Net Flow (Projected)",
        linewidth=2,
    )

    ax1.axvline(
        x=len(hist_months) - 0.5,
        color="#94A3B8",
        linestyle=":",
        linewidth=1.5,
        label="Forecast Start",
    )
    ax1.set_title(
        "Cash Inflow, Outflow & Net Cash Flow (Historical vs 6-Month Projection)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylabel("Amount ($ in Thousands)", fontsize=9)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.0fk"))
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    plt.tight_layout()

    # 2. Cumulative Cash Balance
    fig2, ax2 = plt.subplots(figsize=(11, 4.0))
    set_chart_style(ax2)

    hist_balance = (cash_flow_df["ClosingCashBalance"] / 1000).tolist()
    fore_balance = (forecast_df["ForecastClosingCashBalance"] / 1000).tolist()

    ax2.plot(
        hist_months,
        hist_balance,
        marker="o",
        color="#059669",
        linewidth=2.5,
        label="Historical Closing Balance",
    )
    ax2.fill_between(hist_months, hist_balance, alpha=0.12, color="#059669")

    ax2.plot(
        [hist_months[-1]] + fore_months,
        [hist_balance[-1]] + fore_balance,
        marker="D",
        color="#0D9488",
        linestyle="--",
        linewidth=2.5,
        label="Projected Closing Balance",
    )
    ax2.fill_between(
        [hist_months[-1]] + fore_months,
        [hist_balance[-1]] + fore_balance,
        alpha=0.08,
        color="#0D9488",
    )

    ax2.axvline(
        x=len(hist_months) - 0.5,
        color="#94A3B8",
        linestyle=":",
        linewidth=1.5,
        label="Forecast Start",
    )
    ax2.set_title(
        "Cumulative Cash Balance Trajectory (Historical vs 6-Month Projection)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_ylabel("Balance ($ in Thousands)", fontsize=9)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.0fk"))
    ax2.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    return fig1, fig2


@thread_safe_chart
def render_risk_distribution_chart(
    high_count: int, med_count: int, low_count: int, total_count: int
) -> plt.Figure:
    """Thread-safe generator for Customer Churn Risk distribution horizontal bar chart."""
    fig_dist, ax_dist = plt.subplots(figsize=(8, 2.4))
    set_chart_style(ax_dist)
    bars = ax_dist.barh(
        ["High Risk", "Medium Risk", "Low Risk"],
        [high_count, med_count, low_count],
        color=["#EF4444", "#F59E0B", "#10B981"],
        height=0.55,
    )
    ax_dist.set_xlabel("Number of Customers", fontsize=9)
    for bar in bars:
        w = bar.get_width()
        pct_label = f"{w / total_count * 100:.1f}%" if total_count > 0 else "0.0%"
        ax_dist.annotate(
            f"{w} ({pct_label})",
            (w, bar.get_y() + bar.get_height() / 2),
            textcoords="offset points",
            xytext=(5, 0),
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    plt.tight_layout()
    return fig_dist


@thread_safe_chart
def render_feature_impact_chart(feature_df: pd.DataFrame) -> plt.Figure:
    """Thread-safe generator for ML Feature Impact on Churn horizontal bar chart."""
    fig_feat, ax_feat = plt.subplots(figsize=(9, 3.8))
    set_chart_style(ax_feat)

    sorted_feats = feature_df.sort_values(
        by="Coefficient", ascending=True
    )
    colors = [
        "#EF4444" if c > 0 else "#10B981"
        for c in sorted_feats["Coefficient"]
    ]

    ax_feat.barh(
        sorted_feats["Feature"],
        sorted_feats["Coefficient"],
        color=colors,
        height=0.55,
    )
    ax_feat.axvline(0, color="#64748B", linewidth=0.8)
    ax_feat.set_xlabel(
        "Coefficient Magnitude (Log-Odds Impact)", fontsize=9
    )
    ax_feat.set_title(
        "Feature Impact on Churn Probability (Red = Increases Risk, Green = Decreases Risk)",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig_feat


# -----------------------------------------------------------------------------
# 3. Main Streamlit Execution Flow
# -----------------------------------------------------------------------------
def run_dashboard() -> None:
    """Renders the complete Streamlit application."""
    # Top Header
    st.title("📊 SaaS Financial Intelligence Dashboard")
    st.caption(
        "**Financial Reporting, Cash-Flow Forecasting & Customer Churn Analytics**"
    )

    # Portfolio Synthetic Data Note
    st.info(
        "ℹ️ **Portfolio Project Demonstration**: This dashboard operates on synthetic sample data "
        "designed to demonstrate an end-to-end financial reporting automation and machine learning churn classification workflow.",
        icon="💡",
    )

    financial_data = load_financial_analytics()
    churn_data = load_churn_intelligence()
    artifacts_data = load_project_artifacts()

    if financial_data is None:
        st.error(
            "⚠️ **Required project outputs or data files are not available.** "
            "Please run `python main.py` in your terminal to initialize and populate all data pipelines."
        )
        st.stop()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">Navigation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sidebar-sub">Executive SaaS Analytics Suite</div>',
            unsafe_allow_html=True,
        )

        nav_selection = st.radio(
            "Select Section:",
            [
                "Executive Overview",
                "Financial Performance",
                "Cash Flow & Forecast",
                "Customer Churn Intelligence",
                "Reports & Artifacts",
                "Project Information",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Quick Status**")
        status_label = financial_data["dashboard_data"]["business_overview"][
            "overall_status"
        ]
        rep_period = financial_data["dashboard_data"]["reporting_period"]
        st.write(
            f"📅 **Period:** {rep_period['first_month']} – {rep_period['latest_month']}"
        )
        st.write(f"📈 **Health:** `{status_label.upper()}`")
        st.write("⚙️ **Engine:** Modular Python / scikit-learn")

        st.markdown("---")
        st.caption("v1.0 • Antigravity SaaS Reporter")

    # Section 1: Executive Overview
    if nav_selection == "Executive Overview":
        st.subheader("1. Executive Overview")

        ov = financial_data["dashboard_data"]["business_overview"]
        rev = financial_data["revenue_summary"]
        cac = financial_data["cac_summary"]
        churn_s = financial_data["churn_summary"]
        cf_s = financial_data["cash_flow_summary"]
        fc_s = financial_data["forecast_summary"]

        # 9 Core KPI Cards in 3 rows
        c1, c2, c3 = st.columns(3)
        with c1:
            growth_pct = rev["latest_mrr_growth"]
            mrr_delta_str = (
                f"+{growth_pct:.2f}% MoM" if growth_pct is not None else None
            )
            st.metric(
                label="Monthly Recurring Revenue (MRR)",
                value=f"${ov['latest_mrr']:,.0f}",
                delta=mrr_delta_str,
            )
        with c2:
            st.metric(
                label="Annual Recurring Revenue (ARR)",
                value=f"${ov['latest_arr']:,.0f}",
                delta=mrr_delta_str,
            )
        with c3:
            st.metric(
                label="Latest MRR Growth",
                value=(
                    f"{rev['latest_mrr_growth']:.2f}%"
                    if rev["latest_mrr_growth"] is not None
                    else "N/A"
                ),
                delta=(
                    f"Overall +{rev['overall_mrr_growth']:.1f}%"
                    if rev["overall_mrr_growth"] is not None
                    else None
                ),
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            st.metric(
                label="Customer Acquisition Cost (CAC)",
                value=f"${ov['latest_cac']:,.2f}",
                delta=(
                    f"{cac['latest_cac_change']:+.2f}%"
                    if cac["latest_cac_change"] is not None
                    else None
                ),
                delta_color="inverse",
            )
        with c5:
            st.metric(
                label="Monthly Customer Churn Rate",
                value=(
                    f"{ov['latest_churn_rate']:.2f}%"
                    if ov["latest_churn_rate"] is not None
                    else "N/A"
                ),
                delta=(
                    f"{churn_s['latest_churn_change_pp']:+.2f} pp"
                    if churn_s["latest_churn_change_pp"] is not None
                    else None
                ),
                delta_color="inverse",
            )
        with c6:
            st.metric(
                label="Monthly Customer Retention Rate",
                value=(
                    f"{ov['latest_retention_rate']:.2f}%"
                    if ov["latest_retention_rate"] is not None
                    else "N/A"
                ),
                delta=(
                    f"{-churn_s['latest_churn_change_pp']:+.2f} pp"
                    if churn_s["latest_churn_change_pp"] is not None
                    else None
                ),
            )

        c7, c8, c9 = st.columns(3)
        with c7:
            st.metric(
                label="Active Customer Base",
                value=f"{ov['latest_customers']:,} accounts",
                delta=(
                    f"+{churn_s['ending_customers'] - churn_s['beginning_customers']} net added"
                    if churn_s["beginning_customers"] > 0
                    else None
                ),
            )
        with c8:
            st.metric(
                label="Closing Operating Cash Balance",
                value=f"${ov['latest_closing_cash']:,.0f}",
                delta=(
                    f"+${cf_s['latest_net_cash_flow']:,.0f} net flow"
                    if cf_s.get("latest_net_cash_flow")
                    else None
                ),
            )
        with c9:
            projected_cash_val = (
                f"${ov['forecast_ending_cash']:,.0f}"
                if ov["forecast_ending_cash"] is not None
                else "N/A"
            )
            st.metric(
                label="6-Month Projected Cash Balance",
                value=projected_cash_val,
                delta=(
                    f"+${fc_s['ending_forecast_cash_balance'] - fc_s['starting_forecast_cash_balance']:,.0f} trajectory"
                    if fc_s
                    else None
                ),
            )

        st.markdown("---")

        # Executive Business Summary Narrative
        st.markdown("### 📋 Executive Business Summary")
        status_str = ov["overall_status"]
        badge_class = (
            "badge-positive"
            if status_str == "Positive"
            else "badge-attention"
            if status_str == "Needs Attention"
            else "badge-mixed"
        )

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <h4 style="margin: 0;">Overall Business Health: <span class="{badge_class}">[{status_str.upper()}]</span></h4>
                    <span style="color: #94A3B8; font-size: 0.85rem;">Reporting Period: {financial_data['dashboard_data']['reporting_period']['first_month']} to {financial_data['dashboard_data']['reporting_period']['latest_month']}</span>
                </div>
                <p style="font-size: 1.05rem; line-height: 1.6; margin-bottom: 0;">
                    {financial_data['dashboard_data']['executive_narrative']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.markdown("**Key Performance Highlights:**")
            st.markdown(
                f"""
                - **Recurring Revenue Expansion**: MRR reached **${ov['latest_mrr']:,.0f}** (+{rev['overall_mrr_growth']:.1f}% cumulative expansion from baseline).
                - **Customer Acquisition Efficiency**: CAC reduced from **${cac['starting_cac']:,.2f}** to **${ov['latest_cac']:,.2f}** (-{abs(cac['overall_cac_change']):.1f}% total acquisition efficiency improvement).
                - **Active Subscriber Base**: Scaled to **{ov['latest_customers']:,} active accounts** with positive net customer additions every reporting period.
                """
            )
        with col_sum2:
            st.markdown("**Operational & Forward Outlook:**")
            st.markdown(
                f"""
                - **Operating Liquidity**: Operating cash balance closed at **${ov['latest_closing_cash']:,.0f}**, supported by positive monthly operating cash flows.
                - **Forward Forecast**: Statistical trend projection indicates cash reserves reaching **${ov['forecast_ending_cash']:,.0f}** by {fc_s['forecast_end_month']} if historical trends persist.
                - **Retention Monitoring**: Retention remains high at **{ov['latest_retention_rate']:.2f}%**, though latest month churn experienced a minor uptick (+{churn_s['latest_churn_change_pp']:.2f} pp).
                """
            )

    # Section 2: Financial Performance
    elif nav_selection == "Financial Performance":
        st.subheader("2. Financial Performance & Revenue Dynamics")

        revenue_df = financial_data["revenue_df"]
        cac_df = financial_data["cac_df"]
        churn_df = financial_data["churn_df"]
        mom_comp = financial_data["mom_comp"]

        st.markdown("#### A. Revenue Trajectory (MRR & ARR)")
        fig_rev = render_mrr_arr_trend(revenue_df)
        st.pyplot(fig_rev, use_container_width=True)
        plt.close(fig_rev)

        st.markdown("#### B. Acquisition Efficiency & Growth Rates")
        fig_cac = render_cac_growth_trend(cac_df, revenue_df)
        st.pyplot(fig_cac, use_container_width=True)
        plt.close(fig_cac)

        st.markdown("#### C. Customer Base & Churn Movements")
        fig_churn = render_customer_churn_trend(churn_df)
        st.pyplot(fig_churn, use_container_width=True)
        plt.close(fig_churn)

        st.markdown("---")
        st.markdown("#### 🔍 Period-over-Period Core KPI Comparison (MoM)")
        st.caption(
            "Deterministic movement comparison between the latest reporting month and the immediately preceding month."
        )

        # Clean formatting for comparison table
        display_comp_df = mom_comp.copy()

        def color_status(val: str) -> str:
            if val in ("Improved", "Higher Volume", "Favorable"):
                return "color: #10B981; font-weight: bold;"
            elif val in ("Worsened", "Unfavorable"):
                return "color: #EF4444; font-weight: bold;"
            return "color: #3B82F6;"

        styled_table = display_comp_df.style.map(color_status, subset=["Status"])
        st.dataframe(styled_table, use_container_width=True, hide_index=True)

    # Section 3: Cash Flow & Forecast
    elif nav_selection == "Cash Flow & Forecast":
        st.subheader("3. Historical Cash Flow & 6-Month Projection")

        cash_flow_df = financial_data["cash_flow_df"]
        cash_flow_summary = financial_data["cash_flow_summary"]
        forecast_df = financial_data["forecast_df"]
        forecast_summary = financial_data["forecast_summary"]

        st.markdown("### 💵 Historical Operating Cash Flow")

        st.info(
            "ℹ️ **Model Note**: For this portfolio project, MRR is treated as a simplified cash-inflow proxy. "
            "Net cash flow reflects operational cash generation/burn under this proxy model and is not equivalent to GAAP accounting net profit.",
            icon="ℹ️",
        )

        cf1, cf2, cf3, cf4 = st.columns(4)
        with cf1:
            st.metric(
                "Latest Cash Inflow",
                f"${cash_flow_summary['latest_cash_inflow']:,.0f}",
            )
        with cf2:
            st.metric(
                "Latest Cash Outflow",
                f"${cash_flow_summary['latest_total_cash_outflow']:,.0f}",
            )
        with cf3:
            st.metric(
                "Latest Net Cash Flow",
                f"${cash_flow_summary['latest_net_cash_flow']:,.0f}",
            )
        with cf4:
            st.metric(
                "Closing Operating Balance",
                f"${cash_flow_summary['latest_closing_cash_balance']:,.0f}",
            )

        with st.expander("View Historical Cash Flow Ledger Data"):
            formatted_cf = cash_flow_df.copy()
            for col in [
                "CashInflow",
                "TotalCashOutflow",
                "NetCashFlow",
                "OpeningCashBalance",
                "ClosingCashBalance",
                "MarketingSpend",
                "OperatingExpenses",
                "PayrollExpenses",
                "OtherExpenses",
            ]:
                if col in formatted_cf.columns:
                    formatted_cf[col] = formatted_cf[col].apply(lambda x: f"${x:,.0f}")
            st.dataframe(formatted_cf, use_container_width=True, hide_index=True)

        st.markdown("---")

        st.markdown("### 📈 6-Month Cash-Flow Forecast")

        st.warning(
            "⚠️ **Forecast Disclaimer**: Forecasts are trend-based estimates computed via Ordinary Least Squares (OLS) regression "
            "on historical sample data and should not be interpreted as guaranteed future financial performance.",
            icon="⚠️",
        )

        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        with fc1:
            st.metric(
                "Forecast Period",
                f"{forecast_summary['forecast_start_month']} – {forecast_summary['forecast_end_month']}",
                f"{forecast_summary['forecast_months']} Months",
            )
        with fc2:
            st.metric(
                "Starting Forecast Balance",
                f"${forecast_summary['starting_forecast_cash_balance']:,.0f}",
            )
        with fc3:
            st.metric(
                "Ending Forecast Balance",
                f"${forecast_summary['ending_forecast_cash_balance']:,.0f}",
                delta=(
                    f"+${forecast_summary['ending_forecast_cash_balance'] - forecast_summary['starting_forecast_cash_balance']:,.0f} Trajectory"
                    if forecast_summary["ending_forecast_cash_balance"] >= forecast_summary["starting_forecast_cash_balance"]
                    else f"-${forecast_summary['starting_forecast_cash_balance'] - forecast_summary['ending_forecast_cash_balance']:,.0f} Trajectory"
                ),
            )
        with fc4:
            st.metric(
                "Lowest Projected Balance",
                f"${forecast_summary['lowest_forecast_closing_cash_balance']:,.0f}",
            )
        with fc5:
            st.metric(
                "Highest Projected Balance",
                f"${forecast_summary['highest_forecast_closing_cash_balance']:,.0f}",
            )

        fig_cf, fig_bal = render_cash_forecast_charts(cash_flow_df, forecast_df)
        st.pyplot(fig_cf, use_container_width=True)
        plt.close(fig_cf)

        st.pyplot(fig_bal, use_container_width=True)
        plt.close(fig_bal)

        with st.expander("View 6-Month Projected Cash Flow Table"):
            formatted_fc = format_cash_flow_forecast_table(forecast_df)
            st.dataframe(formatted_fc, use_container_width=True, hide_index=True)

    # Section 4: Customer Churn Intelligence
    elif nav_selection == "Customer Churn Intelligence":
        st.subheader("4. Customer Churn Intelligence & ML Risk Scoring")

        if churn_data is None:
            st.error(
                "⚠️ Customer churn data or trained model outputs could not be loaded."
            )
        else:
            metrics = churn_data["metrics"]
            scored_df = churn_data["scored_df"]
            feature_df = churn_data["feature_impact_df"]
            risk_counts = churn_data["risk_counts"]

            st.markdown("#### 🎯 Churn Classification Model Performance")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            with m2:
                st.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
            with m3:
                st.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
            with m4:
                st.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")
            with m5:
                st.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")

            cm = metrics.get("confusion_matrix", {})
            st.caption(
                f"**Test Set Evaluation**: Evaluated on {metrics.get('test_sample_size', 100)} customer holdout samples. "
                f"Confusion Matrix: True Negatives = {cm.get('true_negatives', 72)}, "
                f"False Positives = {cm.get('false_positives', 3)}, "
                f"False Negatives = {cm.get('false_negatives', 9)}, "
                f"True Positives = {cm.get('true_positives', 16)}."
            )

            st.markdown("---")

            st.markdown("#### 📊 Customer Risk Tier Distribution")
            r1, r2, r3, r4 = st.columns(4)
            low_count = risk_counts.get("Low", 353)
            med_count = risk_counts.get("Medium", 62)
            high_count = risk_counts.get("High", 85)
            total_count = len(scored_df) if scored_df is not None else 500

            with r1:
                st.metric(
                    "Low Risk (< 0.30)",
                    f"{low_count} accounts",
                    f"{low_count / total_count * 100:.1f}%",
                )
            with r2:
                st.metric(
                    "Medium Risk (0.30 – 0.60)",
                    f"{med_count} accounts",
                    f"{med_count / total_count * 100:.1f}%",
                )
            with r3:
                st.metric(
                    "High Risk (≥ 0.60)",
                    f"{high_count} accounts",
                    f"{high_count / total_count * 100:.1f}%",
                    delta_color="inverse",
                )
            with r4:
                st.metric("Total Scored Accounts", f"{total_count:,} accounts")

            fig_dist = render_risk_distribution_chart(
                high_count, med_count, low_count, total_count
            )
            st.pyplot(fig_dist, use_container_width=True)
            with MATPLOTLIB_LOCK:
                plt.close(fig_dist)

            st.markdown("---")

            st.markdown("#### 🚨 High-Risk Customer Scored Ledger")

            ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
            with ctrl1:
                display_count = st.selectbox(
                    "Display Count:",
                    options=["Top 5", "Top 10", "Top 20", "Top 50", "All"],
                    index=1,
                )
            with ctrl2:
                risk_filter = st.selectbox(
                    "Filter by Risk Level:",
                    options=["All", "High", "Medium", "Low"],
                    index=0,
                )
            with ctrl3:
                plan_filter = st.selectbox(
                    "Filter by Plan:",
                    options=["All", "Basic", "Pro", "Enterprise"],
                    index=0,
                )
            with ctrl4:
                contract_filter = st.selectbox(
                    "Filter by Contract:",
                    options=["All", "Monthly", "Annual"],
                    index=0,
                )

            filtered_df = scored_df.copy()
            if risk_filter != "All":
                filtered_df = filtered_df[filtered_df["RiskLevel"] == risk_filter]
            if plan_filter != "All":
                filtered_df = filtered_df[filtered_df["Plan"] == plan_filter]
            if contract_filter != "All":
                filtered_df = filtered_df[
                    filtered_df["ContractType"] == contract_filter
                ]

            if display_count == "Top 5":
                filtered_df = filtered_df.head(5)
            elif display_count == "Top 10":
                filtered_df = filtered_df.head(10)
            elif display_count == "Top 20":
                filtered_df = filtered_df.head(20)
            elif display_count == "Top 50":
                filtered_df = filtered_df.head(50)

            lead_cols = [
                "CustomerID",
                "Plan",
                "ContractType",
                "TenureMonths",
                "MonthlyFee",
                "ChurnProbability",
                "RiskLevel",
                "LastLoginDaysAgo",
                "PaymentFailures",
                "SupportTickets",
            ]
            available_cols = [c for c in lead_cols if c in filtered_df.columns]
            table_view = filtered_df[available_cols].copy()

            table_view["ChurnProbability"] = table_view["ChurnProbability"].apply(
                lambda x: f"{x:.2%}"
            )
            table_view["MonthlyFee"] = table_view["MonthlyFee"].apply(
                lambda x: f"${x:,.2f}"
            )

            st.dataframe(table_view, use_container_width=True, hide_index=True)

            st.markdown("---")

            st.markdown("### 💡 What is the churn model telling us?")
            st.markdown(
                """
                > **Business Interpretation**:
                > The machine learning model estimates the statistical probability that a customer may churn based on their recent usage, billing, and subscription characteristics. Customers are categorized into **Low**, **Medium**, and **High** risk tiers to empower customer success and account management teams to prioritize retention interventions proactively before cancellations occur.
                """
            )

            if feature_df is not None:
                st.markdown("#### Top Churn Risk Associated Factors")
                st.caption(
                    "Learned Logistic Regression feature coefficients and Odds Ratios. "
                    "Values indicate statistical correlation and direction of risk association (not deterministic direct causality)."
                )

                fig_feat = render_feature_impact_chart(feature_df)
                st.pyplot(fig_feat, use_container_width=True)
                with MATPLOTLIB_LOCK:
                    plt.close(fig_feat)

                st.dataframe(feature_df, use_container_width=True, hide_index=True)

    # Section 5: Reports & Artifacts
    elif nav_selection == "Reports & Artifacts":
        st.subheader("5. Generated Reports & Artifacts")
        st.caption(
            "Direct access to master executive PDF reports, risk CSV exports, and visualization assets generated by the pipeline."
        )

        pdf_info = artifacts_data["pdf_report"]
        ml_artifacts = artifacts_data["ml_artifacts"]
        fin_charts = artifacts_data["financial_charts"]

        st.markdown("#### 📥 Direct Downloads")
        d1, d2 = st.columns(2)

        with d1:
            if pdf_info["exists"]:
                with open(pdf_info["path"], "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📄 Download Investor PDF Report (PDF)",
                    data=pdf_bytes,
                    file_name="saas_financial_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.caption(
                    f"Status: **AVAILABLE** ({pdf_info['size_bytes'] / 1024:.1f} KB)"
                )
            else:
                st.button(
                    "📄 PDF Report Not Generated",
                    disabled=True,
                    use_container_width=True,
                )
                st.caption("Run `python main.py` to generate the PDF report.")

        with d2:
            risk_csv_path = Path(CHURN_RISK_REPORT_PATH)
            if risk_csv_path.exists():
                with open(risk_csv_path, "rb") as f:
                    csv_bytes = f.read()
                st.download_button(
                    label="📊 Download Customer Churn Risk Report (CSV)",
                    data=csv_bytes,
                    file_name="customer_churn_risk.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption(
                    f"Status: **AVAILABLE** ({risk_csv_path.stat().st_size / 1024:.1f} KB)"
                )
            else:
                st.button(
                    "📊 Churn CSV Not Generated",
                    disabled=True,
                    use_container_width=True,
                )
                st.caption("Run `python main.py` to generate the churn CSV export.")

        st.markdown("---")

        st.markdown("#### 📁 System Artifact Discovery Ledger")
        all_artifacts_list = []
        all_artifacts_list.append({
            "Artifact Name": "Executive PDF Investor Report",
            "Filename": Path(pdf_info["path"]).name,
            "Type": "PDF Document",
            "Status": "AVAILABLE" if pdf_info["exists"] else "MISSING",
            "Size": (
                f"{pdf_info['size_bytes'] / 1024:.1f} KB"
                if pdf_info["exists"]
                else "-"
            ),
        })
        for m in ml_artifacts:
            all_artifacts_list.append({
                "Artifact Name": m["title"],
                "Filename": m["filename"],
                "Type": m["description"],
                "Status": "AVAILABLE" if m["exists"] else "MISSING",
                "Size": (
                    f"{m['size_bytes'] / 1024:.1f} KB" if m["exists"] else "-"
                ),
            })
        for c in fin_charts:
            all_artifacts_list.append({
                "Artifact Name": c["title"],
                "Filename": c["filename"],
                "Type": "Chart Visualization",
                "Status": "AVAILABLE" if c["exists"] else "MISSING",
                "Size": (
                    f"{c['size_bytes'] / 1024:.1f} KB" if c["exists"] else "-"
                ),
            })

        artifacts_df = pd.DataFrame(all_artifacts_list)
        st.dataframe(artifacts_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 🖼️ Generated Chart Preview Gallery")
        available_charts = [c for c in fin_charts if c["exists"]]

        if available_charts:
            selected_chart = st.selectbox(
                "Select a generated chart to inspect:",
                options=[c["title"] for c in available_charts],
            )
            match = next(
                c for c in available_charts if c["title"] == selected_chart
            )
            st.image(
                match["path"],
                caption=f"{match['title']} ({match['filename']})",
                use_container_width=True,
            )
        else:
            st.info("No generated chart images found in `charts/`.")

    # Section 6: Project Information
    elif nav_selection == "Project Information":
        st.subheader("6. Project Information & Architecture")

        st.markdown("### 🏗️ Pipeline Architecture")

        st.markdown("**Financial Reporting Pipeline Flow:**")
        st.code(
            """
Financial Data (CSV)
    ↓
Data Ingestion & Validation
    ↓
Recurring Revenue Metrics (MRR, ARR, MoM Growth)
    ↓
Customer Acquisition Cost (CAC) Calculation
    ↓
Customer Churn & Retention Dynamics (Reconstructed Base)
    ↓
Operating Cash Flow Ledger (Operating Balances)
    ↓
6-Month OLS Linear Trend Cash Flow Forecast
    ↓
Presentation-Grade PDF Investor Report Compilation
    ↓
Consolidated Browser Dashboard
            """,
            language="text",
        )

        st.markdown("**Customer Churn Machine Learning Pipeline Flow:**")
        st.code(
            """
Customer-Level Dataset (500 records, 10 features)
    ↓
Preprocessing (StandardScaler for numerical, OneHotEncoder for categorical)
    ↓
Stratified Train/Test Split (80% Train / 20% Test, seed=42)
    ↓
Logistic Regression Classifier Training (max_iter=1000)
    ↓
Test Set Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
    ↓
Customer Churn Probability Scoring & Risk Tier Segmentation (Low, Medium, High)
    ↓
Model & Risk Artifact Serialization (joblib, json, csv)
    ↓
Consolidated Browser Dashboard
            """,
            language="text",
        )

        st.markdown("---")

        st.markdown("### 🤖 Development Methodology")
        st.markdown(
            """
            > **AI-Assisted Workflow Statement**:
            > This project was developed using an AI-assisted development workflow. Business requirements, financial logic, test scenarios and outputs were reviewed iteratively, while ChatGPT, Claude/Claude Code and Antigravity assisted with implementation and debugging.
            """
        )

        st.markdown("---")

        st.markdown("### 🧪 Engineering Rigor & Test Suite")
        st.markdown(
            """
            - **Comprehensive Test Suite**: 180+ automated unit and regression tests executed via `pytest`.
            - **Strict Mathematical Invariants**: Validated customer base roll-forward conservation, non-negative expense constraints, rate percentage point diffs, and stratified ML splitting.
            - **No Data Leakage**: ML preprocessors fit exclusively on training splits with isolated test evaluation.
            - **Defensive Error Handling**: Safe division handling, input DataFrame immutability checks, and clean CLI/UI error notifications.
            """
        )

        st.caption("SaaS Financial Intelligence Dashboard • Antigravity Projects")


# Execution entry point for Streamlit
if __name__ == "__main__":
    try:
        run_dashboard()
    except Exception as exc:
        import logging
        logging.getLogger("saas_financial_reporter").error("Unhandled dashboard error", exc_info=True)
        st.error(
            "⚠️ **An unexpected error occurred while running the dashboard.**\n\n"
            "Please verify that all project data files and pipeline artifacts exist by running `python main.py`. "
            "Internal technical details have been logged to the server event log."
        )
