"""
Automated Investor / Management PDF Report Generator (Step 10)
=============================================================================
Generates a publication-ready, multi-page executive financial PDF report for
management review, board reporting, and investor presentations.

Architecture Rules:
1. Reuses calculations from existing modules:
   - revenue_dashboard.py
   - cac_calculator.py
   - churn_retention_analyzer.py
   - kpi_comparator.py
   - cash_flow_analyzer.py
   - cash_flow_forecaster.py
   - chart_generator.py
2. Does NOT duplicate business or financial logic.
3. Formats tables, embeds available charts gracefully, and presents clear
   disclosures, assumptions, and modeling limitations.
4. Handles missing optional charts with warnings rather than crashing.
=============================================================================
"""

from datetime import datetime
import os
from pathlib import Path
import tempfile
from typing import Union, Optional, Dict, Any, List
import logging
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas
from src.security import safe_xml_escape

# Import project modules to reuse existing calculations
from src.revenue_dashboard import (
    calculate_revenue_metrics,
    generate_revenue_summary,
    format_revenue_dashboard,
)
from src.cac_calculator import (
    calculate_cac_metrics,
    generate_cac_summary,
    format_cac_table,
)
from src.churn_retention_analyzer import (
    calculate_churn_retention_metrics,
    generate_churn_retention_summary,
    format_churn_retention_table,
)
from src.kpi_comparator import (
    compare_latest_vs_previous,
    generate_trend_summary,
    format_comparison_table,
)
from src.cash_flow_analyzer import (
    analyze_cash_flow,
    generate_cash_flow_summary,
    format_cash_flow_table,
    DEFAULT_OPENING_CASH_BALANCE,
)
from src.cash_flow_forecaster import (
    build_cash_flow_forecast,
    generate_forecast_summary,
    format_cash_flow_forecast_table,
    DEFAULT_FORECAST_HORIZON,
)
from src.chart_generator import generate_all_charts

logger = logging.getLogger(__name__)


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render total page count
    and professional running headers and footers on every page.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                40,
                755,
                "SaaS Financial Reporter — Executive Management & Investor Report",
            )
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(40, 748, 572, 748)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(40, 42, 572, 42)

        gen_date = datetime.now().strftime("%B %d, %Y")
        self.drawString(40, 30, f"Confidential — For Management & Investor Review Only | Generated: {gen_date}")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 30, page_str)

        self.restoreState()


def _get_report_styles() -> Dict[str, ParagraphStyle]:
    """Builds a curated palette of clean typography styles for the PDF."""
    base_styles = getSampleStyleSheet()

    styles = {
        "ReportTitle": ParagraphStyle(
            "ReportTitle",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=4,
        ),
        "ReportSubtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=12,
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "SubSectionHeading": ParagraphStyle(
            "SubSectionHeading",
            parent=base_styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#334155"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "BodyText": ParagraphStyle(
            "ReportBodyText",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6,
        ),
        "CalloutText": ParagraphStyle(
            "CalloutText",
            parent=base_styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#1E40AF"),
            spaceAfter=6,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=1,  # Centered
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#1E293B"),
            alignment=1,  # Centered
        ),
        "TableCellLeft": ParagraphStyle(
            "TableCellLeft",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#1E293B"),
            alignment=0,  # Left
        ),
        "SummaryKey": ParagraphStyle(
            "SummaryKey",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#475569"),
        ),
        "SummaryVal": ParagraphStyle(
            "SummaryVal",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#0F172A"),
        ),
        "Disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=4,
        ),
    }
    return styles


def _df_to_reportlab_table(
    df: pd.DataFrame,
    styles: Dict[str, ParagraphStyle],
    col_widths: Optional[List[float]] = None,
    align_left_cols: Optional[List[int]] = None,
) -> Table:
    """Converts a formatted pandas DataFrame into a styled ReportLab Table."""
    if align_left_cols is None:
        align_left_cols = [0]

    header_row = [Paragraph(safe_xml_escape(col), styles["TableHeader"]) for col in df.columns]
    data_rows = []
    for _, row in df.iterrows():
        row_cells = []
        for c_idx, val in enumerate(row):
            style = styles["TableCellLeft"] if c_idx in align_left_cols else styles["TableCell"]
            row_cells.append(Paragraph(safe_xml_escape(val), style))
        data_rows.append(row_cells)

    table_data = [header_row] + data_rows
    t = Table(table_data, colWidths=col_widths)

    t_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ])

    # Alternate row shading
    for i in range(1, len(table_data)):
        bg_color = colors.HexColor("#F8FAFC") if i % 2 == 1 else colors.white
        t_style.add("BACKGROUND", (0, i), (-1, i), bg_color)

    t.setStyle(t_style)
    return t


def _safe_add_image(chart_path: Union[str, Path], width: float = 6.2 * inch, height: float = 3.1 * inch) -> Optional[Image]:
    """Safely loads and proportionally sizes a chart image; returns None if missing."""
    p = Path(chart_path)
    if not p.exists() or p.stat().st_size == 0:
        logger.warning(f"Chart image not found or empty at: {p}")
        return None
    try:
        img = Image(str(p), width=width, height=height)
        img.hAlign = "CENTER"
        return img
    except Exception as e:
        logger.warning(f"Failed to load chart image {p}: {e}")
        return None


def generate_investor_report(
    df: pd.DataFrame,
    output_path: Union[str, Path] = "reports/saas_financial_report.pdf",
    charts_dir: Union[str, Path] = "charts",
    opening_cash_balance: float = DEFAULT_OPENING_CASH_BALANCE,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
) -> Path:
    """
    Master function: generates the complete SaaS Financial Performance PDF report.

    Args:
        df: Validated SaaS financial DataFrame.
        output_path: Destination path for the generated PDF.
        charts_dir: Directory where KPI charts are stored/generated.
        opening_cash_balance: Configured starting cash balance.
        forecast_horizon: Number of months to forecast (default 6).

    Returns:
        Path: Resolved path to the generated PDF file.

    Raises:
        ValueError: If df is empty or missing foundational columns.
    """
    if df is None or df.empty:
        raise ValueError("Cannot generate PDF report from an empty DataFrame.")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    charts_path = Path(charts_dir)
    charts_path.mkdir(parents=True, exist_ok=True)

    # 1. Ensure all calculations are produced via existing modules
    revenue_df = calculate_revenue_metrics(df)
    rev_summary = generate_revenue_summary(revenue_df)
    rev_table = format_revenue_dashboard(revenue_df)

    cac_df = calculate_cac_metrics(df)
    cac_summary = generate_cac_summary(cac_df)
    cac_table = format_cac_table(cac_df)

    churn_df = calculate_churn_retention_metrics(df)
    churn_summary = generate_churn_retention_summary(churn_df)
    churn_table = format_churn_retention_table(churn_df)

    mom_comp = compare_latest_vs_previous(df)
    trend_summary = generate_trend_summary(df)
    mom_table = format_comparison_table(mom_comp)

    cash_flow_df = analyze_cash_flow(df, opening_cash_balance=opening_cash_balance)
    cash_flow_sum = generate_cash_flow_summary(cash_flow_df, opening_cash_balance=opening_cash_balance)
    cash_flow_table = format_cash_flow_table(cash_flow_df)

    forecast_df = build_cash_flow_forecast(
        cash_flow_df,
        horizon=forecast_horizon,
        opening_cash_balance=opening_cash_balance,
    )
    forecast_sum = generate_forecast_summary(forecast_df)
    forecast_table = format_cash_flow_forecast_table(forecast_df)

    # Ensure charts are generated on disk
    generate_all_charts(cash_flow_df, charts_path, forecast_df=forecast_df)

    # 2. Build Document Flow using temporary file for atomic persistence
    temp_fd, temp_path_str = tempfile.mkstemp(
        dir=out_file.parent,
        prefix=f".tmp_{out_file.stem}_",
        suffix=out_file.suffix,
    )
    os.close(temp_fd)
    temp_pdf_path = Path(temp_path_str)

    doc = SimpleDocTemplate(
        str(temp_pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=46,
    )

    styles = _get_report_styles()
    story = []

    # ── COVER / TITLE ────────────────────────────────────────────────────────
    story.append(Paragraph("SaaS Financial Performance Report", styles["ReportTitle"]))
    first_month = str(df["Month"].iloc[0])
    last_month = str(df["Month"].iloc[-1])
    fc_first = forecast_sum["forecast_start_month"]
    fc_last = forecast_sum["forecast_end_month"]
    gen_date = datetime.now().strftime("%B %d, %Y")

    subtitle_text = (
        f"<b>Historical Reporting Period:</b> {safe_xml_escape(first_month)} — {safe_xml_escape(last_month)} &nbsp;|&nbsp; "
        f"<b>Forecast Period:</b> {safe_xml_escape(fc_first)} — {safe_xml_escape(fc_last)} &nbsp;|&nbsp; "
        f"<b>Generated:</b> {gen_date}"
    )
    story.append(Paragraph(subtitle_text, styles["ReportSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0F172A"), spaceAfter=10))

    # ── 1. EXECUTIVE SUMMARY ────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", styles["SectionHeading"]))
    
    exec_summary_data = [
        [
            Paragraph("Latest Month MRR", styles["SummaryKey"]),
            Paragraph(f"${rev_summary['latest_mrr']:,.0f}", styles["SummaryVal"]),
            Paragraph("Ending Active Customers", styles["SummaryKey"]),
            Paragraph(f"{churn_summary['ending_customers']:,}", styles["SummaryVal"]),
        ],
        [
            Paragraph("Annual Recurring Revenue (ARR)", styles["SummaryKey"]),
            Paragraph(f"${rev_summary['latest_arr']:,.0f}", styles["SummaryVal"]),
            Paragraph("Customer Acquisition Cost (CAC)", styles["SummaryKey"]),
            Paragraph(f"${cac_summary['latest_cac']:,.2f}" if cac_summary["latest_cac"] else "N/A", styles["SummaryVal"]),
        ],
        [
            Paragraph("Latest MoM MRR Growth", styles["SummaryKey"]),
            Paragraph(f"{rev_summary['latest_mrr_growth']:.2f}%" if rev_summary["latest_mrr_growth"] else "N/A", styles["SummaryVal"]),
            Paragraph("Latest Monthly Churn Rate", styles["SummaryKey"]),
            Paragraph(f"{churn_summary['latest_churn_rate']:.2f}%" if churn_summary["latest_churn_rate"] else "N/A", styles["SummaryVal"]),
        ],
        [
            Paragraph("Latest Net Cash Flow", styles["SummaryKey"]),
            Paragraph(f"+${cash_flow_sum['latest_net_cash_flow']:,.0f}" if cash_flow_sum["latest_net_cash_flow"] >= 0 else f"-${abs(cash_flow_sum['latest_net_cash_flow']):,.0f}", styles["SummaryVal"]),
            Paragraph("Latest Retention Rate", styles["SummaryKey"]),
            Paragraph(f"{churn_summary['latest_retention_rate']:.2f}%" if churn_summary["latest_retention_rate"] else "N/A", styles["SummaryVal"]),
        ],
        [
            Paragraph("Historical Closing Cash", styles["SummaryKey"]),
            Paragraph(f"${cash_flow_sum['latest_closing_cash_balance']:,.0f}", styles["SummaryVal"]),
            Paragraph("6-Month Projected Cash", styles["SummaryKey"]),
            Paragraph(f"${forecast_sum['ending_forecast_cash_balance']:,.0f}", styles["SummaryVal"]),
        ],
    ]
    exec_table = Table(exec_summary_data, colWidths=[130, 135, 140, 135])
    exec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 6))

    narrative_text = (
        f"<b>Performance Overview:</b> {safe_xml_escape(trend_summary['narrative'])} "
        f"{safe_xml_escape(cash_flow_sum['interpretation'])} {safe_xml_escape(forecast_sum['interpretation'])}"
    )
    story.append(Paragraph(narrative_text, styles["BodyText"]))
    story.append(Spacer(1, 8))

    # ── 2. REVENUE PERFORMANCE ──────────────────────────────────────────────
    story.append(Paragraph("2. Recurring Revenue Performance (MRR & ARR)", styles["SectionHeading"]))
    story.append(_df_to_reportlab_table(rev_table, styles, col_widths=[90, 140, 150, 160]))
    story.append(Spacer(1, 6))

    # Embed revenue charts
    mrr_img = _safe_add_image(charts_path / "mrr_trend.png", width=5.8 * inch, height=2.6 * inch)
    if mrr_img:
        story.append(KeepTogether([mrr_img, Spacer(1, 8)]))

    arr_img = _safe_add_image(charts_path / "arr_trend.png", width=5.8 * inch, height=2.6 * inch)
    if arr_img:
        story.append(KeepTogether([arr_img, Spacer(1, 8)]))

    # ── 3. CUSTOMER ACQUISITION (CAC) ────────────────────────────────────────
    story.append(Paragraph("3. Customer Acquisition Cost (CAC) & Efficiency", styles["SectionHeading"]))
    story.append(_df_to_reportlab_table(cac_table, styles, col_widths=[80, 110, 120, 110, 120]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Acquisition Efficiency Status:</b> {safe_xml_escape(cac_summary['interpretation'])} Starting CAC of ${cac_summary['starting_cac']:.2f} moved to ${cac_summary['latest_cac']:.2f} ({cac_summary['overall_cac_change']:+.2f}% overall change).", styles["BodyText"]))
    story.append(Spacer(1, 4))

    cac_img = _safe_add_image(charts_path / "cac_trend.png", width=5.8 * inch, height=2.6 * inch)
    if cac_img:
        story.append(KeepTogether([cac_img, Spacer(1, 8)]))

    # ── 4. CHURN & RETENTION ANALYSIS ────────────────────────────────────────
    story.append(Paragraph("4. Customer Base Roll-Forward, Churn & Retention", styles["SectionHeading"]))
    story.append(_df_to_reportlab_table(churn_table, styles, col_widths=[65, 65, 55, 60, 65, 75, 75, 80]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Retention Dynamics:</b> {safe_xml_escape(churn_summary['interpretation'])} Customer base expanded from {df['NewCustomers'].iloc[0] - df['ChurnedCustomers'].iloc[0]} to {churn_summary['ending_customers']} active subscribers across {churn_summary['total_new_customers']} total additions and {churn_summary['total_churned_customers']} churned accounts.", styles["BodyText"]))
    story.append(Paragraph("<i>Note: Customer base is reconstructed from monthly aggregated customer movement totals.</i>", styles["Disclaimer"]))
    story.append(Spacer(1, 4))

    churn_img = _safe_add_image(charts_path / "churn_retention_trend.png", width=5.8 * inch, height=2.6 * inch)
    if churn_img:
        story.append(KeepTogether([churn_img, Spacer(1, 8)]))

    cust_img = _safe_add_image(charts_path / "customer_base_growth.png", width=5.8 * inch, height=2.6 * inch)
    if cust_img:
        story.append(KeepTogether([cust_img, Spacer(1, 8)]))

    # ── 5. HISTORICAL KPI COMPARISON ─────────────────────────────────────────
    story.append(Paragraph("5. Period-over-Period KPI Comparison (Latest vs Previous Month)", styles["SectionHeading"]))
    story.append(_df_to_reportlab_table(mom_table, styles, col_widths=[110, 85, 85, 85, 95, 80]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Overall Period Health:</b> {safe_xml_escape(trend_summary['overall_status'])}. Key movements show consistent revenue addition alongside controlled customer acquisition costs.", styles["BodyText"]))
    story.append(Spacer(1, 8))

    # ── 6. HISTORICAL CASH FLOW ANALYSIS ─────────────────────────────────────
    story.append(Paragraph("6. Historical Operating Cash Flow & Cash Balance", styles["SectionHeading"]))
    story.append(_df_to_reportlab_table(cash_flow_table, styles, col_widths=[75, 90, 90, 95, 95, 95]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"<b>Cash Position Summary:</b> Starting from an initial cash balance of ${cash_flow_sum['starting_cash_balance']:,.0f}, "
        f"cumulative cash inflow reached ${cash_flow_sum['total_historical_cash_inflow']:,.0f} against total operating outflow of ${cash_flow_sum['total_historical_cash_outflow']:,.0f}, "
        f"yielding cumulative net cash flow of +${cash_flow_sum['cumulative_net_cash_flow']:,.0f} and a closing cash position of {safe_xml_escape(cash_flow_sum['latest_cash_position'])}.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 4))

    net_cf_img = _safe_add_image(charts_path / "net_cash_flow_trend.png", width=5.8 * inch, height=2.5 * inch)
    if net_cf_img:
        story.append(KeepTogether([net_cf_img, Spacer(1, 8)]))

    bal_img = _safe_add_image(charts_path / "cash_balance_trend.png", width=5.8 * inch, height=2.5 * inch)
    if bal_img:
        story.append(KeepTogether([bal_img, Spacer(1, 8)]))

    # ── 7. SIX-MONTH CASH FLOW FORECAST ──────────────────────────────────────
    story.append(Paragraph("7. Six-Month Cash Flow Forecast (Statistical Trend Extrapolation)", styles["SectionHeading"]))
    story.append(Paragraph(
        f"<b>FORECAST WINDOW:</b> {forecast_sum['forecast_start_month']} to {forecast_sum['forecast_end_month']} (6 Projected Months). "
        "Forecast estimates individual revenue and expense categories independently via linear regression trend fitting.",
        styles["CalloutText"],
    ))
    story.append(Spacer(1, 2))
    story.append(_df_to_reportlab_table(forecast_table, styles, col_widths=[75, 90, 90, 95, 95, 95]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"<b>Forecast Outlook:</b> {forecast_sum['interpretation']} "
        f"Projected monthly cash inflow expands from ${forecast_sum['first_forecast_cash_inflow']:,.0f} to ${forecast_sum['last_forecast_cash_inflow']:,.0f}, "
        f"while closing cash balances are estimated to progress from ${forecast_sum['starting_forecast_cash_balance']:,.0f} to ${forecast_sum['ending_forecast_cash_balance']:,.0f}.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 4))

    fc_cf_img = _safe_add_image(charts_path / "cash_flow_forecast.png", width=5.8 * inch, height=2.6 * inch)
    if fc_cf_img:
        story.append(KeepTogether([fc_cf_img, Spacer(1, 8)]))

    fc_bal_img = _safe_add_image(charts_path / "cash_balance_forecast.png", width=5.8 * inch, height=2.6 * inch)
    if fc_bal_img:
        story.append(KeepTogether([fc_bal_img, Spacer(1, 8)]))

    # ── 8. ASSUMPTIONS & MODELING LIMITATIONS ────────────────────────────────
    story.append(Paragraph("8. Disclosures, Assumptions & Limitations", styles["SectionHeading"]))
    disclaimers = [
        "1. <b>Illustrative Sample Data:</b> The dataset contains sample portfolio data for operational demonstration and does not reflect actual proprietary records of any business entity.",
        "2. <b>Operating Cash Flow Simplification:</b> Monthly Recurring Revenue (MRR) is treated as monthly cash inflow. In enterprise accounting, recognized revenue, invoiced amounts, and collected cash differ due to invoicing cycles and payment collection lags.",
        "3. <b>Customer Base Reconstruction:</b> Active customer balances are reconstructed chronologically from aggregated monthly customer addition and churn counts rather than individual subscriber event logs.",
        "4. <b>Statistical Trend Forecasting:</b> Forecasting utilizes ordinary least squares (OLS) linear regression on historical sequence indices. It is an extrapolation of historical trajectory, not a guarantee of future business performance.",
        "5. <b>Excluded Factors:</b> The forecast model does not account for seasonality, pricing overhauls, equity/debt financings, macroeconomic downturns, or unexpected operational shocks.",
        "6. <b>Floor Constraint:</b> Forecast revenue and expense line components are non-negative (floored at zero). Net cash flow and cash balances are unconstrained.",
        "7. <b>Not GAAP/IFRS Accounting:</b> This report is generated by a financial analytics engine and does not constitute audited financial statements under GAAP or IFRS frameworks.",
        "8. <b>Not Financial Advice:</b> This report is intended strictly for internal management reporting and educational/demonstration review. It does not constitute investment or financial advisory guidance.",
    ]
    for d in disclaimers:
        story.append(Paragraph(d, styles["Disclaimer"]))

    # 3. Build the PDF document and atomically commit to destination
    try:
        doc.build(story, canvasmaker=NumberedCanvas)
        os.replace(temp_pdf_path, out_file)
    except Exception:
        if temp_pdf_path.exists():
            try:
                temp_pdf_path.unlink()
            except OSError:
                pass
        raise
    return out_file
