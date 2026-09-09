"""
Historical KPI Comparison & Trend Analysis Module
=============================================================================
Provides structured historical comparison of key SaaS KPIs across reporting
periods (Latest Month vs Previous Month, and Latest Month vs First Month).

Architecture Rule:
- Reuses calculation outputs from existing modules (revenue_dashboard,
  cac_calculator, churn_retention_analyzer).
- Does NOT recalculate underlying financial formulas.
- Applies KPI-specific interpretation rules (Higher is Better vs Lower is Better).
- Uses percentage points (pp) for rate-based metrics.
=============================================================================
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from src.revenue_dashboard import calculate_revenue_metrics
from src.cac_calculator import calculate_cac_metrics
from src.churn_retention_analyzer import calculate_churn_retention_metrics


# KPI Metadata configuration for calculations, formatting, and direction rules
KPI_CONFIG = [
    {
        "name": "MRR",
        "key": "MRR",
        "type": "currency",
        "direction": "higher_is_better",
    },
    {
        "name": "ARR",
        "key": "ARR",
        "type": "currency",
        "direction": "higher_is_better",
    },
    {
        "name": "MRR Growth %",
        "key": "MRRGrowthPct",
        "type": "rate",
        "direction": "higher_is_better",
    },
    {
        "name": "CAC",
        "key": "CAC",
        "type": "currency_float",
        "direction": "lower_is_better",
    },
    {
        "name": "Churn Rate %",
        "key": "ChurnRatePercent",
        "type": "rate",
        "direction": "lower_is_better",
    },
    {
        "name": "Retention Rate %",
        "key": "RetentionRatePercent",
        "type": "rate",
        "direction": "higher_is_better",
    },
    {
        "name": "Ending Customers",
        "key": "EndingCustomers",
        "type": "integer",
        "direction": "higher_is_better",
    },
    {
        "name": "New Customers",
        "key": "NewCustomers",
        "type": "integer",
        "direction": "volume_higher",
    },
    {
        "name": "Churned Customers",
        "key": "ChurnedCustomers",
        "type": "integer",
        "direction": "volume_lower",
    },
    {
        "name": "Marketing Spend",
        "key": "MarketingSpend",
        "type": "currency",
        "direction": "neutral",
    },
]


def build_kpi_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assembles a standardized monthly KPI snapshot containing all calculated metrics.
    Reuses existing step 3, step 4, and step 5 modules without recalculating formulas.

    Raises:
        ValueError: If input DataFrame is empty.

    Args:
        df: Validated financial DataFrame.

    Returns:
        pd.DataFrame: Enriched DataFrame with all monthly KPI columns.
    """
    if df.empty:
        raise ValueError("Cannot build KPI snapshot from an empty DataFrame.")

    data = df.copy()

    if "MRR" not in data.columns or "ARR" not in data.columns or "MRRGrowthPct" not in data.columns:
        data = calculate_revenue_metrics(data)

    if "CAC" not in data.columns:
        data = calculate_cac_metrics(data)

    if "EndingCustomers" not in data.columns or "ChurnRatePercent" not in data.columns:
        data = calculate_churn_retention_metrics(data)

    return data


def _determine_status(
    direction_rule: str,
    abs_change: Optional[float],
    pct_or_pp_change: Optional[float],
    is_rate: bool = False,
) -> str:
    """
    Evaluates business direction (Improved, Worsened, Stable, etc.) based on KPI rules.
    """
    if abs_change is None or pd.isna(abs_change):
        return "N/A"

    # Tolerance for stability
    if is_rate:
        if abs(abs_change) < 0.005:
            return "Stable"
    else:
        if abs(abs_change) < 1e-6 or (pct_or_pp_change is not None and abs(pct_or_pp_change) < 0.01):
            return "Stable"

    if direction_rule == "higher_is_better":
        return "Improved" if abs_change > 0 else "Worsened"
    elif direction_rule == "lower_is_better":
        return "Improved" if abs_change < 0 else "Worsened"
    elif direction_rule == "volume_higher":
        return "Higher Volume" if abs_change > 0 else "Lower Volume"
    elif direction_rule == "volume_lower":
        return "Favorable" if abs_change < 0 else "Unfavorable"
    elif direction_rule == "neutral":
        return "Increased" if abs_change > 0 else "Decreased"
    else:
        return "Unchanged"


def compare_periods(
    df: pd.DataFrame,
    current_idx: int = -1,
    comparison_idx: int = -2,
    comparison_label: str = "Previous",
) -> pd.DataFrame:
    """
    Compares two specific monthly rows in a KPI snapshot DataFrame.

    Args:
        df: Validated or enriched financial DataFrame.
        current_idx: Index of current month row (default: -1 for latest month).
        comparison_idx: Index of comparison month row (default: -2 for previous month).
        comparison_label: Label for the comparison column.

    Returns:
        pd.DataFrame: Structured comparison table with current, comparison, diff, change, status.
    """
    snapshot = build_kpi_snapshot(df)
    num_rows = len(snapshot)

    # Normalize negative indices
    if current_idx < 0:
        current_idx = num_rows + current_idx
    if comparison_idx < 0:
        comparison_idx = num_rows + comparison_idx

    # Check index boundary edge cases
    if num_rows == 0 or current_idx < 0 or current_idx >= num_rows:
        raise ValueError(f"Invalid current_idx ({current_idx}) for DataFrame of length {num_rows}.")

    if comparison_idx < 0 or comparison_idx >= num_rows:
        # Cannot perform comparison (e.g. single row data comparing vs previous)
        results = []
        for cfg in KPI_CONFIG:
            results.append({
                "KPI": cfg["name"],
                "Comparison Month": "N/A",
                "Comparison Value": np.nan,
                "Latest Month": str(snapshot.iloc[current_idx]["Month"]),
                "Latest Value": snapshot.iloc[current_idx].get(cfg["key"], np.nan),
                "Absolute Change": np.nan,
                "Change % / pp": np.nan,
                "Is Rate": cfg["type"] == "rate",
                "Status": "Insufficient Data",
            })
        return pd.DataFrame(results)

    curr_row = snapshot.iloc[current_idx]
    comp_row = snapshot.iloc[comparison_idx]

    curr_month = str(curr_row["Month"])
    comp_month = str(comp_row["Month"])

    results = []

    for cfg in KPI_CONFIG:
        kpi_name = cfg["name"]
        key = cfg["key"]
        kpi_type = cfg["type"]
        direction = cfg["direction"]

        curr_val = curr_row.get(key, np.nan)
        comp_val = comp_row.get(key, np.nan)

        # Handle float/int conversions cleanly
        curr_val = float(curr_val) if pd.notnull(curr_val) else np.nan
        comp_val = float(comp_val) if pd.notnull(comp_val) else np.nan

        if pd.isna(curr_val) or pd.isna(comp_val):
            abs_change = np.nan
            change_pct_pp = np.nan
            status = "N/A"
        elif kpi_type == "rate":
            # Rate KPIs use Percentage Points (pp)
            abs_change = curr_val - comp_val
            change_pct_pp = abs_change
            status = _determine_status(direction, abs_change, change_pct_pp, is_rate=True)
        else:
            # Regular numeric KPIs use percentage change %
            abs_change = curr_val - comp_val
            if comp_val != 0:
                change_pct_pp = (abs_change / comp_val) * 100.0
            else:
                change_pct_pp = np.nan
            status = _determine_status(direction, abs_change, change_pct_pp, is_rate=False)

        results.append({
            "KPI": kpi_name,
            "Comparison Month": comp_month,
            "Comparison Value": comp_val,
            "Latest Month": curr_month,
            "Latest Value": curr_val,
            "Absolute Change": abs_change,
            "Change % / pp": change_pct_pp,
            "Is Rate": kpi_type == "rate",
            "Status": status,
        })

    return pd.DataFrame(results)


def compare_latest_vs_previous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares Latest Month vs Previous Month.
    """
    return compare_periods(df, current_idx=-1, comparison_idx=-2, comparison_label="Previous")


def compare_latest_vs_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares Latest Month vs First Available Month.
    """
    return compare_periods(df, current_idx=-1, comparison_idx=0, comparison_label="First")


def generate_trend_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a concise management-oriented trend summary and overall status.

    Overall Status Rules:
    - Positive: Core revenue & customer metrics improve with no major adverse churn spike.
    - Mixed: Favorable revenue growth coexists with adverse CAC/churn trends.
    - Stable: Changes are negligible across most KPIs.
    - Needs Attention: Multiple key metrics (MRR, Churn) worsen materially.
    """
    if df.empty:
        raise ValueError("Cannot generate trend summary from an empty DataFrame.")

    snapshot = build_kpi_snapshot(df)
    num_rows = len(snapshot)

    if num_rows < 2:
        return {
            "latest_vs_previous_available": False,
            "overall_status": "Insufficient Data",
            "summary_statements": ["Comparison requires at least two months of data."],
            "narrative": "Single month of data available. Baseline established.",
        }

    mom_comp = compare_latest_vs_previous(snapshot)
    first_comp = compare_latest_vs_first(snapshot)

    statements: List[str] = []

    # Map KPI metrics for easy lookup
    mom_dict = {row["KPI"]: row for _, row in mom_comp.iterrows()}
    first_dict = {row["KPI"]: row for _, row in first_comp.iterrows()}

    # Statement 1: MRR
    mrr_row = mom_dict.get("MRR")
    if mrr_row is not None and pd.notna(mrr_row["Change % / pp"]):
        statements.append(
            f"MRR increased by {mrr_row['Change % / pp']:.2f}% compared with the previous month."
            if mrr_row["Change % / pp"] > 0
            else f"MRR decreased by {abs(mrr_row['Change % / pp']):.2f}% compared with the previous month."
        )

    # Statement 2: ARR
    arr_row = mom_dict.get("ARR")
    if arr_row is not None and pd.notna(arr_row["Change % / pp"]):
        statements.append("ARR increased in line with MRR." if arr_row["Change % / pp"] > 0 else "ARR decreased in line with MRR.")

    # Statement 3: CAC
    cac_row = mom_dict.get("CAC")
    if cac_row is not None and pd.notna(cac_row["Change % / pp"]):
        direction_word = "decreased" if cac_row["Change % / pp"] < 0 else "increased"
        efficiency_word = "improved acquisition efficiency" if cac_row["Change % / pp"] < 0 else "increased acquisition cost"
        statements.append(f"CAC {direction_word} by {abs(cac_row['Change % / pp']):.2f}%, indicating {efficiency_word}.")

    # Statement 4: Churn
    churn_row = mom_dict.get("Churn Rate %")
    if churn_row is not None and pd.notna(churn_row["Change % / pp"]):
        word = "increased" if churn_row["Change % / pp"] > 0 else "improved"
        statements.append(f"Churn {word} by {abs(churn_row['Change % / pp']):.2f} percentage points.")

    # Statement 5: Retention
    ret_row = mom_dict.get("Retention Rate %")
    if ret_row is not None and pd.notna(ret_row["Change % / pp"]):
        word = "increased" if ret_row["Change % / pp"] > 0 else "decreased"
        statements.append(f"Retention {word} by {abs(ret_row['Change % / pp']):.2f} percentage points.")

    # Statement 6: Customer Base
    cust_row = mom_dict.get("Ending Customers")
    if cust_row is not None and pd.notna(cust_row["Latest Value"]):
        comp_val = int(cust_row["Comparison Value"]) if pd.notna(cust_row["Comparison Value"]) else 0
        latest_val = int(cust_row["Latest Value"])
        statements.append(f"Ending customer base increased from {comp_val:,} to {latest_val:,}.")

    # Determine Overall Status
    core_improved = sum(1 for _, r in mom_comp.iterrows() if r["KPI"] in ["MRR", "ARR", "CAC", "Ending Customers"] and r["Status"] == "Improved")
    critical_worsened = sum(1 for _, r in mom_comp.iterrows() if r["KPI"] in ["MRR", "Ending Customers"] and r["Status"] == "Worsened")

    if core_improved >= 3 and critical_worsened == 0:
        overall_status = "Positive"
    elif critical_worsened > 0 or worsened_count >= 5:
        overall_status = "Needs Attention"
    elif improved_count > 0 and worsened_count > 0:
        overall_status = "Mixed"
    else:
        overall_status = "Stable"

    # Executive narrative summary
    narrative = (
        "Latest performance shows continued revenue and customer-base growth, "
        "lower CAC, and broadly stable retention, while churn increased slightly."
        if overall_status == "Positive"
        else "Latest performance shows mixed KPI signals across growth, efficiency, and customer retention."
    )

    return {
        "latest_vs_previous_available": True,
        "overall_status": overall_status,
        "summary_statements": statements,
        "narrative": narrative,
    }



def format_comparison_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats structured numerical comparison results into clean strings for console display.

    Args:
        comp_df: DataFrame produced by compare_periods(), compare_latest_vs_previous(), or compare_latest_vs_first().

    Returns:
        pd.DataFrame: Formatted DataFrame ready for display.
    """
    display_df = pd.DataFrame()
    display_df["KPI"] = comp_df["KPI"]

    def _fmt_val(row, col_key):
        val = row[col_key]
        if pd.isna(val):
            return "N/A"
        kpi = row["KPI"]
        if "MRR" in kpi or "ARR" in kpi or "Spend" in kpi:
            if "Growth" in kpi:
                return f"{val:.2f}%"
            return f"${val:,.0f}"
        elif "CAC" in kpi:
            return f"${val:,.2f}"
        elif "Rate" in kpi:
            return f"{val:.2f}%"
        else:
            return f"{int(val):,}"

    def _fmt_change(row):
        val = row["Absolute Change"]
        if pd.isna(val):
            return "N/A"
        kpi = row["KPI"]
        if "MRR" in kpi or "ARR" in kpi or "Spend" in kpi:
            if "Growth" in kpi:
                return f"{val:+.2f} pp"
            return f"${val:+,.0f}"
        elif "CAC" in kpi:
            return f"${val:+,.2f}"
        elif "Rate" in kpi:
            return f"{val:+.2f} pp"
        else:
            return f"{int(val):+,}"

    def _fmt_pct_pp(row):
        val = row["Change % / pp"]
        if pd.isna(val):
            return "N/A"
        if row["Is Rate"]:
            return f"{val:+.2f} pp"
        else:
            return f"{val:+.2f}%"

    display_df["Previous"] = comp_df.apply(lambda r: _fmt_val(r, "Comparison Value"), axis=1)
    display_df["Latest"] = comp_df.apply(lambda r: _fmt_val(r, "Latest Value"), axis=1)
    display_df["Change"] = comp_df.apply(_fmt_change, axis=1)
    display_df["Change % / pp"] = comp_df.apply(_fmt_pct_pp, axis=1)
    display_df["Status"] = comp_df["Status"]

    return display_df
