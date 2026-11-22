"""
pipeline.py
===========
Pipeline Coordinator & Orchestration Engine for SaaS Financial Reporter.
Step 14: Full Pipeline Integration, Error Handling, and Execution Reliability.

Orchestrates all financial reporting and machine learning modules while maintaining:
- Loose coupling between layers
- Clear separation between financial data and customer-level churn ML data
- Deterministic execution status reporting (SUCCESS, PARTIAL_SUCCESS, FAILED)
- Safe error severity handling (Critical errors stop execution; optional failures log warnings)
- Automatic directory management and artifact validation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.config import (
    CHARTS_DIR,
    CHURN_DATA_PATH,
    CHURN_METRICS_PATH,
    CHURN_MODEL_PATH,
    CHURN_PROVENANCE_PATH,
    CHURN_RISK_REPORT_PATH,
    DATA_DIR,
    FINANCIAL_DATA_PATH,
    INITIAL_CASH_BALANCE,
    MODELS_DIR,
    PDF_REPORT_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.exceptions import (
    ArtifactValidationError,
    ChurnDataError,
    ChurnModelError,
    FinancialCalculationError,
    FinancialDataError,
    PipelineError,
    ReportGenerationError,
)
from src.security import sanitize_dataframe_for_csv, atomic_write_bytes

# Core Financial Modules
from src.data_loader import DataValidationError, load_financial_data
from src.revenue_dashboard import (
    calculate_revenue_metrics,
    format_revenue_dashboard,
    generate_revenue_summary,
)
from src.cac_calculator import (
    calculate_cac_metrics,
    format_cac_table,
    generate_cac_summary,
)
from src.churn_retention_analyzer import (
    calculate_churn_retention_metrics,
    format_churn_retention_table,
    generate_churn_retention_summary,
)
from src.chart_generator import generate_all_charts
from src.kpi_comparator import (
    compare_latest_vs_first,
    compare_latest_vs_previous,
    format_comparison_table,
    generate_trend_summary,
)
from src.cash_flow_analyzer import (
    analyze_cash_flow,
    format_cash_flow_table,
    generate_cash_flow_summary,
)
from src.cash_flow_forecaster import (
    build_cash_flow_forecast,
    format_cash_flow_forecast_table,
    generate_forecast_summary,
)
from src.pdf_report_generator import generate_investor_report
from src.reporting_dashboard import build_dashboard_data, display_dashboard

# Churn ML Module
from src.churn_model import (
    assign_churn_risk,
    build_churn_pipeline,
    evaluate_churn_model,
    generate_confusion_matrix_chart,
    generate_roc_curve_chart,
    get_feature_coefficients,
    load_churn_data,
    predict_churn_probabilities,
    save_churn_metrics,
    save_churn_model,
    split_churn_data,
    train_churn_model,
)

# Configure logging
logger = logging.getLogger("saas_financial_reporter")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def prepare_output_directories(
    dirs: Optional[List[Union[str, Path]]] = None,
) -> List[Path]:
    """
    Ensure all required output directories exist.

    Parameters
    ----------
    dirs : list of str or Path, optional
        Directories to initialize. Defaults to CHARTS_DIR, REPORTS_DIR, MODELS_DIR.

    Returns
    -------
    list of Path
        Resolved Paths that were confirmed or created.
    """
    if dirs is None:
        dirs = [CHARTS_DIR, REPORTS_DIR, MODELS_DIR]

    created_paths = []
    for d in dirs:
        p = Path(d).resolve()
        p.mkdir(parents=True, exist_ok=True)
        created_paths.append(p)
    return created_paths


def validate_generated_artifacts(
    artifacts: Dict[str, Union[str, Path]],
    required_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Check the presence and non-zero byte size of generated artifact files.

    Parameters
    ----------
    artifacts : dict
        Mapping of artifact name to its expected filesystem Path.
    required_keys : list of str, optional
        Subset of keys that are mandatory for SUCCESS status.

    Returns
    -------
    dict
        Artifact validation results including status per file, missing files, and total counts.
    """
    if required_keys is None:
        required_keys = []

    results: Dict[str, Dict[str, Any]] = {}
    missing_required: List[str] = []
    zero_byte_files: List[str] = []
    available_count = 0

    for name, path in artifacts.items():
        p = Path(path)
        is_required = name in required_keys
        exists = p.exists()
        size_bytes = p.stat().st_size if exists else 0

        if exists and size_bytes > 0:
            status = "AVAILABLE"
            available_count += 1
        elif exists and size_bytes == 0:
            status = "EMPTY_FILE"
            zero_byte_files.append(name)
            if is_required:
                missing_required.append(name)
        else:
            status = "MISSING"
            if is_required:
                missing_required.append(name)

        results[name] = {
            "path": str(p),
            "exists": exists,
            "size_bytes": size_bytes,
            "status": status,
            "is_required": is_required,
        }

    return {
        "artifacts": results,
        "available_count": available_count,
        "total_checked": len(artifacts),
        "missing_required": missing_required,
        "zero_byte_files": zero_byte_files,
        "all_required_valid": len(missing_required) == 0 and len(zero_byte_files) == 0,
    }


def run_financial_pipeline(
    financial_data_path: Union[str, Path] = FINANCIAL_DATA_PATH,
    charts_dir: Union[str, Path] = CHARTS_DIR,
    pdf_path: Union[str, Path] = PDF_REPORT_PATH,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute the 10-stage financial analytics and reporting workflow (Steps 2–11).

    Parameters
    ----------
    financial_data_path : str or Path
        Path to raw monthly financial dataset.
    charts_dir : str or Path
        Output directory for generated KPI charts.
    pdf_path : str or Path
        Destination path for master PDF investor report.
    verbose : bool, default True
        Whether to print stage progression to stdout.

    Returns
    -------
    dict
        Financial pipeline result dictionary containing status, outputs, warnings, and errors.
    """
    data_path = Path(financial_data_path)
    charts_out = Path(charts_dir)
    pdf_out = Path(pdf_path)

    warnings: List[str] = []
    errors: List[str] = []

    if verbose:
        print("\n" + "=" * 65)
        print("          SAAS FINANCIAL REPORTING PIPELINE")
        print("=" * 65)

    # 1. Data Ingestion & Validation
    if verbose:
        print(f"\n[1/10] Loading & validating financial data from: {data_path.name}...")
    try:
        raw_df = load_financial_data(data_path)
        if verbose:
            print("SUCCESS: Financial data validated.")
    except (FileNotFoundError, DataValidationError, ValueError) as err:
        error_msg = f"Financial data validation failed: {err}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {
            "status": "failed",
            "warnings": warnings,
            "errors": errors,
            "data": None,
        }
    except Exception as exc:
        error_msg = f"Unexpected error loading financial data: {exc}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {
            "status": "failed",
            "warnings": warnings,
            "errors": errors,
            "data": None,
        }

    # 2. Revenue Metrics
    if verbose:
        print("\n[2/10] Calculating recurring revenue metrics (MRR, ARR, Growth)...")
    try:
        revenue_df = calculate_revenue_metrics(raw_df)
        rev_summary = generate_revenue_summary(revenue_df)
        if verbose:
            print("SUCCESS: Revenue metrics calculated.")
    except Exception as exc:
        error_msg = f"Revenue metric calculation failed: {exc}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "data": None}

    # 3. CAC Metrics
    if verbose:
        print("\n[3/10] Calculating Customer Acquisition Cost (CAC) metrics...")
    try:
        cac_df = calculate_cac_metrics(raw_df)
        cac_sum = generate_cac_summary(raw_df)
        if verbose:
            print("SUCCESS: CAC metrics calculated.")
    except Exception as exc:
        error_msg = f"CAC calculation failed: {exc}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "data": None}

    # 4. Churn & Retention Metrics
    if verbose:
        print("\n[4/10] Calculating Customer Churn & Retention metrics...")
    try:
        churn_df = calculate_churn_retention_metrics(raw_df)
        churn_sum = generate_churn_retention_summary(raw_df)
        if verbose:
            print("SUCCESS: Churn & retention metrics calculated.")
    except Exception as exc:
        error_msg = f"Churn & retention calculation failed: {exc}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "data": None}

    # 5. Financial Charts (Optional / Recoverable)
    if verbose:
        print("\n[5/10] Generating financial charts & KPI visualizations...")
    chart_paths = {}
    try:
        chart_paths = generate_all_charts(raw_df, charts_out)
        if verbose:
            print(f"SUCCESS: {len(chart_paths)} financial charts generated.")
    except Exception as exc:
        warn_msg = f"One or more financial charts failed to generate: {exc}"
        warnings.append(warn_msg)
        if verbose:
            print(f"WARNING: {warn_msg}")

    # 6. Historical KPI Comparison
    if verbose:
        print("\n[6/10] Running historical KPI comparison & trend analysis...")
    try:
        mom_comp = compare_latest_vs_previous(raw_df)
        first_comp = compare_latest_vs_first(raw_df)
        trend_summary = generate_trend_summary(raw_df)
        if verbose:
            print("SUCCESS: KPI comparison completed.")
    except Exception as exc:
        error_msg = f"KPI comparison failed: {exc}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "data": None}

    # 7. Cash Flow Analysis
    if verbose:
        print("\n[7/10] Running historical cash-flow analysis...")
    try:
        cash_flow_df = analyze_cash_flow(raw_df)
        cash_flow_sum = generate_cash_flow_summary(cash_flow_df)
        if verbose:
            print("SUCCESS: Cash flow analysis completed.")
    except Exception as exc:
        error_msg = f"Cash flow analysis failed: {exc}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "data": None}

    # 8. Cash Flow Forecasting
    if verbose:
        print("\n[8/10] Forecasting next 6 months cash flow...")
    try:
        forecast_df = build_cash_flow_forecast(cash_flow_df)
        forecast_sum = generate_forecast_summary(forecast_df)
        if verbose:
            print("SUCCESS: Cash flow forecasting completed.")
    except Exception as exc:
        error_msg = f"Cash flow forecasting failed: {exc}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "data": None}

    # 9. PDF Investor Report Generation (Optional / Recoverable)
    if verbose:
        print("\n[9/10] Generating automated PDF investor report...")
    generated_pdf_path = None
    try:
        generated_pdf_path = generate_investor_report(
            raw_df,
            output_path=pdf_out,
            charts_dir=charts_out,
        )
        if verbose:
            print(f"SUCCESS: PDF report generated at {generated_pdf_path.name}.")
    except Exception as exc:
        warn_msg = f"PDF investor report generation failed: {exc}"
        warnings.append(warn_msg)
        if verbose:
            print(f"WARNING: {warn_msg}")

    # 10. Consolidated Reporting Dashboard
    if verbose:
        print("\n[10/10] Building consolidated reporting dashboard...")
    dashboard_data = None
    try:
        dashboard_data = build_dashboard_data(
            raw_df,
            charts_dir=charts_out,
            pdf_path=pdf_out,
        )
        if verbose:
            display_dashboard(dashboard_data)
    except Exception as exc:
        warn_msg = f"Dashboard rendering encountered an issue: {exc}"
        warnings.append(warn_msg)
        if verbose:
            print(f"WARNING: {warn_msg}")

    status = "partial_success" if warnings else "success"

    return {
        "status": status,
        "warnings": warnings,
        "errors": errors,
        "data": {
            "raw_df": raw_df,
            "revenue_df": revenue_df,
            "revenue_summary": rev_summary,
            "cac_df": cac_df,
            "cac_summary": cac_sum,
            "churn_df": churn_df,
            "churn_summary": churn_sum,
            "kpi_comparison_mom": mom_comp,
            "kpi_comparison_first": first_comp,
            "trend_summary": trend_summary,
            "cash_flow_df": cash_flow_df,
            "cash_flow_summary": cash_flow_sum,
            "forecast_df": forecast_df,
            "forecast_summary": forecast_sum,
            "chart_paths": chart_paths,
            "pdf_path": generated_pdf_path,
            "dashboard_data": dashboard_data,
        },
    }


def run_churn_pipeline(
    churn_data_path: Union[str, Path] = CHURN_DATA_PATH,
    models_dir: Union[str, Path] = MODELS_DIR,
    reports_dir: Union[str, Path] = REPORTS_DIR,
    charts_dir: Union[str, Path] = CHARTS_DIR,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute the customer-level churn prediction ML pipeline (Step 13).

    Parameters
    ----------
    churn_data_path : str or Path
        Path to customer churn CSV dataset.
    models_dir : str or Path
        Directory to save serialized model and metrics JSON.
    reports_dir : str or Path
        Directory to save customer risk scoring CSV.
    charts_dir : str or Path
        Directory to save confusion matrix and ROC curve charts.
    verbose : bool, default True
        Whether to print progress to stdout.

    Returns
    -------
    dict
        Churn ML pipeline result dictionary.
    """
    data_path = Path(churn_data_path)
    models_out = Path(models_dir)
    reports_out = Path(reports_dir)
    charts_out = Path(charts_dir)

    warnings: List[str] = []
    errors: List[str] = []

    model_path = models_out / "churn_model.joblib"
    metrics_path = models_out / "churn_model_metrics.json"
    risk_report_path = reports_out / "customer_churn_risk.csv"
    cm_chart_path = charts_out / "churn_confusion_matrix.png"
    roc_chart_path = charts_out / "churn_roc_curve.png"

    if verbose:
        print("\n" + "=" * 65)
        print("          CUSTOMER CHURN PREDICTION & RISK SCORING (ML)")
        print("=" * 65)

    # 1. Load Churn Data
    if verbose:
        print(f"\n[ML-1] Loading customer churn data from: {data_path.name}...")
    try:
        churn_df = load_churn_data(data_path)
        if verbose:
            print(f"SUCCESS: Loaded {len(churn_df)} customer records.")
    except (FileNotFoundError, ValueError) as err:
        error_msg = f"Customer churn data loading failed: {err}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "metrics": None}
    except Exception as exc:
        error_msg = f"Unexpected error loading customer churn data: {exc}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "metrics": None}

    # Verify binary target classes
    unique_targets = set(churn_df["Churned"].dropna().unique())
    if len(unique_targets) < 2:
        error_msg = f"Customer churn dataset contains only 1 target class {unique_targets}. Minimum 2 classes required for ML."
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "metrics": None}

    # 2. Split Data
    if verbose:
        print("[ML-2] Splitting dataset (80% Train / 20% Test, Stratified, seed=42)...")
    try:
        X_train, X_test, y_train, y_test = split_churn_data(churn_df, test_size=0.20, random_state=42)
    except Exception as exc:
        error_msg = f"Dataset train/test split failed: {exc}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "metrics": None}

    # 3. Build & Train Pipeline
    if verbose:
        print("[ML-3] Training Logistic Regression pipeline with StandardScaler & OneHotEncoder...")
    try:
        pipeline = build_churn_pipeline(random_state=42)
        train_churn_model(pipeline, X_train, y_train)
        if verbose:
            print("SUCCESS: Churn pipeline trained.")
    except Exception as exc:
        error_msg = f"Model training failed: {exc}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "metrics": None}

    # 4. Evaluate Model
    if verbose:
        print("[ML-4] Evaluating model performance on test set...")
    try:
        metrics = evaluate_churn_model(pipeline, X_test, y_test)
        if verbose:
            print("\n" + "-" * 65)
            print("                 CHURN MODEL EVALUATION METRICS")
            print("-" * 65)
            print(f"Accuracy         : {metrics['accuracy'] * 100:.2f}%")
            print(f"Precision        : {metrics['precision'] * 100:.2f}%")
            print(f"Recall           : {metrics['recall'] * 100:.2f}%")
            print(f"F1-Score         : {metrics['f1_score'] * 100:.2f}%")
            print(f"ROC-AUC Score    : {metrics['roc_auc']:.4f}")
            cm = metrics["confusion_matrix"]
            print(f"Confusion Matrix : TN={cm['true_negatives']}, FP={cm['false_positives']}, FN={cm['false_negatives']}, TP={cm['true_positives']}")
            print("-" * 65)
    except Exception as exc:
        error_msg = f"Model evaluation failed: {exc}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "metrics": None}

    # 5. Extract Feature Importance & Odds Ratios
    coef_df = None
    try:
        coef_df = get_feature_coefficients(pipeline)
        if verbose:
            print("\n--- TOP CHURN RISK FACTORS ---")
            print(coef_df.to_string(index=False))
    except Exception as exc:
        warn_msg = f"Feature coefficient extraction warning: {exc}"
        warnings.append(warn_msg)
        if verbose:
            print(f"WARNING: {warn_msg}")

    # 6. Score All Customers & Export Risk CSV
    scored_df = None
    try:
        scored_df = predict_churn_probabilities(pipeline, churn_df)
        reports_out.mkdir(parents=True, exist_ok=True)
        # SEC-01 & CONC-02: Defensively sanitize against spreadsheet formula injection (CWE-1236)
        # and atomically persist to prevent partial writes during concurrent pipelines
        clean_risk_df = sanitize_dataframe_for_csv(scored_df)
        csv_bytes = clean_risk_df.to_csv(index=False).encode("utf-8")
        atomic_write_bytes(risk_report_path, csv_bytes)

        if verbose:
            risk_counts = scored_df["RiskLevel"].value_counts().to_dict()
            print("\n--- CUSTOMER RISK TIER DISTRIBUTION ---")
            print(f"Low Risk (< 0.30)        : {risk_counts.get('Low', 0)} customers")
            print(f"Medium Risk (0.30 - 0.60): {risk_counts.get('Medium', 0)} customers")
            print(f"High Risk (>= 0.60)      : {risk_counts.get('High', 0)} customers")
            print(f"Total Scored             : {len(scored_df)} customers")

            print("\n--- TOP 5 HIGHEST PREDICTED CHURN RISK CUSTOMERS ---")
            top_5_cols = ["CustomerID", "Plan", "ContractType", "TenureMonths", "MonthlyFee", "ChurnProbability", "RiskLevel"]
            print(scored_df[top_5_cols].head(5).to_string(index=False))
    except Exception as exc:
        error_msg = f"Customer risk scoring failed: {exc}"
        errors.append(error_msg)
        if verbose:
            print(f"CRITICAL ERROR: {error_msg}")
        return {"status": "failed", "warnings": warnings, "errors": errors, "metrics": None}

    # 7. Persist Model & Metrics Artifacts
    try:
        save_churn_model(pipeline, model_path)
        save_churn_metrics(metrics, metrics_path)
        if verbose:
            print(f"\nSUCCESS: Churn model saved to {model_path.name}.")
    except Exception as exc:
        warn_msg = f"Model artifact persistence warning: {exc}"
        warnings.append(warn_msg)
        if verbose:
            print(f"WARNING: {warn_msg}")

    # 8. Generate Churn Visualizations (Optional / Recoverable)
    try:
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        generate_confusion_matrix_chart(y_test, y_pred, cm_chart_path)
        generate_roc_curve_chart(y_test, y_proba, roc_chart_path)
        if verbose:
            print("SUCCESS: Churn evaluation charts generated.")
    except Exception as exc:
        warn_msg = f"Churn visualization generation warning: {exc}"
        warnings.append(warn_msg)
        if verbose:
            print(f"WARNING: {warn_msg}")

    if verbose:
        print("\nPortfolio Disclaimer:")
        print("  The churn model is trained on a synthetic customer-level portfolio dataset")
        print("  and is intended to demonstrate an end-to-end ML workflow, not production churn prediction.")
        print("=" * 65)

    status = "partial_success" if warnings else "success"

    return {
        "status": status,
        "warnings": warnings,
        "errors": errors,
        "metrics": metrics,
        "pipeline": pipeline,
        "scored_df": scored_df,
        "coef_df": coef_df,
    }


def run_full_pipeline(
    project_root: Optional[Union[str, Path]] = None,
    run_churn_ml: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrate the complete SaaS financial reporting and ML workflow.

    Parameters
    ----------
    project_root : str or Path, optional
        Project root directory. Defaults to PROJECT_ROOT.
    run_churn_ml : bool, default True
        Whether to execute the customer churn ML pipeline.
    verbose : bool, default True
        Whether to print stage output and summary to stdout.

    Returns
    -------
    dict
        Unified execution result dictionary.
    """
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    data_dir = root / "data"
    charts_dir = root / "charts"
    reports_dir = root / "reports"
    models_dir = root / "models"

    financial_data_path = data_dir / "saas_financial_data.csv"
    churn_data_path = data_dir / "customer_churn_data.csv"
    pdf_report_path = reports_dir / "saas_financial_report.pdf"

    # Step 1: Initialize output directories
    if verbose:
        print("=" * 65)
        print("              SaaS Financial Reporter - Pipeline")
        print("=" * 65)
        print("\n[Init] Initializing project output directories...")

    prepare_output_directories([charts_dir, reports_dir, models_dir])
    if verbose:
        print("SUCCESS: Output directories ready (charts/, reports/, models/).")

    # Step 2-11: Financial Reporting Pipeline
    financial_res = run_financial_pipeline(
        financial_data_path=financial_data_path,
        charts_dir=charts_dir,
        pdf_path=pdf_report_path,
        verbose=verbose,
    )

    # Step 13: Customer Churn ML Pipeline
    churn_res = None
    if run_churn_ml:
        churn_res = run_churn_pipeline(
            churn_data_path=churn_data_path,
            models_dir=models_dir,
            reports_dir=reports_dir,
            charts_dir=charts_dir,
            verbose=verbose,
        )

    # Step 14: Artifact Validation
    artifacts_to_check = {
        "PDF Investor Report": pdf_report_path,
        "MRR Trend Chart": charts_dir / "mrr_trend.png",
        "ARR Trend Chart": charts_dir / "arr_trend.png",
        "CAC Trend Chart": charts_dir / "cac_trend.png",
        "Churn & Retention Chart": charts_dir / "churn_retention_trend.png",
        "Customer Base Growth Chart": charts_dir / "customer_base_growth.png",
        "MRR Growth Rate Chart": charts_dir / "mrr_growth_rate.png",
        "Net Cash Flow Chart": charts_dir / "net_cash_flow_trend.png",
        "Cash Balance Chart": charts_dir / "cash_balance_trend.png",
        "Cash Flow Forecast Chart": charts_dir / "cash_flow_forecast.png",
        "Cash Balance Forecast Chart": charts_dir / "cash_balance_forecast.png",
    }

    required_artifacts = ["PDF Investor Report"]

    if run_churn_ml:
        artifacts_to_check.update({
            "Churn Model Binary": models_dir / "churn_model.joblib",
            "Churn Metrics JSON": models_dir / "churn_model_metrics.json",
            "Customer Risk Report": reports_dir / "customer_churn_risk.csv",
            "Churn Confusion Matrix": charts_dir / "churn_confusion_matrix.png",
            "Churn ROC Curve": charts_dir / "churn_roc_curve.png",
        })
        required_artifacts.extend(["Churn Model Binary", "Customer Risk Report"])

    artifact_validation = validate_generated_artifacts(artifacts_to_check, required_keys=required_artifacts)

    # Aggregate warnings and errors
    total_warnings: List[str] = list(financial_res["warnings"])
    total_errors: List[str] = list(financial_res["errors"])

    if churn_res is not None:
        total_warnings.extend(churn_res["warnings"])
        total_errors.extend(churn_res["errors"])

    if not artifact_validation["all_required_valid"]:
        for missing in artifact_validation["missing_required"]:
            total_errors.append(f"Required artifact is missing or empty: {missing}")

    # Determine overall status
    if financial_res["status"] == "failed" or (churn_res and churn_res["status"] == "failed") or len(total_errors) > 0:
        overall_status = "FAILED"
    elif len(total_warnings) > 0 or financial_res["status"] == "partial_success" or (churn_res and churn_res["status"] == "partial_success"):
        overall_status = "PARTIAL_SUCCESS"
    else:
        overall_status = "SUCCESS"

    result = {
        "overall_status": overall_status,
        "financial_pipeline": {
            "status": financial_res["status"],
            "warnings": financial_res["warnings"],
            "errors": financial_res["errors"],
        },
        "churn_pipeline": {
            "status": churn_res["status"] if churn_res else "SKIPPED",
            "warnings": churn_res["warnings"] if churn_res else [],
            "errors": churn_res["errors"] if churn_res else [],
        } if run_churn_ml else None,
        "warnings_count": len(total_warnings),
        "errors_count": len(total_errors),
        "warnings": total_warnings,
        "errors": total_errors,
        "artifact_validation": artifact_validation,
    }

    if verbose:
        display_pipeline_summary(result)

    return result


def display_pipeline_summary(result: Dict[str, Any]) -> None:
    """
    Print the consolidated pipeline execution summary to the console.

    Parameters
    ----------
    result : dict
        Unified pipeline execution result from `run_full_pipeline`.
    """
    fin_status = result["financial_pipeline"]["status"].upper()
    churn_status = result["churn_pipeline"]["status"].upper() if result["churn_pipeline"] else "SKIPPED"
    overall = result["overall_status"]
    warnings_cnt = result["warnings_count"]
    errors_cnt = result["errors_count"]
    artifacts = result["artifact_validation"]

    print("\n" + "=" * 65)
    print("                    PIPELINE SUMMARY")
    print("=" * 65)
    print(f"Financial Reporting Pipeline : {fin_status}")
    print(f"Customer Churn ML Pipeline    : {churn_status}")
    print(f"\nWarnings                      : {warnings_cnt}")
    print(f"Errors                        : {errors_cnt}")

    print("\nGenerated Artifacts:")
    for name, info in artifacts["artifacts"].items():
        print(f"  {name:<28}: {info['status']}")

    print("-" * 65)
    print(f"Overall Execution Status      : {overall}")
    print("=" * 65 + "\n")
