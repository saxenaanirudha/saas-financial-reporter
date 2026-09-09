"""
test_pipeline.py
================
Integration tests for Step 14: Full Pipeline Integration, Error Handling, and Execution Reliability.
Verifies directory preparation, end-to-end execution, critical error handling,
optional failure recovery, artifact validation, immutability, and structured reporting.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.pipeline import (
    prepare_output_directories,
    validate_generated_artifacts,
    run_financial_pipeline,
    run_churn_pipeline,
    run_full_pipeline,
    display_pipeline_summary,
)
from src.config import FINANCIAL_DATA_PATH, CHURN_DATA_PATH


# ── FIXTURES ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_workspace(tmp_path: Path):
    """Set up an isolated workspace with temporary data, charts, reports, and models directories."""
    data_dir = tmp_path / "data"
    charts_dir = tmp_path / "charts"
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"
    data_dir.mkdir()

    # Copy real sample data into isolated workspace
    fin_df = pd.read_csv(FINANCIAL_DATA_PATH)
    fin_df.to_csv(data_dir / "saas_financial_data.csv", index=False)

    churn_df = pd.read_csv(CHURN_DATA_PATH)
    churn_df.to_csv(data_dir / "customer_churn_data.csv", index=False)

    return {
        "root": tmp_path,
        "data_dir": data_dir,
        "charts_dir": charts_dir,
        "reports_dir": reports_dir,
        "models_dir": models_dir,
        "fin_csv": data_dir / "saas_financial_data.csv",
        "churn_csv": data_dir / "customer_churn_data.csv",
    }


# ── 1. DIRECTORY INITIALIZATION TESTS ─────────────────────────────────────────

def test_prepare_output_directories_creates_all(tmp_path: Path):
    d1 = tmp_path / "custom_charts"
    d2 = tmp_path / "custom_reports"
    d3 = tmp_path / "custom_models"

    assert not d1.exists()
    assert not d2.exists()
    assert not d3.exists()

    paths = prepare_output_directories([d1, d2, d3])
    assert len(paths) == 3
    assert d1.exists() and d1.is_dir()
    assert d2.exists() and d2.is_dir()
    assert d3.exists() and d3.is_dir()


# ── 2. END-TO-END PIPELINE TESTS ──────────────────────────────────────────────

def test_run_financial_pipeline_valid_data(temp_workspace):
    res = run_financial_pipeline(
        financial_data_path=temp_workspace["fin_csv"],
        charts_dir=temp_workspace["charts_dir"],
        pdf_path=temp_workspace["reports_dir"] / "test_report.pdf",
        verbose=False,
    )
    assert res["status"] in ["success", "partial_success"]
    assert len(res["errors"]) == 0
    assert res["data"] is not None
    assert "revenue_summary" in res["data"]
    assert "cash_flow_summary" in res["data"]


def test_run_churn_pipeline_valid_data(temp_workspace):
    res = run_churn_pipeline(
        churn_data_path=temp_workspace["churn_csv"],
        models_dir=temp_workspace["models_dir"],
        reports_dir=temp_workspace["reports_dir"],
        charts_dir=temp_workspace["charts_dir"],
        verbose=False,
    )
    assert res["status"] in ["success", "partial_success"]
    assert len(res["errors"]) == 0
    assert res["metrics"] is not None
    assert "accuracy" in res["metrics"]
    assert (temp_workspace["models_dir"] / "churn_model.joblib").exists()
    assert (temp_workspace["reports_dir"] / "customer_churn_risk.csv").exists()


def test_run_full_pipeline_structured_result_keys(temp_workspace):
    res = run_full_pipeline(project_root=temp_workspace["root"], run_churn_ml=True, verbose=False)
    assert "overall_status" in res
    assert "financial_pipeline" in res
    assert "churn_pipeline" in res
    assert "warnings_count" in res
    assert "errors_count" in res
    assert "warnings" in res
    assert "errors" in res
    assert "artifact_validation" in res


def test_run_full_pipeline_success_status(temp_workspace):
    res = run_full_pipeline(project_root=temp_workspace["root"], run_churn_ml=True, verbose=False)
    assert res["overall_status"] == "SUCCESS"
    assert res["financial_pipeline"]["status"] == "success"
    assert res["churn_pipeline"]["status"] == "success"
    assert res["errors_count"] == 0
    assert res["warnings_count"] == 0


def test_run_full_pipeline_skip_churn_ml(temp_workspace):
    res = run_full_pipeline(project_root=temp_workspace["root"], run_churn_ml=False, verbose=False)
    assert res["overall_status"] == "SUCCESS"
    assert res["financial_pipeline"]["status"] == "success"
    assert res["churn_pipeline"] is None


# ── 3. CRITICAL ERROR HANDLING TESTS ──────────────────────────────────────────

def test_missing_financial_csv_causes_failure(temp_workspace):
    bad_path = temp_workspace["data_dir"] / "non_existent.csv"
    res = run_financial_pipeline(financial_data_path=bad_path, verbose=False)
    assert res["status"] == "failed"
    assert len(res["errors"]) > 0
    assert "not found" in res["errors"][0] or "failed" in res["errors"][0]


def test_invalid_financial_data_causes_failure(temp_workspace):
    bad_csv = temp_workspace["data_dir"] / "bad_financial.csv"
    pd.DataFrame({"WrongColumn": [1, 2, 3]}).to_csv(bad_csv, index=False)
    res = run_financial_pipeline(financial_data_path=bad_csv, verbose=False)
    assert res["status"] == "failed"
    assert len(res["errors"]) > 0


def test_missing_churn_csv_causes_failure_when_enabled(temp_workspace):
    bad_path = temp_workspace["data_dir"] / "non_existent_churn.csv"
    res = run_churn_pipeline(churn_data_path=bad_path, verbose=False)
    assert res["status"] == "failed"
    assert len(res["errors"]) > 0


def test_single_class_churn_data_causes_failure(temp_workspace):
    single_class_df = pd.read_csv(temp_workspace["churn_csv"])
    single_class_df["Churned"] = 0  # Only class 0
    single_class_path = temp_workspace["data_dir"] / "single_class_churn.csv"
    single_class_df.to_csv(single_class_path, index=False)

    res = run_churn_pipeline(churn_data_path=single_class_path, verbose=False)
    assert res["status"] == "failed"
    assert any("1 target class" in err or "Minimum 2 classes" in err for err in res["errors"])


# ── 4. OPTIONAL FAILURE RECOVERY & WARNING TESTS ──────────────────────────────

def test_optional_chart_failure_recorded_as_warning(temp_workspace, monkeypatch):
    import src.pipeline as pl

    # Simulate chart generator error
    def failing_chart_generator(*args, **kwargs):
        raise RuntimeError("Simulated chart rendering failure")

    monkeypatch.setattr(pl, "generate_all_charts", failing_chart_generator)

    res = pl.run_financial_pipeline(
        financial_data_path=temp_workspace["fin_csv"],
        charts_dir=temp_workspace["charts_dir"],
        pdf_path=temp_workspace["reports_dir"] / "test.pdf",
        verbose=False,
    )
    assert res["status"] == "partial_success"
    assert len(res["warnings"]) > 0
    assert "Simulated chart rendering failure" in res["warnings"][0]


def test_partial_success_status_on_optional_warning(temp_workspace, monkeypatch):
    import src.pipeline as pl

    def failing_chart_generator(*args, **kwargs):
        raise RuntimeError("Simulated non-critical failure")

    monkeypatch.setattr(pl, "generate_all_charts", failing_chart_generator)

    res = pl.run_full_pipeline(project_root=temp_workspace["root"], run_churn_ml=False, verbose=False)
    assert res["overall_status"] == "PARTIAL_SUCCESS"
    assert res["warnings_count"] > 0
    assert res["errors_count"] == 0


# ── 5. ARTIFACT VALIDATION TESTS ──────────────────────────────────────────────

def test_validate_artifacts_detects_missing_file(tmp_path: Path):
    artifacts = {
        "Real File": tmp_path / "exists.txt",
        "Missing File": tmp_path / "missing.txt",
    }
    (tmp_path / "exists.txt").write_text("content")

    val = validate_generated_artifacts(artifacts, required_keys=["Missing File"])
    assert not val["all_required_valid"]
    assert "Missing File" in val["missing_required"]
    assert val["artifacts"]["Missing File"]["status"] == "MISSING"
    assert val["artifacts"]["Real File"]["status"] == "AVAILABLE"


def test_validate_artifacts_detects_zero_byte_file(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    artifacts = {"Empty File": empty_file}

    val = validate_generated_artifacts(artifacts, required_keys=["Empty File"])
    assert not val["all_required_valid"]
    assert "Empty File" in val["zero_byte_files"]
    assert val["artifacts"]["Empty File"]["status"] == "EMPTY_FILE"


def test_validate_artifacts_passes_valid_files(tmp_path: Path):
    valid_file = tmp_path / "valid.txt"
    valid_file.write_text("hello world")
    artifacts = {"Valid File": valid_file}

    val = validate_generated_artifacts(artifacts, required_keys=["Valid File"])
    assert val["all_required_valid"]
    assert val["available_count"] == 1
    assert val["artifacts"]["Valid File"]["status"] == "AVAILABLE"


# ── 6. DATASET ISOLATION & IMMUTABILITY TESTS ─────────────────────────────────

def test_financial_and_churn_datasets_remain_separate(temp_workspace):
    fin_df = pd.read_csv(temp_workspace["fin_csv"])
    churn_df = pd.read_csv(temp_workspace["churn_csv"])

    assert "MRR" in fin_df.columns
    assert "CustomerID" not in fin_df.columns
    assert "CustomerID" in churn_df.columns
    assert "MRR" not in churn_df.columns


def test_pipeline_does_not_mutate_input_data(temp_workspace):
    fin_before = pd.read_csv(temp_workspace["fin_csv"])
    churn_before = pd.read_csv(temp_workspace["churn_csv"])

    run_full_pipeline(project_root=temp_workspace["root"], run_churn_ml=True, verbose=False)

    fin_after = pd.read_csv(temp_workspace["fin_csv"])
    churn_after = pd.read_csv(temp_workspace["churn_csv"])

    pd.testing.assert_frame_equal(fin_before, fin_after)
    pd.testing.assert_frame_equal(churn_before, churn_after)


# ── 7. SUMMARY DISPLAY TEST ───────────────────────────────────────────────────

def test_display_pipeline_summary_prints_cleanly(capsys):
    dummy_result = {
        "overall_status": "SUCCESS",
        "financial_pipeline": {"status": "success", "warnings": [], "errors": []},
        "churn_pipeline": {"status": "success", "warnings": [], "errors": []},
        "warnings_count": 0,
        "errors_count": 0,
        "artifact_validation": {
            "artifacts": {
                "PDF Investor Report": {"status": "AVAILABLE"},
                "Churn Model Binary": {"status": "AVAILABLE"},
            }
        },
    }
    display_pipeline_summary(dummy_result)
    captured = capsys.readouterr()
    assert "PIPELINE SUMMARY" in captured.out
    assert "Overall Execution Status      : SUCCESS" in captured.out
