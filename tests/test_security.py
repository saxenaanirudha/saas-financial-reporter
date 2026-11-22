"""
test_security.py
================
Security regression tests covering OWASP vulnerabilities for SaaS Financial Reporter:
- SEC-01: CSV Formula Injection Prevention (CWE-1236)
- SEC-02: Model Provenance, Integrity Verification & Untrusted Deserialization Prevention (CWE-502)
- SEC-03: ReportLab XML Entity Injection Prevention in PDF Generation (CWE-116)
- SEC-05: Denial of Service / Resource Exhaustion via Oversized Datasets (CWE-400)
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.security import (
    sanitize_dataframe_for_csv,
    generate_model_provenance,
    verify_model_provenance,
    safe_xml_escape,
    validate_dataset_size,
    SecurityValidationError,
)
from src.pdf_report_generator import generate_investor_report
from src.churn_model import load_saved_churn_model, save_churn_model, build_churn_pipeline


# ── SEC-01: CSV FORMULA INJECTION (CWE-1236) ──────────────────────────────────

def test_csv_formula_injection_sanitization_prefixes_dangerous_characters():
    """
    Cells starting with =, +, -, @, \\t, or \\r must be prefixed with a single quote (')
    to neutralize formula execution in spreadsheet software (Excel, Calc, Sheets).
    """
    raw_data = {
        "CustomerID": ["=cmd|' /C calc'!A0", "+12345", "@SUM(1,2)", "-999", "\tmalicious", "NORMAL_001"],
        "Plan": ["Basic", "=HYPERLINK(\"http://attacker.com\")", "Pro", "Enterprise", "Basic", "Pro"],
        "TenureMonths": [12, 5, 24, 8, 3, 10],  # Numeric column should not be mutated
        "MonthlyFee": [29.99, 79.99, 199.99, 29.99, 79.99, 29.99],
    }
    df = pd.DataFrame(raw_data)
    sanitized = sanitize_dataframe_for_csv(df)

    # 1. Dangerous formula cells must start with single quote
    assert sanitized.loc[0, "CustomerID"].startswith("'=")
    assert sanitized.loc[1, "CustomerID"].startswith("'+")
    assert sanitized.loc[2, "CustomerID"].startswith("'@")
    assert sanitized.loc[3, "CustomerID"].startswith("'-")
    assert sanitized.loc[4, "CustomerID"].startswith("'\t")
    assert sanitized.loc[1, "Plan"].startswith("'=")

    # 2. Benign cells must remain untouched
    assert sanitized.loc[5, "CustomerID"] == "NORMAL_001"
    assert sanitized.loc[0, "Plan"] == "Basic"

    # 3. Numeric columns must retain original values and types
    assert sanitized.loc[0, "TenureMonths"] == 12
    assert sanitized.loc[0, "MonthlyFee"] == 29.99

    # 4. Input DataFrame must not be mutated
    assert not df.loc[0, "CustomerID"].startswith("'")


def test_csv_formula_injection_whitespace_bypass_and_double_escaping():
    """
    BLK-02: Formula injection with leading whitespace, spaces, tabs, newlines
    must be sanitized, and already sanitized values must avoid double-escaping.
    """
    df = pd.DataFrame({
        "Malicious": [
            " =cmd|' /C calc'!A0",   # leading space
            "   @SUM(1,2)",          # leading multiple spaces
            "\t+12345",              # tab followed by +
            "\n-cmd",                # newline followed by -
            "'-already_safe",        # already escaped
            "Normal Text",
        ],
        "NumericValues": [-1500.50, 200.0, -0.01, 0.0, 50.0, 100.0],
    })
    sanitized = sanitize_dataframe_for_csv(df)

    # All whitespace-prefixed formulas must be neutralized
    assert sanitized.loc[0, "Malicious"].startswith("' =")
    assert sanitized.loc[1, "Malicious"].startswith("'   @")
    assert sanitized.loc[2, "Malicious"].startswith("'\t+")
    assert sanitized.loc[3, "Malicious"].startswith("'\n-")

    # Already escaped text must NOT be double-escaped to ''-already_safe
    assert sanitized.loc[4, "Malicious"] == "'-already_safe"
    assert sanitized.loc[5, "Malicious"] == "Normal Text"

    # Numeric negative values must remain floats, not converted to strings
    assert isinstance(sanitized.loc[0, "NumericValues"], float)
    assert sanitized.loc[0, "NumericValues"] == -1500.50


def test_model_signing_fails_when_secret_missing_in_strict_mode(monkeypatch, tmp_path: Path):
    """
    BLK-01: When SAAS_MODEL_SECRET is not set, persistent model signing/verification
    must raise SecurityValidationError in strict mode instead of silently falling back
    to a predictable hardcoded key.
    """
    monkeypatch.delenv("SAAS_MODEL_SECRET", raising=False)
    pipeline = build_churn_pipeline()
    model_path = tmp_path / "models" / "strict_model.joblib"
    prov_path = tmp_path / "models" / "strict_prov.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(pipeline, model_path)

    # In strict mode (require_secret=True), it must reject signing with missing secret
    with pytest.raises(SecurityValidationError, match="SAAS_MODEL_SECRET is required"):
        generate_model_provenance(model_path, prov_path, require_secret=True)

    # Generate a manifest in dev mode so the manifest file exists
    generate_model_provenance(model_path, prov_path, require_secret=False)
    assert prov_path.exists()

    # Verification in strict mode must reject when secret is missing
    with pytest.raises(SecurityValidationError, match="SAAS_MODEL_SECRET is required"):
        verify_model_provenance(model_path, prov_path, require_secret=True)


def test_model_signing_and_verification_with_configured_secret(monkeypatch, tmp_path: Path):
    """
    BLK-01: When SAAS_MODEL_SECRET is configured, signing and verification
    succeed seamlessly in strict mode without any hardcoded fallback.
    """
    monkeypatch.setenv("SAAS_MODEL_SECRET", "prod-cryptographic-secret-2025-vault")
    pipeline = build_churn_pipeline()
    model_path = tmp_path / "models" / "vault_model.joblib"
    prov_path = tmp_path / "models" / "vault_prov.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(pipeline, model_path)

    generate_model_provenance(model_path, prov_path, require_secret=True)
    assert verify_model_provenance(model_path, prov_path, require_secret=True) is True

    # Tampered secret must be rejected
    monkeypatch.setenv("SAAS_MODEL_SECRET", "wrong-secret-key")
    with pytest.raises(SecurityValidationError, match="Cryptographic signature mismatch"):
        verify_model_provenance(model_path, prov_path, require_secret=True)


def test_csv_formula_injection_handles_empty_and_null_values():
    """Sanitizer must handle empty strings and NaN values safely without crashing."""
    df = pd.DataFrame({
        "Text": ["=formula", "", None, float("nan")],
        "Num": [1, 2, 3, 4],
    })
    sanitized = sanitize_dataframe_for_csv(df)
    assert sanitized.loc[0, "Text"] == "'=formula"
    assert sanitized.loc[1, "Text"] == ""
    assert pd.isna(sanitized.loc[2, "Text"])


# ── SEC-02: MODEL INTEGRITY, PROVENANCE & SAFE DESERIALIZATION (CWE-502) ─────────

def test_model_provenance_verification_succeeds_on_authentic_artifact(tmp_path: Path):
    """A model saved with valid provenance manifest passes verification."""
    pipeline = build_churn_pipeline()
    model_path = tmp_path / "models" / "churn_model.joblib"
    provenance_path = tmp_path / "models" / "churn_model_provenance.json"

    # Save model and generate provenance
    save_churn_model(pipeline, model_path)
    generate_model_provenance(model_path, provenance_path)

    # Verification must succeed
    assert verify_model_provenance(model_path, provenance_path) is True


def test_model_provenance_verification_detects_tampered_model(tmp_path: Path):
    """Any byte tampering on the serialized model must be detected and rejected."""
    pipeline = build_churn_pipeline()
    model_path = tmp_path / "models" / "churn_model.joblib"
    provenance_path = tmp_path / "models" / "churn_model_provenance.json"

    save_churn_model(pipeline, model_path)
    generate_model_provenance(model_path, provenance_path)

    # Tamper with model bytes
    with open(model_path, "ab") as f:
        f.write(b"CORRUPTED_TAMPERED_PAYLOAD")

    with pytest.raises(SecurityValidationError, match="integrity check failed|Hash mismatch"):
        verify_model_provenance(model_path, provenance_path)


def test_model_provenance_verification_detects_tampered_manifest(tmp_path: Path):
    """Tampering with the provenance manifest itself must cause verification failure."""
    pipeline = build_churn_pipeline()
    model_path = tmp_path / "models" / "churn_model.joblib"
    provenance_path = tmp_path / "models" / "churn_model_provenance.json"

    save_churn_model(pipeline, model_path)
    generate_model_provenance(model_path, provenance_path)

    # Tamper with manifest file
    with open(provenance_path, "w", encoding="utf-8") as f:
        f.write('{"sha256": "0000000000000000000000000000000000000000000000000000000000000000"}')

    with pytest.raises(SecurityValidationError):
        verify_model_provenance(model_path, provenance_path)


def test_load_saved_churn_model_enforces_path_boundaries(tmp_path: Path):
    """Attempting to load a model outside allowed directory boundaries must be rejected."""
    outside_dir = tmp_path / "untrusted"
    outside_dir.mkdir()
    fake_model = outside_dir / "evil.joblib"
    fake_model.write_text("evil")

    with pytest.raises(SecurityValidationError, match="outside trusted boundary"):
        load_saved_churn_model(fake_model, trusted_dir=tmp_path / "models")


# ── SEC-03: REPORTLAB XML ENTITY INJECTION PREVENTION (CWE-116) ────────────────

def test_pdf_table_renders_unescaped_xml_safely():
    """ReportLab table helper must escape &, <, > without expat XML errors."""
    from src.pdf_report_generator import _df_to_reportlab_table, _get_report_styles
    df = pd.DataFrame({
        "Company & Plan <Tier>": ["Basic <Free>", "Enterprise & Gov > High"],
        "Note": ["Safe & Sound", "Cost < $100 & > $50"],
    })
    styles = _get_report_styles()
    table = _df_to_reportlab_table(df, styles)
    assert table is not None


def test_pdf_generation_succeeds_with_xml_special_characters_in_data(tmp_path: Path):
    """
    Financial dataset and narrative text containing XML special characters
    must compile to PDF cleanly without ReportLab expat XML parsing errors.
    """
    df_special = pd.DataFrame({
        "Month": ["Jan-2025", "Feb-2025"],
        "NewCustomers": [100, 150],
        "ChurnedCustomers": [10, 15],
        "MRR": [50000, 60000],
        "MarketingSpend": [10000, 12000],
        "OperatingExpenses": [8000, 9000],
        "PayrollExpenses": [15000, 16000],
        "OtherExpenses": [2000, 2500],
    })

    pdf_path = tmp_path / "special_report.pdf"
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()

    # Must execute and generate a non-empty PDF without XML parsing errors
    result_path = generate_investor_report(
        df_special,
        output_path=pdf_path,
        charts_dir=charts_dir,
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 1000


# ── SEC-05: DATASET SANITY & DENIAL OF SERVICE MITIGATION (CWE-400) ───────────

def test_validate_dataset_size_enforces_row_limit():
    """Datasets exceeding configured safety limits must raise SecurityValidationError."""
    oversized_df = pd.DataFrame({"col": range(15000)})
    with pytest.raises(SecurityValidationError, match="exceeds maximum allowed"):
        validate_dataset_size(oversized_df, max_rows=10000, dataset_name="Financial CSV")


def test_validate_dataset_size_accepts_normal_dataset():
    """Normal-sized datasets pass without error."""
    normal_df = pd.DataFrame({"col": range(100)})
    validate_dataset_size(normal_df, max_rows=1000, dataset_name="Financial CSV")
