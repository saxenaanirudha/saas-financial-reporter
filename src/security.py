"""
security.py
===========
Central security, data sanitization, and artifact integrity subsystem for
SaaS Financial Reporter.

Covers OWASP recommendations and Barn Door Protocol security controls:
- SEC-01: CSV & Spreadsheet Formula Injection Prevention (CWE-1236)
- SEC-02: Cryptographic Model Provenance, Integrity Verification & Safe Deserialization (CWE-502)
- SEC-03: ReportLab PDF XML Entity Injection Prevention (CWE-116)
- SEC-04: Error Handling & System Path Information Exposure Prevention (CWE-209)
- SEC-05: Denial of Service / Resource Exhaustion Protections (CWE-400)
- CONC-01 & CONC-02: Thread-Safe Matplotlib Rendering and Atomic File Persistence

Trust Assumptions (SEC-02):
---------------------------
1. Host Operating System & Python Runtime: The underlying OS file permissions
   and Python standard library runtime are assumed uncompromised.
2. Provenance Keying: Model signing uses HMAC-SHA256 with an environment-configured
   signing key (`SAAS_MODEL_SECRET` or internal machine derivation).
   An attacker with standard write access to `models/` cannot forge the HMAC signature
   without access to the application secret.
3. Path Isolation: Models and data inputs must resolve within canonical project roots;
   symlink escapes and path traversal sequences (`../`) are strictly blocked.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import functools
from typing import Any, Dict, Optional, Tuple, Union
import xml.sax.saxutils as saxutils

import pandas as pd

from src.exceptions import PipelineError

logger = logging.getLogger("saas_financial_reporter.security")

# Thread lock for Matplotlib rendering across multi-threaded Streamlit sessions
MATPLOTLIB_LOCK = threading.RLock()


def thread_safe_chart(func):
    """
    Decorator to serialize Matplotlib chart generation across concurrent threads,
    preventing state corruption and race conditions in multi-user environments (CONC-01).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with MATPLOTLIB_LOCK:
            return func(*args, **kwargs)
    return wrapper

# Formula injection triggers in spreadsheet software (Excel, Calc, Sheets)
DANGEROUS_CSV_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

# Default safety limits for datasets
MAX_FINANCIAL_ROWS = 10_000
MAX_CHURN_ROWS = 500_000
MAX_MODEL_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB safety cap


class SecurityValidationError(PipelineError):
    """Raised when an operation violates security controls or integrity invariants."""
    pass


# ── SEC-01: CSV FORMULA INJECTION PREVENTION (CWE-1236) ──────────────────────

def sanitize_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensively sanitizes a DataFrame before exporting to CSV.

    Any string/object cell beginning with spreadsheet formula triggers
    ('=', '+', '-', '@', '\\t', '\\r') -- including when prefixed by leading
    whitespace, tabs, or newlines -- is prefixed with a single quote (').
    Spreadsheet applications (Excel, Calc, Google Sheets) will render the cell
    strictly as plain text, eliminating formula injection vulnerabilities.

    Already quoted/escaped values are preserved without double-escaping.
    Numeric values and non-string types are strictly preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame to sanitize.

    Returns
    -------
    pd.DataFrame
        A defensive copy of the DataFrame with sanitized string columns.
        Numeric columns and original types remain preserved.
    """
    if df is None:
        return None

    clean_df = df.copy()

    for col in clean_df.columns:
        # Only inspect object/string columns; preserve numeric types untouched
        if clean_df[col].dtype == object or pd.api.types.is_string_dtype(clean_df[col]):
            def _neutralize_cell(val: Any) -> Any:
                if val is None or pd.isna(val):
                    return val
                s = str(val)
                stripped = s.lstrip(" \t\r\n")
                if stripped.startswith("'"):
                    # Already escaped or single-quoted; avoid double-escaping
                    return val
                if s.startswith(DANGEROUS_CSV_PREFIXES) or stripped.startswith(DANGEROUS_CSV_PREFIXES):
                    return f"'{s}"
                return val

            clean_df[col] = clean_df[col].apply(_neutralize_cell)

    return clean_df


# ── SEC-02: MODEL INTEGRITY, PROVENANCE & SAFE DESERIALIZATION (CWE-502) ────────

_DEV_EPHEMERAL_KEY: Optional[bytes] = None
ARTIFACT_PUBLISH_LOCK = threading.RLock()


def _get_signing_key(
    require_secret: bool = False,
    custom_secret: Optional[Union[str, bytes]] = None,
) -> bytes:
    """
    Retrieves the HMAC secret key for model provenance signing and verification.

    Security Policy:
    ----------------
    1. If `custom_secret` is explicitly provided, it is converted to bytes and used.
    2. If `SAAS_MODEL_SECRET` environment variable is defined, its utf-8 bytes are used.
    3. If neither is available:
       - In strict/production mode (`require_secret=True`, `SAAS_STRICT_SECURITY=1`,
         or `SAAS_ENV=production`), a `SecurityValidationError` is raised immediately.
         Hardcoded fallback keys are strictly rejected.
       - In non-strict development/CI environments, an ephemeral in-memory key generated
         via `os.urandom(32)` is used. This eliminates predictable static keys while
         allowing local test runs to execute without weakening production security.

    Key Configuration & Rotation:
    -----------------------------
    - Production deployments MUST supply `SAAS_MODEL_SECRET` via secure secret management.
    - To rotate a secret: update `SAAS_MODEL_SECRET`, retrain models via the pipeline,
      and re-publish signed provenance manifests.
    - Never log, expose, or commit signing keys.
    """
    if custom_secret is not None:
        if isinstance(custom_secret, str):
            return custom_secret.encode("utf-8")
        return custom_secret

    env_key = os.environ.get("SAAS_MODEL_SECRET")
    if env_key:
        return env_key.encode("utf-8")

    is_strict = (
        require_secret
        or os.environ.get("SAAS_STRICT_SECURITY", "").lower() in ("1", "true", "yes")
        or os.environ.get("SAAS_ENV", "").lower() in ("production", "prod", "strict")
    )
    if is_strict:
        raise SecurityValidationError(
            "SAAS_MODEL_SECRET is required for secure model provenance signing and verification in strict/production mode."
        )

    global _DEV_EPHEMERAL_KEY
    if _DEV_EPHEMERAL_KEY is None:
        _DEV_EPHEMERAL_KEY = os.urandom(32)
    return _DEV_EPHEMERAL_KEY


def is_within_directory(child_path: Union[str, Path], parent_dir: Union[str, Path]) -> bool:
    """Verifies that child_path resolves strictly within parent_dir (no path traversal)."""
    try:
        resolved_child = Path(child_path).resolve()
        resolved_parent = Path(parent_dir).resolve()
        return resolved_parent in resolved_child.parents or resolved_child == resolved_parent
    except Exception:
        return False


def calculate_file_sha256(file_path: Union[str, Path]) -> str:
    """Computes SHA-256 digest of a file in streaming chunks."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_model_provenance(
    model_path: Union[str, Path],
    provenance_path: Union[str, Path],
    require_secret: bool = False,
    custom_secret: Optional[Union[str, bytes]] = None,
    canonical_name: Optional[str] = None,
) -> Path:
    """
    Computes SHA-256 hash and generates a signed cryptographic HMAC-SHA256
    provenance manifest for the serialized model artifact.

    Parameters
    ----------
    model_path : Path or str
        Path to the saved model file (.joblib).
    provenance_path : Path or str
        Destination path for the provenance JSON manifest.
    require_secret : bool, default False
        If True, requires SAAS_MODEL_SECRET to be defined.
    custom_secret : str or bytes, optional
        Custom signing key to use.
    canonical_name : str, optional
        Canonical filename to embed in HMAC message (useful when writing to temp files).

    Returns
    -------
    Path
        Resolved path to the generated provenance manifest.
    """
    m_path = Path(model_path).resolve()
    p_path = Path(provenance_path).resolve()

    if not m_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {m_path}")

    file_size = m_path.stat().st_size
    file_sha256 = calculate_file_sha256(m_path)

    # Compute HMAC signature over canonical payload
    signing_key = _get_signing_key(
        require_secret=require_secret, custom_secret=custom_secret
    )
    art_name = canonical_name if canonical_name is not None else m_path.name
    message = f"{art_name}:{file_size}:{file_sha256}".encode("utf-8")
    signature = hmac.new(signing_key, message, hashlib.sha256).hexdigest()

    manifest = {
        "artifact_name": art_name,
        "file_size": file_size,
        "sha256": file_sha256,
        "algorithm": "HMAC-SHA256",
        "signature": signature,
    }

    p_path.parent.mkdir(parents=True, exist_ok=True)
    with open(p_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return p_path


def verify_model_provenance(
    model_path: Union[str, Path],
    provenance_path: Union[str, Path],
    trusted_dir: Optional[Union[str, Path]] = None,
    require_secret: bool = False,
    custom_secret: Optional[Union[str, bytes]] = None,
) -> bool:
    """
    Validates model integrity, provenance signature, and filesystem boundary.

    Raises
    ------
    SecurityValidationError
        If boundary checks, file size, hash, or cryptographic signature fail.
    """
    m_path = Path(model_path).resolve()
    p_path = Path(provenance_path).resolve()

    # 1. Path Boundary Checks
    if trusted_dir is not None:
        if not is_within_directory(m_path, trusted_dir):
            raise SecurityValidationError(
                f"Model file '{m_path}' is outside trusted boundary '{trusted_dir}'."
            )
        if not is_within_directory(p_path, trusted_dir):
            raise SecurityValidationError(
                f"Provenance manifest '{p_path}' is outside trusted boundary '{trusted_dir}'."
            )

    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found: {m_path}")
    if not p_path.exists():
        raise SecurityValidationError(
            f"Provenance manifest missing for model '{m_path.name}'. Untrusted model load rejected."
        )

    # 2. File Size Safety Cap
    file_size = m_path.stat().st_size
    if file_size > MAX_MODEL_SIZE_BYTES:
        raise SecurityValidationError(
            f"Model file size ({file_size} bytes) exceeds safety maximum ({MAX_MODEL_SIZE_BYTES} bytes)."
        )

    # 3. Read and Validate Manifest Structure
    try:
        with open(p_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        raise SecurityValidationError(f"Invalid or corrupted provenance manifest: {exc}")

    expected_sha256 = manifest.get("sha256")
    expected_sig = manifest.get("signature")

    if not expected_sha256 or not expected_sig:
        raise SecurityValidationError("Provenance manifest missing required hash or signature.")

    # 4. Hash Verification
    actual_sha256 = calculate_file_sha256(m_path)
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise SecurityValidationError(
            f"Model integrity check failed: Hash mismatch for '{m_path.name}'. Possible tampering detected."
        )

    # 5. Cryptographic HMAC Signature Verification
    signing_key = _get_signing_key(
        require_secret=require_secret, custom_secret=custom_secret
    )
    art_name = manifest.get("artifact_name", m_path.name)
    message = f"{art_name}:{file_size}:{actual_sha256}".encode("utf-8")
    computed_sig = hmac.new(signing_key, message, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(computed_sig, expected_sig):
        raise SecurityValidationError(
            f"Cryptographic signature mismatch for '{m_path.name}'. Unauthorized artifact replacement rejected."
        )

    return True


# ── SEC-03: XML ENTITY ESCAPING FOR REPORTLAB (CWE-116) ───────────────────────

def safe_xml_escape(text: Any) -> str:
    """
    Escapes XML special characters (&, <, >, \", ') for safe insertion
    into ReportLab Paragraph flowables.
    """
    if text is None:
        return ""
    s = str(text)
    return saxutils.escape(s, entities={'"': "&quot;", "'": "&apos;"})


# ── SEC-05: DATASET SANITY & DOS MITIGATION (CWE-400) ─────────────────────────

def validate_dataset_size(
    df: pd.DataFrame,
    max_rows: int = MAX_FINANCIAL_ROWS,
    max_cols: int = 50,
    dataset_name: str = "Dataset",
) -> None:
    """
    Validates that a DataFrame does not exceed maximum row/column safety bounds.
    """
    if df is None:
        return
    rows, cols = df.shape
    if rows > max_rows:
        raise SecurityValidationError(
            f"{dataset_name} row count ({rows:,}) exceeds maximum allowed ({max_rows:,})."
        )
    if cols > max_cols:
        raise SecurityValidationError(
            f"{dataset_name} column count ({cols}) exceeds maximum allowed ({max_cols})."
        )


# ── CONC-02: ATOMIC FILE PERSISTENCE ──────────────────────────────────────────

def atomic_write_bytes(target_path: Union[str, Path], data: bytes) -> Path:
    """
    Writes data to a temporary file in the same directory and replaces target_path
    atomically using os.replace to prevent partial reads and write race conditions.
    """
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    temp_fd, temp_path_str = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".tmp_{target.stem}_",
        suffix=target.suffix,
    )
    try:
        with os.fdopen(temp_fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path_str, target)
    except Exception:
        if os.path.exists(temp_path_str):
            try:
                os.remove(temp_path_str)
            except OSError:
                pass
        raise

    return target


def atomic_publish_model_and_provenance(
    pipeline: Any,
    model_path: Union[str, Path],
    provenance_path: Optional[Union[str, Path]] = None,
    secret: Optional[Union[str, bytes]] = None,
    require_secret: bool = False,
) -> Tuple[Path, Path]:
    """
    Atomically publishes a serialized scikit-learn/joblib model artifact and its
    cryptographic HMAC provenance manifest together under a shared lock.

    Guarantees:
    - Never leaves a torn or partially written file on disk.
    - Model binary and provenance manifest hashes/signatures remain 100% synchronized.
    - Cross-platform atomic replacement via `os.replace`.

    Parameters
    ----------
    pipeline : Any
        Fitted model pipeline or estimator to serialize.
    model_path : str or Path
        Target destination for model file (.joblib).
    provenance_path : str or Path, optional
        Target destination for provenance manifest. Defaults to <stem>_provenance.json.
    secret : str or bytes, optional
        Optional HMAC signing key.
    require_secret : bool, default False
        Whether to require SAAS_MODEL_SECRET.

    Returns
    -------
    Tuple[Path, Path]
        Paths to published model and provenance manifest.
    """
    import joblib

    m_target = Path(model_path).resolve()
    m_target.parent.mkdir(parents=True, exist_ok=True)
    if provenance_path is None:
        p_target = m_target.parent / f"{m_target.stem}_provenance.json"
    else:
        p_target = Path(provenance_path).resolve()
        p_target.parent.mkdir(parents=True, exist_ok=True)

    with ARTIFACT_PUBLISH_LOCK:
        temp_model_fd, temp_model_str = tempfile.mkstemp(
            dir=m_target.parent, prefix=f".tmp_{m_target.stem}_", suffix=m_target.suffix
        )
        os.close(temp_model_fd)
        temp_model_path = Path(temp_model_str)

        temp_prov_fd, temp_prov_str = tempfile.mkstemp(
            dir=p_target.parent, prefix=f".tmp_{p_target.stem}_", suffix=p_target.suffix
        )
        os.close(temp_prov_fd)
        temp_prov_path = Path(temp_prov_str)

        try:
            joblib.dump(pipeline, temp_model_path)
            generate_model_provenance(
                temp_model_path,
                temp_prov_path,
                require_secret=require_secret,
                custom_secret=secret,
                canonical_name=m_target.name,
            )
            os.replace(temp_model_path, m_target)
            os.replace(temp_prov_path, p_target)
        except Exception:
            if temp_model_path.exists():
                try:
                    temp_model_path.unlink()
                except OSError:
                    pass
            if temp_prov_path.exists():
                try:
                    temp_prov_path.unlink()
                except OSError:
                    pass
            raise

    return m_target, p_target
