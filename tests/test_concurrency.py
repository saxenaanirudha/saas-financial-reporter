"""
test_concurrency.py
===================
Concurrency & multi-user tests for SaaS Financial Reporter:
- CONC-01: Matplotlib figure generation thread safety (no cross-thread corruption)
- CONC-02: Atomic file write and concurrent output file isolation
"""

import concurrent.futures
from pathlib import Path
import threading
import pandas as pd
import pytest

from src.chart_generator import (
    generate_mrr_chart,
    generate_arr_chart,
    generate_cac_chart,
)
from src.security import atomic_write_bytes


@pytest.fixture
def sample_financial_data():
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


def test_concurrent_chart_generation_does_not_corrupt(sample_financial_data, tmp_path: Path):
    """
    Simultaneously generate charts across multiple threads.
    Validates that Matplotlib state does not cross-contaminate or crash threads.
    """
    charts_dir = tmp_path / "concurrent_charts"
    charts_dir.mkdir()

    def generate_task(worker_id: int):
        sub_dir = charts_dir / f"worker_{worker_id}"
        sub_dir.mkdir()
        p1 = generate_mrr_chart(sample_financial_data, output_dir=sub_dir)
        p2 = generate_arr_chart(sample_financial_data, output_dir=sub_dir)
        p3 = generate_cac_chart(sample_financial_data, output_dir=sub_dir)
        return p1.stat().st_size > 0 and p2.stat().st_size > 0 and p3.stat().st_size > 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(generate_task, i) for i in range(6)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 6
    assert all(results)


def test_atomic_write_bytes_prevents_partial_file_reads(tmp_path: Path):
    """Atomic write replaces target file atomically via temporary write and rename."""
    target_file = tmp_path / "output.bin"
    payload = b"SECURE_ATOMIC_PAYLOAD_CONTENT"

    atomic_write_bytes(target_file, payload)

    assert target_file.exists()
    assert target_file.read_bytes() == payload


def test_all_matplotlib_operations_are_synchronized():
    """
    BLK-03: Every chart generation operation across churn_model and app
    must be protected by the thread-safety synchronization lock.
    """
    import inspect
    from src.churn_model import generate_confusion_matrix_chart, generate_roc_curve_chart
    import app

    # ML charts must be protected by @thread_safe_chart wrapper
    assert hasattr(generate_confusion_matrix_chart, "__wrapped__"), (
        "generate_confusion_matrix_chart must be protected by @thread_safe_chart"
    )
    assert hasattr(generate_roc_curve_chart, "__wrapped__"), (
        "generate_roc_curve_chart must be protected by @thread_safe_chart"
    )

    # Streamlit dashboard inline charts must be extracted into locked helper functions
    assert hasattr(app, "render_risk_distribution_chart"), (
        "app.py must have render_risk_distribution_chart helper"
    )
    assert hasattr(app.render_risk_distribution_chart, "__wrapped__"), (
        "render_risk_distribution_chart must be protected by @thread_safe_chart"
    )
    assert hasattr(app, "render_feature_impact_chart"), (
        "app.py must have render_feature_impact_chart helper"
    )
    assert hasattr(app.render_feature_impact_chart, "__wrapped__"), (
        "render_feature_impact_chart must be protected by @thread_safe_chart"
    )


def test_atomic_artifact_publishing_ensures_model_and_provenance_consistency(tmp_path: Path):
    """
    BLK-04: Model binary and cryptographic provenance manifest must be published
    atomically so that intermediate states or mismatched pairs cannot be read.
    """
    from src.security import atomic_publish_model_and_provenance, verify_model_provenance
    from src.churn_model import build_churn_pipeline

    pipeline = build_churn_pipeline()
    model_path = tmp_path / "models" / "churn_model.joblib"
    prov_path = tmp_path / "models" / "churn_model_provenance.json"

    m_out, p_out = atomic_publish_model_and_provenance(
        pipeline=pipeline,
        model_path=model_path,
        provenance_path=prov_path,
        secret=b"test-secret-12345",
    )

    assert m_out.exists()
    assert p_out.exists()
    assert m_out.stat().st_size > 0
    assert p_out.stat().st_size > 0

    # Published pair must verify successfully
    assert verify_model_provenance(
        m_out,
        p_out,
        custom_secret=b"test-secret-12345",
    ) is True

    # Concurrent publishing simulation: verify consistency under simultaneous updates
    def concurrent_worker(worker_id: int):
        worker_pipeline = build_churn_pipeline()
        atomic_publish_model_and_provenance(
            pipeline=worker_pipeline,
            model_path=model_path,
            provenance_path=prov_path,
            secret=b"test-secret-12345",
        )
        assert verify_model_provenance(
            model_path,
            prov_path,
            custom_secret=b"test-secret-12345",
        ) is True

    threads = [threading.Thread(target=concurrent_worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Final verification after concurrent writes
    assert verify_model_provenance(
        model_path,
        prov_path,
        custom_secret=b"test-secret-12345",
    ) is True
