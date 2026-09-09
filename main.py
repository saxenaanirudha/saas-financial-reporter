"""
=============================================================================
SaaS Financial Reporting Automation
=============================================================================
Main application entry point.
Step 14: Full Pipeline Integration, Error Handling, and Execution Reliability.
=============================================================================
"""

import sys
from pathlib import Path
from src.pipeline import run_full_pipeline


def main() -> None:
    """
    Main entry point for running the end-to-end SaaS Financial Reporting
    and Customer Churn ML pipelines.

    Execution Flow:
    1. Resolves project root dynamically.
    2. Executes both the financial reporting and churn ML pipelines via `run_full_pipeline`.
    3. Exits with status code 0 on SUCCESS/PARTIAL_SUCCESS, or 1 on critical failure.
    """
    # Dynamic path resolution ensures portability across environments
    project_root = Path(__file__).resolve().parent

    # Run the comprehensive pipeline (ingestion, analytics, charts, PDF, ML scoring)
    result = run_full_pipeline(
        project_root=project_root,
        run_churn_ml=True,
        verbose=True,
    )

    # Exit code contract: Non-zero exit code if critical steps failed
    if result["overall_status"] == "FAILED":
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
