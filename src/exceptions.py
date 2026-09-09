"""
exceptions.py
=============
Lightweight domain exceptions for SaaS Financial Reporter.
Step 14: Full Pipeline Integration, Error Handling, and Execution Reliability.
"""


class PipelineError(Exception):
    """Base exception for all pipeline orchestration and execution errors."""
    pass


class FinancialDataError(PipelineError):
    """Raised when monthly financial data is missing, invalid, or corrupted."""
    pass


class FinancialCalculationError(PipelineError):
    """Raised when core financial or KPI calculations encounter numerical/domain issues."""
    pass


class ChurnDataError(PipelineError):
    """Raised when customer-level churn data fails validation or has single-class targets."""
    pass


class ChurnModelError(PipelineError):
    """Raised when churn model training, evaluation, or scoring encounters an unrecoverable failure."""
    pass


class ReportGenerationError(PipelineError):
    """Raised when PDF report or chart generation fails critically."""
    pass


class ArtifactValidationError(PipelineError):
    """Raised when required output artifacts are missing or empty."""
    pass
