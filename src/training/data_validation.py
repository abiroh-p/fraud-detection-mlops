"""
Data validation for the training pipeline.

Validates a feature DataFrame before training begins.
Catches data quality issues early — before they silently
corrupt model weights and degrade production performance.

In production, companies like Spotify and Uber run validation
gates that block training if data quality drops below a threshold.
This is exactly that gate.
"""

from dataclasses import dataclass, field

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cfg = load_config("model_config.yaml")
_feature_cfg = _cfg["features"]
_training_cfg = _cfg["training"]


@dataclass
class ValidationReport:
    """
    Holds the results of a data validation run.

    Using a dataclass instead of a plain dict gives us:
    - Type hints on every field
    - Clean __repr__ for logging
    - Immutable-by-default fields with field()
    """

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    fraud_count: int = 0
    fraud_rate: float = 0.0

    def add_error(self, message: str) -> None:
        """Add a blocking error — will cause validation to fail."""
        self.errors.append(message)
        self.passed = False

    def add_warning(self, message: str) -> None:
        """Add a non-blocking warning — logged but training continues."""
        self.warnings.append(message)

    def log_summary(self) -> None:
        """Print a human-readable validation summary."""
        status = "PASSED" if self.passed else "FAILED"
        logger.info("── Data Validation %s ──────────────────", status)
        logger.info("  Rows        : %d", self.row_count)
        logger.info("  Fraud count : %d", self.fraud_count)
        logger.info("  Fraud rate  : %.2f%%", self.fraud_rate * 100)

        for warning in self.warnings:
            logger.warning("  WARNING: %s", warning)

        for error in self.errors:
            logger.error("  ERROR: %s", error)


def validate_training_data(df: pd.DataFrame) -> ValidationReport:
    """
    Run all validation checks on a feature DataFrame.

    Checks performed:
    1. Minimum row count — not enough data = unreliable model
    2. Required columns present — missing features = training crash
    3. No NaN values in feature columns — silently corrupt gradients
    4. Target column exists and is binary — wrong target = wrong model
    5. Fraud rate sanity check — if 0% or 100% fraud, data is broken
    6. Feature value ranges — catch obviously wrong values

    Args:
        df: DataFrame of engineered feature vectors.

    Returns:
        ValidationReport with passed=True if all checks pass.

    Raises:
        DataValidationError: If validation fails (passed=False).
    """
    report = ValidationReport()
    report.row_count = len(df)

    # ── Check 1: Minimum sample count ────────────────────────────
    min_samples = _training_cfg["min_samples"]
    if len(df) < min_samples:
        report.add_error(
            f"Insufficient training data: {len(df)} rows found, "
            f"minimum required is {min_samples}. "
            f"Run the simulator longer to collect more transactions."
        )

    # ── Check 2: Required feature columns present ─────────────────
    required_features = _feature_cfg["numeric_features"]
    target_col = _feature_cfg["target"]
    required_cols = required_features + [target_col]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        report.add_error(
            f"Missing required columns: {missing_cols}. "
            f"Check that FeatureEngineer produces all expected features."
        )

    # ── Check 3: No NaN values in feature columns ─────────────────
    if not missing_cols:
        nan_counts = df[required_features].isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if not nan_cols.empty:
            report.add_error(
                f"NaN values detected in feature columns: "
                f"{nan_cols.to_dict()}. "
                f"Check feature engineering for edge cases."
            )

    # ── Check 4: Target column is binary ─────────────────────────
    if target_col in df.columns:
        unique_values = set(df[target_col].unique())
        if not unique_values.issubset({0, 1}):
            report.add_error(
                f"Target column '{target_col}' contains non-binary values: "
                f"{unique_values}. Expected only 0 and 1."
            )

        # ── Check 5: Fraud rate sanity ────────────────────────────
        report.fraud_count = int(df[target_col].sum())
        report.fraud_rate = report.fraud_count / max(len(df), 1)

        if report.fraud_rate == 0.0:
            report.add_error(
                "No fraud cases found in training data. "
                "The model cannot learn to detect fraud without positive examples."
            )
        elif report.fraud_rate == 1.0:
            report.add_error(
                "All transactions are labeled as fraud. "
                "This indicates a data pipeline error."
            )
        elif report.fraud_rate < 0.005:
            report.add_warning(
                f"Very low fraud rate: {report.fraud_rate:.3%}. "
                f"Consider collecting more data or adjusting scale_pos_weight."
            )

    # ── Check 6: Feature value ranges ────────────────────────────
    if "amount" in df.columns:
        if (df["amount"] < 0).any():
            report.add_error(
                "Negative transaction amounts detected. "
                "Check transaction simulator or data source."
            )
        if (df["amount"] > 50_000).any():
            report.add_warning(
                f"Transactions above $50,000 detected "
                f"({(df['amount'] > 50_000).sum()} rows). "
                f"Verify these are legitimate."
            )

    if "hour_of_day" in df.columns:
        if not df["hour_of_day"].between(0, 23).all():
            report.add_error(
                "Invalid hour_of_day values detected. Expected range 0-23."
            )

    report.log_summary()

    if not report.passed:
        raise DataValidationError(
            f"Data validation failed with {len(report.errors)} error(s). "
            f"See log output above for details."
        )

    return report
