"""
Data drift detector for the fraud detection pipeline.

Compares the current distribution of incoming feature vectors
against the reference distribution from training time.

When drift is detected, it signals the retraining trigger
to schedule a new training run.

Statistical tests used:
- Population Stability Index (PSI) — industry standard in banking
- Kolmogorov-Smirnov test — non-parametric distribution comparison
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cfg = load_config("monitoring_config.yaml")
_alert_threshold = _cfg["metrics"]["fraud_rate_alert_threshold"]

# PSI thresholds — industry standard from banking risk management
PSI_NO_DRIFT = 0.1
PSI_MODERATE_DRIFT = 0.2
# KS test significance level
KS_ALPHA = 0.05


@dataclass
class FeatureDriftResult:
    """Drift test results for a single feature."""

    feature_name: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    drift_detected: bool
    severity: str  # "none", "moderate", "high"


@dataclass
class DriftReport:
    """
    Complete drift report across all features.

    Aggregates individual feature results into an overall
    drift verdict used by the retraining trigger.
    """

    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    overall_drift_detected: bool = False
    drifted_features: list[str] = field(default_factory=list)
    high_drift_features: list[str] = field(default_factory=list)

    def log_summary(self) -> None:
        """Print a human-readable drift summary."""
        status = "DRIFT DETECTED" if self.overall_drift_detected else "NO DRIFT"
        logger.info("── Drift Report: %s ─────────────────────", status)
        logger.info("  Total features checked : %d", len(self.feature_results))
        logger.info("  Drifted features       : %d", len(self.drifted_features))
        logger.info("  High drift features    : %s", self.high_drift_features)

        for result in self.feature_results:
            if result.drift_detected:
                logger.warning(
                    "  DRIFT [%s] %s — PSI=%.3f KS_p=%.4f",
                    result.severity.upper(),
                    result.feature_name,
                    result.psi,
                    result.ks_p_value,
                )


def _compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI = Σ (current% - reference%) × ln(current% / reference%)

    PSI measures how much a population has shifted relative to
    a baseline. Originally developed for credit scoring models
    at banks — now standard across all financial ML systems.

    Args:
        reference: Reference distribution (training data).
        current: Current distribution (live data).
        n_bins: Number of bins for discretizing continuous features.

    Returns:
        PSI score. Higher = more drift.
    """
    # Create bins based on reference distribution
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    # Count observations in each bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Convert to percentages — add small epsilon to avoid log(0)
    epsilon = 1e-6
    ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * n_bins)
    cur_pct = (cur_counts + epsilon) / (len(current) + epsilon * n_bins)

    # PSI formula
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _classify_psi(psi: float) -> tuple[bool, str]:
    """
    Classify PSI score into drift severity.

    Returns:
        Tuple of (drift_detected, severity_label)
    """
    if psi < PSI_NO_DRIFT:
        return False, "none"
    elif psi < PSI_MODERATE_DRIFT:
        return True, "moderate"
    else:
        return True, "high"


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features_to_check: list[str] | None = None,
) -> DriftReport:
    """
    Run drift detection across all specified features.

    Compares the current distribution of incoming features
    against the reference distribution from training time.

    Args:
        reference_df: DataFrame from training time (reference).
        current_df: DataFrame of recent incoming features.
        features_to_check: List of feature columns to check.
                           If None, checks all numeric columns.

    Returns:
        DriftReport with per-feature results and overall verdict.
    """
    if features_to_check is None:
        features_to_check = [
            col
            for col in reference_df.select_dtypes(include=[np.number]).columns
            if col in current_df.columns
        ]

    report = DriftReport()

    for feature in features_to_check:
        ref_values = reference_df[feature].dropna().values
        cur_values = current_df[feature].dropna().values

        if len(ref_values) < 30 or len(cur_values) < 30:
            logger.warning(
                "Skipping drift check for '%s' — insufficient data " "(ref=%d, cur=%d)",
                feature,
                len(ref_values),
                len(cur_values),
            )
            continue

        # PSI test
        psi = _compute_psi(ref_values, cur_values)
        drift_detected, severity = _classify_psi(psi)

        # KS test — independent second opinion
        ks_stat, ks_p_value = stats.ks_2samp(ref_values, cur_values)
        ks_drift = ks_p_value < KS_ALPHA

        # Drift confirmed if BOTH tests agree
        confirmed_drift = drift_detected and ks_drift

        result = FeatureDriftResult(
            feature_name=feature,
            psi=round(psi, 4),
            ks_statistic=round(float(ks_stat), 4),
            ks_p_value=round(float(ks_p_value), 4),
            drift_detected=confirmed_drift,
            severity=severity if confirmed_drift else "none",
        )

        report.feature_results.append(result)

        if confirmed_drift:
            report.drifted_features.append(feature)
            if severity == "high":
                report.high_drift_features.append(feature)

    # Overall drift if any feature drifted
    report.overall_drift_detected = len(report.drifted_features) > 0
    report.log_summary()

    return report
