"""
Retraining trigger for the fraud detection pipeline.

Runs on a schedule, collects recent feature vectors,
compares them against the reference distribution,
and triggers retraining if drift is detected.

This closes the MLOps loop:
  Data → Features → Model → Predictions → Drift Check → Retrain → Model
                                              ▲                        │
                                              └────────────────────────┘

In production this would be orchestrated by Airflow or Prefect.
For our pipeline, it runs as a standalone script on a cron schedule.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd

from src.features.feature_engineering import FeatureEngineer
from src.features.kafka_consumer import TransactionConsumer
from src.monitoring.drift_detector import detect_drift
from src.training.model import FEATURE_COLUMNS
from src.training.train import run_training
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cfg = load_config("model_config.yaml")
_mlflow_cfg = _cfg["mlflow"]
_monitoring_cfg = load_config("monitoring_config.yaml")

# Minimum transactions needed for a reliable drift check
MIN_SAMPLES_FOR_DRIFT = 200

# Reference data path — saved during training
REFERENCE_DATA_PATH = Path("data/reference/reference_features.parquet")


def collect_recent_features(num_transactions: int = 500) -> pd.DataFrame:
    """
    Collect recent transactions from Kafka and engineer features.

    Args:
        num_transactions: Number of recent transactions to sample.

    Returns:
        DataFrame of recent feature vectors.
    """
    logger.info(
        "Collecting %d recent transactions for drift check...",
        num_transactions,
    )

    consumer = TransactionConsumer(group_id="drift-detector")
    engineer = FeatureEngineer()
    records = []

    for transaction in consumer.consume(max_messages=num_transactions):
        features = engineer.transform(transaction)
        records.append(features)

    df = pd.DataFrame(records)
    logger.info("Collected %d feature vectors.", len(df))
    return df


def save_reference_data(df: pd.DataFrame) -> None:
    """
    Save the training feature distribution as reference data.

    Call this once after training to establish the baseline.
    Subsequent drift checks compare against this baseline.

    Args:
        df: DataFrame of training feature vectors.
    """
    REFERENCE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[FEATURE_COLUMNS].to_parquet(REFERENCE_DATA_PATH, index=False)
    logger.info(
        "Saved reference data to %s (%d rows)",
        REFERENCE_DATA_PATH,
        len(df),
    )


def load_reference_data() -> pd.DataFrame | None:
    """
    Load the reference feature distribution from disk.

    Returns:
        Reference DataFrame, or None if not found.
    """
    if not REFERENCE_DATA_PATH.exists():
        logger.warning(
            "Reference data not found at %s. "
            "Run save_reference_data() after training first.",
            REFERENCE_DATA_PATH,
        )
        return None

    df = pd.read_parquet(REFERENCE_DATA_PATH)
    logger.info(
        "Loaded reference data: %d rows, %d features",
        len(df),
        len(df.columns),
    )
    return df


def promote_model_to_champion(model_name: str, version: str) -> None:
    """
    Promote a model version to the 'champion' alias in MLflow.

    The serving layer loads the champion alias — promoting here
    means the new model is picked up on next serving restart
    without any code change.

    Args:
        model_name: Registered model name in MLflow.
        version: Version number to promote.
    """
    mlflow.set_tracking_uri(_mlflow_cfg["tracking_uri"])
    client = mlflow.tracking.MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version,
    )

    logger.info(
        "Promoted model '%s' version %s to 'champion' alias.",
        model_name,
        version,
    )


def run_drift_check_and_retrain(
    num_recent_transactions: int = 500,
    force_retrain: bool = False,
) -> dict:
    """
    Main entry point for the drift detection and retraining loop.

    Steps:
    1. Load reference distribution
    2. Collect recent feature vectors from Kafka
    3. Run drift detection
    4. If drift detected (or forced), trigger retraining
    5. Promote new model to champion alias

    Args:
        num_recent_transactions: How many recent transactions to check.
        force_retrain: Skip drift check and retrain unconditionally.

    Returns:
        Dict with drift results and retraining outcome.
    """
    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "drift_detected": False,
        "retrained": False,
        "new_model_version": None,
    }

    # ── Step 1: Load reference data ───────────────────────────────
    reference_df = load_reference_data()
    if reference_df is None and not force_retrain:
        logger.error(
            "Cannot run drift check without reference data. " "Train a model first."
        )
        return result

    # ── Step 2: Collect recent features ───────────────────────────
    current_df = collect_recent_features(num_recent_transactions)

    if len(current_df) < MIN_SAMPLES_FOR_DRIFT:
        logger.warning(
            "Insufficient recent data for drift check: %d samples "
            "(minimum %d). Skipping.",
            len(current_df),
            MIN_SAMPLES_FOR_DRIFT,
        )
        return result

    # ── Step 3: Run drift detection ───────────────────────────────
    if reference_df is not None:
        report = detect_drift(
            reference_df=reference_df,
            current_df=current_df[FEATURE_COLUMNS],
            features_to_check=FEATURE_COLUMNS,
        )
        result["drift_detected"] = report.overall_drift_detected
        result["drifted_features"] = report.drifted_features
    else:
        logger.info("No reference data — skipping drift check.")

    # ── Step 4: Retrain if needed ─────────────────────────────────
    should_retrain = force_retrain or result["drift_detected"]

    if should_retrain:
        reason = "forced" if force_retrain else "drift detected"
        logger.info("Triggering retraining (%s)...", reason)

        run_id = run_training(num_transactions=2000)
        result["retrained"] = True
        result["run_id"] = run_id

        # ── Step 5: Promote new model ─────────────────────────────
        mlflow.set_tracking_uri(_mlflow_cfg["tracking_uri"])
        client = mlflow.tracking.MlflowClient()

        # Get the version created by this run
        versions = client.search_model_versions(
            f"name='{_mlflow_cfg['registered_model_name']}'"
        )
        latest = max(versions, key=lambda v: int(v.version))

        promote_model_to_champion(
            model_name=_mlflow_cfg["registered_model_name"],
            version=latest.version,
        )

        result["new_model_version"] = latest.version
        logger.info(
            "Retraining complete. New champion: version %s",
            latest.version,
        )
    else:
        logger.info("No drift detected. Retraining not required.")

    return result


if __name__ == "__main__":
    # Run with force_retrain=True to test the full pipeline
    outcome = run_drift_check_and_retrain(
        num_recent_transactions=500,
        force_retrain=True,
    )
    logger.info("Outcome: %s", json.dumps(outcome, indent=2))
