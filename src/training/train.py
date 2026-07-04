"""
Main training pipeline entry point.

Orchestrates the full training workflow:
  1. Collect feature vectors from Kafka
  2. Validate data quality
  3. Split into train/test sets
  4. Train the model pipeline
  5. Evaluate on test set
  6. Log everything to MLflow
  7. Register best model in MLflow Model Registry

Run this script to train a new model:
    python -m src.training.train
"""

import json

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.feature_engineering import FeatureEngineer
from src.features.kafka_consumer import TransactionConsumer
from src.training.data_validation import validate_training_data
from src.training.evaluate import evaluate_model
from src.training.model import build_pipeline, get_feature_matrix, get_target_series
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cfg = load_config("model_config.yaml")
_mlflow_cfg = _cfg["mlflow"]
_training_cfg = _cfg["training"]


def collect_training_data(num_transactions: int = 2000) -> pd.DataFrame:
    """
    Consume transactions from Kafka and engineer features.

    In a production system this would read from a PostgreSQL
    table of historical labeled transactions. For our pipeline,
    we consume directly from Kafka and engineer features on the fly.

    Args:
        num_transactions: Number of transactions to collect.

    Returns:
        DataFrame of engineered feature vectors with labels.
    """
    logger.info(
        "Collecting %d transactions from Kafka...", num_transactions
    )

    consumer = TransactionConsumer(group_id="training-pipeline")
    engineer = FeatureEngineer()
    records = []

    for transaction in consumer.consume(max_messages=num_transactions):
        features = engineer.transform(transaction)
        records.append(features)

    df = pd.DataFrame(records)

    logger.info(
        "Collected %d records. Fraud rate: %.2f%%",
        len(df),
        df["is_fraud"].mean() * 100,
    )

    return df


def run_training(num_transactions: int = 2000) -> str:
    """
    Execute the full training pipeline under an MLflow run.

    Every training run is tracked in MLflow with:
    - Parameters: hyperparameters, data size, feature list
    - Metrics: precision, recall, F1, AUC, fraud catch rate
    - Artifacts: the trained model pipeline
    - Tags: who ran it, what git commit, dataset size

    Args:
        num_transactions: Number of transactions to train on.

    Returns:
        MLflow run ID for this training run.
    """
    # ── MLflow Setup ──────────────────────────────────────────────
    mlflow.set_tracking_uri(_mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(_mlflow_cfg["experiment_name"])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run started. Run ID: %s", run_id)

        # ── Step 1: Collect Data ──────────────────────────────────
        df = collect_training_data(num_transactions)

        # ── Step 2: Validate Data ─────────────────────────────────
        # Raises DataValidationError if data is bad — run fails cleanly
        report = validate_training_data(df)

        # ── Step 3: Prepare Features and Target ───────────────────
        X = get_feature_matrix(df)
        y = get_target_series(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=_training_cfg["test_size"],
            random_state=_training_cfg["random_state"],
            # stratify ensures fraud rate is preserved in both splits
            # Without this, by chance the test set could have 0 fraud
            stratify=y,
        )

        logger.info(
            "Train size: %d | Test size: %d | Train fraud: %.2f%%",
            len(X_train),
            len(X_test),
            y_train.mean() * 100,
        )

        # ── Step 4: Log Parameters to MLflow ──────────────────────
        mlflow.log_params(_cfg["model"]["hyperparameters"])
        mlflow.log_params(
            {
                "num_transactions": num_transactions,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "fraud_rate": round(report.fraud_rate, 4),
                "feature_count": len(X.columns),
            }
        )

        # Log the full feature list as a JSON artifact
        # This lets us reproduce the exact feature set later
        mlflow.log_text(
            json.dumps(list(X.columns), indent=2),
            "feature_columns.json",
        )

        # ── Step 5: Train ─────────────────────────────────────────
        logger.info("Training model...")
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        logger.info("Training complete.")

        # ── Step 6: Evaluate ──────────────────────────────────────
        metrics = evaluate_model(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics.to_dict())

        # ── Step 7: Log Model and Register ───────────────────────
        # Log the full pipeline (scaler + classifier) as one artifact
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            # Signature captures input schema — enables serving validation
            signature=mlflow.models.infer_signature(
                X_train, pipeline.predict(X_train)
            ),
            # registered_model_name triggers automatic registration
            registered_model_name=_mlflow_cfg["registered_model_name"],
            # MLflow 3.x security: explicitly trust XGBoost types
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        logger.info(
            "Model registered as '%s' in MLflow Model Registry.",
            _mlflow_cfg["registered_model_name"],
        )
        logger.info(
            "View this run: %s/#/experiments/1/runs/%s",
            _mlflow_cfg["tracking_uri"],
            run_id,
        )

        return run_id


if __name__ == "__main__":
    run_id = run_training(num_transactions=2000)
    logger.info("Training pipeline complete. Run ID: %s", run_id)
