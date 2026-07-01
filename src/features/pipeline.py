"""
Feature engineering pipeline entry point.

Connects the three components:
  TransactionConsumer → FeatureEngineer → FeatureVectorProducer

Run this as a long-running service. It will continuously consume
raw transactions, engineer features, and publish feature vectors.
"""

from src.features.feature_engineering import FeatureEngineer
from src.features.kafka_consumer import TransactionConsumer
from src.features.kafka_producer import FeatureVectorProducer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(max_messages: int | None = None) -> None:
    """
    Run the feature engineering pipeline.

    Args:
        max_messages: Stop after N messages. None = run forever.
    """
    logger.info("Starting feature engineering pipeline.")

    consumer = TransactionConsumer()
    engineer = FeatureEngineer()
    producer = FeatureVectorProducer()

    try:
        for transaction in consumer.consume(max_messages=max_messages):
            features = engineer.transform(transaction)
            producer.publish(features)

            if features["is_fraud"]:
                logger.info(
                    "FRAUD detected — user=%s amount=%.2f speed=%.1f km/h",
                    features["user_id"],
                    features["amount"],
                    features["speed_from_last_txn_kmh"],
                )

    finally:
        producer.flush()
        logger.info("Feature pipeline shut down cleanly.")


if __name__ == "__main__":
    run_pipeline()
