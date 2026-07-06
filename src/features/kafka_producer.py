"""
Kafka producer for publishing feature vectors.

Reads from FeatureEngineer output (a plain dict) and publishes
it as JSON to the 'feature-vectors' topic.

Design decision: This is intentionally a thin wrapper around
confluent_kafka.Producer. Its only job is serialization and
publishing. No business logic lives here.
"""

import json
from typing import Any

from confluent_kafka import Producer

from src.utils.config_loader import get_kafka_config
from src.utils.exceptions import KafkaPublishError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureVectorProducer:
    """
    Publishes feature vectors to the 'feature-vectors' Kafka topic.

    Usage:
        producer = FeatureVectorProducer()
        producer.publish(feature_dict)
        producer.flush()
    """

    def __init__(self) -> None:
        """
        Initialize producer and resolve target topic from config.
        """
        self._config = get_kafka_config()
        self._topic = self._config["topics"]["feature_vectors"]

        producer_cfg = self._config.get("producer", {})

        self._producer = Producer(
            {
                "bootstrap.servers": self._config["bootstrap_servers"],
                "acks": producer_cfg.get("acks", "all"),
                "retries": producer_cfg.get("retries", 3),
                "retry.backoff.ms": producer_cfg.get("retry_backoff_ms", 300),
                "compression.type": "snappy",
            }
        )

        logger.info("FeatureVectorProducer initialized. Topic: %s", self._topic)

    def publish(self, features: dict[str, Any]) -> None:
        """
        Serialize and publish a feature vector to Kafka.

        The message key is user_id — same reason as the transaction
        simulator. All feature vectors for the same user land in the
        same partition, preserving per-user ordering.

        Args:
            features: Feature dict produced by FeatureEngineer.transform().

        Raises:
            KafkaPublishError: If the message cannot be queued.
        """
        user_id = features.get("user_id", "unknown")

        try:
            self._producer.produce(
                topic=self._topic,
                key=user_id.encode("utf-8"),
                value=json.dumps(features, default=str).encode("utf-8"),
                callback=self._delivery_callback,
            )
            # Poll triggers pending delivery callbacks
            self._producer.poll(0)

        except BufferError as e:
            raise KafkaPublishError(
                f"Producer queue full while publishing feature vector "
                f"for user '{user_id}': {e}"
            ) from e

    def flush(self, timeout: float = 30.0) -> None:
        """
        Block until all queued messages are delivered.

        Always call this before shutdown. Without it, messages still
        sitting in the producer's internal buffer will be silently lost
        when the process exits.

        Args:
            timeout: Maximum seconds to wait for delivery confirmation.
        """
        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning(
                "%d message(s) were not delivered within timeout.", remaining
            )
        else:
            logger.debug("All messages flushed successfully.")

    @staticmethod
    def _delivery_callback(err: Exception | None, msg: object) -> None:
        """
        Called asynchronously after each message delivery attempt.

        Args:
            err: None on success, Exception on failure.
            msg: The delivered message object.
        """
        if err is not None:
            logger.error("Feature vector delivery failed: %s", err)
        else:
            logger.debug(
                "Feature vector delivered to %s [partition %d] @ offset %d",
                msg.topic(),
                msg.partition(),
                msg.offset(),
            )
