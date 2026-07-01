"""
Transaction simulator — generates realistic banking transactions
and publishes them to the Kafka raw-transactions topic.

In production, this role is played by the actual bank's
transaction processing system (e.g., Visa's VisaNet).

For our pipeline, this simulator:
- Generates statistically realistic transaction data
- Injects a configurable fraud rate (~2% matches real-world rates)
- Publishes continuously to Kafka
- Handles Kafka connection failures with retries
"""

import json
import random
import time
from datetime import datetime, timezone
from typing import Optional

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.data.schemas import MerchantCategory, Transaction
from src.utils.config_loader import get_kafka_config
from src.utils.exceptions import KafkaPublishError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────

# Real-world fraud rate is approximately 0.1% to 2%
# We use 5% here so we have enough fraud examples to train on
FRAUD_RATE = 0.05

# Realistic US merchant location bounds
LAT_RANGE = (25.0, 48.0)
LON_RANGE = (-125.0, -67.0)

# Typical transaction amounts by category (mean, std_dev)
AMOUNT_PROFILES: dict[MerchantCategory, tuple[float, float]] = {
    MerchantCategory.GROCERY: (65.0, 30.0),
    MerchantCategory.ELECTRONICS: (350.0, 200.0),
    MerchantCategory.RESTAURANT: (45.0, 25.0),
    MerchantCategory.TRAVEL: (800.0, 400.0),
    MerchantCategory.ONLINE: (120.0, 80.0),
    MerchantCategory.GAS: (55.0, 20.0),
    MerchantCategory.PHARMACY: (40.0, 25.0),
    MerchantCategory.ATM: (200.0, 100.0),
    MerchantCategory.OTHER: (90.0, 50.0),
}


def _delivery_callback(err: Optional[Exception], msg: object) -> None:
    """
    Called by the Kafka producer after each message is delivered or fails.

    Kafka's producer is asynchronous — it sends messages in the background.
    This callback tells us whether delivery succeeded or failed.

    Args:
        err: None if delivery succeeded, Exception if it failed.
        msg: The message object that was delivered.
    """
    if err is not None:
        logger.error("Message delivery failed: %s", err)
    else:
        logger.debug(
            "Message delivered to %s [partition %d] @ offset %d",
            msg.topic(),
            msg.partition(),
            msg.offset(),
        )


def _create_topic_if_not_exists(
    admin_client: AdminClient,
    topic_name: str,
    num_partitions: int,
    replication_factor: int,
) -> None:
    """
    Idempotently create a Kafka topic.

    'Idempotent' means calling this multiple times has the same effect
    as calling it once — if the topic already exists, we do nothing.
    This is a critical pattern in distributed systems.

    Args:
        admin_client: Kafka admin client.
        topic_name: Name of the topic to create.
        num_partitions: Number of partitions for the topic.
        replication_factor: Number of replicas per partition.
    """
    existing = admin_client.list_topics(timeout=10).topics

    if topic_name in existing:
        logger.info("Topic '%s' already exists, skipping creation", topic_name)
        return

    new_topic = NewTopic(
        topic=topic_name,
        num_partitions=num_partitions,
        replication_factor=replication_factor,
    )

    futures = admin_client.create_topics([new_topic])

    for topic, future in futures.items():
        try:
            future.result()
            logger.info("Created Kafka topic: %s", topic)
        except Exception as e:
            # Topic might have been created by another process between our
            # check and our create — that is fine, we just log and continue
            logger.warning("Could not create topic %s: %s", topic, e)


def _generate_transaction(user_ids: list[str]) -> Transaction:
    """
    Generate a single realistic transaction.

    Fraud transactions are made distinguishable by:
    - Higher amounts (thieves max out cards)
    - More likely to be ONLINE or ELECTRONICS categories
    - Slight location randomization (not matching user's home location)

    Args:
        user_ids: Pool of user IDs to assign transactions to.

    Returns:
        A validated Transaction instance.
    """
    is_fraud = random.random() < FRAUD_RATE
    category = random.choice(list(MerchantCategory))

    # Fraud transactions skew toward high-value categories
    if is_fraud:
        category = random.choices(
            population=[
                MerchantCategory.ONLINE,
                MerchantCategory.ELECTRONICS,
                MerchantCategory.TRAVEL,
                MerchantCategory.ATM,
            ],
            weights=[0.35, 0.30, 0.20, 0.15],
        )[0]

    # Generate amount from a normal distribution for realism
    mean, std = AMOUNT_PROFILES[category]
    amount = abs(random.gauss(mean, std))

    # Fraud transactions tend to have unusually high amounts
    if is_fraud:
        amount *= random.uniform(1.5, 4.0)

    return Transaction(
        user_id=random.choice(user_ids),
        amount=amount,
        merchant_id=f"merchant_{random.randint(1000, 9999)}",
        merchant_category=category,
        timestamp=datetime.now(timezone.utc),
        latitude=random.uniform(*LAT_RANGE),
        longitude=random.uniform(*LON_RANGE),
        is_fraud=is_fraud,
    )


class TransactionSimulator:
    """
    Continuously generates and publishes banking transactions to Kafka.

    Design pattern: This class follows the Single Responsibility Principle —
    it only knows about generating and publishing transactions. It does not
    know about feature engineering, models, or the database.

    Usage:
        simulator = TransactionSimulator()
        simulator.run(transactions_per_second=10)
    """

    def __init__(self) -> None:
        """
        Initialize the simulator by loading config and connecting to Kafka.
        """
        self._config = get_kafka_config()
        self._topic = self._config["topics"]["raw_transactions"]
        self._producer = self._build_producer()
        self._admin = self._build_admin_client()

        # Create topic on startup if it doesn't exist
        _create_topic_if_not_exists(
            admin_client=self._admin,
            topic_name=self._topic,
            num_partitions=self._config["num_partitions"],
            replication_factor=self._config["replication_factor"],
        )

        # A realistic pool of 1000 users
        self._user_ids = [f"user_{i:04d}" for i in range(1, 1001)]

        logger.info(
            "TransactionSimulator initialized. Topic: %s", self._topic
        )

    def _build_producer(self) -> Producer:
        """
        Build and return a configured Kafka Producer.

        The producer config maps directly to librdkafka configuration.
        confluent-kafka is a thin Python wrapper around librdkafka (C library),
        which is why it is significantly faster than kafka-python.
        """
        producer_cfg = self._config.get("producer", {})

        return Producer(
            {
                "bootstrap.servers": self._config["bootstrap_servers"],
                "acks": producer_cfg.get("acks", "all"),
                "retries": producer_cfg.get("retries", 3),
                "retry.backoff.ms": producer_cfg.get("retry_backoff_ms", 300),
                # Compress messages — reduces network bandwidth significantly
                "compression.type": "snappy",
            }
        )

    def _build_admin_client(self) -> AdminClient:
        """Build and return a Kafka AdminClient for topic management."""
        return AdminClient(
            {"bootstrap.servers": self._config["bootstrap_servers"]}
        )

    def publish(self, transaction: Transaction) -> None:
        """
        Serialize and publish a single transaction to Kafka.

        The message key is the user_id. This ensures all transactions
        for the same user land in the same partition, which preserves
        ordering per user — critical for feature engineering later.

        Args:
            transaction: A validated Transaction instance.

        Raises:
            KafkaPublishError: If the message cannot be queued.
        """
        try:
            self._producer.produce(
                topic=self._topic,
                # Key ensures same-user ordering across partitions
                key=transaction.user_id.encode("utf-8"),
                # Value is the full transaction as JSON bytes
                value=transaction.model_dump_json().encode("utf-8"),
                # Called asynchronously when delivery is confirmed
                callback=_delivery_callback,
            )
            # Poll triggers delivery callbacks — call after each produce
            self._producer.poll(0)

        except BufferError as e:
            # Producer's internal queue is full — broker might be slow
            raise KafkaPublishError(
                f"Kafka producer queue is full. "
                f"Broker may be overloaded: {e}"
            ) from e

    def run(
        self,
        transactions_per_second: float = 5.0,
        max_transactions: Optional[int] = None,
    ) -> None:
        """
        Main loop — generate and publish transactions continuously.

        Args:
            transactions_per_second: Target publish rate.
            max_transactions: Stop after this many transactions.
                              None means run forever (production mode).
        """
        interval = 1.0 / transactions_per_second
        count = 0

        logger.info(
            "Starting simulator at %.1f TPS. Topic: %s",
            transactions_per_second,
            self._topic,
        )

        try:
            while True:
                transaction = _generate_transaction(self._user_ids)
                self.publish(transaction)

                count += 1
                if count % 100 == 0:
                    logger.info(
                        "Published %d transactions (%d fraud)",
                        count,
                        int(count * FRAUD_RATE),
                    )

                if max_transactions and count >= max_transactions:
                    logger.info("Reached max_transactions limit: %d", count)
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Simulator interrupted by user.")
        finally:
            # flush() blocks until all queued messages are delivered
            # This ensures no messages are lost on shutdown
            logger.info("Flushing remaining messages...")
            self._producer.flush(timeout=30)
            logger.info("Simulator stopped. Total published: %d", count)


if __name__ == "__main__":
    simulator = TransactionSimulator()
    simulator.run(transactions_per_second=2.0)
