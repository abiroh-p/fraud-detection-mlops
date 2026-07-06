"""
Kafka consumer for the feature engineering pipeline.

Reads raw Transaction JSON messages from the 'raw-transactions' topic,
deserializes them, and yields validated Transaction objects.

Design decision: This class only handles Kafka concerns — connecting,
polling, deserializing, and committing offsets. It knows nothing about
feature engineering. That separation makes both classes independently
testable.
"""

import json
from collections.abc import Iterator

from confluent_kafka import Consumer, KafkaException, Message

from src.data.schemas import Transaction
from src.utils.config_loader import get_kafka_config
from src.utils.exceptions import KafkaConsumeError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransactionConsumer:
    """
    Kafka consumer that yields deserialized Transaction objects.

    Why manual offset commits?
    With auto-commit, Kafka marks a message as processed the moment
    it is received — even if your code crashes before actually handling it.
    With manual commit, we only mark a message as processed AFTER our
    feature engineering succeeds. This gives us at-least-once delivery:
    no message is ever silently lost.

    Usage:
        consumer = TransactionConsumer()
        for transaction in consumer.consume():
            features = engineer.transform(transaction)
    """

    def __init__(self, group_id: str | None = None) -> None:
        """
        Initialize and subscribe to the raw-transactions topic.

        Args:
            group_id: Consumer group ID. If None, uses value from config.
                      Overridable for testing — each test gets its own group
                      so they don't interfere with each other's offsets.
        """
        self._config = get_kafka_config()
        self._topic = self._config["topics"]["raw_transactions"]

        consumer_cfg = self._config.get("consumer", {})
        resolved_group_id = group_id or consumer_cfg.get("group_id", "feature-pipeline")

        self._consumer = Consumer(
            {
                "bootstrap.servers": self._config["bootstrap_servers"],
                "group.id": resolved_group_id,
                # earliest = start from beginning if no offset saved yet
                # latest  = only read new messages (use in production
                #           when you don't want to reprocess old data)
                "auto.offset.reset": consumer_cfg.get("auto_offset_reset", "earliest"),
                # We commit manually — never automatically
                "enable.auto.commit": False,
            }
        )

        self._consumer.subscribe([self._topic])
        logger.info(
            "Consumer subscribed to '%s' with group '%s'",
            self._topic,
            resolved_group_id,
        )

    def consume(
        self,
        poll_timeout_seconds: float = 1.0,
        max_messages: int | None = None,
    ) -> Iterator[Transaction]:
        """
        Poll Kafka and yield one Transaction at a time.

        This is a generator — it yields transactions one by one
        as they arrive. The caller processes each one, then the
        loop continues polling.

        Why a generator instead of a callback?
        Generators let the caller control the loop. If the caller
        wants to stop after 100 messages (e.g. in tests), it just
        breaks out of the for loop. A callback-based design makes
        that much harder.

        Args:
            poll_timeout_seconds: How long to wait for a message
                before returning None and trying again. 1 second is
                standard — keeps the loop responsive to Ctrl+C.
            max_messages: Stop after this many messages. None = forever.

        Yields:
            Validated Transaction objects.

        Raises:
            KafkaConsumeError: On unrecoverable Kafka errors.
        """
        count = 0

        try:
            while True:
                msg: Message | None = self._consumer.poll(timeout=poll_timeout_seconds)

                if msg is None:
                    # No message arrived within the timeout — normal,
                    # just poll again
                    continue

                if msg.error():
                    raise KafkaConsumeError(f"Kafka consumer error: {msg.error()}")

                transaction = self._deserialize(msg)

                if transaction is None:
                    # Deserialization failed — skip this message
                    # We still commit so we don't get stuck retrying
                    # a permanently malformed message
                    self._commit(msg)
                    continue

                yield transaction

                # Commit AFTER yielding — meaning AFTER the caller has
                # finished processing this transaction
                self._commit(msg)

                count += 1
                if max_messages and count >= max_messages:
                    logger.info("Reached max_messages limit: %d", count)
                    break

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user.")
        finally:
            self.close()

    def _deserialize(self, msg: Message) -> Transaction | None:
        """
        Deserialize a raw Kafka message into a Transaction object.

        Two failure modes handled separately:
        - JSON parse error: message bytes are not valid JSON
        - Pydantic validation error: JSON is valid but fields are wrong

        Both are logged and return None so the pipeline skips the
        bad message rather than crashing entirely.

        Args:
            msg: Raw Kafka message.

        Returns:
            Validated Transaction, or None if deserialization fails.
        """
        try:
            raw = json.loads(msg.value().decode("utf-8"))
            return Transaction(**raw)

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON from topic '%s' offset %d: %s",
                msg.topic(),
                msg.offset(),
                e,
            )
            return None

        except Exception as e:
            logger.error(
                "Failed to validate Transaction at offset %d: %s",
                msg.offset(),
                e,
            )
            return None

    def _commit(self, msg: Message) -> None:
        """
        Commit the offset for a processed message.

        Committing offset N means: 'I have successfully processed
        all messages up to and including N. If I restart, begin from N+1.'

        Args:
            msg: The message whose offset we are committing.
        """
        try:
            self._consumer.commit(message=msg, asynchronous=False)
        except KafkaException as e:
            # Log but don't raise — a commit failure is not fatal.
            # The message will be reprocessed on restart (at-least-once).
            logger.warning("Offset commit failed: %s", e)

    def close(self) -> None:
        """
        Cleanly close the consumer connection.

        Always call this on shutdown. It:
        - Commits any pending offsets
        - Releases partition assignments back to the group
        - Allows other consumers to take over immediately
        """
        logger.info("Closing Kafka consumer.")
        self._consumer.close()
