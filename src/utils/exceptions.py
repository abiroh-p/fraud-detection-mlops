"""
Custom exception hierarchy for the fraud detection pipeline.

Why custom exceptions?
- Self-documenting error messages in stack traces
- Allows callers to catch specific failure types
- Enables different retry strategies per exception type
- Standard practice at Amazon, Google, and Netflix

Hierarchy:
    FraudDetectionError (base)
    ├── KafkaError
    │   ├── KafkaPublishError
    │   └── KafkaConsumeError
    ├── ModelError
    │   ├── ModelLoadError
    │   └── ModelPredictionError
    ├── DataValidationError
    └── ConfigurationError
"""


class FraudDetectionError(Exception):
    """
    Base exception for all pipeline errors.

    Every custom exception inherits from this so callers can catch
    all pipeline errors with a single except clause if needed.
    """

    pass


# ── Kafka Exceptions ──────────────────────────────────────────────


class KafkaError(FraudDetectionError):
    """Base class for all Kafka-related errors."""

    pass


class KafkaPublishError(KafkaError):
    """
    Raised when a message cannot be published to a Kafka topic.

    Common causes:
    - Broker is unreachable
    - Topic does not exist
    - Message is too large (default max is 1MB)
    """

    pass


class KafkaConsumeError(KafkaError):
    """
    Raised when a message cannot be consumed from a Kafka topic.

    Common causes:
    - Deserialization failure
    - Consumer group rebalancing timeout
    """

    pass


# ── Model Exceptions ──────────────────────────────────────────────


class ModelError(FraudDetectionError):
    """Base class for all model-related errors."""

    pass


class ModelLoadError(ModelError):
    """
    Raised when a model cannot be loaded from the registry.

    Common causes:
    - MLflow server is unreachable
    - Model version does not exist
    - Artifact is corrupted
    """

    pass


class ModelPredictionError(ModelError):
    """
    Raised when inference fails on a feature vector.

    Common causes:
    - Feature vector has wrong shape
    - Missing expected feature columns
    - NaN values in input
    """

    pass


# ── Data Exceptions ───────────────────────────────────────────────


class DataValidationError(FraudDetectionError):
    """
    Raised when incoming data fails schema or quality checks.

    Common causes:
    - Missing required fields
    - Values outside expected range
    - Wrong data types
    """

    pass


# ── Config Exceptions ─────────────────────────────────────────────


class ConfigurationError(FraudDetectionError):
    """
    Raised when a required configuration value is missing or invalid.

    Common causes:
    - YAML file not found
    - Required key missing from config
    - Environment variable not set
    """

    pass
