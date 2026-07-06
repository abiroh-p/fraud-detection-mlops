"""
Configuration loader for YAML config files.

Why YAML configs instead of hardcoding?
- Change Kafka broker address without touching Python code
- Different configs for dev/staging/production environments
- Non-engineers can adjust thresholds without code changes
- The 12-Factor App methodology (factor III) requires this
"""

from pathlib import Path
from typing import Any

import yaml

from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# The configs/ directory is always relative to the project root
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(filename: str) -> dict[str, Any]:
    """
    Load and parse a YAML configuration file from the configs/ directory.

    Args:
        filename: Name of the YAML file, e.g. "kafka_config.yaml"

    Returns:
        Dictionary of configuration values.

    Raises:
        ConfigurationError: If the file does not exist or cannot be parsed.

    Example:
        >>> config = load_config("kafka_config.yaml")
        >>> broker = config["kafka"]["bootstrap_servers"]
        'localhost:9092'
    """
    config_path = _CONFIG_DIR / filename

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}\n"
            f"Expected location: {_CONFIG_DIR}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logger.debug("Loaded config from %s", config_path)
        return config

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML file {filename}: {e}") from e


def get_kafka_config() -> dict[str, Any]:
    """
    Convenience function — load the Kafka configuration.

    Returns:
        The 'kafka' section of kafka_config.yaml

    Example:
        >>> kafka_cfg = get_kafka_config()
        >>> kafka_cfg["bootstrap_servers"]
        'localhost:9092'
    """
    config = load_config("kafka_config.yaml")
    return config["kafka"]
