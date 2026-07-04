"""
Prometheus metrics definitions for the fraud detection serving layer.

This module defines all metrics as module-level singletons.
They are registered with the Prometheus client once at import time
and updated throughout the application lifetime.

Why module-level singletons?
Prometheus metrics must be registered exactly once. If you create
a new Counter() inside a function that gets called multiple times,
you get a 'Duplicated timeseries' error. Module-level constants
are instantiated once when the module is first imported.

Metric types used here:
- Counter   : monotonically increasing number (total requests, total fraud)
- Histogram : distribution of values (latency, amount)
- Gauge     : value that can go up and down (model info)
"""

from prometheus_client import Counter, Gauge, Histogram, Info

from src.utils.config_loader import load_config

_cfg = load_config("monitoring_config.yaml")
_buckets = _cfg["metrics"]["latency_buckets"]

# ── Request Metrics ───────────────────────────────────────────────

PREDICTION_REQUESTS_TOTAL = Counter(
    name="fraud_prediction_requests_total",
    documentation="Total number of prediction requests received.",
    labelnames=["status"],  # status = "success" or "error"
)
"""
Counter incremented on every prediction request.

Labels let us split by outcome:
  fraud_prediction_requests_total{status="success"} 9821
  fraud_prediction_requests_total{status="error"}   3

In PromQL (Prometheus Query Language):
  rate(fraud_prediction_requests_total[5m])
  → requests per second over the last 5 minutes
"""

PREDICTION_LATENCY_SECONDS = Histogram(
    name="fraud_prediction_latency_seconds",
    documentation="Time taken to produce a fraud prediction in seconds.",
    buckets=_buckets,
)
"""
Histogram tracking inference latency.

In PromQL:
  histogram_quantile(0.99, fraud_prediction_latency_seconds_bucket)
  → 99th percentile latency (p99)

This is the standard SLO metric: "99% of predictions complete in Xms"
"""

# ── Fraud Metrics ─────────────────────────────────────────────────

FRAUD_PREDICTIONS_TOTAL = Counter(
    name="fraud_predictions_total",
    documentation="Total number of transactions flagged as fraudulent.",
)
"""
Counter incremented only when is_fraud=True.

In PromQL:
  rate(fraud_predictions_total[5m])
  /
  rate(fraud_prediction_requests_total[5m])
  → rolling fraud rate over 5 minutes

Alert fires when this exceeds fraud_rate_alert_threshold.
"""

PREDICTION_AMOUNT = Histogram(
    name="fraud_prediction_amount_usd",
    documentation="Distribution of transaction amounts scored.",
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 50000],
)
"""
Histogram of transaction amounts.

Lets us see if the amount distribution shifts over time —
a sign that the incoming data no longer matches training data.
This is a simple form of data drift detection.
"""

FRAUD_PROBABILITY = Histogram(
    name="fraud_prediction_probability",
    documentation="Distribution of raw fraud probability scores.",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
"""
Histogram of model output probabilities.

In a healthy model, most scores should be near 0 (not fraud)
with a small spike near 1 (clear fraud).

If this distribution shifts — e.g. many scores cluster around 0.5
— the model is uncertain and may be experiencing drift.
"""

# ── Model Info ────────────────────────────────────────────────────

MODEL_INFO = Info(
    name="fraud_model",
    documentation="Information about the currently loaded model.",
)
"""
Info metric stores static key-value metadata about the model.

In Grafana, this appears as a label on all model metrics,
letting you correlate performance changes with model versions.

Set once at startup:
  MODEL_INFO.info({
      "version": "latest version",
      "name": "fraud-detector",
  })
"""

MODEL_LOAD_STATUS = Gauge(
    name="fraud_model_loaded",
    documentation="1 if the model is loaded and ready, 0 otherwise.",
)
"""
Gauge that is 1 when healthy, 0 when not.

In Grafana, this powers a simple status indicator:
  Green = model loaded
  Red   = model not loaded

Alert fires immediately if this drops to 0 in production.
"""


def record_prediction(
    fraud_probability: float,
    is_fraud: bool,
    amount: float,
    latency_seconds: float,
) -> None:
    """
    Record all metrics for one completed prediction.

    Call this after every successful prediction in the API route.
    Centralizing metric updates here means routes stay clean —
    they call one function instead of updating five metrics manually.

    Args:
        fraud_probability: Raw model output score (0-1).
        is_fraud: Binary fraud decision.
        amount: Transaction amount in USD.
        latency_seconds: Total inference time in seconds.
    """
    PREDICTION_REQUESTS_TOTAL.labels(status="success").inc()
    PREDICTION_LATENCY_SECONDS.observe(latency_seconds)
    PREDICTION_AMOUNT.observe(amount)
    FRAUD_PROBABILITY.observe(fraud_probability)

    if is_fraud:
        FRAUD_PREDICTIONS_TOTAL.inc()


def record_prediction_error() -> None:
    """
    Record a failed prediction attempt.

    Separating error counts from success counts lets us
    compute error rate independently of fraud rate.
    """
    PREDICTION_REQUESTS_TOTAL.labels(status="error").inc()


def set_model_loaded(version: str, name: str) -> None:
    """
    Mark the model as loaded in Prometheus metrics.

    Call this once after successful model load at startup.

    Args:
        version: Model version string from MLflow.
        name: Registered model name from MLflow.
    """
    MODEL_LOAD_STATUS.set(1)
    MODEL_INFO.info({"version": version, "name": name})


def set_model_unloaded() -> None:
    """Mark the model as unloaded — triggers alert in Grafana."""
    MODEL_LOAD_STATUS.set(0)
