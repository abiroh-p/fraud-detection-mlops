"""
Pydantic schemas for the FastAPI serving layer.

These are separate from src/data/schemas.py intentionally.

src/data/schemas.py  — defines the Transaction (internal pipeline format)
src/serving/schemas.py — defines the API contract (what callers send/receive)

Why separate?
The internal Transaction schema may evolve independently of the API
contract. Mixing them creates tight coupling between the data pipeline
and the public API — a common mistake in early ML systems.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Feature vector sent by the caller for fraud scoring.

    The caller is responsible for engineering these features
    before calling the API. In our pipeline, the feature
    engineering pipeline does this automatically.

    All fields match exactly the FEATURE_COLUMNS list in model_config.yaml.
    """

    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category_encoded: int = Field(..., ge=0, le=8)
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    is_night: int = Field(..., ge=0, le=1)
    txn_count_1h: int = Field(..., ge=0)
    txn_amount_sum_1h: float = Field(..., ge=0)
    txn_count_24h: int = Field(..., ge=0)
    txn_amount_sum_24h: float = Field(..., ge=0)
    amount_vs_user_mean: float = Field(...)
    user_mean_amount: float = Field(..., ge=0)
    user_txn_count_total: int = Field(..., ge=0)
    is_high_value_for_user: int = Field(..., ge=0, le=1)
    dist_from_last_txn_km: float = Field(..., ge=0)
    speed_from_last_txn_kmh: float = Field(..., ge=0)
    is_impossible_travel: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    """
    Fraud prediction returned by the API.

    We return both the probability and a binary decision.
    The caller can use the probability for risk scoring
    and the decision for immediate block/allow logic.
    """

    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability this transaction is fraudulent (0-1)",
    )
    is_fraud: bool = Field(
        ...,
        description="Binary fraud decision at the configured threshold",
    )
    model_version: str = Field(
        ...,
        description="MLflow model version used for this prediction",
    )
    threshold: float = Field(
        ...,
        description="Decision threshold used to produce is_fraud",
    )


class HealthResponse(BaseModel):
    """Response for the /health endpoint — used by Kubernetes."""

    status: str
    model_loaded: bool
    model_version: str | None
