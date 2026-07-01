"""
Pydantic schemas for transaction data.

Why Pydantic schemas?
- Automatic validation: wrong types raise clear errors immediately
- Auto-generated documentation in FastAPI
- Single source of truth for data shape across all services
- Used by Google, Stripe, and FastAPI itself

A Transaction represents one card payment attempt.
"""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class MerchantCategory(str, Enum):
    """
    Standardized merchant category codes (simplified from ISO 18245).

    Using an Enum instead of raw strings means:
    - Invalid categories are caught at validation time
    - IDE autocomplete works
    - Refactoring is safe
    """
    GROCERY = "grocery"
    ELECTRONICS = "electronics"
    RESTAURANT = "restaurant"
    TRAVEL = "travel"
    ONLINE = "online"
    GAS = "gas"
    PHARMACY = "pharmacy"
    ATM = "atm"
    OTHER = "other"


class Transaction(BaseModel):
    """
    Represents a single banking transaction event.

    This schema is used by:
    - TransactionSimulator (creates instances)
    - Kafka producer (serializes to JSON)
    - Feature pipeline (deserializes from JSON)
    - Data validation layer (validates before training)
    """

    transaction_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this transaction",
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Anonymized user identifier",
    )
    amount: float = Field(
        ...,
        gt=0.0,
        le=50_000.0,
        description="Transaction amount in USD. Must be positive.",
    )
    merchant_id: str = Field(
        ...,
        description="Anonymized merchant identifier",
    )
    merchant_category: MerchantCategory = Field(
        ...,
        description="Category of the merchant",
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp when the transaction occurred",
    )
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Transaction location latitude",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Transaction location longitude",
    )
    is_fraud: bool = Field(
        default=False,
        description="Ground truth fraud label. Unknown at inference time.",
    )

    @field_validator("amount")
    @classmethod
    def round_amount(cls, v: float) -> float:
        """Round amount to 2 decimal places — cents precision."""
        return round(v, 2)

    model_config = {
        # Allow serialization of UUID and datetime to JSON
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    }


class TransactionBatch(BaseModel):
    """
    A batch of transactions — used for bulk processing in training pipeline.
    """
    transactions: list[Transaction]
    batch_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def size(self) -> int:
        """Number of transactions in the batch."""
        return len(self.transactions)

    @property
    def fraud_count(self) -> int:
        """Number of fraudulent transactions in the batch."""
        return sum(1 for t in self.transactions if t.is_fraud)
