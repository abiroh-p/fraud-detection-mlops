"""
Shared pytest fixtures for the fraud detection test suite.

Fixtures defined here are automatically available to all test files
without any import. pytest discovers conftest.py automatically.

Why fixtures?
- Avoid repeating setup code in every test
- Centralize test data creation
- Make tests readable — the fixture name describes what it provides
"""

from datetime import UTC, datetime

import pytest

from src.data.schemas import MerchantCategory, Transaction
from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_transaction() -> Transaction:
    """
    A single legitimate transaction for testing.
    Used as a baseline — all fields valid, no fraud signals.
    """
    return Transaction(
        user_id="user_0001",
        amount=65.0,
        merchant_id="merchant_1234",
        merchant_category=MerchantCategory.GROCERY,
        timestamp=datetime(2024, 1, 15, 14, 30, 0, tzinfo=UTC),
        latitude=40.71,
        longitude=-74.00,
        is_fraud=False,
    )


@pytest.fixture
def fraud_transaction() -> Transaction:
    """
    A high-risk fraudulent transaction for testing.
    High amount, night time, online category.
    """
    return Transaction(
        user_id="user_0002",
        amount=4500.0,
        merchant_id="merchant_9999",
        merchant_category=MerchantCategory.ELECTRONICS,
        timestamp=datetime(2024, 1, 15, 3, 15, 0, tzinfo=UTC),
        latitude=51.50,
        longitude=-0.12,
        is_fraud=True,
    )


@pytest.fixture
def feature_engineer() -> FeatureEngineer:
    """
    A fresh FeatureEngineer instance with empty history.
    Each test gets its own instance — no shared state between tests.
    """
    return FeatureEngineer()


@pytest.fixture
def sample_feature_dict(
    feature_engineer: FeatureEngineer,
    sample_transaction: Transaction,
) -> dict:
    """
    A complete feature dictionary ready for model input.
    Produced by transforming the sample_transaction.
    """
    return feature_engineer.transform(sample_transaction)
