"""
Unit tests for the feature engineering module.

These tests verify:
- Feature output has correct keys
- Velocity features accumulate correctly
- Location features compute correctly
- Impossible travel detection works
- First transaction has zero history features

No Kafka, no MLflow, no Docker required.
All tests run against pure Python functions.
"""

from datetime import UTC, datetime, timedelta

from src.data.schemas import MerchantCategory, Transaction
from src.features.feature_engineering import _haversine_km
from src.training.model import FEATURE_COLUMNS


class TestHaversine:
    """Tests for the geographic distance calculation."""

    def test_same_location_is_zero(self):
        """Distance from a point to itself must be zero."""
        dist = _haversine_km(40.71, -74.00, 40.71, -74.00)
        assert dist == 0.0

    def test_known_distance_nyc_to_london(self):
        """
        NYC to London is approximately 5,570 km.
        We allow 1% tolerance for floating point differences.
        """
        dist = _haversine_km(40.71, -74.00, 51.50, -0.12)
        assert abs(dist - 5570) < 100

    def test_distance_is_symmetric(self):
        """Distance A→B must equal distance B→A."""
        d1 = _haversine_km(40.71, -74.00, 34.05, -118.24)
        d2 = _haversine_km(34.05, -118.24, 40.71, -74.00)
        assert abs(d1 - d2) < 0.001


class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""

    def test_output_contains_all_model_features(
        self, feature_engineer, sample_transaction
    ):
        """
        Every key in FEATURE_COLUMNS must be present in the output.
        This test acts as a contract — if you add a feature to the
        config but forget to add it to feature_engineering.py, this fails.
        """
        features = feature_engineer.transform(sample_transaction)
        for col in FEATURE_COLUMNS:
            assert col in features, f"Missing feature: {col}"

    def test_first_transaction_has_zero_history(
        self, feature_engineer, sample_transaction
    ):
        """
        A user's first transaction has no history.
        All velocity and location features must be zero.
        """
        features = feature_engineer.transform(sample_transaction)

        assert features["txn_count_1h"] == 0
        assert features["txn_count_24h"] == 0
        assert features["txn_amount_sum_1h"] == 0.0
        assert features["txn_amount_sum_24h"] == 0.0
        assert features["dist_from_last_txn_km"] == 0.0
        assert features["speed_from_last_txn_kmh"] == 0.0
        assert features["is_impossible_travel"] == 0
        assert features["user_txn_count_total"] == 0

    def test_velocity_accumulates_across_transactions(self, feature_engineer):
        """
        After N transactions, velocity counts must reflect history.
        """
        base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)
        user_id = "user_test_velocity"

        # Send 3 transactions 10 minutes apart
        for i in range(3):
            txn = Transaction(
                user_id=user_id,
                amount=100.0,
                merchant_id="merchant_1",
                merchant_category=MerchantCategory.GROCERY,
                timestamp=base_time + timedelta(minutes=10 * i),
                latitude=40.71,
                longitude=-74.00,
                is_fraud=False,
            )
            features = feature_engineer.transform(txn)

        # The 3rd transaction should see 2 previous ones in 1h window
        assert features["txn_count_1h"] == 2
        assert features["txn_amount_sum_1h"] == 200.0
        assert features["txn_count_24h"] == 2

    def test_impossible_travel_detected(self, feature_engineer):
        """
        Two transactions in different continents within minutes
        must trigger the impossible travel flag.
        """
        user_id = "user_test_travel"
        base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)

        # First transaction in New York
        txn1 = Transaction(
            user_id=user_id,
            amount=50.0,
            merchant_id="merchant_1",
            merchant_category=MerchantCategory.GROCERY,
            timestamp=base_time,
            latitude=40.71,
            longitude=-74.00,
            is_fraud=False,
        )
        feature_engineer.transform(txn1)

        # Second transaction in London 5 minutes later — impossible
        txn2 = Transaction(
            user_id=user_id,
            amount=50.0,
            merchant_id="merchant_2",
            merchant_category=MerchantCategory.GROCERY,
            timestamp=base_time + timedelta(minutes=5),
            latitude=51.50,
            longitude=-0.12,
            is_fraud=True,
        )
        features = feature_engineer.transform(txn2)

        assert features["is_impossible_travel"] == 1
        assert features["speed_from_last_txn_kmh"] > 900

    def test_night_transaction_flag(self, feature_engineer):
        """Transactions between 10pm and 6am must be flagged as night."""
        night_txn = Transaction(
            user_id="user_night",
            amount=100.0,
            merchant_id="merchant_1",
            merchant_category=MerchantCategory.ATM,
            timestamp=datetime(2024, 1, 15, 3, 0, 0, tzinfo=UTC),
            latitude=40.71,
            longitude=-74.00,
            is_fraud=False,
        )
        features = feature_engineer.transform(night_txn)
        assert features["is_night"] == 1

    def test_weekend_flag(self, feature_engineer):
        """Transactions on Saturday/Sunday must be flagged as weekend."""
        # January 20, 2024 is a Saturday
        weekend_txn = Transaction(
            user_id="user_weekend",
            amount=100.0,
            merchant_id="merchant_1",
            merchant_category=MerchantCategory.GROCERY,
            timestamp=datetime(2024, 1, 20, 14, 0, 0, tzinfo=UTC),
            latitude=40.71,
            longitude=-74.00,
            is_fraud=False,
        )
        features = feature_engineer.transform(weekend_txn)
        assert features["is_weekend"] == 1

    def test_high_value_flag(self, feature_engineer):
        """
        A transaction 3x above user mean must be flagged.
        Build history first, then send high-value transaction.
        """
        user_id = "user_highvalue"
        base_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC)

        # Build history: 5 transactions at $100 each
        for i in range(5):
            txn = Transaction(
                user_id=user_id,
                amount=100.0,
                merchant_id="merchant_1",
                merchant_category=MerchantCategory.GROCERY,
                timestamp=base_time + timedelta(days=i),
                latitude=40.71,
                longitude=-74.00,
                is_fraud=False,
            )
            feature_engineer.transform(txn)

        # Now send a $500 transaction — 5x the mean of $100
        high_value_txn = Transaction(
            user_id=user_id,
            amount=500.0,
            merchant_id="merchant_1",
            merchant_category=MerchantCategory.ELECTRONICS,
            timestamp=base_time + timedelta(days=6),
            latitude=40.71,
            longitude=-74.00,
            is_fraud=True,
        )
        features = feature_engineer.transform(high_value_txn)
        assert features["is_high_value_for_user"] == 1
