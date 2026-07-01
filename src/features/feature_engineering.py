"""
Feature engineering for fraud detection.

Transforms raw Transaction objects into numerical feature vectors
that a machine learning model can consume.

Design decision: This module is intentionally pure Python with no
Kafka dependency. It takes a Transaction and returns a dict.
This makes it independently testable without any infrastructure.
"""

import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from src.data.schemas import MerchantCategory, Transaction
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────

_cfg = load_config("pipeline_config.yaml")["feature_engineering"]

SHORT_WINDOW_MINUTES: int = _cfg["velocity_windows"]["short_minutes"]
LONG_WINDOW_HOURS: int = _cfg["velocity_windows"]["long_hours"]
MAX_SPEED_KMH: float = _cfg["max_realistic_speed_kmh"]
MAX_HISTORY: int = _cfg["max_history_per_user"]
HIGH_VALUE_MULTIPLIER: float = _cfg["high_value_threshold_multiplier"]

# ── Merchant category → integer mapping ──────────────────────────
# ML models need numbers, not strings.
# We use a fixed mapping so the encoding is stable across runs.
CATEGORY_ENCODING: dict[MerchantCategory, int] = {
    MerchantCategory.GROCERY: 0,
    MerchantCategory.ELECTRONICS: 1,
    MerchantCategory.RESTAURANT: 2,
    MerchantCategory.TRAVEL: 3,
    MerchantCategory.ONLINE: 4,
    MerchantCategory.GAS: 5,
    MerchantCategory.PHARMACY: 6,
    MerchantCategory.ATM: 7,
    MerchantCategory.OTHER: 8,
}


def _haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate the great-circle distance between two GPS coordinates.

    The Haversine formula accounts for the curvature of the Earth.
    Euclidean distance would give wrong results for geographic coordinates.

    Args:
        lat1, lon1: First point in decimal degrees.
        lat2, lon2: Second point in decimal degrees.

    Returns:
        Distance in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers

    # Convert degrees to radians — math.sin/cos expect radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class FeatureEngineer:
    """
    Stateful feature engineer that maintains per-user transaction history.

    Why stateful?
    Velocity features (how many transactions in the last hour?) require
    memory of past events. This class holds that memory in-memory.

    Production note:
    In a multi-instance deployment, this state would live in Redis
    so all instances share the same view of each user's history.
    For now, a single instance with in-memory state is correct.

    Usage:
        engineer = FeatureEngineer()
        features = engineer.transform(transaction)
    """

    def __init__(self) -> None:
        """
        Initialize per-user history store.

        defaultdict(deque) means: if a user_id is seen for the first time,
        automatically create an empty deque for them. No KeyError ever.

        deque(maxlen=MAX_HISTORY) means: when the deque is full, the oldest
        item is automatically dropped. This is our memory guard.
        """
        self._history: dict[str, deque[Transaction]] = defaultdict(
            lambda: deque(maxlen=MAX_HISTORY)
        )

    def transform(self, transaction: Transaction) -> dict[str, Any]:
        """
        Transform a raw Transaction into a feature vector.

        This is the main public method. It:
        1. Computes all features using the current history
        2. Appends the transaction to history for future use
        3. Returns the feature dict

        The order matters — we compute features BEFORE updating history
        so the current transaction is not included in its own velocity.

        Args:
            transaction: A validated Transaction object.

        Returns:
            Dictionary of feature name → numeric value.
            Also includes metadata fields for logging/debugging.
        """
        history = self._history[transaction.user_id]

        features = {
            # ── Metadata (not used by model, used for logging) ────
            "transaction_id": str(transaction.transaction_id),
            "user_id": transaction.user_id,
            "timestamp": transaction.timestamp.isoformat(),

            # ── Raw transaction features ──────────────────────────
            "amount": transaction.amount,
            "merchant_category_encoded": CATEGORY_ENCODING.get(
                transaction.merchant_category, 8
            ),
            "hour_of_day": transaction.timestamp.hour,
            "day_of_week": transaction.timestamp.weekday(),  # 0=Monday, 6=Sunday
            "is_weekend": int(transaction.timestamp.weekday() >= 5),
            "is_night": int(
                transaction.timestamp.hour < 6
                or transaction.timestamp.hour >= 22
            ),

            # ── Velocity features ─────────────────────────────────
            **self._velocity_features(transaction, history),

            # ── User behaviour features ───────────────────────────
            **self._user_behaviour_features(transaction, history),

            # ── Location features ─────────────────────────────────
            **self._location_features(transaction, history),

            # ── Label (only available in training data) ───────────
            "is_fraud": int(transaction.is_fraud),
        }

        # Now update history with the current transaction
        self._history[transaction.user_id].append(transaction)

        return features

    def _velocity_features(
        self,
        transaction: Transaction,
        history: deque[Transaction],
    ) -> dict[str, Any]:
        """
        Count and sum transactions within rolling time windows.

        These are among the strongest fraud signals:
        - A fraudster who steals a card number often makes many small
          purchases quickly before the victim notices.
        - Velocity spikes are a near-universal fraud indicator.

        Args:
            transaction: The current transaction.
            history: Past transactions for this user.

        Returns:
            Dict with count and sum features for short and long windows.
        """
        now = transaction.timestamp
        short_cutoff = now.timestamp() - (SHORT_WINDOW_MINUTES * 60)
        long_cutoff = now.timestamp() - (LONG_WINDOW_HOURS * 3600)

        short_count = 0
        short_sum = 0.0
        long_count = 0
        long_sum = 0.0

        for past_txn in history:
            past_ts = past_txn.timestamp.timestamp()

            if past_ts >= long_cutoff:
                long_count += 1
                long_sum += past_txn.amount

            if past_ts >= short_cutoff:
                short_count += 1
                short_sum += past_txn.amount

        return {
            "txn_count_1h": short_count,
            "txn_amount_sum_1h": short_sum,
            "txn_count_24h": long_count,
            "txn_amount_sum_24h": long_sum,
        }

    def _user_behaviour_features(
        self,
        transaction: Transaction,
        history: deque[Transaction],
    ) -> dict[str, Any]:
        """
        Compare this transaction against the user's historical baseline.

        A $5,000 purchase is normal for one user and suspicious for another.
        Personalised deviation features capture this relative context.

        Args:
            transaction: The current transaction.
            history: Past transactions for this user.

        Returns:
            Dict with deviation and average features.
        """
        if not history:
            # First-ever transaction for this user — no baseline yet
            return {
                "amount_vs_user_mean": 0.0,
                "user_mean_amount": 0.0,
                "user_txn_count_total": 0,
                "is_high_value_for_user": 0,
            }

        amounts = [t.amount for t in history]
        user_mean = sum(amounts) / len(amounts)

        # How many standard deviations above the mean?
        # Called a z-score in statistics
        amount_deviation = (
            (transaction.amount - user_mean) / user_mean
            if user_mean > 0
            else 0.0
        )

        return {
            "amount_vs_user_mean": round(amount_deviation, 4),
            "user_mean_amount": round(user_mean, 2),
            "user_txn_count_total": len(history),
            "is_high_value_for_user": int(
                transaction.amount > user_mean * HIGH_VALUE_MULTIPLIER
            ),
        }

    def _location_features(
        self,
        transaction: Transaction,
        history: deque[Transaction],
    ) -> dict[str, Any]:
        """
        Detect impossible travel and location anomalies.

        If a user swipes their card in Los Angeles and then in London
        20 minutes later, that is physically impossible — strong fraud signal.

        We compute:
        - Distance from last transaction (km)
        - Speed required to travel that distance (km/h)
        - Whether that speed exceeds our threshold (impossible travel flag)

        Args:
            transaction: The current transaction.
            history: Past transactions for this user.

        Returns:
            Dict with distance and speed features.
        """
        if not history:
            return {
                "dist_from_last_txn_km": 0.0,
                "speed_from_last_txn_kmh": 0.0,
                "is_impossible_travel": 0,
            }

        last = history[-1]  # Most recent past transaction

        distance_km = _haversine_km(
            last.latitude, last.longitude,
            transaction.latitude, transaction.longitude,
        )

        # Time elapsed in hours between transactions
        time_diff_hours = (
            transaction.timestamp.timestamp() - last.timestamp.timestamp()
        ) / 3600.0

        # Avoid division by zero for near-simultaneous transactions
        if time_diff_hours > 0:
            speed_kmh = distance_km / time_diff_hours
        else:
            speed_kmh = 0.0

        return {
            "dist_from_last_txn_km": round(distance_km, 2),
            "speed_from_last_txn_kmh": round(speed_kmh, 2),
            "is_impossible_travel": int(speed_kmh > MAX_SPEED_KMH),
        }
