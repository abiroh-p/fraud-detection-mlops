"""
Unit tests for the FraudPredictor serving layer.

These tests use mocking to avoid needing a real MLflow server.
Mocking replaces a real dependency with a fake one that behaves
predictably — essential for fast, reliable unit tests.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.serving.predictor import FraudPredictor
from src.serving.schemas import PredictionRequest
from src.utils.exceptions import ModelPredictionError


@pytest.fixture
def mock_model():
    """
    A fake sklearn pipeline that returns predictable probabilities.
    predict_proba returns [[0.1, 0.9]] meaning 90% fraud probability.
    """
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.1, 0.9]])
    model.predict.return_value = np.array([1])
    return model


@pytest.fixture
def loaded_predictor(mock_model):
    """
    A FraudPredictor with the model already loaded (mocked).
    Skips MLflow connection entirely.
    """
    predictor = FraudPredictor()
    predictor._model = mock_model
    predictor._model_version = "v1-test"
    return predictor


@pytest.fixture
def sample_request():
    """A valid PredictionRequest with high-risk features."""
    return PredictionRequest(
        amount=4500.0,
        merchant_category_encoded=1,
        hour_of_day=3,
        day_of_week=6,
        is_weekend=1,
        is_night=1,
        txn_count_1h=5,
        txn_amount_sum_1h=12000.0,
        txn_count_24h=12,
        txn_amount_sum_24h=25000.0,
        amount_vs_user_mean=4.2,
        user_mean_amount=80.0,
        user_txn_count_total=3,
        is_high_value_for_user=1,
        dist_from_last_txn_km=8500.0,
        speed_from_last_txn_kmh=95000.0,
        is_impossible_travel=1,
    )


class TestFraudPredictor:

    def test_predict_returns_response(self, loaded_predictor, sample_request):
        """Prediction returns a properly structured PredictionResponse."""
        response = loaded_predictor.predict(sample_request)

        assert response.fraud_probability == pytest.approx(0.9, abs=0.01)
        assert response.is_fraud is True
        assert response.model_version == "v1-test"
        assert response.threshold == 0.5

    def test_predict_not_fraud_below_threshold(
        self, loaded_predictor, mock_model, sample_request
    ):
        """Low probability score must result in is_fraud=False."""
        mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])

        response = loaded_predictor.predict(sample_request)

        assert response.fraud_probability == pytest.approx(0.05, abs=0.01)
        assert response.is_fraud is False

    def test_predict_raises_when_model_not_loaded(self, sample_request):
        """Predict must raise ModelPredictionError if model not loaded."""
        predictor = FraudPredictor()
        # Model is None by default — not loaded

        with pytest.raises(ModelPredictionError):
            predictor.predict(sample_request)

    def test_is_loaded_false_before_load(self):
        """is_loaded must be False before load() is called."""
        predictor = FraudPredictor()
        assert predictor.is_loaded is False

    def test_is_loaded_true_after_mock_load(self, loaded_predictor):
        """is_loaded must be True when model is set."""
        assert loaded_predictor.is_loaded is True

    def test_custom_threshold(self, loaded_predictor, mock_model, sample_request):
        """
        With a low threshold (0.1), even a 0.9 score is flagged.
        With a high threshold (0.95), a 0.9 score is NOT flagged.
        """
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

        # High threshold — 0.9 does not exceed 0.95
        response = loaded_predictor.predict(sample_request, threshold=0.95)
        assert response.is_fraud is False

        # Low threshold — 0.9 exceeds 0.1
        response = loaded_predictor.predict(sample_request, threshold=0.1)
        assert response.is_fraud is True
