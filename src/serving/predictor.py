"""
Model predictor for the fraud detection serving layer.

Responsible for:
- Loading the trained pipeline from MLflow Model Registry
- Running inference on incoming feature vectors
- Managing model version information for audit logging

Design pattern: Singleton-style loader — the model is loaded
once at startup and reused for every request. Loading a model
on every request would add 500ms+ latency per call.
"""

import pandas as pd

from src.serving.schemas import PredictionRequest, PredictionResponse
from src.training.model import FEATURE_COLUMNS
from src.utils.config_loader import load_config
from src.utils.exceptions import ModelLoadError, ModelPredictionError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cfg = load_config("model_config.yaml")
_mlflow_cfg = _cfg["mlflow"]

# Default decision threshold — can be overridden per request
DEFAULT_THRESHOLD = 0.5


class FraudPredictor:
    """
    Loads and serves the fraud detection model.

    The model is loaded from MLflow Model Registry using the
    'champion' alias — the model marked as the current best
    model for production use.

    Why 'champion' alias instead of a version number?
    If we hardcode version=3, we must redeploy the serving
    layer every time we train a better model. With the champion
    alias, we promote a new model in MLflow and the serving
    layer picks it up on next restart — no code change needed.

    Usage:
        predictor = FraudPredictor()
        predictor.load()
        response = predictor.predict(request)
    """

    def __init__(self) -> None:
        self._model = None
        self._model_version: str | None = None
        self._model_name = _mlflow_cfg["registered_model_name"]
        # Environment variable takes priority over config file
        # This allows Docker to override localhost with container name
        import os

        self._tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            _mlflow_cfg["tracking_uri"],
        )

    def load(self) -> None:
        """
        Load the latest model version from MLflow Model Registry.

        Uses MlflowClient directly to get the latest version metadata,
        then constructs the artifact URI manually to avoid DNS rebinding
        protection issues when running inside Docker.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self._tracking_uri)
        client = MlflowClient(tracking_uri=self._tracking_uri)

        try:
            # Get latest version metadata directly
            # Get all versions and pick the latest by version number
            all_versions = client.search_model_versions(f"name='{self._model_name}'")
            if not all_versions:
                raise ModelLoadError(
                    f"No versions found for model '{self._model_name}'"
                )

            latest_version = max(all_versions, key=lambda v: int(v.version))
            run_id = latest_version.run_id
            version_num = latest_version.version
            model_uri = f"runs:/{run_id}/model"

            logger.info(
                "Loading model '%s' version %s (run_id=%s)...",
                self._model_name,
                version_num,
                run_id,
            )

            self._model = mlflow.sklearn.load_model(model_uri)
            self._model_version = f"v{version_num}"

            logger.info(
                "Model loaded successfully: %s v%s",
                self._model_name,
                version_num,
            )

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model '{self._model_name}': {e}"
            ) from e

    @property
    def is_loaded(self) -> bool:
        """Return True if the model is loaded and ready."""
        return self._model is not None

    @property
    def model_version(self) -> str:
        """Return the loaded model version label."""
        return self._model_version or "unknown"

    def predict(
        self,
        request: PredictionRequest,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> PredictionResponse:
        """
        Run inference on a single prediction request.

        Steps:
        1. Convert Pydantic request to DataFrame
        2. Enforce canonical column order
        3. Run pipeline.predict_proba()
        4. Apply threshold to get binary decision
        5. Return structured response

        Args:
            request: Validated PredictionRequest from FastAPI.
            threshold: Fraud decision threshold (default 0.5).

        Returns:
            PredictionResponse with probability and decision.

        Raises:
            ModelPredictionError: If inference fails.
        """
        if not self.is_loaded:
            raise ModelPredictionError(
                "Model is not loaded. Call load() before predict()."
            )

        try:
            # Convert request to DataFrame with canonical column order
            # model_dump() returns a plain dict from the Pydantic model
            feature_dict = request.model_dump()
            df = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]

            # predict_proba returns [[prob_legit, prob_fraud]]
            # We take index [0][1] — first row, fraud probability
            fraud_probability = float(self._model.predict_proba(df)[0][1])

            is_fraud = fraud_probability >= threshold

            logger.debug(
                "Prediction: prob=%.4f fraud=%s threshold=%.2f",
                fraud_probability,
                is_fraud,
                threshold,
            )

            return PredictionResponse(
                fraud_probability=round(fraud_probability, 6),
                is_fraud=is_fraud,
                model_version=self.model_version,
                threshold=threshold,
            )

        except Exception as e:
            raise ModelPredictionError(f"Inference failed: {e}") from e
