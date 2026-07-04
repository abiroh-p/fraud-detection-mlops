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
import mlflow.sklearn

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
        self._tracking_uri = _mlflow_cfg["tracking_uri"]

    def load(self) -> None:
        """
        Load the latest model version from MLflow Model Registry.

        We try loading the 'champion' alias first — this is the
        promoted production model. If no champion exists yet
        (e.g. first deployment), we fall back to the latest version.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        import mlflow
        mlflow.set_tracking_uri(self._tracking_uri)

        # Try champion alias first, fall back to latest version
        for alias_or_version in ["champion", None]:
            try:
                if alias_or_version == "champion":
                    model_uri = (
                        f"models:/{self._model_name}@champion"
                    )
                    label = "champion alias"
                else:
                    model_uri = (
                        f"models:/{self._model_name}/latest"
                    )
                    label = "latest version"

                logger.info(
                    "Loading model '%s' (%s) from MLflow...",
                    self._model_name,
                    label,
                )

                self._model = mlflow.sklearn.load_model(
                    model_uri,
                    # Trust XGBoost types when loading
                    dst_path=None,
                )
                self._model_version = label
                logger.info(
                    "Model loaded successfully: %s (%s)",
                    self._model_name,
                    label,
                )
                return

            except Exception as e:
                if alias_or_version == "champion":
                    logger.warning(
                        "Champion alias not found, trying latest version. "
                        "Reason: %s", e
                    )
                    continue
                else:
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
            fraud_probability = float(
                self._model.predict_proba(df)[0][1]
            )

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
            raise ModelPredictionError(
                f"Inference failed: {e}"
            ) from e
