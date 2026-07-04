"""
FastAPI application for the fraud detection serving layer.

Exposes three endpoints:
- POST /predict  — score a transaction for fraud
- GET  /health   — liveness check for Kubernetes
- GET  /ready    — readiness check for Kubernetes

Why separate /health and /ready?
Kubernetes uses two different probes:

Liveness probe  → /health
  "Is the process alive?"
  If this fails, Kubernetes restarts the pod.
  Should always return 200 as long as the process is running.

Readiness probe → /ready
  "Is the pod ready to serve traffic?"
  If this fails, Kubernetes removes the pod from the load balancer
  but does NOT restart it. Used during model loading — the pod is
  alive but not ready until the model finishes downloading.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.serving.middleware import LoggingMiddleware
from src.serving.predictor import FraudPredictor
from src.serving.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.utils.exceptions import ModelLoadError, ModelPredictionError
from src.utils.logger import get_logger
from src.monitoring.metrics_collector import (
    record_prediction,
    record_prediction_error,
    set_model_loaded,
    set_model_unloaded,
)
from prometheus_client import make_asgi_app

logger = get_logger(__name__)

# ── Predictor singleton ───────────────────────────────────────────
# One instance shared across all requests.
# Instantiated here at module level so it is accessible to routes.
predictor = FraudPredictor()


# ── Lifespan — startup and shutdown logic ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.

    This is the modern replacement for @app.on_event("startup").
    FastAPI deprecated the event-based approach in favour of lifespan
    because it handles both startup and shutdown in one place and
    works correctly with pytest's async test client.
    """
    # ── Startup ───────────────────────────────────────────────────
    logger.info("Starting Fraud Detection API...")
    try:
        predictor.load()
        set_model_loaded(
            version=predictor.model_version,
            name="fraud-detector",
        )
        logger.info("API ready. Model loaded: %s", predictor.model_version)
    except ModelLoadError as e:
        set_model_unloaded()
        logger.error("Failed to load model on startup: %s", e)

    yield  # API is now serving requests

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("Shutting down Fraud Detection API.")


# ── App instance ──────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring for banking transactions.",
    version="1.0.0",
    lifespan=lifespan,
)

# Register middleware — applied to every request automatically
app.add_middleware(LoggingMiddleware)


# ── Exception handlers ────────────────────────────────────────────
@app.exception_handler(ModelPredictionError)
async def prediction_error_handler(
    request: Request, exc: ModelPredictionError
) -> JSONResponse:
    """
    Convert ModelPredictionError into a clean 500 response.

    Without this, FastAPI would return a raw Python traceback
    to the caller — a security risk and a poor API experience.
    """
    logger.error("Prediction error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# ── Routes ────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a transaction for fraud",
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Accept a feature vector and return a fraud prediction.

    FastAPI automatically:
    - Parses the JSON body into a PredictionRequest
    - Validates all fields (types, ranges) via Pydantic
    - Returns 422 Unprocessable Entity if validation fails
    - Serializes the PredictionResponse back to JSON

    This means our route handler contains zero boilerplate —
    only business logic.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Try again in a moment.",
        )

    import time
    start = time.perf_counter()
    try:
        response = predictor.predict(request)
        latency = time.perf_counter() - start
        record_prediction(
            fraud_probability=response.fraud_probability,
            is_fraud=response.is_fraud,
            amount=request.amount,
            latency_seconds=latency,
        )
        return response
    except ModelPredictionError:
        record_prediction_error()
        raise


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
)
async def health() -> HealthResponse:
    """
    Liveness probe — always returns 200 if the process is running.
    Kubernetes restarts the pod if this returns non-200.
    """
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version if predictor.is_loaded else None,
    )


@app.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness check",
)
async def ready() -> HealthResponse:
    """
    Readiness probe — returns 503 until the model is loaded.
    Kubernetes removes the pod from the load balancer if this
    returns non-200, preventing traffic from hitting an unready pod.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not yet loaded.",
        )

    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_version=predictor.model_version,
    )


# Mount Prometheus metrics endpoint
# Prometheus scrapes this URL every 15 seconds
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
app.mount("/metrics/", metrics_app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serving.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",  # Uvicorn logs — we handle app logs ourselves
    )
