"""
FastAPI middleware for the fraud detection serving layer.

Middleware sits between the HTTP server and your route handlers.
Every request passes through it — before and after your handler runs.

We use it for:
- Request/response logging with timing
- Adding correlation IDs for distributed tracing
- Catching unhandled exceptions cleanly

In production, this is where you would add:
- Authentication (JWT validation)
- Rate limiting
- Request size limits
"""

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with timing and correlation ID.

    A correlation ID is a unique ID generated per request.
    It is added to every log line so you can trace a single
    request across multiple services in a distributed system.

    Example log output:
        REQUEST  POST /predict | id=abc123
        RESPONSE POST /predict | id=abc123 | 200 | 8.3ms
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Intercept every request, add correlation ID, log timing.

        Args:
            request: Incoming HTTP request.
            call_next: The next handler in the chain (your route function).

        Returns:
            HTTP response with correlation ID header added.
        """
        # Generate a unique ID for this request
        correlation_id = str(uuid.uuid4())[:8]

        # Attach to request state so route handlers can access it
        request.state.correlation_id = correlation_id

        logger.info(
            "REQUEST  %s %s | id=%s",
            request.method,
            request.url.path,
            correlation_id,
        )

        start_time = time.perf_counter()

        # Call the actual route handler
        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "RESPONSE %s %s | id=%s | status=%d | %.1fms",
            request.method,
            request.url.path,
            correlation_id,
            response.status_code,
            duration_ms,
        )

        # Add correlation ID to response headers
        # Callers can log this ID to correlate their logs with ours
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"

        return response
