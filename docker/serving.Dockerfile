FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition first — Docker layer caching
# If pyproject.toml hasn't changed, this layer is cached
COPY pyproject.toml .

# Install only production dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["python", "-m", "src.serving.app"]
