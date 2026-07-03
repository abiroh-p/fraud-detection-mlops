FROM ghcr.io/mlflow/mlflow:v2.13.0

# Install psycopg2 so MLflow can connect to PostgreSQL
# We use the binary version to avoid needing libpq-dev build tools
RUN pip install --no-cache-dir psycopg2-binary==2.9.9
