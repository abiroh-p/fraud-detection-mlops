# Real-Time Fraud Detection MLOps Pipeline

A production-grade machine learning system that detects fraudulent banking transactions in real time.

## Architecture
Transactions → Kafka → Feature Pipeline → ML Model → Prediction API → Monitoring

## Tech Stack

| Layer | Technology |
|---|---|
| Stream Ingestion | Apache Kafka |
| Feature Engineering | Python, Faust |
| Model Training | Scikit-Learn, XGBoost, MLflow |
| Model Serving | FastAPI |
| Data Versioning | DVC |
| Monitoring | Prometheus, Grafana, Evidently |
| Orchestration | Kubernetes |
| CI/CD | GitHub Actions |
| Database | PostgreSQL |

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/fraud-detection-mlops.git
cd fraud-detection-mlops

# 2. Install dependencies
make install-dev

# 3. Start all services
make docker-up

# 4. Run tests
make test
```

## Project Structure
src/

├── data/         # Transaction simulation and schemas

├── features/     # Kafka-based feature engineering pipeline

├── training/     # Model training and evaluation

├── serving/      # FastAPI prediction service

├── monitoring/   # Drift detection and Prometheus metrics

└── utils/        # Shared logging, config, exceptions

## Development

```bash
make lint         # Run linter
make format       # Run formatter
make test-unit    # Run fast unit tests
make test-int     # Run integration tests (requires Docker)
```
