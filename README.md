# Real-Time Fraud Detection MLOps Pipeline

A production-grade machine learning system that detects fraudulent banking
transactions in real time using streaming data, automated retraining, and
full observability.

---

## Architecture
Transactions → Kafka → Feature Pipeline → ML Model → Prediction API
│
Prometheus ← /metrics
│
Grafana
│
Drift Detector
│
Retraining Trigger
│
MLflow Registry

---

## Tech Stack

| Layer | Technology |
|---|---|
| Stream Ingestion | Apache Kafka (KRaft) |
| Feature Engineering | Python, Haversine, Velocity Windows |
| Model Training | XGBoost, Scikit-Learn Pipeline |
| Experiment Tracking | MLflow |
| Model Serving | FastAPI, Uvicorn |
| Data Versioning | DVC |
| Monitoring | Prometheus, Grafana |
| Drift Detection | PSI + Kolmogorov-Smirnov Tests |
| Orchestration | Kubernetes, HPA |
| CI/CD | GitHub Actions |
| Database | PostgreSQL |
| Containerization | Docker, Docker Compose |

---

## Features

- **Stream processing** — Kafka ingests transactions and routes feature vectors
- **17 engineered features** — velocity windows, impossible travel detection, user behaviour baselines
- **Sub-10ms inference** — XGBoost pipeline served via FastAPI
- **Automated drift detection** — PSI + KS tests across all features
- **Auto-retraining** — new model trained, evaluated, and promoted to champion automatically
- **Zero-downtime deploys** — Kubernetes rolling update strategy
- **Full observability** — Prometheus metrics, Grafana dashboards, structured logging
- **CI/CD pipeline** — lint, test, build Docker on every push

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Git

### 1. Clone and install

```bash
git clone https://github.com/your-username/fraud-detection-mlops.git
cd fraud-detection-mlops
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Start all services

```bash
docker compose up -d
```

Services started:
- Kafka at `localhost:9092`
- Kafka UI at `localhost:8080`
- MLflow at `localhost:5000`
- Fraud Detection API at `localhost:8000`
- Prometheus at `localhost:9090`
- Grafana at `localhost:3000`

### 4. Generate training data

```bash
# Terminal 1 — start transaction simulator
python -m src.data.transaction_simulator

# Terminal 2 — run feature pipeline
python -m src.features.pipeline
```

### 5. Train the model

```bash
python -m src.training.train
```

### 6. Score a transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 4500.0,
    "merchant_category_encoded": 1,
    "hour_of_day": 3,
    "day_of_week": 6,
    "is_weekend": 1,
    "is_night": 1,
    "txn_count_1h": 5,
    "txn_amount_sum_1h": 12000.0,
    "txn_count_24h": 12,
    "txn_amount_sum_24h": 25000.0,
    "amount_vs_user_mean": 4.2,
    "user_mean_amount": 80.0,
    "user_txn_count_total": 3,
    "is_high_value_for_user": 1,
    "dist_from_last_txn_km": 8500.0,
    "speed_from_last_txn_kmh": 95000.0,
    "is_impossible_travel": 1
  }'
```

Response:
```json
{
  "fraud_probability": 0.987813,
  "is_fraud": true,
  "model_version": "v2",
  "threshold": 0.5
}
```

---

## Project Structure
fraud-detection-mlops/
├── src/
│   ├── data/           # Transaction simulation and schemas
│   ├── features/       # Kafka-based feature engineering pipeline
│   ├── training/       # Model training, evaluation, data validation
│   ├── serving/        # FastAPI prediction service
│   ├── monitoring/     # Drift detection and Prometheus metrics
│   └── utils/          # Shared logging, config, exceptions
├── tests/
│   ├── unit/           # Fast tests, no infrastructure required
│   └── integration/    # Tests requiring Docker services
├── configs/            # YAML configuration files
├── kubernetes/         # Kubernetes manifests
├── docker/             # Dockerfiles per service
├── mlflow/             # MLflow server Dockerfile
└── .github/workflows/  # CI/CD pipelines

---

## Development

```bash
make install-dev    # Install all dependencies
make lint           # Run ruff linter
make format         # Run black formatter
make test-unit      # Run fast unit tests
make test-int       # Run integration tests (requires Docker)
make docker-up      # Start all services
make docker-down    # Stop all services
```

---

## CI/CD

Every push to `main` triggers:

1. **Lint** — Ruff checks code quality
2. **Format** — Black verifies formatting
3. **Test** — 16 unit tests run against pure Python
4. **Build** — Docker image built and verified

---

## Monitoring

| Metric | Query |
|---|---|
| Request rate | `rate(fraud_prediction_requests_total[5m])` |
| P99 latency | `histogram_quantile(0.99, fraud_prediction_latency_seconds_bucket)` |
| Fraud rate | `rate(fraud_predictions_total[5m]) / rate(fraud_prediction_requests_total[5m])` |
| Model status | `fraud_model_loaded` |

---

## Kubernetes Deployment

```bash
# Start Minikube
minikube start --driver=docker --memory=3000 --cpus=2

# Build image inside Minikube
eval $(minikube docker-env)
docker build -t fraud-detection-serving:latest -f docker/serving.Dockerfile .

# Deploy
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/serving/

# Check status
kubectl get all -n fraud-detection
```

---

## License

MIT
