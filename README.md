# MNIST Production-Style ML Pipeline (Study Project)

A toy problem (MNIST digit classification) wrapped in the **system architecture of a real
production ML product**. Every component here is a scaled-down version of something you
would find at a company running image classification in production (e.g. check-digit
reading at a bank — the original use case for MNIST).

## Architecture

```
                        ┌────────────────────────────────────────────┐
                        │                DEVELOPMENT                 │
                        │                                            │
   torchvision MNIST ──▶│  src/train.py ──▶ MLflow (experiments,     │
                        │        │           model registry)         │
                        │        ▼                                   │
                        │  src/evaluate.py  (QUALITY GATE:           │
                        │        │           beats incumbent? per-   │
                        │        │           digit slice checks?)    │
                        │        ▼                                   │
                        │  src/export_onnx.py ──▶ models/model.onnx  │
                        └────────┬───────────────────────────────────┘
                                 │  promote
                                 ▼
┌─────────────┐   HTTP    ┌──────────────┐   logs preds   ┌──────────────┐
│   Client    │──────────▶│ serving/     │───────────────▶│  PostgreSQL  │
│ (curl, app) │  /predict │ FastAPI+ONNX │                │  metrics +   │
└─────────────┘           └──────────────┘                │  predictions │
                                 ▲                        └──────┬───────┘
                                 │ health, canary                │
                          ┌──────┴───────┐                       ▼
                          │ monitoring/  │                ┌──────────────┐
                          │ drift.py     │◀───────────────│  Streamlit   │
                          │ (PSI on      │    reads       │  dashboard   │
                          │  inputs)     │                └──────────────┘
                          └──────────────┘
```

## Toy → Real-world mapping (the point of this project)

| This repo                          | Production equivalent                              |
|------------------------------------|----------------------------------------------------|
| `src/train.py` + MLflow            | Scheduled training pipeline (Airflow/Kubeflow) logging to a managed registry |
| `src/evaluate.py` quality gate     | CI/CD promotion gate: new model must beat incumbent, incl. per-slice metrics |
| `src/export_onnx.py`               | Model packaging step (ONNX/TorchScript) in the release pipeline |
| `serving/app.py` FastAPI + ONNX RT | TorchServe/Triton behind a load balancer with autoscaling |
| Prediction logging to Postgres     | Inference event stream (Kafka → warehouse)         |
| `monitoring/drift.py` (PSI)        | Evidently/Arize drift monitors triggering retraining alerts |
| `dashboard/app.py` Streamlit       | Grafana + ML observability dashboards              |
| `tests/`                           | Data validation (Great Expectations), contract tests in CI |
| `.github/workflows/ci.yml`         | Full train→gate→deploy CD pipeline                 |
| `docker-compose.yml`               | Kubernetes/ECS + Terraform                         |

## Quickstart

```bash

# 1. Start services: Postgres, MLflow, API, dashboard
docker compose up -d --build

# 2. Install training deps natively (fast inner loop, GPU-capable in WSL2)
pip install -r requirements.txt

# 3. Train — logs metrics to MLflow (localhost:5000) and Postgres
python src/train.py --epochs 3 --lr 0.001 --run-name baseline

# 4. Quality gate — compares this run to the current "incumbent"
python src/evaluate.py --run-name baseline

# 5. Export the promoted model for serving
python src/export_onnx.py --output models/model.onnx
docker compose restart api    # picks up the new model

# 6. Hit the API
python tests/send_test_requests.py          # sends real MNIST test images
curl http://localhost:8000/health

# 7. Watch it all
#    MLflow:    http://localhost:5000   (experiments, registry)
#    Dashboard: http://localhost:8501   (curves, confusion matrix, drift)

# 8. Simulate drift and see the monitor catch it
python monitoring/simulate_drift.py         # sends distorted images
python monitoring/drift.py                  # prints PSI report; dashboard shows it too
```

## Study guide — suggested order

1. **`src/train.py`** — see how every run is tied to params + metrics (reproducibility).
2. **`src/evaluate.py`** — the quality gate. Note the *per-digit slice check*: overall
   accuracy can improve while digit-7 accuracy regresses. Gates catch that.
3. **`serving/app.py`** — model as a service: stable ONNX artifact, health endpoint,
   input validation, prediction logging. The model file is *not* the product; this is.
4. **`monitoring/drift.py`** — PSI (Population Stability Index) on input pixel stats.
   Run the drift simulation and watch the number cross the alert threshold.
5. **`.github/workflows/ci.yml`** — how the gate becomes automatic.

## Ports

| Service   | URL                    |
|-----------|------------------------|
| FastAPI   | http://localhost:8000  |
| MLflow    | http://localhost:5000  |
| Dashboard | http://localhost:8501  |
| Postgres  | localhost:5432         |
