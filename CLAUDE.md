# MNIST Production-Style Pipeline

Study project: toy MNIST model wrapped in production ML system architecture.

## Environment
- Development happens natively in WSL2 Ubuntu; services run in Docker.
- Postgres from native scripts: `postgresql://mnist:mnist@localhost:5432/experiments`
- Postgres from inside compose network: host is `db`, not `localhost`.
- MLflow UI: http://localhost:5000 · API: http://localhost:8000 · Dashboard: http://localhost:8501

## Commands
- Start services: `docker compose up -d --build`
- Train: `python src/train.py --epochs 3 --run-name <name>`
- Quality gate + promote: `python src/evaluate.py --run-name <name>`
- Export for serving: `python src/export_onnx.py --output models/model.onnx` then `docker compose restart api`
- Populate prediction log: `python tests/send_test_requests.py`
- Drift check: `python monitoring/drift.py` (simulate with `python monitoring/simulate_drift.py`)
- Contract tests (API must be up): `pytest tests/test_api_contract.py -v`

## Conventions
- Serving never imports training code — it only loads the ONNX artifact from models/.
- Input normalization constants (0.1307 / 0.3081) must match between src/train.py and serving/app.py.
- evaluate.py is the only place that sets runs.promoted; never flip it manually.
