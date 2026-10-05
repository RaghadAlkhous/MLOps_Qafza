# Olist Late Delivery Prediction — MLOps Pipeline

End-to-end MLOps project predicting whether an Olist order will be delivered **Late** or **On Time**, covering relational ingestion → feature engineering → model training/registry → containerized API → monitoring → CI/CD.

## Target & Metrics

- Target: `late` (1 = delivered after estimated date, 0 = on time).
- Prediction point: order approval time (leakage-prone features excluded).
- Imbalance: late ≈ 8.11% → primary metric = Average Precision (PR-AUC).
- Model: Random Forest, tuned on validation, evaluated once on test.
- Order-level table: 99,441 rows; stratified split 70/15/15 (`random_state=42`).

| Metric | Test |
|---|---|
| Average Precision | 0.287 |
| ROC-AUC | 0.790 |
| Precision (Late) | 0.352 |
| Recall (Late) | 0.322 |
| F1 (Late) | 0.336 |
| Accuracy | 0.897 |

## Architecture

```
PostgreSQL (Docker)
  → Notebooks 01–06 (read/join → label → split → EDA → features → train/tune/eval)
  → artifacts/ (model + transformers + feature_list, DVC-tracked)
  → MLflow Model Registry (olist-late-delivery-rf, alias: Production)
  → FastAPI Inference Service (Docker)
  → Monitoring (/metrics, /monitoring/stats, /monitoring/drift, prediction logs)
  → CI/CD (GitHub Actions: lint → format → test → build → push to GHCR)
```

## Structure

```
MLOps_Task/
├── app/                  # FastAPI (main.py, schemas.py)
├── src/                  # pipeline, features, preprocessing, encoders, validation, monitoring, logger, config
├── config/               # YAML config
├── sql/                  # schema.sql
├── gx/                   # Great Expectations suites
├── scripts/              # entrypoint.sh, register_in_container.py, register_ci_model.py
├── notebooks/            # 01–06 (outputs cleared)
├── tests/                # pytest (features, validation, api, model)
├── requirements/         # runtime.txt, dev.txt
├── docs/                 # monitoring_alerts.md
├── .github/workflows/    # ci.yml
├── artifacts/            # model + transformers + feature_list (DVC-tracked)
├── Dockerfile
├── docker-compose.yml
├── .pre-commit-config.yaml
├── ruff.toml
├── pytest.ini
└── README.md
```

## Quick Start

```bash
cp .env.example .env
pip install "dvc[s3]" && dvc pull
docker-compose up -d
curl http://localhost:8000/health
```

`artifacts/` is mounted at runtime (not baked into the image). The container auto-registers the model into an internal MLflow registry on first boot, then starts Uvicorn.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + model version |
| GET | `/model` | Model name / alias / version |
| POST | `/predict` | Single-order prediction |
| POST | `/predict/batch` | Batch prediction |
| GET | `/metrics` | Prometheus metrics |
| GET | `/monitoring/stats` | Requests, latency, error rate, distribution, drift |
| GET | `/monitoring/drift` | Drift vs. baseline late-rate |

## Validation

Two layers: Pydantic (types/ranges → 422), then Great Expectations (business rules, e.g. `customer_state` regex → 400).

## Monitoring

Middleware-based, dependency-free. Tracks request count, latency, error rate, prediction distribution, and drift against `EXPECTED_LATE_RATE=0.0811`. Predictions logged to `logs/predictions.jsonl`. Alert thresholds in `docs/monitoring_alerts.md`.

## CI/CD

GitHub Actions on every push/PR: `ruff check` → `ruff format --check` → `pytest` → Docker build & push to GHCR (build gated by `needs: test`). Local gates via pre-commit (`ruff-check` before `ruff-format`). Real-model integration tests gated behind `requires_real_model` (CI uses a clean registry):

```bash
pytest tests/ -v -m "requires_real_model"      # local, real artifacts
pytest tests/ -v -m "not requires_real_model"  # CI-style
```

## Data Versioning

`data/raw/` excluded from VCS. `artifacts/` tracked via DVC (`artifacts.dvc`). Fixed seeds; transformations fitted on training data only.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · MLflow · Great Expectations · FastAPI · Uvicorn · PostgreSQL · Docker · Docker Compose · DVC · GitHub Actions · Ruff · Pytest · Pre-commit

## Reviewer Notes

- `.env` not committed; use `.env.example`.
- `artifacts/` bundled for out-of-the-box run; otherwise `dvc pull`.
- Notebooks delivered with cleared outputs; re-run to regenerate figures.
