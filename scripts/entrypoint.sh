#!/bin/bash
set -e

echo "=== Starting Olist Late Delivery Prediction Service ==="

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:/app/mlruns}"

echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"

echo "Checking if registered model is usable..."
if python -c "
import mlflow, os
from mlflow import MlflowClient

mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'file:/app/mlruns'))
client = MlflowClient()
try:
    mv = client.get_model_version_by_alias('olist-late-delivery-rf', 'Production')
    source = str(getattr(mv, 'source', '') or '')
    if source.lower().startswith('c:') or 'c:/' in source.lower():
        print('Model source points to Windows path:', source)
        raise SystemExit(1)
    print(f'Model found in registry: {mv.name} v{mv.version}')
except SystemExit:
    raise
except Exception as e:
    print('Model not found or invalid:', e)
    raise SystemExit(1)
"; then
    echo "Using existing registered model."
else
    echo "Registering model inside container from artifacts..."
    python scripts/register_in_container.py
fi

echo "Starting uvicorn server on port 8000..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
