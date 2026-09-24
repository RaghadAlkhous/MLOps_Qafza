import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os

import joblib
import mlflow
import mlflow.sklearn

# Critical for unpickling custom sklearn transformers
from src.encoders import FrequencyEncoder

sys.modules["__main__"].FrequencyEncoder = FrequencyEncoder


def register():
    artifacts_dir = Path("/app/artifacts")

    print("Loading model from artifacts...")
    model = joblib.load(artifacts_dir / "notebook_06" / "final_model.joblib")

    print("Loading metadata...")
    with open(artifacts_dir / "notebook_06" / "results_summary.json", encoding="utf-8") as f:
        metadata = json.load(f)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:/app/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Using tracking URI: {tracking_uri}")

    mlflow.set_experiment("olist-late-delivery")

    best_params = metadata.get("best_parameters") or {}
    test_results = metadata.get("test_results") or {}

    print("Starting MLflow run...")
    with mlflow.start_run(run_name="container_register") as run:
        mlflow.log_params({k: str(v) for k, v in best_params.items()})

        test_metrics = {
            f"test_{k}": float(v) for k, v in test_results.items() if isinstance(v, (int, float))
        }
        mlflow.log_metrics(test_metrics)

        print("Logging sklearn model...")
        mlflow.sklearn.log_model(model, "model")

        print("Logging preprocessing artifacts...")
        mlflow.log_artifact(
            str(artifacts_dir / "notebook_05" / "transformers" / "preprocessor.joblib")
        )
        mlflow.log_artifact(
            str(artifacts_dir / "notebook_05" / "transformers" / "city_frequency_encoder.joblib")
        )
        mlflow.log_artifact(
            str(artifacts_dir / "notebook_05" / "transformers" / "zip_frequency_encoder.joblib")
        )
        mlflow.log_artifact(str(artifacts_dir / "notebook_05" / "feature_list.json"))

        model_uri = f"runs:/{run.info.run_id}/model"

    model_name = "olist-late-delivery-rf"
    print(f"Registering model: {model_name}")
    result = mlflow.register_model(model_uri, model_name)

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name=model_name, alias="Production", version=result.version)
    print(f"SUCCESS: Model registered as {model_name} v{result.version} with alias 'Production'")


if __name__ == "__main__":
    register()
