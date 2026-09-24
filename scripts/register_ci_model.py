"""
Register a trained model in MLflow for CI testing.
Creates a dummy model if artifacts are not present.
"""

import json
import os
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "olist-late-delivery-rf"


def register_model():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # Check if Production alias already exists
    try:
        existing_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if any(v.alias == "Production" for v in existing_versions):
            print(f"Model {MODEL_NAME} with Production alias already exists. Skipping.")
            return
    except Exception as e:
        print(f"Checking existing models: {e}")

    artifacts_dir = Path("artifacts")
    notebook_06 = artifacts_dir / "notebook_06"
    notebook_05_transformers = artifacts_dir / "notebook_05" / "transformers"

    # Create dummy artifacts if they don't exist (for CI environment)
    if not notebook_06.exists() or not (notebook_06 / "final_model.joblib").exists():
        print("Artifacts not found. Creating dummy model and artifacts for CI...")

        dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy_model.fit([[0]], [0])

        notebook_06.mkdir(parents=True, exist_ok=True)
        joblib.dump(dummy_model, notebook_06 / "final_model.joblib")

        results = {
            "best_parameters": {},
            "test_results": {"average_precision": 0.5, "roc_auc": 0.5},
        }
        with open(notebook_06 / "results_summary.json", "w") as f:
            json.dump(results, f)

        notebook_05_transformers.mkdir(parents=True, exist_ok=True)
        preprocessor = StandardScaler()
        preprocessor.fit([[0]])
        joblib.dump(preprocessor, notebook_05_transformers / "preprocessor.joblib")
        joblib.dump({}, notebook_05_transformers / "city_frequency_encoder.joblib")
        joblib.dump({}, notebook_05_transformers / "zip_frequency_encoder.joblib")

        feature_list_path = artifacts_dir / "notebook_05" / "feature_list.json"
        feature_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(feature_list_path, "w") as f:
            json.dump([], f)

    # Load model and metadata
    model = joblib.load(notebook_06 / "final_model.joblib")
    with open(notebook_06 / "results_summary.json") as f:
        metadata = json.load(f)

    print(f"Registering model: {MODEL_NAME}")

    with mlflow.start_run(run_name="ci_registration") as run:
        best_params = metadata.get("best_parameters", {})
        mlflow.log_params({k: str(v) for k, v in best_params.items()})

        test_results = metadata.get("test_results", {})
        mlflow.log_metrics(
            {k: float(v) for k, v in test_results.items() if isinstance(v, (int, float))}
        )

        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

        if notebook_05_transformers.exists():
            mlflow.log_artifact(str(notebook_05_transformers / "preprocessor.joblib"))
            mlflow.log_artifact(str(notebook_05_transformers / "city_frequency_encoder.joblib"))
            mlflow.log_artifact(str(notebook_05_transformers / "zip_frequency_encoder.joblib"))

        feature_list_path = artifacts_dir / "notebook_05" / "feature_list.json"
        if feature_list_path.exists():
            mlflow.log_artifact(str(feature_list_path))

        run_id = run.info.run_id

    # Set Production alias
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    new_version = next((v for v in versions if v.run_id == run_id), None)

    if new_version:
        client.set_registered_model_alias(
            name=MODEL_NAME, alias="Production", version=new_version.version
        )
        print(f"Successfully registered {MODEL_NAME} v{new_version.version} with Production alias")
    else:
        print("WARNING: Could not find newly created model version")


if __name__ == "__main__":
    register_model()
