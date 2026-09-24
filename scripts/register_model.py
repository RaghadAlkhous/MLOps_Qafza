import sys
from pathlib import Path

# Make the project root importable regardless of how this script is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import mlflow.sklearn
import joblib
import json
from src.config import CONFIG

def register():
    artifacts_dir = Path(CONFIG["paths"]["artifacts_dir"])
    
    model = joblib.load(artifacts_dir / "notebook_06" / "final_model.joblib")
    with open(artifacts_dir / "notebook_06" / "results_summary.json") as f:
        metadata = json.load(f)
        
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("olist-late-delivery")
    
    with mlflow.start_run(run_name="register_task2_model") as run:
        mlflow.log_params(metadata["best_parameters"])
        test_metrics = {f"test_{k}": v for k, v in metadata["test_results"].items() if isinstance(v, (int, float))}
        mlflow.log_metrics(test_metrics)
        
        mlflow.sklearn.log_model(model, "model")
        
        mlflow.log_artifact(str(artifacts_dir / "notebook_05" / "transformers" / "preprocessor.joblib"))
        mlflow.log_artifact(str(artifacts_dir / "notebook_05" / "transformers" / "city_frequency_encoder.joblib"))
        mlflow.log_artifact(str(artifacts_dir / "notebook_05" / "transformers" / "zip_frequency_encoder.joblib"))
        mlflow.log_artifact(str(artifacts_dir / "notebook_05" / "feature_list.json"))
        
        model_uri = f"runs:/{run.info.run_id}/model"
        
    model_name = "olist-late-delivery-rf"
    result = mlflow.register_model(model_uri, model_name)
    
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name=model_name, alias="Production", version=result.version)
    print(f" Model registered: {model_name} v{result.version} (Production)")

if __name__ == "__main__":
    register()
