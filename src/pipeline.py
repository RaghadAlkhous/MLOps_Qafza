import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import pandas as pd
from mlflow import MlflowClient

from src.config import CONFIG
from src.encoders import FrequencyEncoder
from src.features import create_features
from src.logger import get_logger
from src.preprocessing import apply_frequency_encoding, apply_preprocessor
from src.validation import validate_order_data

sys.modules["__main__"].FrequencyEncoder = FrequencyEncoder

logger = get_logger(__name__)


class InferencePipeline:
    def __init__(self):
        logger.info("Initializing InferencePipeline from MLflow Registry...")

        # Use environment variable or construct relative path
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:/app/mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient()
        model_name = "olist-late-delivery-rf"

        try:
            mv = client.get_model_version_by_alias(model_name, "Production")
            self.model_version = f"{model_name}:{mv.version}"
            run_id = mv.run_id

            # CRITICAL: Construct model URI manually to avoid Windows absolute paths
            # Instead of using the stored artifact location (which has C:/Users/...),
            # we construct a relative path based on the current tracking URI
            model_uri = f"runs:/{run_id}/model"

            logger.info(f"Loading model from URI: {model_uri}")
            self.model = mlflow.sklearn.load_model(model_uri)

            cache_dir = Path(CONFIG["paths"]["models_dir"]) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            client.download_artifacts(run_id, "preprocessor.joblib", str(cache_dir))
            client.download_artifacts(run_id, "city_frequency_encoder.joblib", str(cache_dir))
            client.download_artifacts(run_id, "zip_frequency_encoder.joblib", str(cache_dir))
            client.download_artifacts(run_id, "feature_list.json", str(cache_dir))

            self.artifacts = {
                "city_encoder": joblib.load(cache_dir / "city_frequency_encoder.joblib"),
                "zip_encoder": joblib.load(cache_dir / "zip_frequency_encoder.joblib"),
                "preprocessor": joblib.load(cache_dir / "preprocessor.joblib"),
                "feature_list": json.load(open(cache_dir / "feature_list.json")),
            }
            logger.info(f"Loaded model {self.model_version} from MLflow Registry.")

        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            raise

    def predict(self, raw_input: pd.DataFrame) -> Dict[str, Any]:
        start_time = time.time()
        n_rows = raw_input.shape[0]
        logger.info(f"Received prediction request for {n_rows} order(s).")

        try:
            # Step 0: Validate Input Data
            validation_result = validate_order_data(raw_input)
            if not validation_result["success"]:
                status = validation_result["status"]
                error_msg = f"Data validation failed: {validation_result['details']}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "status": status,
                    "latency_ms": round((time.time() - start_time) * 1000, 2),
                }

            # Step 1: Build derived features
            df_features = create_features(raw_input)

            # Step 2: Apply frequency encoding
            df_encoded = apply_frequency_encoding(
                df_features, self.artifacts["city_encoder"], self.artifacts["zip_encoder"]
            )

            # Step 3: Apply main preprocessor
            X_final = apply_preprocessor(
                df_encoded, self.artifacts["preprocessor"], self.artifacts["feature_list"]
            )

            # Step 4: Make predictions (Vectorized for Batch support)
            proba_late = self.model.predict_proba(X_final)[:, 1].tolist()
            prediction_classes = self.model.predict(X_final).tolist()

            elapsed_time = time.time() - start_time

            # Single row request
            if n_rows == 1:
                result = {
                    "prediction": int(prediction_classes[0]),
                    "label": "late" if prediction_classes[0] == 1 else "on_time",
                    "probability": float(proba_late[0]),
                    "model_version": self.model_version,
                    "latency_ms": round(elapsed_time * 1000, 2),
                    "status": "success",
                }
                logger.info(f"Prediction successful: {result}")
                return result

            # Batch request
            else:
                results = []
                for i in range(n_rows):
                    results.append(
                        {
                            "prediction": int(prediction_classes[i]),
                            "label": "late" if prediction_classes[i] == 1 else "on_time",
                            "probability": float(proba_late[i]),
                            "model_version": self.model_version,
                            "status": "success",
                        }
                    )
                logger.info(f"Batch prediction successful for {n_rows} orders.")
                return {
                    "predictions": results,
                    "latency_ms": round(elapsed_time * 1000, 2),
                    "status": "success",
                }

        except KeyError as e:
            error_msg = f"Missing required column in input data: {str(e)}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "status": "engine_error",
                "latency_ms": round((time.time() - start_time) * 1000, 2),
            }

        except Exception as e:
            error_msg = f"Unexpected error during inference: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "error": error_msg,
                "status": "engine_error",
                "latency_ms": round((time.time() - start_time) * 1000, 2),
            }
