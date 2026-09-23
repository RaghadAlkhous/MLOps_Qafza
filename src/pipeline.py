import pandas as pd
import time
import json
from typing import Dict, Any

from src.features import create_features
from src.preprocessing import (
    load_artifacts,
    apply_frequency_encoding,
    apply_preprocessor
)
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

class InferencePipeline:
    def __init__(self):
        logger.info("Initializing InferencePipeline and loading artifacts...")
        self.artifacts = load_artifacts()
        self.model_version = CONFIG["model"]["version"]
        logger.info(f"Artifacts loaded successfully. Model version: {self.model_version}")

    def predict(self, raw_input: pd.DataFrame) -> Dict[str, Any]:
        start_time = time.time()
        
        # Log input (truncate if too large)
        input_summary = raw_input.shape[0]
        logger.info(f"Received prediction request for {input_summary} order(s).")

        try:
            # Step 1: Build derived features
            df_features = create_features(raw_input)
            
            # Step 2: Apply frequency encoding
            df_encoded = apply_frequency_encoding(
                df_features,
                self.artifacts["city_encoder"],
                self.artifacts["zip_encoder"]
            )
            
            # Step 3: Apply main preprocessor
            X_final = apply_preprocessor(
                df_encoded,
                self.artifacts["preprocessor"],
                self.artifacts["feature_list"]
            )
            
            # Step 4: Make prediction
            prob_late = self.artifacts["model"].predict_proba(X_final)[0][1]
            prediction_class = self.artifacts["model"].predict(X_final)[0]
            
            elapsed_time = time.time() - start_time
            
            result = {
                "prediction": int(prediction_class),
                "label": "late" if prediction_class == 1 else "on_time",
                "probability": float(prob_late),
                "model_version": self.model_version,
                "latency_ms": round(elapsed_time * 1000, 2)
            }
            
            logger.info(f"Prediction successful: {result}")
            return result

        except KeyError as e:
            error_msg = f"Missing required column in input data: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "status": "failed"}
            
        except Exception as e:
            error_msg = f"Unexpected error during inference: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg, "status": "failed"}
