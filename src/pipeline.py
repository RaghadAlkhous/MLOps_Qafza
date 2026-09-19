import pandas as pd
import time
from typing import Dict, Any

from src.features import create_features
from src.preprocessing import (
    load_artifacts,
    apply_frequency_encoding,
    apply_preprocessor
)
from src.config import CONFIG

class InferencePipeline:
    """
    Production inference pipeline for late delivery prediction.
    Loads fitted artifacts once and reuses them for all predictions.
    """
    
    def __init__(self):
        """Load all artifacts once during initialization."""
        self.artifacts = load_artifacts()
        self.model_version = CONFIG["model"]["version"]
    
    def predict(self, raw_input: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the full inference pipeline on raw input data.
        """
        start_time = time.time()
        
        # Step 1: Build derived features (MUST happen first to keep customer_city/zip)
        df_features = create_features(raw_input)
        
        # Step 2: Apply frequency encoding (city/zip -> frequency values)
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
        
        return {
            "prediction": int(prediction_class),
            "label": "late" if prediction_class == 1 else "on_time",
            "probability": float(prob_late),
            "model_version": self.model_version,
            "latency_ms": round(elapsed_time * 1000, 2)
        }
