import sys
from pathlib import Path
from typing import List

import joblib
import pandas as pd

# Import FrequencyEncoder so joblib can unpickle it
from src.encoders import FrequencyEncoder

sys.modules["__main__"].FrequencyEncoder = FrequencyEncoder

from src.config import CONFIG


def load_artifacts():
    """Load all fitted artifacts from the artifacts directory."""
    artifacts_dir = Path(CONFIG["paths"]["artifacts_dir"])

    artifacts = {
        "city_encoder": joblib.load(
            artifacts_dir / "notebook_05" / "transformers" / "city_frequency_encoder.joblib"
        ),
        "zip_encoder": joblib.load(
            artifacts_dir / "notebook_05" / "transformers" / "zip_frequency_encoder.joblib"
        ),
        "preprocessor": joblib.load(
            artifacts_dir / "notebook_05" / "transformers" / "preprocessor.joblib"
        ),
        "model": joblib.load(artifacts_dir / "notebook_06" / "final_model.joblib"),
    }

    # Load feature list
    import json

    with open(artifacts_dir / "notebook_05" / "feature_list.json", "r") as f:
        artifacts["feature_list"] = json.load(f)

    return artifacts


def apply_frequency_encoding(df: pd.DataFrame, city_encoder, zip_encoder) -> pd.DataFrame:
    """
    Apply frequency encoding to high-cardinality categorical features.
    This must be done BEFORE the main preprocessor.
    """
    df_encoded = df.copy()

    # Apply frequency encoding
    df_encoded["city_frequency"] = city_encoder.transform(df_encoded["customer_city"]).ravel()
    df_encoded["zip_frequency"] = zip_encoder.transform(
        df_encoded["customer_zip_code_prefix"].astype(str)
    ).ravel()

    # Drop original high-cardinality columns
    df_encoded = df_encoded.drop(columns=["customer_city", "customer_zip_code_prefix"])

    return df_encoded


def apply_preprocessor(df: pd.DataFrame, preprocessor, feature_list: List[str]) -> pd.DataFrame:
    """
    Apply the fitted ColumnTransformer and enforce exact feature order.
    """
    # Apply preprocessing
    X_processed = preprocessor.transform(df)

    # Convert to DataFrame and enforce exact feature order
    X_final = pd.DataFrame(X_processed, columns=feature_list, index=df.index)

    return X_final
