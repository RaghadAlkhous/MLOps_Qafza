import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build derived features from raw input data.
    This function must match exactly what was used in Notebook 05.

    Args:
        df: Raw input DataFrame with columns like order_purchase_timestamp,
            customer_city, etc.

    Returns:
        DataFrame with engineered features ready for preprocessing
    """
    out = pd.DataFrame(index=df.index)

    purchase = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")

    approved = pd.to_datetime(df["order_approved_at"], errors="coerce")

    estimated = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")

    # Temporal features
    out["purchase_year"] = purchase.dt.year

    purchase_month = purchase.dt.month
    out["purchase_month_sin"] = np.sin(2 * np.pi * purchase_month / 12)
    out["purchase_month_cos"] = np.cos(2 * np.pi * purchase_month / 12)

    purchase_dow = purchase.dt.dayofweek
    out["purchase_dow_sin"] = np.sin(2 * np.pi * purchase_dow / 7)
    out["purchase_dow_cos"] = np.cos(2 * np.pi * purchase_dow / 7)

    purchase_hour = purchase.dt.hour
    out["purchase_hour_sin"] = np.sin(2 * np.pi * purchase_hour / 24)
    out["purchase_hour_cos"] = np.cos(2 * np.pi * purchase_hour / 24)

    day_of_week = purchase.dt.dayofweek
    out["is_weekend"] = np.where(day_of_week.isna(), np.nan, (day_of_week >= 5).astype(int))

    # Approval timing
    approval_delay = (approved - purchase).dt.total_seconds() / 3600
    out["approval_delay_hours"] = approval_delay
    out["approval_delay_missing"] = approval_delay.isna().astype(int)

    # Estimated delivery window
    estimated_lead = (estimated - purchase).dt.total_seconds() / 86400
    out["estimated_delivery_lead_days"] = estimated_lead

    # Order structure
    out["item_count"] = df["item_count"]
    out["unique_products"] = df["unique_products"]
    out["unique_sellers"] = df["unique_sellers"]

    # Monetary / shipping
    out["total_price"] = df["total_price"]
    out["total_freight_value"] = df["total_freight_value"]

    out["freight_ratio"] = np.where(
        df["total_price"] > 0, df["total_freight_value"] / df["total_price"], 0.0
    )

    # Payment
    out["payment_count"] = df["payment_count"]
    out["payment_types_count"] = df["payment_types_count"]

    # Geography
    out["customer_state"] = df["customer_state"]
    out["customer_city"] = df["customer_city"]
    out["customer_zip_code_prefix"] = df["customer_zip_code_prefix"].astype("string")

    return out
