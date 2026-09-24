from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class OrderInput(BaseModel):
    """Schema for a single order input."""
    order_purchase_timestamp: str = Field(..., examples=["2017-08-15 10:30:00"])
    order_approved_at: str = Field(..., examples=["2017-08-15 10:30:00"])
    order_estimated_delivery_date: str = Field(..., examples=["2017-08-25 00:00:00"])
    item_count: int = Field(..., ge=1, examples=[2])
    unique_products: int = Field(..., ge=1, examples=[2])
    unique_sellers: int = Field(..., ge=1, examples=[1])
    total_price: float = Field(..., ge=0, examples=[150.50])
    total_freight_value: float = Field(..., ge=0, examples=[25.00])
    payment_count: int = Field(..., ge=1, examples=[1])
    payment_types_count: int = Field(..., ge=1, examples=[1])
    customer_state: str = Field(..., min_length=2, max_length=2, examples=["SP"])
    customer_city: str = Field(..., examples=["sao paulo"])
    customer_zip_code_prefix: str = Field(..., examples=["01000"])


class PredictionResult(BaseModel):
    """Schema for a single prediction result."""
    prediction: int = Field(..., description="0 = on_time, 1 = late")
    label: Literal["on_time", "late"] = Field(..., description="Human-readable label")
    probability: float = Field(..., ge=0, le=1, description="Probability of being late")
    model_version: str = Field(..., description="Model version from MLflow registry")
    latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")
    status: Literal["success"] = Field(..., description="Status of the prediction")


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    predictions: List[PredictionResult] = Field(..., description="List of predictions")
    latency_ms: float = Field(..., description="Total batch inference latency in milliseconds")
    status: Literal["success"] = Field(..., description="Status of the batch prediction")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str = Field(..., description="Error message")
    status: Literal["rejected", "engine_error"] = Field(..., description="Error status")
    latency_ms: Optional[float] = Field(None, description="Time taken before failure in milliseconds")


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: Literal["healthy"] = Field(..., description="Service health status")
    model_version: str = Field(..., description="Currently loaded model version")


class ModelInfoResponse(BaseModel):
    """Schema for model information response."""
    model_name: str = Field(..., description="Name of the registered model")
    model_version: str = Field(..., description="Version of the loaded model")
    alias: str = Field(..., description="Model alias (e.g., Production)")
    primary_metric: str = Field("average_precision", description="Primary evaluation metric")
