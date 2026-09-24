from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import (
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    OrderInput,
    PredictionResult,
)
from src.logger import get_logger
from src.pipeline import InferencePipeline

logger = get_logger(__name__)


# Singleton: Load the pipeline once at startup
pipeline: InferencePipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the inference pipeline at startup and cleanup at shutdown."""
    global pipeline
    logger.info("Starting up: Loading InferencePipeline...")
    pipeline = InferencePipeline()
    logger.info(f"Pipeline loaded successfully. Model version: {pipeline.model_version}")
    yield
    logger.info("Shutting down: Cleanup complete.")


app = FastAPI(
    title="Olist Late Delivery Prediction API",
    description="Production inference service for predicting late deliveries",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Health check endpoint.
    Returns the service status and the currently loaded model version.
    """
    return HealthResponse(
        status="healthy",
        model_version=pipeline.model_version,
    )


@app.get("/model", response_model=ModelInfoResponse, tags=["System"])
def get_model_info():
    """
    Get information about the currently loaded model.
    Returns the model name, version, and alias.
    """
    model_name, version = pipeline.model_version.split(":")
    return ModelInfoResponse(
        model_name=model_name,
        model_version=pipeline.model_version,
        alias="Production",
        primary_metric="average_precision",
    )


@app.post(
    "/predict",
    response_model=PredictionResult,
    responses={
        400: {"model": ErrorResponse, "description": "Validation failed"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Prediction"],
)
def predict_single(order: OrderInput):
    """
    Predict whether a single order will be delivered late or on time.

    Returns:
        - **prediction**: 0 (on_time) or 1 (late)
        - **label**: "on_time" or "late"
        - **probability**: Probability of being late (0.0 to 1.0)
        - **model_version**: Version of the model used
        - **latency_ms**: Inference time in milliseconds
    """
    # Convert Pydantic model to DataFrame
    df = pd.DataFrame([order.model_dump()])

    # Run inference
    result = pipeline.predict(df)

    # Handle errors
    if result["status"] == "rejected":
        raise HTTPException(status_code=400, detail=result)
    elif result["status"] == "engine_error":
        raise HTTPException(status_code=500, detail=result)

    return PredictionResult(**result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation failed"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Prediction"],
)
def predict_batch(orders: List[OrderInput]):
    """
    Predict whether multiple orders will be delivered late or on time.

    Returns:
        - **predictions**: List of prediction results
        - **latency_ms**: Total batch inference time in milliseconds
    """
    if not orders:
        raise HTTPException(status_code=400, detail={"error": "Empty batch", "status": "rejected"})

    # Convert Pydantic models to DataFrame
    df = pd.DataFrame([order.model_dump() for order in orders])

    # Run inference
    result = pipeline.predict(df)

    # Handle errors
    if result["status"] == "rejected":
        raise HTTPException(status_code=400, detail=result)
    elif result["status"] == "engine_error":
        raise HTTPException(status_code=500, detail=result)

    return BatchPredictionResponse(**result)


# Custom exception handler to return structured error responses
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status": "engine_error",
            "latency_ms": None,
        },
    )


from src.monitoring import setup_monitoring

setup_monitoring(app)
