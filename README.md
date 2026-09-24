# MLOps Task 3 - Production Inference Pipeline

This repository contains the production-ready inference pipeline for the Olist Late Delivery Prediction model.

## Project Structure
- pp/: FastAPI application
- src/: Core inference pipeline modules
- config/: Configuration files
- 	ests/: Unit and integration tests
- equirements/: Pinned dependencies

## Setup
1. Install dependencies: pip install -r requirements/dev.txt
2. Run tests: pytest
3. Start API: uvicorn app.main:app --reload

## Inference
The pipeline loads the fitted transformers and the trained RandomForestClassifier from the rtifacts/ directory (managed by DVC).
