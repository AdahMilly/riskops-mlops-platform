import os
from typing import Any

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

from riskops.data.features import build_features
from riskops.data.validation import (
    validate_inference_transaction,
)
from riskops.ml.features import select_features
from riskops.ml.registry import (
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
)
from riskops.serving.schemas import (
    RiskResponse,
    TransactionRequest,
)

MODEL_NAME = os.getenv(
    "RISKOPS_MODEL_NAME",
    REGISTERED_MODEL_NAME,
)

MODEL_ALIAS = os.getenv(
    "RISKOPS_MODEL_ALIAS",
    PRODUCTION_ALIAS,
)


app = FastAPI(
    title="RiskOps Fraud Detection API",
    description=("Production inference service for transaction fraud-risk scoring."),
    version="1.0.0",
)


_model: Any | None = None


def load_production_model() -> Any:
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    return mlflow.sklearn.load_model(model_uri)


def get_model() -> Any:
    global _model

    if _model is None:
        _model = load_production_model()

    return _model


def determine_risk_level(
    probability: float,
) -> str:
    if probability >= 0.70:
        return "HIGH"

    if probability >= 0.30:
        return "MEDIUM"

    return "LOW"


def prepare_transaction(
    transaction: TransactionRequest,
) -> pd.DataFrame:
    raw_transaction = pd.DataFrame([transaction.model_dump()])

    validated_transaction = validate_inference_transaction(raw_transaction)

    engineered_features = build_features(validated_transaction)

    model_features = select_features(engineered_features)

    return model_features


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
    }


@app.get("/ready")
def readiness() -> dict[str, str]:
    try:
        get_model()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {error}",
        ) from error

    return {
        "status": "ready",
    }


@app.post(
    "/predict",
    response_model=RiskResponse,
)
def predict(
    transaction: TransactionRequest,
) -> RiskResponse:
    try:
        model = get_model()

        features = prepare_transaction(transaction)

        probability = float(model.predict_proba(features)[0, 1])

        risk_level = determine_risk_level(probability)

        return RiskResponse(
            fraud_probability=probability,
            risk_level=risk_level,
            model_name=MODEL_NAME,
            model_version=MODEL_ALIAS,
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error
