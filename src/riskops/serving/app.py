import os

from fastapi import FastAPI, HTTPException

from riskops.ml.registry import (
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
)
from riskops.ml.tracking import configure_mlflow
from riskops.serving.risk import determine_risk_level
from riskops.serving.schemas import (
    RiskResponse,
    TransactionRequest,
)
from riskops.serving.service import PredictionService

MODEL_NAME = os.getenv(
    "RISKOPS_MODEL_NAME",
    REGISTERED_MODEL_NAME,
)

MODEL_ALIAS = os.getenv(
    "RISKOPS_MODEL_ALIAS",
    PRODUCTION_ALIAS,
)

configure_mlflow()

prediction_service = PredictionService(
    model_name=MODEL_NAME,
    model_alias=MODEL_ALIAS,
)


app = FastAPI(
    title="RiskOps Fraud Detection API",
    description=("Production inference service for transaction fraud-risk scoring."),
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""

    return {"status": "healthy"}


@app.get("/ready")
def readiness() -> dict[str, str]:
    """Verify that the production model can be loaded."""

    try:
        prediction_service.get_model()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {error}",
        ) from error

    return {"status": "ready"}


@app.post(
    "/predict",
    response_model=RiskResponse,
)
def predict(
    transaction: TransactionRequest,
) -> RiskResponse:
    """Score a transaction using the production model."""

    try:
        probability = prediction_service.predict_probability(transaction)

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
