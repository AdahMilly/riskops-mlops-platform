import os

from fastapi import FastAPI, HTTPException

from riskops.investigation.graph import build_investigation_graph
from riskops.investigation.schemas import InvestigationRequest
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

investigation_graph = build_investigation_graph()


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


@app.post("/predict", response_model=RiskResponse)
def predict(transaction: TransactionRequest) -> RiskResponse:
    """Score a transaction and investigate high-risk transactions."""

    try:
        probability = prediction_service.predict_probability(transaction)
        risk_level = determine_risk_level(probability)

        investigation = None

        if risk_level == "HIGH":
            investigation_request = InvestigationRequest(
                transaction_id=transaction.transaction_id,
                customer_id=transaction.customer_id,
                fraud_probability=probability,
                risk_level=risk_level,
                amount=transaction.amount,
                currency=transaction.currency,
                merchant_category=transaction.merchant_category,
                country=transaction.country,
                payment_method=transaction.payment_method,
                device_id=transaction.device_id,
                is_international=transaction.is_international,
                transactions_last_24h=transaction.transactions_last_24h,
                amount_last_24h=transaction.amount_last_24h,
            )

            result = investigation_graph.invoke(
                {
                    "request": investigation_request,
                }
            )

            investigation = result["report"]

        return RiskResponse(
            fraud_probability=probability,
            risk_level=risk_level,
            model_name=MODEL_NAME,
            model_version=MODEL_ALIAS,
            investigation=investigation,
        )

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error
