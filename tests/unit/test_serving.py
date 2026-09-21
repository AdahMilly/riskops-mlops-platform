from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from riskops.serving.app import app
from riskops.serving.schemas import TransactionRequest
from riskops.serving.service import PredictionService

client = TestClient(app)


def build_transaction() -> TransactionRequest:
    return TransactionRequest(
        transaction_id="txn-001",
        customer_id="cust-001",
        timestamp="2026-01-15T14:30:00",
        amount=1500.0,
        currency="KES",
        merchant_category="electronics",
        country="KE",
        payment_method="card",
        device_id="device-001",
        is_international=False,
        customer_age_days=500,
        transactions_last_24h=3,
        amount_last_24h=5000.0,
    )


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prepare_transaction_builds_model_features() -> None:
    service = PredictionService(
        model_name="riskops-fraud-model",
        model_alias="production",
    )

    features = service.prepare_transaction(build_transaction())

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1
    assert "amount" in features.columns
    assert "hour" in features.columns
    assert "day_of_week" in features.columns
    assert "amount_to_24h_ratio" in features.columns
    assert "velocity_risk" in features.columns
    assert "high_value_transaction" in features.columns
    assert "international_velocity_risk" in features.columns
    assert "international_high_value" in features.columns
    assert "transaction_velocity_per_age" in features.columns
    assert "amount_share_of_24h" in features.columns
    assert "is_fraud" not in features.columns


def test_predict_probability() -> None:
    service = PredictionService(
        model_name="riskops-fraud-model",
        model_alias="production",
    )

    class FakeModel:
        def predict_proba(
            self,
            features: pd.DataFrame,
        ) -> np.ndarray:
            assert len(features) == 1

            return np.array([[0.20, 0.80]])

    with patch.object(
        service,
        "get_model",
        return_value=FakeModel(),
    ):
        probability = service.predict_probability(build_transaction())

    assert probability == 0.80


def test_predict_returns_risk_response() -> None:
    with patch("riskops.serving.app.prediction_service") as mock_service:
        mock_service.predict_probability.return_value = 0.80

        response = client.post(
            "/predict",
            json=build_transaction().model_dump(mode="json"),
        )

    assert response.status_code == 200

    body = response.json()

    assert body["fraud_probability"] == 0.80
    assert body["risk_level"] == "HIGH"
    assert body["model_name"] == "riskops-fraud-model"
    assert body["model_version"] == "production"


def test_predict_returns_medium_risk() -> None:
    with patch("riskops.serving.app.prediction_service") as mock_service:
        mock_service.predict_probability.return_value = 0.50

        response = client.post(
            "/predict",
            json=build_transaction().model_dump(mode="json"),
        )

    assert response.status_code == 200

    body = response.json()

    assert body["fraud_probability"] == 0.50
    assert body["risk_level"] == "MEDIUM"


def test_predict_returns_low_risk() -> None:
    with patch("riskops.serving.app.prediction_service") as mock_service:
        mock_service.predict_probability.return_value = 0.10

        response = client.post(
            "/predict",
            json=build_transaction().model_dump(mode="json"),
        )

    assert response.status_code == 200

    body = response.json()

    assert body["fraud_probability"] == 0.10
    assert body["risk_level"] == "LOW"


def test_readiness_when_model_available() -> None:
    with patch("riskops.serving.app.prediction_service") as mock_service:
        mock_service.get_model.return_value = object()

        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_when_model_unavailable() -> None:
    with patch("riskops.serving.app.prediction_service") as mock_service:
        mock_service.get_model.side_effect = RuntimeError("model unavailable")

        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Model unavailable: model unavailable"}
