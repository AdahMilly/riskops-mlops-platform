from datetime import datetime
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from riskops.serving.app import (
    app,
    determine_risk_level,
    prepare_transaction,
)
from riskops.serving.schemas import TransactionRequest

client = TestClient(app)


def build_transaction() -> TransactionRequest:
    return TransactionRequest(
        transaction_id="txn-001",
        customer_id="customer-001",
        timestamp=datetime(
            2026,
            1,
            1,
            14,
            30,
        ),
        amount=1500.00,
        currency="KES",
        merchant_category="retail",
        country="KE",
        payment_method="card",
        device_id="device-001",
        is_international=False,
        customer_age_days=365,
        transactions_last_24h=3,
        amount_last_24h=2500.00,
    )


def test_low_risk() -> None:
    assert determine_risk_level(0.10) == "LOW"


def test_medium_risk() -> None:
    assert determine_risk_level(0.50) == "MEDIUM"


def test_high_risk() -> None:
    assert determine_risk_level(0.90) == "HIGH"


def test_low_boundary() -> None:
    assert determine_risk_level(0.29) == "LOW"


def test_medium_boundary() -> None:
    assert determine_risk_level(0.30) == "MEDIUM"


def test_high_boundary() -> None:
    assert determine_risk_level(0.70) == "HIGH"


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }


def test_readiness_when_model_available() -> None:
    with patch(
        "riskops.serving.app.get_model",
        return_value=object(),
    ):
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
    }


def test_readiness_when_model_unavailable() -> None:
    with patch(
        "riskops.serving.app.get_model",
        side_effect=Exception("model unavailable"),
    ):
        response = client.get("/ready")

    assert response.status_code == 503
    assert "Model unavailable" in response.json()["detail"]


def test_prepare_transaction_builds_model_features() -> None:
    transaction = build_transaction()

    features = prepare_transaction(transaction)

    assert isinstance(
        features,
        pd.DataFrame,
    )

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


def test_predict_returns_risk_response() -> None:
    transaction = build_transaction()

    with patch("riskops.serving.app.get_model") as mocked_get_model:
        mocked_get_model.return_value.predict_proba.return_value = pd.DataFrame(
            [[0.20, 0.80]]
        ).to_numpy()

        response = client.post(
            "/predict",
            json=transaction.model_dump(mode="json"),
        )

    assert response.status_code == 200

    body = response.json()

    assert body["fraud_probability"] == 0.80
    assert body["risk_level"] == "HIGH"
    assert body["model_name"] == "riskops-fraud-model"
    assert body["model_version"] == "production"
