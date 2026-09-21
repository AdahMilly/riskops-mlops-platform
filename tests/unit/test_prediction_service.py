import numpy as np
import pandas as pd
import pytest

from riskops.serving.schemas import TransactionRequest
from riskops.serving.service import PredictionService


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


def test_prepare_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert "is_fraud" not in features.columns


def test_predict_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(
        service,
        "get_model",
        lambda: FakeModel(),
    )

    probability = service.predict_probability(build_transaction())

    assert probability == pytest.approx(0.80)
