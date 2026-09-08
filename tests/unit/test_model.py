import pandas as pd

from riskops.ml.features import FEATURE_COLUMNS
from riskops.ml.model import build_model_pipeline


def test_model_pipeline_can_fit_and_predict():
    dataframe = pd.DataFrame(
        {
            "amount": [100.0, 1500.0, 200.0, 2500.0],
            "customer_age_days": [100, 20, 200, 10],
            "transactions_last_24h": [2, 15, 1, 20],
            "amount_last_24h": [500.0, 3000.0, 200.0, 5000.0],
            "hour": [10, 23, 14, 2],
            "day_of_week": [1, 5, 2, 6],
            "amount_to_24h_ratio": [0.2, 0.5, 1.0, 0.5],
            "velocity_risk": [0, 1, 0, 1],
            "high_value_transaction": [0, 1, 0, 1],
            "international_velocity_risk": [0, 1, 0, 1],
            "international_high_value": [0, 1, 0, 1],
            "transaction_velocity_per_age": [0.02, 0.75, 0.005, 2.0],
            "amount_share_of_24h": [0.2, 0.5, 1.0, 0.5],
            "currency": ["KES", "USD", "KES", "EUR"],
            "merchant_category": [
                "retail",
                "electronics",
                "fuel",
                "gaming",
            ],
            "country": ["KE", "US", "KE", "GB"],
            "payment_method": [
                "mobile_money",
                "card",
                "bank_transfer",
                "card",
            ],
            "is_international": [False, True, False, True],
        }
    )

    assert list(dataframe.columns) == list(FEATURE_COLUMNS)

    target = pd.Series([0, 1, 0, 1])

    model = build_model_pipeline()

    model.fit(dataframe, target)

    predictions = model.predict(dataframe)
    probabilities = model.predict_proba(dataframe)

    assert len(predictions) == 4
    assert probabilities.shape == (4, 2)
