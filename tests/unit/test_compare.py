import pandas as pd

from riskops.ml.compare import compare_models
from riskops.ml.model import build_logistic_regression


def test_compare_models_returns_metrics():
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
            "is_fraud": [0, 1, 0, 1],
        }
    )

    results = compare_models(
        train_data=dataframe,
        validation_data=dataframe,
        models={
            "logistic_regression": build_logistic_regression,
        },
    )

    assert "logistic_regression" in results

    metrics = results["logistic_regression"]

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics
