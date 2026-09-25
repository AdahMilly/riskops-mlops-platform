from typing import Any


def get_customer_profile(
    customer_id: str,
) -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "account_age_days": 500,
        "previous_fraud_cases": 0,
        "account_status": "active",
    }


def get_transaction_history(
    customer_id: str,
) -> list[dict[str, Any]]:
    return [
        {
            "transaction_id": "history-001",
            "customer_id": customer_id,
            "amount": 250.0,
            "currency": "KES",
            "merchant_category": "retail",
            "is_fraud": False,
        },
        {
            "transaction_id": "history-002",
            "customer_id": customer_id,
            "amount": 420.0,
            "currency": "KES",
            "merchant_category": "fuel",
            "is_fraud": False,
        },
    ]


def analyze_velocity(
    transactions_last_24h: int,
    amount_last_24h: float,
) -> dict[str, Any]:
    return {
        "transactions_last_24h": transactions_last_24h,
        "amount_last_24h": amount_last_24h,
        "high_velocity": transactions_last_24h >= 10,
        "high_amount_velocity": amount_last_24h >= 5000,
    }


def explain_model_prediction(
    fraud_probability: float,
) -> dict[str, Any]:
    return {
        "fraud_probability": fraud_probability,
        "risk_band": (
            "high"
            if fraud_probability >= 0.70
            else "medium"
            if fraud_probability >= 0.30
            else "low"
        ),
    }


def match_known_patterns(
    *,
    is_international: bool,
    transactions_last_24h: int,
    amount: float,
) -> list[str]:
    patterns: list[str] = []

    if is_international:
        patterns.append("international_transaction")

    if transactions_last_24h >= 10:
        patterns.append("high_transaction_velocity")

    if amount >= 1000:
        patterns.append("high_value_transaction")

    return patterns
