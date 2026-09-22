from riskops.investigation.tools import (
    analyze_velocity,
    explain_model_prediction,
    get_customer_profile,
    get_transaction_history,
    match_known_patterns,
)


def test_get_customer_profile() -> None:
    profile = get_customer_profile("cust-001")

    assert profile["customer_id"] == "cust-001"
    assert profile["account_status"] == "active"


def test_get_transaction_history() -> None:
    history = get_transaction_history("cust-001")

    assert len(history) == 2
    assert all(transaction["customer_id"] == "cust-001" for transaction in history)


def test_analyze_velocity() -> None:
    result = analyze_velocity(
        transactions_last_24h=12,
        amount_last_24h=6000,
    )

    assert result["high_velocity"] is True
    assert result["high_amount_velocity"] is True


def test_explain_model_prediction() -> None:
    result = explain_model_prediction(0.85)

    assert result["fraud_probability"] == 0.85
    assert result["risk_band"] == "high"


def test_match_known_patterns() -> None:
    patterns = match_known_patterns(
        is_international=True,
        transactions_last_24h=12,
        amount=1500,
    )

    assert "international_transaction" in patterns
    assert "high_transaction_velocity" in patterns
    assert "high_value_transaction" in patterns
