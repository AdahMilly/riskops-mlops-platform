from riskops.investigation.graph import (
    build_investigation_graph,
)
from riskops.investigation.schemas import InvestigationRequest


def build_request(
    *,
    fraud_probability: float = 0.85,
    risk_level: str = "HIGH",
) -> InvestigationRequest:
    return InvestigationRequest(
        transaction_id="txn-001",
        customer_id="cust-001",
        fraud_probability=fraud_probability,
        risk_level=risk_level,
        amount=1500.0,
        currency="KES",
        merchant_category="retail",
        country="KE",
        payment_method="card",
        device_id="device-001",
        is_international=True,
        transactions_last_24h=12,
        amount_last_24h=6000.0,
    )


def test_investigation_graph_collects_evidence() -> None:
    graph = build_investigation_graph()

    result = graph.invoke(
        {
            "request": build_request(),
        }
    )

    assert result["customer_profile"]["customer_id"] == "cust-001"
    assert len(result["transaction_history"]) == 2

    assert result["velocity_analysis"]["high_velocity"] is True
    assert result["velocity_analysis"]["high_amount_velocity"] is True

    assert result["model_explanation"]["fraud_probability"] == 0.85

    assert "international_transaction" in result["known_patterns"]
    assert "high_transaction_velocity" in result["known_patterns"]
    assert "high_value_transaction" in result["known_patterns"]


def test_high_risk_transaction_is_escalated() -> None:
    graph = build_investigation_graph()

    result = graph.invoke(
        {
            "request": build_request(),
        }
    )

    report = result["report"]

    assert report.transaction_id == "txn-001"
    assert report.risk_level == "HIGH"
    assert report.recommendation == "ESCALATE"


def test_medium_risk_transaction_requires_review() -> None:
    graph = build_investigation_graph()

    result = graph.invoke(
        {
            "request": build_request(
                fraud_probability=0.45,
                risk_level="MEDIUM",
            ),
        }
    )

    assert result["report"].recommendation == "REVIEW"


def test_low_risk_transaction_is_approved() -> None:
    graph = build_investigation_graph()

    result = graph.invoke(
        {
            "request": build_request(
                fraud_probability=0.10,
                risk_level="LOW",
            ),
        }
    )

    assert result["report"].recommendation == "APPROVE"
