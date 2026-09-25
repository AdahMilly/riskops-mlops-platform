from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from riskops.investigation.schemas import (
    InvestigationEvidence,
    InvestigationReport,
    InvestigationRequest,
)
from riskops.investigation.tools import (
    analyze_velocity,
    explain_model_prediction,
    get_customer_profile,
    get_transaction_history,
    match_known_patterns,
)


class InvestigationState(TypedDict, total=False):
    """State carried through the fraud investigation graph."""

    request: InvestigationRequest
    customer_profile: dict[str, object]
    transaction_history: list[dict[str, object]]
    velocity_analysis: dict[str, object]
    model_explanation: dict[str, object]
    known_patterns: list[str]
    report: InvestigationReport


def collect_customer_profile(
    state: InvestigationState,
) -> dict[str, object]:
    """Collect customer account information."""

    request = state["request"]

    return {"customer_profile": get_customer_profile(request.customer_id)}


def collect_transaction_history(
    state: InvestigationState,
) -> dict[str, object]:
    """Collect recent customer transaction history."""

    request = state["request"]

    return {"transaction_history": get_transaction_history(request.customer_id)}


def analyze_transaction_velocity(
    state: InvestigationState,
) -> dict[str, object]:
    """Analyze transaction frequency and amount velocity."""

    request = state["request"]

    return {
        "velocity_analysis": analyze_velocity(
            transactions_last_24h=request.transactions_last_24h,
            amount_last_24h=request.amount_last_24h,
        )
    }


def explain_model(
    state: InvestigationState,
) -> dict[str, object]:
    """Capture structured information about the ML prediction."""

    request = state["request"]

    return {"model_explanation": explain_model_prediction(request.fraud_probability)}


def match_patterns(
    state: InvestigationState,
) -> dict[str, object]:
    """Match transaction characteristics against known risk patterns."""

    request = state["request"]

    return {
        "known_patterns": match_known_patterns(
            is_international=request.is_international,
            transactions_last_24h=request.transactions_last_24h,
            amount=request.amount,
        )
    }


def build_investigation_report(
    state: InvestigationState,
) -> dict[str, InvestigationReport]:
    """Build the final structured investigation report."""

    request = state["request"]

    evidence = InvestigationEvidence(
        customer_profile=state["customer_profile"],
        transaction_history=state["transaction_history"],
        velocity_analysis=state["velocity_analysis"],
        model_explanation=state["model_explanation"],
        known_patterns=state["known_patterns"],
    )

    if request.risk_level == "HIGH":
        recommendation = "ESCALATE"
        rationale = "High-risk transaction requires investigation escalation."
    elif request.risk_level == "MEDIUM":
        recommendation = "REVIEW"
        rationale = "Medium-risk transaction requires manual review."
    else:
        recommendation = "APPROVE"
        rationale = "Low-risk transaction does not require investigation escalation."

    report = InvestigationReport(
        transaction_id=request.transaction_id,
        risk_level=request.risk_level,
        fraud_probability=request.fraud_probability,
        evidence=evidence,
        recommendation=recommendation,
        rationale=rationale,
    )

    return {"report": report}


def build_investigation_graph():
    """Build and compile the deterministic investigation graph."""

    graph = StateGraph(InvestigationState)

    graph.add_node(
        "collect_customer_profile",
        collect_customer_profile,
    )
    graph.add_node(
        "collect_transaction_history",
        collect_transaction_history,
    )
    graph.add_node(
        "analyze_transaction_velocity",
        analyze_transaction_velocity,
    )
    graph.add_node(
        "explain_model",
        explain_model,
    )
    graph.add_node(
        "match_patterns",
        match_patterns,
    )
    graph.add_node(
        "build_investigation_report",
        build_investigation_report,
    )

    graph.add_edge(
        START,
        "collect_customer_profile",
    )
    graph.add_edge(
        "collect_customer_profile",
        "collect_transaction_history",
    )
    graph.add_edge(
        "collect_transaction_history",
        "analyze_transaction_velocity",
    )
    graph.add_edge(
        "analyze_transaction_velocity",
        "explain_model",
    )
    graph.add_edge(
        "explain_model",
        "match_patterns",
    )
    graph.add_edge(
        "match_patterns",
        "build_investigation_report",
    )
    graph.add_edge(
        "build_investigation_report",
        END,
    )

    return graph.compile()
