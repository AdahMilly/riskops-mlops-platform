from typing import Literal

RiskLevel = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
]


def determine_risk_level(
    probability: float,
) -> RiskLevel:
    """Convert fraud probability into an operational risk level."""

    if probability >= 0.70:
        return "HIGH"

    if probability >= 0.30:
        return "MEDIUM"

    return "LOW"
