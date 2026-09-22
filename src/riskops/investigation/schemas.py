from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
]


class InvestigationRequest(BaseModel):
    transaction_id: str
    customer_id: str
    fraud_probability: float = Field(
        ge=0,
        le=1,
    )
    risk_level: RiskLevel
    amount: float = Field(gt=0)
    currency: str
    merchant_category: str
    country: str
    payment_method: str
    device_id: str
    is_international: bool
    transactions_last_24h: int = Field(
        ge=0,
    )
    amount_last_24h: float = Field(
        ge=0,
    )


class InvestigationEvidence(BaseModel):
    customer_profile: dict[str, object] = Field(
        default_factory=dict,
    )
    transaction_history: list[dict[str, object]] = Field(
        default_factory=list,
    )
    velocity_analysis: dict[str, object] = Field(
        default_factory=dict,
    )
    model_explanation: dict[str, object] = Field(
        default_factory=dict,
    )
    known_patterns: list[str] = Field(
        default_factory=list,
    )


class InvestigationReport(BaseModel):
    transaction_id: str
    risk_level: RiskLevel
    fraud_probability: float
    evidence: InvestigationEvidence
    recommendation: Literal[
        "APPROVE",
        "REVIEW",
        "ESCALATE",
    ]
    rationale: str
