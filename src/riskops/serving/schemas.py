from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from riskops.investigation.schemas import InvestigationReport


class TransactionRequest(BaseModel):
    transaction_id: str
    customer_id: str

    timestamp: datetime

    amount: float = Field(gt=0)

    currency: str
    merchant_category: str
    country: str
    payment_method: str

    device_id: str

    is_international: bool

    customer_age_days: int = Field(ge=0)

    transactions_last_24h: int = Field(ge=0)

    amount_last_24h: float = Field(ge=0)


class RiskResponse(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_name: str
    model_version: str
    investigation: InvestigationReport | None = None
