import pandas as pd

from riskops.data.quality import run_quality_checks
from riskops.data.schema import TransactionSchema


def validate_transactions(dataframe: pd.DataFrame) -> pd.DataFrame:
    validated = TransactionSchema.validate(dataframe)

    if validated["amount"].lt(0).any():
        raise ValueError("Transaction amount cannot be negative.")

    if validated["transactions_last_24h"].lt(0).any():
        raise ValueError("Transaction velocity cannot be negative.")

    if validated["amount_last_24h"].lt(0).any():
        raise ValueError("24-hour transaction amount cannot be negative.")

    if validated["customer_age_days"].lt(0).any():
        raise ValueError("Customer age cannot be negative.")

    run_quality_checks(validated)

    return validated
