import pandas as pd

from riskops.data.quality import run_quality_checks
from riskops.data.schema import TransactionSchema


def validate_transactions(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    validated = TransactionSchema.validate(dataframe)

    if validated["amount"].lt(0).any():
        raise ValueError("Transaction amount cannot be negative.")

    if validated["transactions_last_24h"].lt(0).any():
        raise ValueError("Transaction velocity cannot be negative.")

    if validated["amount_last_24h"].lt(0).any():
        raise ValueError("24-hour transaction amount cannot be negative.")

    if validated["customer_age_days"].lt(0).any():
        raise ValueError("Customer age cannot be negative.")

    run_quality_checks(
        validated,
        require_target=False,
    )

    return validated


def validate_inference_transaction(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
        "currency",
        "merchant_category",
        "country",
        "payment_method",
        "device_id",
        "is_international",
        "customer_age_days",
        "transactions_last_24h",
        "amount_last_24h",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing required inference columns: {sorted(missing_columns)}")

    if dataframe.empty:
        raise ValueError("Inference transaction cannot be empty.")

    if dataframe[list(required_columns)].isnull().any().any():
        raise ValueError("Missing values detected in inference transaction.")

    if not pd.api.types.is_datetime64_any_dtype(dataframe["timestamp"]):
        raise ValueError("timestamp must be a datetime column.")

    if dataframe["amount"].lt(0).any():
        raise ValueError("Transaction amount cannot be negative.")

    if dataframe["transactions_last_24h"].lt(0).any():
        raise ValueError("Transaction velocity cannot be negative.")

    if dataframe["amount_last_24h"].lt(0).any():
        raise ValueError("24-hour transaction amount cannot be negative.")

    if dataframe["customer_age_days"].lt(0).any():
        raise ValueError("Customer age cannot be negative.")

    run_quality_checks(dataframe.assign(is_fraud=0))

    return dataframe
