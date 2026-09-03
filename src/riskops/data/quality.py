from collections.abc import Sequence

import pandas as pd

ALLOWED_CURRENCIES = {"KES", "USD", "EUR", "GBP"}
ALLOWED_MERCHANT_CATEGORIES = {
    "retail",
    "ecommerce",
    "fuel",
    "travel",
    "gaming",
    "electronics",
    "financial",
}
ALLOWED_COUNTRIES = {"KE", "UG", "TZ", "NG", "ZA", "GB", "US"}
ALLOWED_PAYMENT_METHODS = {
    "card",
    "mobile_money",
    "bank_transfer",
}

LEAKAGE_COLUMNS = {
    "fraud_confirmed_at",
    "investigation_outcome",
    "chargeback_date",
    "chargeback_amount",
    "fraud_resolution",
}


def run_quality_checks(
    dataframe: pd.DataFrame,
    leakage_columns: Sequence[str] | None = None,
) -> None:
    """Run business and data-quality checks on transaction data."""

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
        "is_fraud",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if dataframe["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs detected.")

    if dataframe[list(required_columns)].isnull().any().any():
        raise ValueError("Missing values detected in required columns.")

    if not pd.api.types.is_datetime64_any_dtype(dataframe["timestamp"]):
        raise ValueError("timestamp must be a datetime column.")

    if not dataframe["currency"].isin(ALLOWED_CURRENCIES).all():
        raise ValueError("Invalid currency detected.")

    if not dataframe["merchant_category"].isin(ALLOWED_MERCHANT_CATEGORIES).all():
        raise ValueError("Invalid merchant category detected.")

    if not dataframe["country"].isin(ALLOWED_COUNTRIES).all():
        raise ValueError("Invalid country detected.")

    if not dataframe["payment_method"].isin(ALLOWED_PAYMENT_METHODS).all():
        raise ValueError("Invalid payment method detected.")

    if not dataframe["is_fraud"].isin([0, 1]).all():
        raise ValueError("is_fraud must contain only 0 or 1.")

    if not dataframe["is_international"].isin([True, False]).all():
        raise ValueError("is_international must contain only True or False.")

    leakage = LEAKAGE_COLUMNS

    if leakage_columns is not None:
        leakage = leakage.union(leakage_columns)

    detected_leakage = leakage.intersection(dataframe.columns)

    if detected_leakage:
        raise ValueError(f"Potential target leakage detected: {sorted(detected_leakage)}")
