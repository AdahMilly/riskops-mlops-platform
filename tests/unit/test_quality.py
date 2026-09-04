import pandas as pd
import pytest

from riskops.data.quality import run_quality_checks


def create_valid_transactions(rows: int = 3) -> pd.DataFrame:
    """Create a valid transaction dataset for quality tests."""

    return pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i:03d}" for i in range(rows)],
            "customer_id": [f"CUST-{i:03d}" for i in range(rows)],
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="h",
            ),
            "amount": [100.0, 250.0, 500.0][:rows],
            "currency": ["KES"] * rows,
            "merchant_category": ["retail"] * rows,
            "country": ["KE"] * rows,
            "payment_method": ["mobile_money"] * rows,
            "device_id": [f"DEVICE-{i:03d}" for i in range(rows)],
            "is_international": [False] * rows,
            "customer_age_days": [500] * rows,
            "transactions_last_24h": [2] * rows,
            "amount_last_24h": [300.0] * rows,
            "is_fraud": [0, 0, 1][:rows],
        }
    )


def test_valid_transactions_pass_quality_checks():
    dataframe = create_valid_transactions()

    run_quality_checks(dataframe)


def test_duplicate_transaction_ids_are_rejected():
    dataframe = create_valid_transactions()
    dataframe.loc[1, "transaction_id"] = dataframe.loc[0, "transaction_id"]

    with pytest.raises(
        ValueError,
        match="Duplicate transaction IDs detected",
    ):
        run_quality_checks(dataframe)


def test_missing_values_are_rejected():
    dataframe = create_valid_transactions()
    dataframe.loc[0, "amount"] = None

    with pytest.raises(
        ValueError,
        match="Missing values detected",
    ):
        run_quality_checks(dataframe)


def test_invalid_currency_is_rejected():
    dataframe = create_valid_transactions()
    dataframe.loc[0, "currency"] = "XYZ"

    with pytest.raises(
        ValueError,
        match="Invalid currency detected",
    ):
        run_quality_checks(dataframe)


def test_invalid_country_is_rejected():
    dataframe = create_valid_transactions()
    dataframe.loc[0, "country"] = "XX"

    with pytest.raises(
        ValueError,
        match="Invalid country detected",
    ):
        run_quality_checks(dataframe)


def test_invalid_payment_method_is_rejected():
    dataframe = create_valid_transactions()
    dataframe.loc[0, "payment_method"] = "crypto"

    with pytest.raises(
        ValueError,
        match="Invalid payment method detected",
    ):
        run_quality_checks(dataframe)


def test_invalid_target_is_rejected():
    dataframe = create_valid_transactions()
    dataframe.loc[0, "is_fraud"] = 2

    with pytest.raises(
        ValueError,
        match="is_fraud must contain only 0 or 1",
    ):
        run_quality_checks(dataframe)


def test_invalid_timestamp_type_is_rejected():
    dataframe = create_valid_transactions()
    dataframe["timestamp"] = dataframe["timestamp"].astype(str)

    with pytest.raises(
        ValueError,
        match="timestamp must be a datetime column",
    ):
        run_quality_checks(dataframe)


def test_target_leakage_is_rejected():
    dataframe = create_valid_transactions()
    dataframe["fraud_confirmed_at"] = pd.Timestamp("2026-01-02")

    with pytest.raises(
        ValueError,
        match="Potential target leakage detected",
    ):
        run_quality_checks(dataframe)


def test_custom_leakage_column_is_rejected():
    dataframe = create_valid_transactions()
    dataframe["internal_fraud_score"] = 0.95

    with pytest.raises(
        ValueError,
        match="Potential target leakage detected",
    ):
        run_quality_checks(
            dataframe,
            leakage_columns=["internal_fraud_score"],
        )
