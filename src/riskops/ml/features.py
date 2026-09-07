from collections.abc import Sequence

import pandas as pd

TARGET_COLUMN = "is_fraud"

NUMERICAL_FEATURES: tuple[str, ...] = (
    "amount",
    "customer_age_days",
    "transactions_last_24h",
    "amount_last_24h",
    "hour",
    "day_of_week",
    "amount_to_24h_ratio",
    "velocity_risk",
    "high_value_transaction",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "currency",
    "merchant_category",
    "country",
    "payment_method",
)

BOOLEAN_FEATURES: tuple[str, ...] = ("is_international",)

FEATURE_COLUMNS: tuple[str, ...] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES


def prepare_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = set(FEATURE_COLUMNS + (TARGET_COLUMN,)) - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing required ML columns: {sorted(missing_columns)}")

    features = dataframe.loc[:, FEATURE_COLUMNS].copy()
    target = dataframe[TARGET_COLUMN].copy()

    return features, target


def validate_feature_columns(
    dataframe: pd.DataFrame,
    expected_columns: Sequence[str] = FEATURE_COLUMNS,
) -> None:
    actual_columns = set(dataframe.columns)
    expected = set(expected_columns)

    missing = expected - actual_columns

    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
