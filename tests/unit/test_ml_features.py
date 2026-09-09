import pandas as pd

from riskops.data.features import build_features


def test_build_features_creates_time_features():
    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 10:00:00"]),
            "amount": [100.0],
            "amount_last_24h": [500.0],
            "transactions_last_24h": [2],
            "customer_age_days": [100],
            "is_international": [False],
        }
    )

    result = build_features(dataframe)

    assert result["hour"].iloc[0] == 10
    assert result["day_of_week"].iloc[0] == 3


def test_build_features_creates_amount_to_24h_ratio():
    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 10:00:00"]),
            "amount": [100.0],
            "amount_last_24h": [500.0],
            "transactions_last_24h": [2],
            "customer_age_days": [100],
            "is_international": [False],
        }
    )

    result = build_features(dataframe)

    assert result["amount_to_24h_ratio"].iloc[0] == 0.2


def test_build_features_creates_velocity_risk():
    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 10:00:00"]),
            "amount": [100.0],
            "amount_last_24h": [500.0],
            "transactions_last_24h": [10],
            "customer_age_days": [100],
            "is_international": [False],
        }
    )

    result = build_features(dataframe)

    assert result["velocity_risk"].iloc[0] == 1


def test_build_features_creates_high_value_transaction():
    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 10:00:00"]),
            "amount": [1000.0],
            "amount_last_24h": [5000.0],
            "transactions_last_24h": [2],
            "customer_age_days": [100],
            "is_international": [False],
        }
    )

    result = build_features(dataframe)

    assert result["high_value_transaction"].iloc[0] == 1


def test_build_features_creates_behavioral_interaction_features():
    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 10:00:00"],
            ),
            "amount": [1500.0],
            "amount_last_24h": [3000.0],
            "transactions_last_24h": [10],
            "customer_age_days": [100],
            "is_international": [True],
        }
    )

    result = build_features(dataframe)

    assert result["international_velocity_risk"].iloc[0] == 1
    assert result["international_high_value"].iloc[0] == 1
    assert result["transaction_velocity_per_age"].iloc[0] == 0.1
    assert result["amount_share_of_24h"].iloc[0] == 0.5
