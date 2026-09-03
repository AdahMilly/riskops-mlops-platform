import pandas as pd
import pytest

from riskops.data.split import split_transactions


def create_transactions(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i:03d}" for i in range(rows)],
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="h",
            ),
        }
    )


def test_split_sizes():
    dataframe = create_transactions()

    train, validation, test = split_transactions(dataframe)

    assert len(train) == 70
    assert len(validation) == 15
    assert len(test) == 15


def test_split_is_chronological():
    dataframe = create_transactions()

    train, validation, test = split_transactions(dataframe)

    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()


def test_split_preserves_all_rows():
    dataframe = create_transactions()

    train, validation, test = split_transactions(dataframe)

    combined = pd.concat([train, validation, test])

    assert len(combined) == len(dataframe)
    assert set(combined["transaction_id"]) == set(dataframe["transaction_id"])


def test_invalid_ratios():
    dataframe = create_transactions()
    with pytest.raises(ValueError):
        split_transactions(
            dataframe,
            train_ratio=0.8,
            validation_ratio=0.3,
        )


def test_empty_dataframe():
    dataframe = pd.DataFrame(columns=["transaction_id", "timestamp"])
    with pytest.raises(ValueError):
        split_transactions(dataframe)
