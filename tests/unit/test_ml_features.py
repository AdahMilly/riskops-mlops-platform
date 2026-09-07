import pandas as pd
import pytest

from riskops.ml.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    prepare_features,
    validate_feature_columns,
)


def test_prepare_features_separates_features_and_target():
    dataframe = pd.DataFrame({column: [0] for column in FEATURE_COLUMNS})
    dataframe["is_fraud"] = [1]

    features, target = prepare_features(dataframe)

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert target.tolist() == [1]


def test_prepare_features_rejects_missing_columns():
    dataframe = pd.DataFrame(
        {
            "amount": [100.0],
            "is_fraud": [0],
        }
    )

    with pytest.raises(ValueError, match="Missing required ML columns"):
        prepare_features(dataframe)


def test_validate_feature_columns_accepts_expected_columns():
    dataframe = pd.DataFrame({column: [0] for column in FEATURE_COLUMNS})

    validate_feature_columns(dataframe)


def test_feature_groups_are_disjoint():
    numerical = set(NUMERICAL_FEATURES)
    categorical = set(CATEGORICAL_FEATURES)

    assert numerical.isdisjoint(categorical)
