import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from riskops.ml.calibration import (
    build_calibrated_model,
    calculate_brier_score,
)


def build_test_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("classifier", LogisticRegression()),
        ]
    )


def test_build_calibrated_model() -> None:
    base_model = build_test_model()

    model = build_calibrated_model(
        base_model,
        method="sigmoid",
        cv=3,
    )

    assert model.estimator is base_model
    assert model.method == "sigmoid"
    assert model.cv == 3


def test_build_calibrated_model_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="Calibration method"):
        build_calibrated_model(
            build_test_model(),
            method="invalid",
        )


def test_build_calibrated_model_rejects_invalid_cv() -> None:
    with pytest.raises(ValueError, match="cv must be at least 2"):
        build_calibrated_model(
            build_test_model(),
            cv=1,
        )


def test_calculate_brier_score() -> None:
    target = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    score = calculate_brier_score(
        target,
        probabilities,
    )

    assert score == pytest.approx(0.025)
