from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

from riskops.ml.features import prepare_features


def build_calibrated_model(
    base_model: Pipeline,
    *,
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("Calibration method must be either 'sigmoid' or 'isotonic'.")

    if cv < 2:
        raise ValueError("cv must be at least 2.")

    return CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv=cv,
    )


def calibrate_model(
    train_data: pd.DataFrame,
    model: Pipeline,
    *,
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    features, target = prepare_features(train_data)

    calibrated_model = build_calibrated_model(
        model,
        method=method,
        cv=cv,
    )

    calibrated_model.fit(features, target)

    return calibrated_model


def calculate_brier_score(
    target: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> float:
    return float(
        brier_score_loss(
            target,
            probabilities,
        )
    )


def evaluate_calibration(
    model: CalibratedClassifierCV,
    validation_data: pd.DataFrame,
) -> dict[str, Any]:
    features, target = prepare_features(validation_data)

    probabilities = model.predict_proba(features)[:, 1]

    return {
        "brier_score": calculate_brier_score(
            target,
            probabilities,
        ),
        "probabilities": probabilities,
    }
