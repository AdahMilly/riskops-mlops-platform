from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from riskops.ml.evaluate import evaluate_predictions
from riskops.ml.features import prepare_features
from riskops.ml.thresholds import analyze_thresholds

ModelBuilder = Callable[[], Pipeline]


def compare_models(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    models: dict[str, ModelBuilder],
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    x_train, y_train = prepare_features(train_data)
    x_validation, y_validation = prepare_features(validation_data)

    results: dict[str, dict[str, Any]] = {}

    for model_name, model_builder in models.items():
        model = model_builder()

        model.fit(x_train, y_train)

        probabilities = model.predict_proba(x_validation)[:, 1]

        metrics = evaluate_predictions(
            y_true=y_validation.to_numpy(),
            probabilities=probabilities,
            threshold=threshold,
        )

        results[model_name] = metrics

    return results


def analyze_model_thresholds(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    models: dict[str, ModelBuilder],
    thresholds: list[float] | None = None,
) -> dict[str, pd.DataFrame]:
    x_train, y_train = prepare_features(train_data)
    x_validation, y_validation = prepare_features(validation_data)

    results: dict[str, pd.DataFrame] = {}

    for model_name, model_builder in models.items():
        model = model_builder()

        model.fit(x_train, y_train)

        probabilities = model.predict_proba(x_validation)[:, 1]

        threshold_results = analyze_thresholds(
            y_true=y_validation.to_numpy(),
            probabilities=probabilities,
            thresholds=thresholds,
        )

        results[model_name] = threshold_results

    return results
