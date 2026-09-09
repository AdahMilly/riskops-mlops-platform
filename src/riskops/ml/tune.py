from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from riskops.ml.evaluate import evaluate_predictions
from riskops.ml.features import prepare_features

ModelBuilder = Callable[[dict[str, Any]], Pipeline]


def tune_model(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    model_builder: ModelBuilder,
    parameter_grid: list[dict[str, Any]],
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate a controlled set of model configurations."""

    x_train, y_train = prepare_features(train_data)
    x_validation, y_validation = prepare_features(validation_data)

    results: list[dict[str, Any]] = []

    for parameters in parameter_grid:
        model = model_builder(parameters)

        model.fit(x_train, y_train)

        probabilities = model.predict_proba(x_validation)[:, 1]

        metrics = evaluate_predictions(
            y_true=y_validation.to_numpy(),
            probabilities=probabilities,
            threshold=threshold,
        )

        results.append(
            {
                "parameters": parameters,
                **metrics,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(
            by=["pr_auc", "recall", "f1"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
