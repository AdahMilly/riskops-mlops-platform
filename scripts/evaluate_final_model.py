from pathlib import Path

import pandas as pd

from riskops.ml.calibration import (
    calculate_brier_score,
    calibrate_model,
)
from riskops.ml.evaluate import evaluate_predictions
from riskops.ml.features import prepare_features
from riskops.ml.model import build_logistic_regression
from riskops.ml.quality_gate import validate_model_quality


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return dataframe


def evaluate_dataset(
    model,
    dataframe: pd.DataFrame,
) -> dict:
    features, target = prepare_features(dataframe)

    probabilities = model.predict_proba(features)[:, 1]

    metrics = evaluate_predictions(
        y_true=target.to_numpy(),
        probabilities=probabilities,
        threshold=0.5,
    )

    metrics["brier_score"] = calculate_brier_score(
        target,
        probabilities,
    )

    return metrics


def print_metrics(
    title: str,
    metrics: dict,
) -> None:
    print(title)
    print("=" * 80)

    for metric, value in metrics.items():
        print(f"{metric}: {value}")


def main() -> None:
    train_data = load_dataset(Path("data/splits/train.csv"))

    validation_data = load_dataset(Path("data/splits/validation.csv"))

    test_data = load_dataset(Path("data/splits/test.csv"))

    model = calibrate_model(
        train_data=train_data,
        model=build_logistic_regression(
            {
                "C": 0.1,
            }
        ),
        method="sigmoid",
        cv=5,
    )

    validation_metrics = evaluate_dataset(
        model,
        validation_data,
    )

    print_metrics(
        "Validation metrics",
        validation_metrics,
    )

    test_metrics = evaluate_dataset(
        model,
        test_data,
    )

    print()
    print_metrics(
        "Final test evaluation",
        test_metrics,
    )

    print()
    print("Quality gate")
    print("=" * 80)

    try:
        validate_model_quality(test_metrics)
        print("PASSED")
    except Exception as error:
        print(f"FAILED — {error}")


if __name__ == "__main__":
    main()
