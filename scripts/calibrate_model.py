from pathlib import Path

import pandas as pd

from riskops.ml.calibration import (
    calibrate_model,
    evaluate_calibration,
)
from riskops.ml.evaluate import evaluate_predictions
from riskops.ml.model import build_logistic_regression


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return dataframe


def main() -> None:
    train_data = load_dataset(Path("data/splits/train.csv"))

    validation_data = load_dataset(Path("data/splits/validation.csv"))

    base_model = build_logistic_regression(
        {
            "C": 0.1,
        }
    )

    calibrated_model = calibrate_model(
        train_data=train_data,
        model=base_model,
        method="sigmoid",
        cv=5,
    )

    calibration_results = evaluate_calibration(
        calibrated_model,
        validation_data,
    )

    probabilities = calibration_results["probabilities"]

    target = validation_data["is_fraud"].to_numpy()

    metrics = evaluate_predictions(
        y_true=target,
        probabilities=probabilities,
        threshold=0.5,
    )

    metrics["brier_score"] = calibration_results["brier_score"]

    print("Calibrated Logistic Regression")
    print("=" * 80)

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    output_directory = Path("data/processed/calibration")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    probabilities_output = pd.DataFrame(
        {
            "actual": target,
            "fraud_probability": probabilities,
        }
    )

    probabilities_output.to_csv(
        output_directory / "validation_probabilities.csv",
        index=False,
    )

    print(
        "\nSaved:",
        output_directory / "validation_probabilities.csv",
    )


if __name__ == "__main__":
    main()
