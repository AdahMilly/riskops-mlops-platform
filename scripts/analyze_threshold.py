from pathlib import Path

import pandas as pd

from riskops.ml.compare import analyze_model_thresholds
from riskops.ml.model import (
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)
from riskops.ml.thresholds import select_candidate_thresholds


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

    models = {
        "logistic_regression": build_logistic_regression,
        "random_forest": build_random_forest,
        "xgboost": build_xgboost,
    }

    results = analyze_model_thresholds(
        train_data=train_data,
        validation_data=validation_data,
        models=models,
    )

    output_directory = Path("data/processed/thresholds")
    output_directory.mkdir(parents=True, exist_ok=True)

    print("Threshold analysis")
    print("=" * 110)

    for model_name, threshold_results in results.items():
        output_path = output_directory / f"{model_name}.csv"

        threshold_results.to_csv(
            output_path,
            index=False,
        )

        print(f"\nModel: {model_name}")
        print("-" * 110)

        display_columns = [
            "threshold",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "flagged",
            "false_positives",
            "false_negatives",
        ]

        print(threshold_results[display_columns].to_string(index=False))

        candidates = select_candidate_thresholds(
            threshold_results,
            minimum_recall=0.70,
            maximum_false_positive_rate=0.25,
        )

        print("\nProvisional candidate thresholds")
        print("-" * 110)

        if candidates.empty:
            print("No thresholds satisfy the provisional constraints.")
        else:
            print(candidates[display_columns].to_string(index=False))

        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
