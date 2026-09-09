from pathlib import Path
from typing import Any

import pandas as pd

from riskops.ml.model import (
    build_logistic_regression,
    build_random_forest,
)
from riskops.ml.tune import tune_model


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

    logistic_parameters: list[dict[str, Any]] = [
        {"C": 0.1},
        {"C": 0.5},
        {"C": 1.0},
        {"C": 2.0},
    ]

    random_forest_parameters: list[dict[str, Any]] = [
        {
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        {
            "max_depth": 8,
            "min_samples_leaf": 1,
        },
        {
            "max_depth": 12,
            "min_samples_leaf": 1,
        },
        {
            "max_depth": None,
            "min_samples_leaf": 2,
        },
        {
            "max_depth": None,
            "min_samples_leaf": 5,
        },
    ]

    output_directory = Path("data/processed/tuning")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Controlled model tuning")
    print("=" * 110)

    experiments = {
        "logistic_regression": (
            build_logistic_regression,
            logistic_parameters,
        ),
        "random_forest": (
            build_random_forest,
            random_forest_parameters,
        ),
    }

    for model_name, (
        model_builder,
        parameter_grid,
    ) in experiments.items():
        results = tune_model(
            train_data=train_data,
            validation_data=validation_data,
            model_builder=model_builder,
            parameter_grid=parameter_grid,
        )

        output_path = output_directory / f"{model_name}.csv"

        results.to_csv(
            output_path,
            index=False,
        )

        print(f"\n{model_name}")
        print("-" * 110)

        display_columns = [
            "parameters",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]

        print(
            results[display_columns].to_string(
                index=False,
            )
        )

        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
