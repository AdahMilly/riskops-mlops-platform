from pathlib import Path

import pandas as pd

from riskops.ml.features import prepare_features
from riskops.ml.importance import (
    calculate_permutation_importance,
    get_model_coefficients,
)
from riskops.ml.model import (
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)


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

    x_train, y_train = prepare_features(train_data)
    x_validation, y_validation = prepare_features(validation_data)

    models = {
        "logistic_regression": build_logistic_regression(),
        "random_forest": build_random_forest(),
        "xgboost": build_xgboost(),
    }

    output_directory = Path("data/processed/feature_importance")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Model feature importance")
    print("=" * 100)

    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("-" * 100)

        model.fit(x_train, y_train)

        importance = calculate_permutation_importance(
            model=model,
            features=x_validation,
            target=y_validation,
        )

        print(
            importance.to_string(
                index=False,
            )
        )

        importance.to_csv(
            output_directory / f"{model_name}_permutation.csv",
            index=False,
        )

        if model_name == "logistic_regression":
            coefficients = get_model_coefficients(model)

            print("\nLogistic regression coefficients")
            print("-" * 100)

            print(
                coefficients.head(20).to_string(
                    index=False,
                )
            )

            coefficients.to_csv(
                output_directory / "logistic_coefficients.csv",
                index=False,
            )


if __name__ == "__main__":
    main()
