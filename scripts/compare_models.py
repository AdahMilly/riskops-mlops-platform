from pathlib import Path

import pandas as pd

from riskops.ml.compare import compare_models
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

    models = {
        "logistic_regression": build_logistic_regression,
        "random_forest": build_random_forest,
        "xgboost": build_xgboost,
    }

    results = compare_models(
        train_data=train_data,
        validation_data=validation_data,
        models=models,
    )

    print("Model comparison")
    print("=" * 80)

    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        print("-" * len(model_name))

        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1:        {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"Threshold: {metrics['threshold']:.4f}")
        print(f"Confusion: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
