from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from riskops.ml.features import prepare_features
from riskops.ml.model import build_model_pipeline


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return dataframe


def train_model(
    train_path: Path,
    validation_path: Path,
) -> tuple[Pipeline, pd.Series, pd.Series]:
    train_data = load_dataset(train_path)
    validation_data = load_dataset(validation_path)

    x_train, y_train = prepare_features(train_data)
    x_validation, y_validation = prepare_features(validation_data)

    model = build_model_pipeline()

    model.fit(x_train, y_train)

    validation_probabilities = pd.Series(
        model.predict_proba(x_validation)[:, 1],
        index=validation_data.index,
        name="fraud_probability",
    )

    return model, y_validation, validation_probabilities
