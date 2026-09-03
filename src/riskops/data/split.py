from pathlib import Path

import pandas as pd


def split_transactions(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1.")

    if "timestamp" not in dataframe.columns:
        raise ValueError("Dataframe must contain a timestamp column.")

    if dataframe.empty:
        raise ValueError("Cannot split an empty dataframe.")

    sorted_data = dataframe.sort_values("timestamp").reset_index(drop=True)

    total_rows = len(sorted_data)
    train_end = int(total_rows * train_ratio)
    validation_end = int(total_rows * (train_ratio + validation_ratio))

    train = sorted_data.iloc[:train_end].copy()
    validation = sorted_data.iloc[train_end:validation_end].copy()
    test = sorted_data.iloc[validation_end:].copy()

    return train, validation, test


def save_splits(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> None:
    train, validation, test = split_transactions(dataframe)

    output_directory.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_directory / "train.csv", index=False)
    validation.to_csv(output_directory / "validation.csv", index=False)
    test.to_csv(output_directory / "test.csv", index=False)
