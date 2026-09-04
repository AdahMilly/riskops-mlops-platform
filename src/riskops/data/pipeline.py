from pathlib import Path

import pandas as pd

from riskops.data.features import build_features
from riskops.data.split import save_splits
from riskops.data.validation import validate_transactions


def process_transactions(
    input_path: Path,
    output_path: Path,
    split_directory: Path | None = None,
) -> None:
    dataframe = pd.read_csv(input_path)

    validated = validate_transactions(dataframe)

    features = build_features(validated)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(output_path, index=False)

    if split_directory is not None:
        save_splits(features, split_directory)
