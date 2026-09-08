from pathlib import Path

import pandas as pd

from riskops.ml.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return dataframe


def analyze_numerical_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for feature in NUMERICAL_FEATURES:
        legitimate = dataframe.loc[
            dataframe["is_fraud"] == 0,
            feature,
        ]

        fraud = dataframe.loc[
            dataframe["is_fraud"] == 1,
            feature,
        ]

        rows.append(
            {
                "feature": feature,
                "legitimate_mean": legitimate.mean(),
                "fraud_mean": fraud.mean(),
                "legitimate_median": legitimate.median(),
                "fraud_median": fraud.median(),
                "mean_ratio": (fraud.mean() / legitimate.mean() if legitimate.mean() != 0 else 0.0),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by="mean_ratio",
        ascending=False,
    )


def analyze_categorical_features(
    dataframe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}

    for feature in CATEGORICAL_FEATURES + BOOLEAN_FEATURES:
        analysis = (
            dataframe.groupby(feature, dropna=False)["is_fraud"]
            .agg(
                transactions="count",
                fraud_count="sum",
                fraud_rate="mean",
            )
            .reset_index()
            .sort_values(
                by="fraud_rate",
                ascending=False,
            )
        )

        results[feature] = analysis

    return results


def main() -> None:
    dataframe = load_dataset(Path("data/splits/train.csv"))

    output_directory = Path("data/processed/feature_analysis")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Feature diagnostics")
    print("=" * 100)

    print("\nNumerical feature analysis")
    print("-" * 100)

    numerical_results = analyze_numerical_features(dataframe)

    print(
        numerical_results.to_string(
            index=False,
        )
    )

    numerical_results.to_csv(
        output_directory / "numerical_features.csv",
        index=False,
    )

    categorical_results = analyze_categorical_features(dataframe)

    for feature, results in categorical_results.items():
        print(f"\n{feature} fraud rates")
        print("-" * 100)
        print(
            results.to_string(
                index=False,
            )
        )

        results.to_csv(
            output_directory / f"{feature}_fraud_rates.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
