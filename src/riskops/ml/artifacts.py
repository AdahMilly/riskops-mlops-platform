from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(
    y_true,
    probabilities,
    output_path: Path,
    *,
    threshold: float = 0.5,
) -> None:
    predictions = (probabilities >= threshold).astype(int)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
    )

    plt.title(f"Fraud Detection Confusion Matrix (threshold={threshold})")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_threshold_analysis(
    threshold_results: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_results.to_csv(
        output_path,
        index=False,
    )


def save_metrics(
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scalar_metrics = {
        key: value for key, value in metrics.items() if isinstance(value, int | float)
    }

    pd.DataFrame([scalar_metrics]).to_csv(
        output_path,
        index=False,
    )


def save_validation_probabilities(
    target: pd.Series,
    probabilities,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        {
            "actual": target.to_numpy(),
            "fraud_probability": probabilities,
        }
    ).to_csv(
        output_path,
        index=False,
    )
