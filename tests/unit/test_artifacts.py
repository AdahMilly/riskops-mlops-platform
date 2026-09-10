import numpy as np
import pandas as pd

from riskops.ml.artifacts import (
    save_confusion_matrix,
    save_metrics,
    save_threshold_analysis,
    save_validation_probabilities,
)


def test_save_confusion_matrix(tmp_path) -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    output_path = tmp_path / "confusion_matrix.png"

    save_confusion_matrix(
        y_true,
        probabilities,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_threshold_analysis(tmp_path) -> None:
    dataframe = pd.DataFrame(
        {
            "threshold": [0.3, 0.5],
            "precision": [0.2, 0.3],
            "recall": [0.8, 0.7],
        }
    )

    output_path = tmp_path / "thresholds.csv"

    save_threshold_analysis(
        dataframe,
        output_path,
    )

    assert output_path.exists()

    saved = pd.read_csv(output_path)

    assert len(saved) == 2
    assert list(saved.columns) == [
        "threshold",
        "precision",
        "recall",
    ]


def test_save_metrics(tmp_path) -> None:
    metrics = {
        "precision": 0.3,
        "recall": 0.7,
        "pr_auc": 0.14,
        "confusion_matrix": [[10, 2], [3, 1]],
    }

    output_path = tmp_path / "metrics.csv"

    save_metrics(
        metrics,
        output_path,
    )

    assert output_path.exists()

    saved = pd.read_csv(output_path)

    assert "precision" in saved.columns
    assert "recall" in saved.columns
    assert "pr_auc" in saved.columns

    assert "confusion_matrix" not in saved.columns


def test_save_validation_probabilities(tmp_path) -> None:
    target = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.2, 0.9])

    output_path = tmp_path / "probabilities.csv"

    save_validation_probabilities(
        target,
        probabilities,
        output_path,
    )

    assert output_path.exists()

    saved = pd.read_csv(output_path)

    assert list(saved.columns) == [
        "actual",
        "fraud_probability",
    ]

    assert len(saved) == 4
