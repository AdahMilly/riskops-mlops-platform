from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def analyze_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Sequence[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(
            np.arange(0.05, 0.55, 0.05),
            2,
        )

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if y_true.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must contain the same number of observations.")

    if y_true.size == 0:
        raise ValueError("Cannot analyze thresholds for empty inputs.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be between 0 and 1.")

        predictions = (probabilities >= threshold).astype(int)

        matrix = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )

        true_negatives, false_positives, false_negatives, true_positives = matrix.ravel()

        false_positive_rate = false_positives / (false_positives + true_negatives)

        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "false_positive_rate": false_positive_rate,
                "flagged": int(predictions.sum()),
                "true_negatives": int(true_negatives),
                "false_positives": int(false_positives),
                "false_negatives": int(false_negatives),
                "true_positives": int(true_positives),
            }
        )

    return pd.DataFrame(rows)


def select_candidate_thresholds(
    threshold_results: pd.DataFrame,
    *,
    minimum_recall: float = 0.70,
    maximum_false_positive_rate: float = 0.25,
) -> pd.DataFrame:
    required_columns = {
        "threshold",
        "recall",
        "false_positive_rate",
    }

    missing_columns = required_columns - set(threshold_results.columns)

    if missing_columns:
        raise ValueError(f"Missing required threshold columns: {sorted(missing_columns)}")

    candidates = threshold_results.loc[
        (threshold_results["recall"] >= minimum_recall)
        & (threshold_results["false_positive_rate"] <= maximum_false_positive_rate)
    ].copy()

    return candidates.sort_values(
        by=["f1", "precision"],
        ascending=False,
    ).reset_index(drop=True)
