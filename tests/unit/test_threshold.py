import numpy as np
import pandas as pd
import pytest

from riskops.ml.thresholds import analyze_thresholds, select_candidate_thresholds


def test_analyze_thresholds_returns_expected_columns():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array(
        [
            0.05,
            0.10,
            0.20,
            0.80,
            0.90,
            0.95,
        ]
    )

    results = analyze_thresholds(
        y_true=y_true,
        probabilities=probabilities,
        thresholds=[0.5],
    )

    assert len(results) == 1
    assert list(results.columns) == [
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "flagged",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]


def test_analyze_thresholds_calculates_perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array(
        [
            0.05,
            0.10,
            0.20,
            0.80,
            0.90,
            0.95,
        ]
    )

    results = analyze_thresholds(
        y_true=y_true,
        probabilities=probabilities,
        thresholds=[0.5],
    )

    row = results.iloc[0]

    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["f1"] == 1.0
    assert row["flagged"] == 3
    assert row["false_positives"] == 0
    assert row["false_negatives"] == 0


def test_analyze_thresholds_rejects_mismatched_lengths():
    with pytest.raises(
        ValueError,
        match="same number of observations",
    ):
        analyze_thresholds(
            y_true=np.array([0, 1]),
            probabilities=np.array([0.5]),
        )


def test_analyze_thresholds_rejects_invalid_probabilities():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        analyze_thresholds(
            y_true=np.array([0, 1]),
            probabilities=np.array([-0.1, 1.1]),
        )


def test_analyze_thresholds_rejects_invalid_thresholds():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        analyze_thresholds(
            y_true=np.array([0, 1]),
            probabilities=np.array([0.2, 0.8]),
            thresholds=[1.5],
        )


def test_analyze_thresholds_counts_confusion_matrix():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array(
        [
            0.10,
            0.20,
            0.80,
            0.30,
            0.70,
            0.90,
        ]
    )

    results = analyze_thresholds(
        y_true=y_true,
        probabilities=probabilities,
        thresholds=[0.5],
    )

    row = results.iloc[0]

    assert row["true_negatives"] == 2
    assert row["false_positives"] == 1
    assert row["false_negatives"] == 1
    assert row["true_positives"] == 2


def test_analyze_thresholds_calculates_false_positive_rate():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array(
        [
            0.10,
            0.20,
            0.80,
            0.30,
            0.70,
            0.90,
        ]
    )

    results = analyze_thresholds(
        y_true=y_true,
        probabilities=probabilities,
        thresholds=[0.5],
    )

    row = results.iloc[0]

    assert row["false_positive_rate"] == 1 / 3


def test_select_candidate_thresholds_applies_constraints():
    results = pd.DataFrame(
        {
            "threshold": [0.20, 0.30, 0.40],
            "precision": [0.08, 0.10, 0.15],
            "recall": [0.80, 0.75, 0.60],
            "f1": [0.14, 0.18, 0.25],
            "false_positive_rate": [0.20, 0.15, 0.10],
        }
    )

    candidates = select_candidate_thresholds(
        results,
        minimum_recall=0.70,
        maximum_false_positive_rate=0.25,
    )

    assert candidates["threshold"].tolist() == [0.30, 0.20]


def test_select_candidate_thresholds_returns_empty_when_no_threshold_matches():
    results = pd.DataFrame(
        {
            "threshold": [0.20, 0.30],
            "precision": [0.08, 0.10],
            "recall": [0.60, 0.65],
            "f1": [0.10, 0.12],
            "false_positive_rate": [0.20, 0.15],
        }
    )

    candidates = select_candidate_thresholds(
        results,
        minimum_recall=0.70,
        maximum_false_positive_rate=0.25,
    )

    assert candidates.empty
