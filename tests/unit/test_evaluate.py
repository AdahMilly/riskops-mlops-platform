import numpy as np

from riskops.ml.evaluate import evaluate_predictions


def test_evaluate_predictions_returns_expected_metrics():
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

    metrics = evaluate_predictions(
        y_true=y_true,
        probabilities=probabilities,
        threshold=0.5,
    )

    assert metrics["threshold"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["confusion_matrix"] == [
        [3, 0],
        [0, 3],
    ]
