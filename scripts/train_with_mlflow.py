from pathlib import Path

import mlflow
import pandas as pd

from riskops.ml.artifacts import (
    save_confusion_matrix,
    save_metrics,
    save_threshold_analysis,
    save_validation_probabilities,
)
from riskops.ml.calibration import (
    calculate_brier_score,
    calibrate_model,
)
from riskops.ml.evaluate import evaluate_predictions
from riskops.ml.features import prepare_features
from riskops.ml.lineage import get_dataset_lineage
from riskops.ml.model import build_logistic_regression
from riskops.ml.quality_gate import validate_model_quality
from riskops.ml.thresholds import analyze_thresholds
from riskops.ml.tracking import configure_tracking

TRAIN_PATH = Path("data/splits/train.csv")

VALIDATION_PATH = Path("data/splits/validation.csv")

ARTIFACT_DIRECTORY = Path("data/processed/mlflow")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return dataframe


def main() -> None:
    configure_tracking()

    train_data = load_dataset(TRAIN_PATH)

    validation_data = load_dataset(VALIDATION_PATH)

    model_parameters = {
        "C": 0.1,
    }

    calibration_parameters = {
        "method": "sigmoid",
        "cv": 5,
    }

    classification_threshold = 0.5

    model = calibrate_model(
        train_data=train_data,
        model=build_logistic_regression(model_parameters),
        **calibration_parameters,
    )

    validation_features, validation_target = prepare_features(validation_data)

    probabilities = model.predict_proba(validation_features)[:, 1]

    metrics = evaluate_predictions(
        y_true=validation_target.to_numpy(),
        probabilities=probabilities,
        threshold=classification_threshold,
    )

    metrics["brier_score"] = calculate_brier_score(
        validation_target,
        probabilities,
    )

    threshold_results = analyze_thresholds(
        y_true=validation_target.to_numpy(),
        probabilities=probabilities,
    )

    lineage = get_dataset_lineage()

    run_artifact_directory = ARTIFACT_DIRECTORY / "current_run"

    run_artifact_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = run_artifact_directory / "metrics.csv"

    confusion_matrix_path = run_artifact_directory / "confusion_matrix.png"

    probabilities_path = run_artifact_directory / "validation_probabilities.csv"

    threshold_path = run_artifact_directory / "threshold_analysis.csv"

    save_metrics(
        metrics,
        metrics_path,
    )

    save_confusion_matrix(
        validation_target.to_numpy(),
        probabilities,
        confusion_matrix_path,
        threshold=classification_threshold,
    )

    save_validation_probabilities(
        validation_target,
        probabilities,
        probabilities_path,
    )

    save_threshold_analysis(
        threshold_results,
        threshold_path,
    )

    quality_gate_passed = True

    try:
        validate_model_quality(metrics)
    except Exception:
        quality_gate_passed = False

    with mlflow.start_run(run_name="logistic-regression-calibrated") as run:
        mlflow.log_params(
            {
                "model": "logistic_regression",
                "C": model_parameters["C"],
                "calibration_method": (calibration_parameters["method"]),
                "calibration_cv": (calibration_parameters["cv"]),
                "classification_threshold": (classification_threshold),
            }
        )

        mlflow.log_metrics(
            {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier_score": metrics["brier_score"],
            }
        )

        mlflow.set_tags(
            {
                "project": "riskops",
                "model_type": "fraud_detection",
                "dataset": "dvc",
                "git_revision": lineage["git_revision"],
                "dvc_lock_sha256": (lineage["dvc_lock_sha256"]),
                "status": "candidate",
                "quality_gate": ("passed" if quality_gate_passed else "failed"),
            }
        )

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(confusion_matrix_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(probabilities_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(threshold_path),
            artifact_path="evaluation",
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
        )

        print("MLflow run created")
        print("=" * 80)
        print(f"Run ID: {run.info.run_id}")
        print("Experiment: riskops-fraud-detection")
        print(
            "Quality gate:",
            "PASSED" if quality_gate_passed else "FAILED",
        )
        print(f"Git revision: {lineage['git_revision']}")
        print(
            "DVC lock SHA256:",
            lineage["dvc_lock_sha256"],
        )
        print()
        print("Artifacts:")
        print(f"- {metrics_path}")
        print(f"- {confusion_matrix_path}")
        print(f"- {probabilities_path}")
        print(f"- {threshold_path}")


if __name__ == "__main__":
    main()
