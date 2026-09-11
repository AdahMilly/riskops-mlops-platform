from typing import Any

import mlflow
from sklearn.pipeline import Pipeline

EXPERIMENT_NAME = "riskops-fraud-detection"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def configure_tracking(
    experiment_name: str = EXPERIMENT_NAME,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_model_run(
    *,
    model: Pipeline,
    model_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    tags: dict[str, str] | None = None,
) -> str:
    with mlflow.start_run() as run:
        mlflow.log_params(parameters)

        scalar_metrics = {
            key: value for key, value in metrics.items() if isinstance(value, int | float)
        }

        mlflow.log_metrics(scalar_metrics)

        mlflow.set_tag(
            "model_name",
            model_name,
        )

        if tags:
            mlflow.set_tags(tags)

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
        )

        return run.info.run_id
