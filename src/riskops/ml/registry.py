from typing import Any

import mlflow
from mlflow import MlflowClient

REGISTERED_MODEL_NAME = "riskops-fraud-model"

CANDIDATE_ALIAS = "candidate"
PRODUCTION_ALIAS = "production"

QUALITY_GATE_TAG = "quality_gate"
PASSED_STATUS = "passed"


class ModelPromotionError(Exception):
    """Raised when a model cannot be promoted."""


def get_client() -> MlflowClient:
    return MlflowClient()


def register_model_from_run(
    run_id: str,
    *,
    model_name: str = REGISTERED_MODEL_NAME,
) -> Any:
    model_uri = f"runs:/{run_id}/model"

    return mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )


def set_candidate_alias(
    model_name: str,
    version: str,
) -> None:
    client = get_client()

    client.set_registered_model_alias(
        model_name,
        CANDIDATE_ALIAS,
        version,
    )


def get_alias_version(
    model_name: str,
    alias: str,
) -> str | None:
    client = get_client()

    try:
        model_version = client.get_model_version_by_alias(
            model_name,
            alias,
        )
    except Exception:
        return None

    return model_version.version


def validate_candidate(
    model_name: str = REGISTERED_MODEL_NAME,
) -> str:
    client = get_client()

    candidate_version = get_alias_version(
        model_name,
        CANDIDATE_ALIAS,
    )

    if candidate_version is None:
        raise ModelPromotionError("Production promotion blocked: no candidate model is registered.")

    model_version = client.get_model_version(
        model_name,
        candidate_version,
    )

    if not model_version.run_id:
        raise ModelPromotionError(
            "Production promotion blocked: candidate model has no source run."
        )

    run = client.get_run(model_version.run_id)

    quality_gate = run.data.tags.get(QUALITY_GATE_TAG)

    if quality_gate != PASSED_STATUS:
        raise ModelPromotionError(
            "Production promotion blocked: "
            f"candidate version {candidate_version} "
            f"has quality gate status '{quality_gate}'."
        )

    return candidate_version


def promote_candidate_to_production(
    model_name: str = REGISTERED_MODEL_NAME,
) -> str:
    candidate_version = validate_candidate(model_name)

    client = get_client()

    client.set_registered_model_alias(
        model_name,
        PRODUCTION_ALIAS,
        candidate_version,
    )

    return candidate_version


def rollback_production(
    model_name: str,
    target_version: str,
) -> str:
    client = get_client()

    try:
        client.get_model_version(
            model_name,
            target_version,
        )
    except Exception as error:
        raise ModelPromotionError(
            f"Rollback blocked: model version {target_version} does not exist."
        ) from error

    client.set_registered_model_alias(
        model_name,
        PRODUCTION_ALIAS,
        target_version,
    )

    return target_version


def load_model_by_alias(
    model_name: str = REGISTERED_MODEL_NAME,
    alias: str = PRODUCTION_ALIAS,
) -> Any:
    model_uri = f"models:/{model_name}@{alias}"

    return mlflow.pyfunc.load_model(model_uri)
