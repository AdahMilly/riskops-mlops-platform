from unittest.mock import Mock, patch

import pytest

from riskops.ml.registry import (
    CANDIDATE_ALIAS,
    PRODUCTION_ALIAS,
    ModelPromotionError,
    get_alias_version,
    promote_candidate_to_production,
    rollback_production,
    validate_candidate,
)


def test_alias_constants() -> None:
    assert CANDIDATE_ALIAS == "candidate"
    assert PRODUCTION_ALIAS == "production"


@patch("riskops.ml.registry.get_client")
def test_get_alias_version(mock_get_client) -> None:
    model_version = Mock()
    model_version.version = "3"

    mock_get_client.return_value.get_model_version_by_alias.return_value = model_version

    result = get_alias_version(
        "riskops-fraud-model",
        "candidate",
    )

    assert result == "3"


@patch("riskops.ml.registry.get_client")
def test_get_alias_version_returns_none_when_missing(
    mock_get_client,
) -> None:
    mock_get_client.return_value.get_model_version_by_alias.side_effect = Exception(
        "Alias not found"
    )

    result = get_alias_version(
        "riskops-fraud-model",
        "candidate",
    )

    assert result is None


@patch("riskops.ml.registry.get_client")
def test_validate_candidate_blocks_failed_quality_gate(
    mock_get_client,
) -> None:
    model_version = Mock()
    model_version.version = "1"
    model_version.run_id = "run-123"

    run = Mock()
    run.data.tags = {
        "quality_gate": "failed",
    }

    client = mock_get_client.return_value

    client.get_model_version_by_alias.return_value = model_version
    client.get_model_version.return_value = model_version
    client.get_run.return_value = run

    with pytest.raises(
        ModelPromotionError,
        match="quality gate status 'failed'",
    ):
        validate_candidate("riskops-fraud-model")


@patch("riskops.ml.registry.get_client")
def test_validate_candidate_allows_passed_quality_gate(
    mock_get_client,
) -> None:
    model_version = Mock()
    model_version.version = "2"
    model_version.run_id = "run-456"

    run = Mock()
    run.data.tags = {
        "quality_gate": "passed",
    }

    client = mock_get_client.return_value

    client.get_model_version_by_alias.return_value = model_version
    client.get_model_version.return_value = model_version
    client.get_run.return_value = run

    result = validate_candidate("riskops-fraud-model")

    assert result == "2"


@patch("riskops.ml.registry.get_client")
def test_promote_candidate_to_production(
    mock_get_client,
) -> None:
    model_version = Mock()
    model_version.version = "2"
    model_version.run_id = "run-456"

    run = Mock()
    run.data.tags = {
        "quality_gate": "passed",
    }

    client = mock_get_client.return_value

    client.get_model_version_by_alias.return_value = model_version
    client.get_model_version.return_value = model_version
    client.get_run.return_value = run

    result = promote_candidate_to_production("riskops-fraud-model")

    assert result == "2"

    client.set_registered_model_alias.assert_called_once_with(
        "riskops-fraud-model",
        "production",
        "2",
    )


@patch("riskops.ml.registry.get_client")
def test_promote_candidate_blocks_missing_candidate(
    mock_get_client,
) -> None:
    mock_get_client.return_value.get_model_version_by_alias.side_effect = Exception(
        "Alias not found"
    )

    with pytest.raises(
        ModelPromotionError,
        match="no candidate model",
    ):
        promote_candidate_to_production("riskops-fraud-model")


@patch("riskops.ml.registry.get_client")
def test_rollback_production(
    mock_get_client,
) -> None:
    client = mock_get_client.return_value

    result = rollback_production(
        "riskops-fraud-model",
        "1",
    )

    assert result == "1"

    client.get_model_version.assert_called_once_with(
        "riskops-fraud-model",
        "1",
    )

    client.set_registered_model_alias.assert_called_once_with(
        "riskops-fraud-model",
        "production",
        "1",
    )


@patch("riskops.ml.registry.get_client")
def test_rollback_blocks_unknown_version(
    mock_get_client,
) -> None:
    mock_get_client.return_value.get_model_version.side_effect = Exception("Version not found")

    with pytest.raises(
        ModelPromotionError,
        match="does not exist",
    ):
        rollback_production(
            "riskops-fraud-model",
            "999",
        )
