import pytest

from riskops.serving.risk import determine_risk_level


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.10, "LOW"),
        (0.29, "LOW"),
        (0.30, "MEDIUM"),
        (0.69, "MEDIUM"),
        (0.70, "HIGH"),
        (0.95, "HIGH"),
    ],
)
def test_determine_risk_level(
    probability: float,
    expected: str,
) -> None:
    assert determine_risk_level(probability) == expected
