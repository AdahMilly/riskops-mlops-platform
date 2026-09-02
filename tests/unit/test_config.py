from riskops.config import settings


def test_application_name():
    assert settings.application_name == "riskops"


def test_default_environment():
    assert settings.environment == "development"
