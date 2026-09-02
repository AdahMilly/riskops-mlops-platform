from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    application_name: str = "riskops"
    environment: str = "development"


settings = Settings()
