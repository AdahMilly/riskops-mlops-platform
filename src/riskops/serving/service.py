from typing import Any

import mlflow.sklearn
import pandas as pd

from riskops.data.features import build_features
from riskops.data.validation import validate_inference_transaction
from riskops.ml.features import select_features
from riskops.serving.schemas import TransactionRequest


class PredictionService:
    """Service responsible for transaction risk prediction."""

    def __init__(
        self,
        model_name: str,
        model_alias: str,
    ) -> None:
        self.model_name = model_name
        self.model_alias = model_alias
        self._model: Any | None = None

    def load_model(self) -> Any:
        """Load the configured MLflow model."""

        model_uri = f"models:/{self.model_name}@{self.model_alias}"

        return mlflow.sklearn.load_model(model_uri)

    def get_model(self) -> Any:
        """Return the cached model."""

        if self._model is None:
            self._model = self.load_model()

        return self._model

    def prepare_transaction(
        self,
        transaction: TransactionRequest,
    ) -> pd.DataFrame:
        """Validate and transform a transaction for inference."""

        raw_transaction = pd.DataFrame([transaction.model_dump()])

        validated_transaction = validate_inference_transaction(raw_transaction)

        engineered_features = build_features(validated_transaction)

        return select_features(engineered_features)

    def predict_probability(
        self,
        transaction: TransactionRequest,
    ) -> float:
        """Return the model fraud probability."""

        model = self.get_model()

        features = self.prepare_transaction(transaction)

        probability = model.predict_proba(features)[0, 1]

        return float(probability)
