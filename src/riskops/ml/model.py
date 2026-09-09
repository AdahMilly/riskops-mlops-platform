from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from riskops.ml.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                list(NUMERICAL_FEATURES),
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "boolean",
                "passthrough",
                list(BOOLEAN_FEATURES),
            ),
        ],
        remainder="drop",
    )


def build_logistic_regression(
    parameters: dict[str, Any] | None = None,
) -> Pipeline:
    parameters = parameters or {}

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        C=parameters.get("C", 1.0),
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def build_random_forest(
    parameters: dict[str, Any] | None = None,
) -> Pipeline:
    parameters = parameters or {}

    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_depth=parameters.get("max_depth"),
        min_samples_leaf=parameters.get("min_samples_leaf", 1),
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def build_model_pipeline() -> Pipeline:
    return build_logistic_regression()


def build_xgboost() -> Pipeline:
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
