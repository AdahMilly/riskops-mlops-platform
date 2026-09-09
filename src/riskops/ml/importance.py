import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def calculate_permutation_importance(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    scoring: str = "average_precision",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate permutation importance for a trained model."""

    result = permutation_importance(
        model,
        features,
        target,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    return importance.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def get_model_coefficients(
    model: Pipeline,
) -> pd.DataFrame:
    """Extract logistic-regression coefficients from a fitted pipeline."""

    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "coef_"):
        raise ValueError("Model classifier does not expose coefficients.")

    preprocessor = model.named_steps["preprocessor"]

    transformed_names = preprocessor.get_feature_names_out()

    coefficients = classifier.coef_[0]

    if len(transformed_names) != len(coefficients):
        raise ValueError("Feature names and coefficients have different lengths.")

    result = pd.DataFrame(
        {
            "feature": transformed_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    )

    return result.sort_values(
        by="absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)
