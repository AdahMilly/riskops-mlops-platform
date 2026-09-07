from typing import Any


class ModelQualityError(Exception):
    def validate_model_quality(
        metrics: dict[str, Any],
        *,
        minimum_recall: float = 0.70,
        minimum_pr_auc: float = 0.20,
    ) -> None:
        recall = metrics["recall"]
        pr_auc = metrics["pr_auc"]

        failures: list[str] = []

        if recall < minimum_recall:
            failures.append(f"recall {recall:.4f} is below minimum {minimum_recall:.4f}")

        if pr_auc < minimum_pr_auc:
            failures.append(f"PR-AUC {pr_auc:.4f} is below minimum {minimum_pr_auc:.4f}")

        if failures:
            raise ModelQualityError("Model quality gate failed: " + "; ".join(failures))
