from pathlib import Path

from riskops.ml.evaluate import evaluate_predictions
from riskops.ml.train import train_model


def main() -> None:
    model, y_validation, validation_probabilities = train_model(
        train_path=Path("data/splits/train.csv"),
        validation_path=Path("data/splits/validation.csv"),
    )

    metrics = evaluate_predictions(
        y_true=y_validation.to_numpy(),
        probabilities=validation_probabilities.to_numpy(),
    )

    print("Baseline model evaluation")
    print("=" * 32)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"\nModel: {model.__class__.__name__}")


if __name__ == "__main__":
    main()
