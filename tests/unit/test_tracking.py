from riskops.ml.tracking import EXPERIMENT_NAME


def test_mlflow_experiment_name() -> None:
    assert EXPERIMENT_NAME == "riskops-fraud-detection"
