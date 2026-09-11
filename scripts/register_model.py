import sys

from riskops.ml.registry import (
    CANDIDATE_ALIAS,
    REGISTERED_MODEL_NAME,
    register_model_from_run,
    set_candidate_alias,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/register_model.py <run_id>")

    run_id = sys.argv[1]

    model_version = register_model_from_run(
        run_id,
        model_name=REGISTERED_MODEL_NAME,
    )

    set_candidate_alias(
        REGISTERED_MODEL_NAME,
        model_version.version,
    )

    print("Model registered")
    print("=" * 80)
    print(f"Model: {REGISTERED_MODEL_NAME}")
    print(f"Version: {model_version.version}")
    print(f"Alias: {CANDIDATE_ALIAS}")
    print(f"Source run: {run_id}")


if __name__ == "__main__":
    main()
