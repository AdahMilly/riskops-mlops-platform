from riskops.ml.registry import (
    REGISTERED_MODEL_NAME,
    ModelPromotionError,
    get_alias_version,
    promote_candidate_to_production,
)


def main() -> None:
    print("Model promotion")
    print("=" * 80)

    model_name = REGISTERED_MODEL_NAME

    candidate_version = get_alias_version(
        model_name,
        "candidate",
    )

    production_version = get_alias_version(
        model_name,
        "production",
    )

    print(f"Model: {model_name}")
    print(f"Candidate version: {candidate_version}")
    print(f"Current production version: {production_version}")

    if candidate_version is None:
        raise SystemExit("Promotion failed: no candidate model is registered.")

    print("\nValidating candidate...")

    try:
        promoted_version = promote_candidate_to_production(model_name)
    except ModelPromotionError as error:
        print(f"\nPromotion: BLOCKED — {error}")
        raise SystemExit(1) from error

    print("\nPromotion: SUCCESS")
    print(f"Previous production version: {production_version}")
    print(f"New production version: {promoted_version}")


if __name__ == "__main__":
    main()
