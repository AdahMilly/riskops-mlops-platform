from pathlib import Path

from riskops.data.pipeline import process_transactions


def main() -> None:
    process_transactions(
        input_path=Path("data/raw/transactions.csv"),
        output_path=Path("data/processed/transactions.csv"),
        split_directory=Path("data/splits"),
    )


if __name__ == "__main__":
    main()
