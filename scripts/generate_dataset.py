from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
NUM_ROWS = 10_000


def generate_transactions() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    timestamps = pd.date_range(
        start="2026-01-01",
        periods=NUM_ROWS,
        freq="15min",
    )
    amounts = rng.lognormal(
        mean=4.0,
        sigma=1.0,
        size=NUM_ROWS,
    ).round(2)
    transactions_last_24h = rng.poisson(
        lam=4,
        size=NUM_ROWS,
    )
    amount_last_24h = (amounts * rng.uniform(1.0, 10.0, NUM_ROWS)).round(2)
    is_international = rng.choice(
        [True, False],
        size=NUM_ROWS,
        p=[0.2, 0.8],
    )
    high_velocity = transactions_last_24h >= 10
    high_amount = amounts >= 1000
    fraud_probability = 0.01 + high_velocity * 0.20 + high_amount * 0.15 + is_international * 0.05
    fraud_probability = np.clip(
        fraud_probability,
        0,
        0.95,
    )
    is_fraud = (rng.random(NUM_ROWS) < fraud_probability).astype(int)
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN-{index:06d}" for index in range(NUM_ROWS)],
            "customer_id": rng.choice(
                [f"CUST-{i:04d}" for i in range(1000)],
                size=NUM_ROWS,
            ),
            "timestamp": timestamps,
            "amount": amounts,
            "currency": rng.choice(
                ["KES", "USD", "EUR", "GBP"],
                size=NUM_ROWS,
                p=[0.7, 0.15, 0.1, 0.05],
            ),
            "merchant_category": rng.choice(
                [
                    "retail",
                    "ecommerce",
                    "fuel",
                    "travel",
                    "gaming",
                    "electronics",
                    "financial",
                ],
                size=NUM_ROWS,
            ),
            "country": rng.choice(
                ["KE", "UG", "TZ", "NG", "ZA", "GB", "US"],
                size=NUM_ROWS,
                p=[0.55, 0.1, 0.08, 0.08, 0.07, 0.06, 0.06],
            ),
            "payment_method": rng.choice(
                ["card", "mobile_money", "bank_transfer"],
                size=NUM_ROWS,
                p=[0.45, 0.4, 0.15],
            ),
            "device_id": rng.choice(
                [f"DEVICE-{i:04d}" for i in range(2000)],
                size=NUM_ROWS,
            ),
            "is_international": is_international,
            "customer_age_days": rng.integers(
                30,
                3000,
                size=NUM_ROWS,
            ),
            "transactions_last_24h": transactions_last_24h,
            "amount_last_24h": amount_last_24h,
            "is_fraud": is_fraud,
        }
    )


def main() -> None:
    output_path = Path("data/raw/transactions.csv")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dataframe = generate_transactions()
    dataframe.to_csv(
        output_path,
        index=False,
    )
    print(f"Generated {len(dataframe):,} transactions at {output_path}")


if __name__ == "__main__":
    main()
